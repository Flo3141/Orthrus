#!/usr/bin/env python3
"""
Unified extraction of Orthrus embeddings using precomputed N-tracks archives
and fine-tuned 4-fold cross-validation model checkpoints.

Exclusively operates on precomputed N-track NPZ archives (from generate_tracks.py). A directory with
fold_0..fold_3 yields out-of-fold validation embeddings plus a test ensemble mean. A single checkpoint
(pretrained or one fold) runs through the same code as the only model, shared by all folds.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from util import (
    reset_peak_memory_stats, print_memory_profile, load_metadata, get_meta_filepath,
    normalize_run_config, diff_run_config, resolve_dataset_key, CV_FOLDS, TEST_SPLITS, assign_splits, require_meta,
    TRUNCATION_MODE, truncate_to_max_length, require_truncation_mode,
)


def get_representation(model: torch.nn.Module, x_tensor: torch.Tensor, lengths_tensor: torch.Tensor) -> torch.Tensor:
    """
    Invokes the representation method on the Orthrus backbone or wrapper model.
    """
    if hasattr(model, "representation"):
        return model.representation(x_tensor, lengths_tensor, channel_last=True)
    elif hasattr(model, "backbone") and hasattr(model.backbone, "representation"):
        return model.backbone.representation(x_tensor, lengths_tensor, channel_last=True)
    else:
        raise AttributeError(f"Loaded model ({type(model).__name__}) has no 'representation' method.")


def extract_embeddings_for_tracks(
    tracks: list,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288,
    desc: str = "Extracting representations",
) -> np.ndarray:
    """
    Extracts Orthrus pooled embeddings using dynamic-length batching.
    Tracks are sorted by length to minimize zero-padding overhead.
    Tracks longer than max_length keep their 3' end (identical to finetune_ntrack.py).
    """
    if len(tracks) == 0:
        return np.zeros((0, model.config.ssm_model_dim), dtype=np.float32)

    sample_data = []
    truncated_count = 0

    for idx, tr in enumerate(tracks):
        if tr.shape[0] > max_length:
            truncated_count += 1
            tr = truncate_to_max_length(tr, max_length)

        sample_data.append({
            "orig_idx": idx,
            "track": tr,
            "length": tr.shape[0],
        })

    if truncated_count > 0:
        print(f"Notice: {truncated_count} sequences were truncated to max_length={max_length} nucleotides.")

    # Sort descending by length to detect peak VRAM early and minimize batch padding
    sorted_samples = sorted(sample_data, key=lambda s: s["length"], reverse=True)
    embeddings_list = [None] * len(sample_data)

    for i in tqdm(range(0, len(sorted_samples), batch_size), desc=desc):
        batch = sorted_samples[i : i + batch_size]
        b_lens = [s["length"] for s in batch]
        max_b_len = max(b_lens)
        n_channels = batch[0]["track"].shape[1]

        batch_arr = np.zeros((len(batch), max_b_len, n_channels), dtype=np.float32)
        for b_idx, s in enumerate(batch):
            l = s["length"]
            batch_arr[b_idx, :l, :] = s["track"]

        x_tensor = torch.from_numpy(batch_arr).to(device)
        lengths_tensor = torch.tensor(b_lens, dtype=torch.long, device=device)

        with torch.no_grad():
            batch_emb = get_representation(model, x_tensor, lengths_tensor)
            batch_emb_np = batch_emb.cpu().float().numpy()

        for b_idx, s in enumerate(batch):
            orig_i = s["orig_idx"]
            embeddings_list[orig_i] = batch_emb_np[b_idx]

    return np.stack(embeddings_list, axis=0)


def resolve_checkpoint(model_source: Union[str, Path]) -> str:
    """Returns the directory holding the model weights, or the input unchanged (e.g. a Hugging Face model ID)."""
    source = str(model_source).strip()
    finetuned_dir = Path(source) / "best_finetuned_backbone"
    return str(finetuned_dir) if finetuned_dir.is_dir() else source


def load_fold_backbone(fold_dir: Union[str, Path], device: torch.device, num_channels: Optional[int] = None) -> torch.nn.Module:
    """
    Loads fine-tuned Orthrus backbone checkpoint or base pretrained model.
    Adapts embedding layer dynamically if num_channels does not match base model in_features.
    """
    ckpt_target = resolve_checkpoint(fold_dir)

    print(f"Loading backbone from: {ckpt_target}...")
    model = AutoModel.from_pretrained(ckpt_target, trust_remote_code=True)

    if num_channels is not None:
        emb_layer = getattr(model, "embedding", None)
        if emb_layer is not None and hasattr(emb_layer, "in_features") and emb_layer.in_features != num_channels:
            raise ValueError(
                f"[Error] Checkpoint channel dimension ({emb_layer.in_features}) does not match dataset tracks ({num_channels})!\n"
                f"Please ensure the model was fine-tuned or converted using convert_6track_to_ntrack.py."
            )

    model = model.to(device)
    model.eval()
    return model


def file_fingerprint(path: Path) -> list:
    """[size, mtime_ns] of a file; changes whenever the file is rewritten."""
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def checkpoint_fingerprint(model_source: Union[str, Path]) -> dict:
    """Fingerprints all weight files of a local checkpoint to detect retrained models in the same directory."""
    ckpt_dir = Path(resolve_checkpoint(model_source))
    if not ckpt_dir.is_dir():  # Hugging Face model ID: no local weight files
        return {}
    weight_files = sorted(p for p in ckpt_dir.iterdir() if p.suffix in (".safetensors", ".bin", ".pt"))
    return {p.name: file_fingerprint(p) for p in weight_files}


def build_fold_assignment(sample_splits: np.ndarray) -> np.ndarray:
    """Labels each sample by the source of its embedding: its validation fold, the test ensemble or unassigned."""
    assignment = np.full(len(sample_splits), "unassigned", dtype=object)
    for k, fold in CV_FOLDS.items():
        assignment[np.isin(sample_splits, fold["val"])] = f"fold_{k}"
    assignment[np.isin(sample_splits, TEST_SPLITS)] = "ensemble_mean"
    return assignment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified 4-Fold CV Orthrus Embedding Extraction for Precomputed N-Tracks Archives"
    )
    parser.add_argument(
        "--tracks_file",
        "--data_path",
        dest="tracks_file",
        type=str,
        required=True,
        help="Path to precomputed N-tracks NPZ archive (e.g. 6-track, 7-track, 8-track, required)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned model directory (with fold_0..fold_3) or pretrained base model checkpoint (e.g. 'quietflamingo/orthrus-large-6-track', required)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for model inference (default: 16)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length; longer transcripts keep their 3' end (default: 12288)",
    )
    parser.add_argument(
        "--start_fresh",
        action="store_true",
        default=False,
        help="Ignore existing intermediate fold checkpoint files and restart extraction from scratch",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Precomputed Track Archive
    tracks_path = Path(args.tracks_file)
    if not tracks_path.exists():
        raise FileNotFoundError(f"[Error] Specified tracks archive does not exist: {tracks_path}")

    print("=" * 75)
    print("      ORTHRUS UNIFIED 4-FOLD CV EMBEDDING EXTRACTION PIPELINE       ")
    print("=" * 75)
    print(f"Tracks archive:     {tracks_path}")
    print(f"Model directory:    {args.model_checkpoint}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Max sequence len:   {args.max_length}")

    npz_data = np.load(tracks_path, allow_pickle=True)
    if "tracks" not in npz_data:
        raise KeyError(
            f"[Error] Required field 'tracks' not found in {tracks_path}! "
            f"Available keys: {list(npz_data.files)}"
        )

    tracks = list(npz_data["tracks"])
    n_samples = len(tracks)
    if n_samples == 0:
        raise RuntimeError(f"[Error] Tracks archive {tracks_path} contains 0 samples!")

    n_channels = tracks[0].shape[1]
    # Finetuning and Ridge always use the 'target' field (no alternative target columns)
    if "target" not in npz_data:
        raise KeyError(f"[Error] Required field 'target' not found in {tracks_path}! Available keys: {list(npz_data.files)}")
    targets = npz_data["target"].astype(np.float32)
    targets_raw = npz_data["target_raw"].astype(np.float32) if "target_raw" in npz_data else np.full(n_samples, np.nan, dtype=np.float32)
    tx_ids = npz_data["ensembl_transcript_id"].astype(str) if "ensembl_transcript_id" in npz_data else np.array([""] * n_samples)
    gene_ids = npz_data["ensembl_gene_id"].astype(str) if "ensembl_gene_id" in npz_data else np.array([""] * n_samples)
    genes = npz_data["gene"].astype(str) if "gene" in npz_data else np.array([""] * n_samples)
    seq_lens = npz_data["seq_lens"].astype(np.int32) if "seq_lens" in npz_data else np.array([len(t) for t in tracks], dtype=np.int32)

    # Load companion metadata strictly (no fallbacks)
    tracks_meta = load_metadata(tracks_path, required=True)

    require_meta(tracks_meta, tracks_path, "dataset", "track_name")
    dataset_name = resolve_dataset_key(tracks_meta["dataset"])
    track_name = str(tracks_meta["track_name"]).strip()

    normalization = str(tracks_meta.get("normalization", "none")).strip().lower()
    require_truncation_mode(tracks_meta, tracks_path)

    print(f"\n[Tracks Info] Loaded {n_samples:,} transcripts with {n_channels} channels.")
    print(f"Dataset identity:   {dataset_name}")
    print(f"Track identity:     {track_name} (Normalization: {normalization.upper()})")

    # 2. Resolve Model Checkpoint & Architecture Mode
    model_checkpoint_str = str(args.model_checkpoint).strip()
    is_hf_model = model_checkpoint_str == "quietflamingo/orthrus-large-6-track" or (
        "/" in model_checkpoint_str and not Path(model_checkpoint_str).exists()
    )

    if is_hf_model:
        is_4fold_dir = False
        model_identifier = model_checkpoint_str
        print(f"\n[Model Detection] Pretrained base model: {model_checkpoint_str}")
    else:
        model_path = Path(model_checkpoint_str)
        if not model_path.exists():
            raise FileNotFoundError(f"[Error] Model checkpoint path not found: {model_path}")

        fold_subdirs = [model_path / f"fold_{k}" for k in CV_FOLDS]
        if all(p.is_dir() for p in fold_subdirs):
            is_4fold_dir = True
            model_identifier = str(model_path)
            print(f"\n[Model Detection] 4-Fold fine-tuned model directory: {model_path}")
        else:
            is_4fold_dir = False
            model_identifier = str(model_path)
            print(f"\n[Model Detection] Single checkpoint directory (shared across all folds): {model_path}")

    # Each model with the folds whose validation splits it embeds: one model per fold, or one shared by all folds
    if is_4fold_dir:
        model_jobs = [(fold_subdirs[k], [k]) for k in CV_FOLDS]
    else:
        model_jobs = [(model_identifier, list(CV_FOLDS))]

    # 3. Resolve Destination File Path strictly from metadata
    # The model was finetuned before if it has 4 Folds or is a fine-tuned fold checkpoint; else it was a base pretrained model
    is_finetuned = is_4fold_dir or (not is_hf_model and ("best_finetuned_backbone" in str(model_path) or "fold_" in str(model_path)))
    var_str = "finetuned" if is_finetuned else "pretrained"
    norm_str = f"_{normalization}" if normalization and normalization != "none" else ""
    save_file = tracks_path.parent / f"{dataset_name}_{track_name}_{var_str}{norm_str}_embeddings.npz"

    save_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = save_file.with_name(f"{save_file.stem}_checkpoint.npz")
    print(f"Destination file:   {save_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing device:   {device}")
    reset_peak_memory_stats(device)

    # 4. Resolve Standardized 10-Fold Splits Lookup Table
    sample_splits = assign_splits(dataset_name, tx_ids)

    fold_assignment = build_fold_assignment(sample_splits)

    test_idx = np.where(np.isin(sample_splits, TEST_SPLITS))[0]
    test_tracks = [tracks[i] for i in test_idx]

    # Restored from the checkpoint, or allocated once the first model provides the embedding dimension
    all_embeddings = None
    test_accum = None

    print(f"\nCV Setup: Total Samples = {n_samples:,}, Test Samples {TEST_SPLITS} = {len(test_idx):,}, Models = {len(model_jobs)}")
    n_unmatched = int(np.sum(sample_splits == -1))
    if n_unmatched > 0:
        print(f"[Notice] {n_unmatched:,} samples had no split assignment in lookup table; their embeddings stay zero.")

    # 5. Execute Embedding Extraction (with Resumable Checkpoint)
    # The run configuration (input tracks + model weights) is stored in the checkpoint, so that a
    # resume never mixes embeddings of different track archives or retrained fold models.
    run_config = normalize_run_config({
        "tracks_file": tracks_path.resolve(),
        "tracks_file_fingerprint": file_fingerprint(tracks_path),
        "model_checkpoint": model_identifier,
        "fold_checkpoints": {f"fold_{k}": checkpoint_fingerprint(model_source) for model_source, folds in model_jobs for k in folds},
        "max_length": args.max_length,
        "truncation": TRUNCATION_MODE,
    })

    completed_folds = set()
    if not args.start_fresh and checkpoint_file.exists():
        print(f"[*] Found existing checkpoint: {checkpoint_file}")
        ckpt_data = None
        try:
            with np.load(checkpoint_file, allow_pickle=True) as ckpt:
                ckpt_data = {k: ckpt[k].copy() for k in ckpt.files}
        except Exception as e:
            print(f"[!] Warning: Failed to load checkpoint ({e}). Starting folds from scratch.")

        if ckpt_data is not None:
            if "run_config" not in ckpt_data:
                raise RuntimeError(
                    f"[Error] Checkpoint '{checkpoint_file}' has no run configuration and cannot be verified. "
                    f"Please rerun with --start_fresh."
                )
            diffs = diff_run_config(json.loads(str(ckpt_data["run_config"])), run_config)
            if diffs:
                diff_str = "\n".join(f"  - {k}: existing={v[0]!r} | current={v[1]!r}" for k, v in diffs.items())
                raise RuntimeError(
                    f"[Error] Checkpoint '{checkpoint_file}' was created with a different configuration:\n{diff_str}\n"
                    f"Resuming would mix both runs. Please rerun with --start_fresh."
                )
            all_embeddings = ckpt_data["embeddings"]
            test_accum = ckpt_data["test_accum"]
            completed_folds = set(ckpt_data["completed_folds"].tolist())
            print(f"[*] Resuming: Run configuration matches, folds {sorted(completed_folds)} already completed.")

    for model_source, folds in model_jobs:
        label = f"Fold {'+'.join(str(k) for k in folds)}"
        if completed_folds.issuperset(folds):
            print(f"\n--- [{label}] Already completed in checkpoint. Skipping ---")
            continue

        val_splits = [s for k in folds for s in CV_FOLDS[k]["val"]]
        val_idx = np.where(np.isin(sample_splits, val_splits))[0]
        val_tracks = [tracks[i] for i in val_idx]

        print(f"\n--- [{label}] Loading model from: {model_source} ---")
        model = load_fold_backbone(model_source, device, num_channels=n_channels)

        if all_embeddings is None:
            embedding_dim = model.config.ssm_model_dim
            all_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
            test_accum = np.zeros((len(test_idx), embedding_dim), dtype=np.float32)

        print(f"[{label}] Extracting Val Embeddings (Splits {val_splits}, {len(val_idx):,} samples)...")
        all_embeddings[val_idx] = extract_embeddings_for_tracks(
            tracks=val_tracks,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            desc=f"{label} Val",
        )

        print(f"[{label}] Extracting Test Set Embeddings (Splits {TEST_SPLITS}, {len(test_idx):,} samples)...")
        test_accum += extract_embeddings_for_tracks(
            tracks=test_tracks,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            desc=f"{label} Test",
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save checkpoint while further folds are pending
        completed_folds.update(folds)
        if len(completed_folds) < len(CV_FOLDS):
            ckpt_dict = {
                "embeddings": all_embeddings,
                "test_accum": test_accum,
                "completed_folds": np.array(list(completed_folds), dtype=np.int32),
                "run_config": json.dumps(run_config, sort_keys=True),
            }
            tmp_ckpt = checkpoint_file.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_ckpt, **ckpt_dict)
            tmp_ckpt.replace(checkpoint_file)
            print(f"[✓] Checkpoint saved after {label} (completed: {sorted(completed_folds)}).")

    # Clean up temporary checkpoint upon full completion
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except OSError:
            pass

    # Test Set: mean over all models (with a single model, its embeddings unchanged)
    print(f"\nComputing Test Set {TEST_SPLITS} Embeddings as mean over {len(model_jobs)} model(s)...")
    all_embeddings[test_idx] = test_accum / len(model_jobs)

    # 6. Save Standardized Output NPZ Archive
    print(f"\nSaving final embeddings to: {save_file}")
    np.savez_compressed(
        save_file,
        embeddings_arr=all_embeddings,
        targets_arr=targets,
        targets_raw_arr=targets_raw,
        ensembl_transcript_id_arr=tx_ids,
        ensembl_gene_id_arr=gene_ids,
        genes_arr=genes,
        seq_lens_arr=seq_lens,
        fold_assignment_arr=fold_assignment,
        # Only metadata not contained in the track .meta.json (single source for dataset, track, normalization)
        source_tracks_meta=str(get_meta_filepath(tracks_path).name),
        variant=str(var_str),
        model_checkpoint=str(model_identifier),
    )

    print("=" * 75)
    print("                    EXTRACTION COMPLETED SUCCESSFULLY                 ")
    print("=" * 75)
    print(f"Embedding array shape:   {all_embeddings.shape}")
    print(f"Track identity:          {track_name} (Normalization: {normalization.upper()})")
    print(f"Saved to:                {save_file}")
    print("=" * 75 + "\n")

    print_memory_profile(device=device, title="ORTHRUS EMBEDDINGS SPEICHER-PROFILING")


if __name__ == "__main__":
    main()
