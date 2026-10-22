#!/usr/bin/env python3
"""
Unified extraction of Orthrus embeddings using precomputed N-tracks archives
and fine-tuned 4-fold cross-validation model checkpoints.

Exclusively operates on precomputed N-track NPZ archives (from generate_miRNA_rbp_tracks.py,
generate_2_rbp_tracks.py, generate_m6a_tracks.py, etc.) and strictly runs 4-fold CV
representation extraction (Out-of-Fold validation + Test ensemble mean).
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

from util import reset_peak_memory_stats, print_memory_profile, load_metadata, save_metadata, get_meta_filepath


SPLITS_LOOKUP_HIPSC_CM = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv"
)
SPLITS_LOOKUP_RNAHL_HUMAN = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/rnahl-human_10folds_lookup.csv"
)
SPLITS_LOOKUP_RNAHL_MOUSE = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/rnahl-mouse_10folds_lookup.csv"
)

SPLITS_LOOKUP_TABLES = {
    "hipsc_cm": SPLITS_LOOKUP_HIPSC_CM,
    "mrna_bench_human": SPLITS_LOOKUP_RNAHL_HUMAN,
    "mrna_bench_mouse": SPLITS_LOOKUP_RNAHL_MOUSE,
}


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
    Extracts 512-dimensional Orthrus pooled embeddings using dynamic-length batching.
    Tracks are sorted by length to minimize zero-padding overhead.
    """
    if len(tracks) == 0:
        return np.zeros((0, 512), dtype=np.float32)

    sample_data = []
    truncated_count = 0

    for idx, tr in enumerate(tracks):
        if tr.shape[0] > max_length:
            truncated_count += 1
            tr = tr[:max_length, :]

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


def load_fold_backbone(fold_dir: Union[str, Path], device: torch.device, num_channels: Optional[int] = None) -> torch.nn.Module:
    """
    Loads fine-tuned Orthrus backbone checkpoint or base pretrained model.
    Adapts embedding layer dynamically if num_channels does not match base model in_features.
    """
    fold_dir_str = str(fold_dir).strip()
    fold_path = Path(fold_dir_str)

    if fold_path.exists() and (fold_path / "best_finetuned_backbone").is_dir():
        ckpt_target = str(fold_path / "best_finetuned_backbone")
    else:
        ckpt_target = fold_dir_str

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
        help="Maximum sequence length (default: 12288 bp)",
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
    targets = npz_data["target"].astype(np.float32) if "target" in npz_data else np.full(n_samples, np.nan, dtype=np.float32)
    targets_raw = npz_data["target_raw"].astype(np.float32) if "target_raw" in npz_data else np.full(n_samples, np.nan, dtype=np.float32)
    tx_ids = npz_data["ensembl_transcript_id"].astype(str) if "ensembl_transcript_id" in npz_data else np.array([""] * n_samples)
    gene_ids = npz_data["ensembl_gene_id"].astype(str) if "ensembl_gene_id" in npz_data else np.array([""] * n_samples)
    genes = npz_data["gene"].astype(str) if "gene" in npz_data else np.array([""] * n_samples)
    seq_lens = npz_data["seq_lens"].astype(np.int32) if "seq_lens" in npz_data else np.array([len(t) for t in tracks], dtype=np.int32)

    # Load companion metadata strictly (no fallbacks)
    tracks_meta = load_metadata(tracks_path, required=True)

    if "dataset" not in tracks_meta:
        raise KeyError(f"[Error] Required metadata key 'dataset' not found in companion metadata for {tracks_path}!")
    dataset_name = str(tracks_meta["dataset"]).strip()

    if "track_name" not in tracks_meta:
        raise KeyError(f"[Error] Required metadata key 'track_name' not found in companion metadata for {tracks_path}!")
    track_name = str(tracks_meta["track_name"]).strip()

    normalization = str(tracks_meta.get("normalization", "none")).strip().lower()

    print(f"\n[Tracks Info] Loaded {n_samples:,} transcripts with {n_channels} channels.")
    print(f"Dataset identity:   {dataset_name}")
    print(f"Track identity:     {track_name} (Normalization: {normalization.upper()})")

    # 2. Resolve Model Checkpoint & Architecture Mode
    model_checkpoint_str = str(args.model_checkpoint).strip()
    is_hf_model = model_checkpoint_str == "quietflamingo/orthrus-large-6-track" or (
        "/" in model_checkpoint_str and not Path(model_checkpoint_str).exists()
    )

    if is_hf_model:
        is_base_model = True
        is_4fold_dir = False
        model_identifier = model_checkpoint_str
        fold_subdirs = []
        print(f"\n[Model Detection] Pretrained base model: {model_checkpoint_str}")
    else:
        model_path = Path(model_checkpoint_str)
        if not model_path.exists():
            raise FileNotFoundError(f"[Error] Model checkpoint path not found: {model_path}")

        # Check for model metadata
        try:
            model_meta = load_metadata(model_path, required=False)
        except Exception:
            model_meta = {}

        fold_subdirs = [model_path / f"fold_{k}" for k in range(4)]
        if all(p.is_dir() for p in fold_subdirs):
            is_4fold_dir = True
            model_identifier = str(model_path)
            print(f"\n[Model Detection] 4-Fold fine-tuned model directory: {model_path}")
        else:
            is_4fold_dir = False
            model_identifier = str(model_path)
            print(f"\n[Model Detection] Single checkpoint directory (shared across all folds): {model_path}")

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
    d_clean = str(dataset_name).strip().lower().replace("-", "_")
    if d_clean in SPLITS_LOOKUP_TABLES:
        lookup_path = SPLITS_LOOKUP_TABLES[d_clean]
    else:
        raise FileNotFoundError(f"[Error] No lookup table found for dataset: {dataset_name} (cleaned: {d_clean})")

    if not lookup_path.exists():
        raise FileNotFoundError(
            f"[Error] 4-Fold cross-validation requires a standardized 10-fold split lookup table! "
            f"Lookup table not found at: {lookup_path}."
        )

    print(f"\n[Splits Lookup] Loading lookup table: {lookup_path}")
    lookup_df = pd.read_csv(lookup_path)
    lookup_tx_col = "ensembl_transcript_id" if "ensembl_transcript_id" in lookup_df.columns else "transcript_id"
    if lookup_tx_col not in lookup_df.columns or "split" not in lookup_df.columns:
        raise KeyError(f"[Error] Required columns ('{lookup_tx_col}', 'split') not found in lookup table {lookup_path}!")

    tx_to_split = dict(
        zip(lookup_df[lookup_tx_col].astype(str).str.strip(), lookup_df["split"].astype(int))
    )
    sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in tx_ids])

    # Canonical 4-fold assignments:
    # Fold 0 (Train [0..5]) -> Val [6, 7]
    # Fold 1 (Train [2..7]) -> Val [0, 1]
    # Fold 2 (Train [0,1,4,5,6,7]) -> Val [2, 3]
    # Fold 3 (Train [0,1,2,3,6,7]) -> Val [4, 5]
    fold_val_splits = {
        0: [6, 7],
        1: [0, 1],
        2: [2, 3],
        3: [4, 5],
    }
    test_splits = [8, 9]

    all_embeddings = np.zeros((n_samples, 512), dtype=np.float32)
    fold_assignment = np.empty(n_samples, dtype=object)

    test_idx = np.where(np.isin(sample_splits, test_splits))[0]
    test_tracks = [tracks[i] for i in test_idx]
    test_accum = np.zeros((len(test_idx), 512), dtype=np.float32)

    print(f"\n4-Fold CV Setup: Total Samples = {n_samples:,}, Test Samples [8, 9] = {len(test_idx):,}")

    # 5. Execute Embedding Extraction
    if not is_4fold_dir:
        print(f"\n[Single Model Mode] Extracting embeddings using model: {model_identifier}")
        print("Model has not seen CV split data; extracting all samples in a single unified pass.")
        model = load_fold_backbone(model_identifier, device, num_channels=tracks[0].shape[1])
        all_embeddings = extract_embeddings_for_tracks(
            tracks=tracks,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            desc="Orthrus Base Model Extraction",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set fold assignments based on standard splits
        for k in range(4):
            val_splits = fold_val_splits[k]
            val_idx = np.where(np.isin(sample_splits, val_splits))[0]
            for i in val_idx:
                fold_assignment[i] = f"fold_{k}"

        if len(test_idx) > 0:
            for i in test_idx:
                fold_assignment[i] = "ensemble_mean"

        unmatched_idx = np.where(sample_splits == -1)[0]
        if len(unmatched_idx) > 0:
            for i in unmatched_idx:
                fold_assignment[i] = "unassigned"

    else:
        # 4-Fold CV Extraction (with Resumable Checkpoint)
        completed_folds = set()
        if not args.start_fresh and checkpoint_file.exists():
            print(f"[*] Found existing checkpoint: {checkpoint_file}")
            try:
                with np.load(checkpoint_file, allow_pickle=True) as ckpt:
                    all_embeddings = ckpt["embeddings"].copy()
                    fold_assignment = ckpt["fold_assignment"].copy()
                    test_accum = ckpt["test_accum"].copy()
                    completed_folds = set(ckpt["completed_folds"].tolist())
                print(f"[*] Resuming: Folds {sorted(completed_folds)} already completed.")
            except Exception as e:
                print(f"[!] Warning: Failed to load checkpoint ({e}). Starting folds from scratch.")

        for k in range(4):
            if k in completed_folds:
                print(f"\n--- [Fold {k}] Already completed in checkpoint. Skipping ---")
                continue

            val_splits = fold_val_splits[k]
            val_idx = np.where(np.isin(sample_splits, val_splits))[0]
            val_tracks = [tracks[i] for i in val_idx]

            fold_dir = fold_subdirs[k]
            print(f"\n--- [Fold {k}] Loading model from: {fold_dir} ---")
            fold_model = load_fold_backbone(fold_dir, device, num_channels=tracks[0].shape[1])

            print(f"[Fold {k}] Extracting Out-of-Fold Val Embeddings (Splits {val_splits}, {len(val_idx):,} samples)...")
            val_emb = extract_embeddings_for_tracks(
                tracks=val_tracks,
                model=fold_model,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                desc=f"Fold {k} Val",
            )
            all_embeddings[val_idx] = val_emb
            for i in val_idx:
                fold_assignment[i] = f"fold_{k}"

            print(f"[Fold {k}] Extracting Test Set Embeddings (Splits {test_splits}, {len(test_idx):,} samples)...")
            fold_test_emb = extract_embeddings_for_tracks(
                tracks=test_tracks,
                model=fold_model,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                desc=f"Fold {k} Test",
            )
            test_accum += fold_test_emb

            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save checkpoint after each fold
            completed_folds.add(k)
            ckpt_dict = {
                "embeddings": all_embeddings,
                "fold_assignment": fold_assignment,
                "test_accum": test_accum,
                "completed_folds": np.array(list(completed_folds), dtype=np.int32),
            }
            tmp_ckpt = checkpoint_file.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_ckpt, **ckpt_dict)
            tmp_ckpt.replace(checkpoint_file)
            print(f"[✓] Checkpoint saved after Fold {k} (completed: {sorted(completed_folds)}).")

        # Clean up temporary checkpoint upon full completion
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
            except OSError:
                pass

        # Ensemble Mean for Test Set [8, 9]
        print("\nComputing Ensemble Mean Embedding for Test Set [8, 9] across all 4 folds...")
        if len(test_idx) > 0:
            all_embeddings[test_idx] = test_accum / 4.0
            for i in test_idx:
                fold_assignment[i] = "ensemble_mean"

        unmatched_idx = np.where(sample_splits == -1)[0]
        if len(unmatched_idx) > 0:
            print(f"[Notice] {len(unmatched_idx):,} samples had no split assignment in lookup table.")
            for i in unmatched_idx:
                fold_assignment[i] = "unassigned"

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
        dataset=str(dataset_name),
        track_name=str(track_name),
        normalization=str(normalization),
        variant=str(var_str),
        model_checkpoint=str(model_identifier),
        is_finetuned=is_finetuned,
        is_4fold_cv=is_4fold_dir,
        source_tracks_file=str(tracks_path.name),
        source_tracks_meta=str(get_meta_filepath(tracks_path).name),
    )

    print("=" * 75)
    print("                    EXTRACTION COMPLETED SUCCESSFULLY                 ")
    print("=" * 75)
    print(f"Embedding array shape:   {all_embeddings.shape}")
    print(f"Track identity:          {track_name} (Normalization: {normalization.upper()})")
    print(f"Saved to:                {save_file}")
    print(f"Standardized NPZ Keys:   ['embeddings_arr', 'targets_arr', 'targets_raw_arr',")
    print(f"                          'ensembl_transcript_id_arr', 'ensembl_gene_id_arr',")
    print(f"                          'genes_arr', 'seq_lens_arr', 'fold_assignment_arr',")
    print(f"                          'dataset', 'track_name', 'normalization',")
    print(f"                          'model_checkpoint', 'is_finetuned', 'is_4fold_cv']")
    print("=" * 75 + "\n")

    print_memory_profile(device=device, title="ORTHRUS EMBEDDINGS SPEICHER-PROFILING")


if __name__ == "__main__":
    main()
