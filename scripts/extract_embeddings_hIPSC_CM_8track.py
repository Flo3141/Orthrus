#!/usr/bin/env python3
"""
Extraction of Orthrus 8-track embeddings for the hIPSC_CM dataset.
Loads the 8-track converted or fine-tuned Orthrus model and extracts 512-dimensional
pooled embeddings directly from the multi-track NPZ file (from generate_trans_factor_tracks.py).

Outputs an NPZ file identical in structure to the 6-track embeddings, enabling
seamless evaluation with train_ridge_regression_hIPSC_CM.py.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


def extract_embeddings_for_8track(
    tracks: list,
    metadata_list: list,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288,
) -> np.ndarray:
    """
    Extracts embeddings using dynamic-length batching (sorting by sequence length to minimize padding).
    """
    print(f"Preparing {len(tracks)} sequences for embedding extraction...")

    sample_data = []
    truncated_count = 0

    for idx, (tr, meta) in enumerate(zip(tracks, metadata_list)):
        if tr.shape[0] > max_length:
            truncated_count += 1
            tr = tr[:max_length, :]

        sample_data.append({
            "orig_idx": idx,
            "track": tr,
            "length": tr.shape[0],
            "meta": meta,
        })

    if truncated_count > 0:
        print(f"Notice: {truncated_count} sequences were truncated to {max_length} bp.")

    # Sort by length to minimize batch padding
    sorted_samples = sorted(sample_data, key=lambda s: s["length"])
    embeddings_list = [None] * len(sample_data)

    print(f"Extracting embeddings with batch size {batch_size} on {device}...")
    for i in tqdm(range(0, len(sorted_samples), batch_size), desc="Computing representations"):
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
            batch_emb = model.representation(x_tensor, lengths_tensor, channel_last=True)
            batch_emb_np = batch_emb.cpu().float().numpy()

        for b_idx, s in enumerate(batch):
            orig_i = s["orig_idx"]
            embeddings_list[orig_i] = batch_emb_np[b_idx]

    all_embeddings = np.stack(embeddings_list, axis=0)
    return all_embeddings


SPLITS_LOOKUP_PATH = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")


def main():
    parser = argparse.ArgumentParser(description="Extract Orthrus 8-track embeddings for hIPSC_CM")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/hIPSC_CM_8track_minmax.npz",
        help="Path to 8-track augmented NPZ file (from generate_trans_factor_tracks.py)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus/orthrus_8track_finetuned_hIPSC_CM",
        help="Path to 8-track model directory (parent folder with fold_0..fold_3, or single model checkpoint)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus",
        help="Directory to save the extracted embeddings",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="orthrus_8track_embeddings_hIPSC_CM.npz",
        help="Filename for the saved embeddings NPZ archive (automatically appended with normalization suffix)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length according to the Orthrus paper (default: 12288)",
    )
    parser.add_argument(
        "--is_finetuned",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether the checkpoint is fine-tuned ('auto', 'true', or 'false')",
    )
    args = parser.parse_args()

    data_file = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / args.output_filename

    # Resolve whether model is fine-tuned or 4-fold
    model_path = Path(str(args.model_checkpoint).strip())
    fold_subdirs = [model_path / f"fold_{k}" for k in range(4)]
    is_4fold = all(p.is_dir() for p in fold_subdirs)

    model_path_str = str(model_path)
    if args.is_finetuned == "true":
        is_finetuned = True
    elif args.is_finetuned == "false":
        is_finetuned = False
    else:
        is_finetuned = (
            is_4fold
            or "finetun" in model_path_str.lower()
            or "checkpoint" in model_path_str.lower()
        )

    # 1. Load multi-track data
    print(f"\nLoading NPZ archive: {data_file}...")
    npz_data = np.load(data_file, allow_pickle=True)

    # Dynamically append normalization suffix if using default output_filename
    norm_val = str(npz_data.get("normalization", "none")).strip().lower()
    if norm_val and norm_val not in ["none", "nan"] and not save_file.stem.endswith(f"_{norm_val}"):
        save_file = save_file.parent / f"{save_file.stem}_{norm_val}{save_file.suffix}"

    print("=" * 70)
    print("         Orthrus 8-Track Embedding Extraction Pipeline          ")
    print("=" * 70)
    print(f"Data file:         {data_file}")
    print(f"Model checkpoint:  {args.model_checkpoint}")
    print(f"Extraction Mode:   {'4-Fold Cross-Validation (Out-of-Fold + Test Ensemble)' if is_4fold else 'Single Model'}")
    print(f"Model variant:     {'Fine-Tuned' if is_finetuned else 'Base 8-Track'}")
    print(f"Output file:       {save_file}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Max sequence len:  {args.max_length}")

    tracks = list(npz_data["tracks"])
    n_samples = len(tracks)
    print(f"Loaded {n_samples} transcripts.")

    # 2. Extract metadata
    tx_ids = np.array([str(npz_data["ensembl_transcript_id"][i]) if "ensembl_transcript_id" in npz_data else f"tx_{i}" for i in range(n_samples)])
    gene_ids = np.array([str(npz_data["ensembl_gene_id"][i]) if "ensembl_gene_id" in npz_data else "" for i in range(n_samples)])
    gene_symbols = np.array([str(npz_data["hgnc_symbol"][i]) if "hgnc_symbol" in npz_data else "" for i in range(n_samples)])
    half_lives = np.array([float(npz_data["half_life"][i]) if "half_life" in npz_data else np.nan for i in range(n_samples)], dtype=np.float32)
    half_lives_transformed = np.array([float(npz_data["half_life_transformed"][i]) if "half_life_transformed" in npz_data else np.nan for i in range(n_samples)], dtype=np.float32)
    rates = np.array([float(npz_data["rate"][i]) if "rate" in npz_data else np.nan for i in range(n_samples)], dtype=np.float32)
    has_mirna = np.array([bool(npz_data["has_mirna"][i]) if "has_mirna" in npz_data else False for i in range(n_samples)], dtype=bool)
    has_eclip = np.array([bool(npz_data["has_eclip"][i]) if "has_eclip" in npz_data else False for i in range(n_samples)], dtype=bool)
    has_gtf = np.array([bool(npz_data["has_gtf"][i]) if "has_gtf" in npz_data else False for i in range(n_samples)], dtype=bool)
    seq_lens = np.array([tr.shape[0] for tr in tracks], dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 3. Extract embeddings
    if is_4fold:
        lookup_path = SPLITS_LOOKUP_PATH
        if not lookup_path.exists():
            raise FileNotFoundError(f"4-fold extraction requires split lookup table at: {lookup_path}")

        print(f"\nLoading split lookup table: {lookup_path}")
        lookup_df = pd.read_csv(lookup_path)
        if "ensembl_transcript_id" not in lookup_df.columns:
            raise KeyError(f"[Error] 'ensembl_transcript_id' column not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}")
        if "split" not in lookup_df.columns:
            raise KeyError(f"[Error] 'split' column not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}")

        tx_to_split = dict(zip(lookup_df["ensembl_transcript_id"].astype(str).str.strip(), lookup_df["split"].astype(int)))
        sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in tx_ids])

        # Canonical 4-fold validation assignments:
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

        print(f"\n4-Fold CV Extraction: Total Samples = {n_samples}, Test Samples [8, 9] = {len(test_idx)}")

        for k in range(4):
            val_splits = fold_val_splits[k]
            val_idx = np.where(np.isin(sample_splits, val_splits))[0]
            val_tracks = [tracks[i] for i in val_idx]

            fold_dir = fold_subdirs[k]
            ckpt_dir = fold_dir / "best_finetuned_backbone" if (fold_dir / "best_finetuned_backbone").is_dir() else fold_dir
            print(f"\n--- [Fold {k}] Loading 8-track model from: {ckpt_dir} ---")
            fold_model = AutoModel.from_pretrained(str(ckpt_dir), trust_remote_code=True)
            fold_model = fold_model.to(device)
            fold_model.eval()

            print(f"[Fold {k}] Extracting Out-of-Fold Val Embeddings (Splits {val_splits}, {len(val_idx)} samples)...")
            val_emb = extract_embeddings_for_8track(
                tracks=val_tracks,
                metadata_list=[],
                model=fold_model,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            all_embeddings[val_idx] = val_emb
            for i in val_idx:
                fold_assignment[i] = f"fold_{k}"

            print(f"[Fold {k}] Extracting Test Set Embeddings (Splits {test_splits}, {len(test_idx)} samples)...")
            fold_test_emb = extract_embeddings_for_8track(
                tracks=test_tracks,
                metadata_list=[],
                model=fold_model,
                device=device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            test_accum += fold_test_emb

            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nComputing Ensemble Mean Embedding for Test Set [8, 9] across all 4 folds...")
        all_embeddings[test_idx] = test_accum / 4.0
        for i in test_idx:
            fold_assignment[i] = "ensemble_mean"

        unmatched_idx = np.where(sample_splits == -1)[0]
        if len(unmatched_idx) > 0:
            print(f"[Notice] {len(unmatched_idx)} samples had no split assignment in lookup table.")
            for i in unmatched_idx:
                fold_assignment[i] = "unassigned"

    else:
        # Single model mode
        print(f"\nLoading 8-track model from '{args.model_checkpoint}' on {device}...")
        model = AutoModel.from_pretrained(args.model_checkpoint, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully.")

        all_embeddings = extract_embeddings_for_8track(
            tracks=tracks,
            metadata_list=[],
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        fold_assignment = np.array(["single_model"] * n_samples, dtype=object)

    # 4. Save in standardized NPZ format
    print(f"\nSaving embeddings to: {save_file}")
    np.savez_compressed(
        save_file,
        embeddings=all_embeddings,
        fold_assignment=fold_assignment,
        half_life_transformed=half_lives_transformed,
        half_life=half_lives,
        rate=rates,
        ensembl_transcript_id=tx_ids,
        ensembl_gene_id=gene_ids,
        hgnc_symbol=gene_symbols,
        has_mirna=has_mirna,
        has_eclip=has_eclip,
        has_gtf=has_gtf,
        seq_lens=seq_lens,
        normalization=str(npz_data.get("normalization", "none")),
        model_checkpoint=model_path_str,
        is_finetuned=is_finetuned,
        is_4fold_cv=is_4fold,
    )

    print(f"Successfully saved! Embedding array shape: {all_embeddings.shape}")
    print(f"Saved to:        {save_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
