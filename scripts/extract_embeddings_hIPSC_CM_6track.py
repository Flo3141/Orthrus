#!/usr/bin/env python3
"""
Extraction of Orthrus 6-track embeddings for the hIPSC_CM dataset (hIPSC_CM_ej_cds.txt).
Runs on GPU cluster.

Track construction from hIPSC_CM tokens:
- Channels 0-3: A, C, G, T/U (4-channel one-hot encoding)
- Channel 4:    CDS track (1.0 on uppercase letters such as 'A' in 'A,t,t', corresponds exactly to frame-0 codon start cds[0::3]=1 in Orthrus)
- Channel 5:    Splice track (1.0 on tokens with 'ej' suffix, marks exon junction boundaries)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel


def seq_to_one_hot(seq: str) -> np.ndarray:
    """
    Converts an RNA/DNA sequence into a 4-channel one-hot encoding.
    Conforms to the Orthrus paper:
      Channel 0: A (Adenine)
      Channel 1: C (Cytosine)
      Channel 2: G (Guanine)
      Channel 3: T / U (Thymine / Uracil)
      All other characters (e.g. 'N') -> [0, 0, 0, 0]
    
    Returns:
        np.ndarray of shape (L, 4) with dtype float32.
    """
    seq_bytes = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    oh = np.zeros((len(seq_bytes), 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0  # 'A'
    oh[seq_bytes == 67, 1] = 1.0  # 'C'
    oh[seq_bytes == 71, 2] = 1.0  # 'G'
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # 'T' (84) or 'U' (85)
    return oh


def parse_saluki_sequence_to_six_track(raw_seq: str) -> np.ndarray:
    """
    Parses the comma-separated Saluki sequence and produces an (L, 6) array:
      - Tracks 0-3: A, C, G, T/U one-hot
      - Track 4:    CDS marker (1.0 on uppercase letters, 0.0 on lowercase letters)
      - Track 5:    Splice marker (1.0 on 'ej' tokens, 0.0 otherwise)
    """
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    if not tokens:
        return np.zeros((0, 6), dtype=np.float32)

    # 1. Extract nucleotide sequence (first character of each token)
    clean_seq = "".join(tok[0] for tok in tokens)
    seq_oh = seq_to_one_hot(clean_seq)  # (L, 4)

    # 2. CDS track: uppercase letter = codon start (1st base of codon)
    cds_track = np.array([1.0 if tok[0].isupper() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    # 3. Splice track: 'ej' in token = exon junction
    splice_track = np.array([1.0 if "ej" in tok.lower() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    six_track = np.concatenate([seq_oh, cds_track, splice_track], axis=1)
    return six_track


def extract_embeddings_for_hIPSC_CM(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288,
) -> dict:
    """
    Extracts Orthrus 6-track embeddings for the hIPSC_CM DataFrame using dynamic length batching.
    """
    print(f"Processing hIPSC_CM dataset with {len(df)} entries...")

    sample_data = []
    skipped_count = 0

    for idx, row in df.iterrows():
        raw_seq = str(row["sequence"])
        six_track = parse_saluki_sequence_to_six_track(raw_seq)

        if six_track.shape[0] > max_length:
            skipped_count += 1
            six_track = six_track[:max_length, :]

        sample_data.append({
            "orig_idx": idx,
            "track": six_track,
            "length": six_track.shape[0],
            "transcript_id": str(row.get("ensembl_transcript_id", "")),
            "gene_id": str(row.get("ensembl_gene_id", "")),
            "gene_symbol": str(row.get("hgnc_symbol", "")),
            "biotype": str(row.get("transcript_biotype", "")),
            "half_life": float(row.get("half_life", np.nan)),
            "half_life_transformed": float(row.get("half_life_transformed", np.nan)),
            "rate": float(row.get("rate", np.nan)),
        })

    if skipped_count > 0:
        print(f"Notice: {skipped_count} sequences were truncated to max_length={max_length} nucleotides.")

def extract_embeddings_for_tracks(
    tracks: list,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288,
    desc: str = "Extracting embeddings",
) -> np.ndarray:
    """
    Extracts Orthrus embeddings for a list of tracks using dynamic length batching.
    """
    if len(tracks) == 0:
        return np.zeros((0, 512), dtype=np.float32)

    sample_data = []
    for idx, tr in enumerate(tracks):
        if tr.shape[0] > max_length:
            tr = tr[:max_length, :]
        sample_data.append({
            "orig_idx": idx,
            "track": tr,
            "length": tr.shape[0],
        })

    # Sort by length to minimize padding within batches
    sorted_samples = sorted(sample_data, key=lambda x: x["length"])
    embeddings_list = [None] * len(sample_data)

    for i in tqdm(range(0, len(sorted_samples), batch_size), desc=desc):
        batch = sorted_samples[i : i + batch_size]
        b_lens = [s["length"] for s in batch]
        max_b_len = max(b_lens)

        batch_arr = np.zeros((len(batch), max_b_len, 6), dtype=np.float32)
        for b_idx, s in enumerate(batch):
            l = s["length"]
            batch_arr[b_idx, :l, :] = s["track"]

        x_tensor = torch.from_numpy(batch_arr).to(device)
        lengths_tensor = torch.tensor(b_lens, dtype=torch.long, device=device)

        with torch.no_grad():
            batch_emb = model.representation(x_tensor, lengths_tensor, channel_last=True)
            batch_emb_np = batch_emb.cpu().numpy()

        for b_idx, s in enumerate(batch):
            orig_i = s["orig_idx"]
            embeddings_list[orig_i] = batch_emb_np[b_idx]

    return np.stack(embeddings_list, axis=0)


def load_model_from_checkpoint(model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Loads a 6-track Orthrus model from a directory or .pt file.
    """
    if model_path.is_dir() and (model_path / "best_finetuned_backbone").is_dir():
        model_path = model_path / "best_finetuned_backbone"

    if model_path.is_file() and model_path.suffix in [".pt", ".pth", ".bin"]:
        print(f"Loading base Orthrus model and restoring weights from checkpoint file: {model_path}...")
        model = AutoModel.from_pretrained("quietflamingo/orthrus-large-6-track", trust_remote_code=True)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_dict[k[len("backbone."):]] = v
            elif not k.startswith("head."):
                backbone_dict[k] = v
        model.load_state_dict(backbone_dict, strict=False)
    else:
        print(f"Loading Orthrus 6-track model from '{model_path}'...")
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

    model = model.to(device)
    model.eval()
    return model


SPLITS_LOOKUP_PATH = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")


def main():
    parser = argparse.ArgumentParser(description="Extract Orthrus 6-track embeddings for hIPSC_CM")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt",
        help="Path to hIPSC_CM data file (tab-separated)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="orthrus_6track_embeddings_hIPSC_CM.npz",
        help="Filename for the saved NPZ archive (automatically appended with _finetuned when fine-tuned)",
    )
    parser.add_argument(
        "--model_checkpoint",
        "--model_name",
        dest="model_checkpoint",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus/orthrus_6track_finetuned_hIPSC_CM",
        help="Path to 6-track model directory (or parent folder with fold_0..fold_3), or Hugging Face identifier",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length according to the Orthrus paper (default: 12288)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(str(args.model_checkpoint).strip())
    fold_subdirs = [model_path / f"fold_{k}" for k in range(4)]
    is_4fold = all(p.is_dir() for p in fold_subdirs)

    model_path_str = str(model_path)
    is_finetuned = (
        is_4fold
        or model_path_str != "quietflamingo/orthrus-large-6-track"
        or "finetun" in model_path_str.lower()
        or "checkpoint" in model_path_str.lower()
    )

    save_file = output_dir / args.output_filename
    if is_finetuned and "finetuned" not in save_file.stem.lower():
        stem = save_file.stem
        save_file = save_file.parent / f"{stem}_finetuned{save_file.suffix}"

    print("=" * 70)
    print("         Orthrus 6-Track Embedding Extraction Pipeline          ")
    print("=" * 70)
    print(f"Data file:         {data_path}")
    print(f"Model checkpoint:  {model_path_str}")
    print(f"Extraction Mode:   {'4-Fold Cross-Validation (Out-of-Fold + Test Ensemble)' if is_4fold else 'Single Model'}")
    print(f"Model variant:     {'Fine-Tuned' if is_finetuned else 'Pretrained Base'}")
    print(f"Output file:       {save_file}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Max sequence len:  {args.max_length}")

    # 1. Load dataset
    print(f"\nLoading hIPSC_CM dataset from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Ensure half_life_transformed exists 
    if "half_life_transformed" not in df.columns:
        raise ValueError(f"Column 'half_life_transformed' not found! Columns are: {list(df.columns)}")

    # Parse Saluki sequence tokens into 6 tracks
    print("\nParsing sequences into 6-channel tracks [A, C, G, U, CDS, Splice]...")
    raw_seqs = df["sequence"].astype(str).values
    tracks = []
    truncated_count = 0
    for s in tqdm(raw_seqs, desc="Parsing 6-tracks"):
        tr = parse_saluki_sequence_to_six_track(s)
        if tr.shape[0] > args.max_length:
            truncated_count += 1
            tr = tr[:args.max_length, :]
        tracks.append(tr)

    if truncated_count > 0:
        print(f"Notice: {truncated_count} sequences were truncated to max_length={args.max_length} nucleotides.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Extract metadata arrays
    transcript_ids = df["ensembl_transcript_id"].astype(str).values if "ensembl_transcript_id" in df.columns else np.array([""] * len(df))
    gene_ids = df["ensembl_gene_id"].astype(str).values if "ensembl_gene_id" in df.columns else np.array([""] * len(df))
    gene_symbols = df["hgnc_symbol"].astype(str).values if "hgnc_symbol" in df.columns else np.array([""] * len(df))
    biotypes = df["transcript_biotype"].astype(str).values if "transcript_biotype" in df.columns else np.array([""] * len(df))
    half_lives = df["half_life"].astype(np.float32).values if "half_life" in df.columns else np.full(len(df), np.nan, dtype=np.float32)
    half_lives_transformed = df["half_life_transformed"].astype(np.float32).values if "half_life_transformed" in df.columns else np.full(len(df), np.nan, dtype=np.float32)
    rates = df["rate"].astype(np.float32).values if "rate" in df.columns else np.full(len(df), np.nan, dtype=np.float32)
    seq_lens = np.array([len(t) for t in tracks], dtype=np.int32)

    # 2. Extract Embeddings
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
        sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in transcript_ids])

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

        all_embeddings = np.zeros((len(df), 512), dtype=np.float32)
        fold_assignment = np.empty(len(df), dtype=object)

        test_idx = np.where(np.isin(sample_splits, test_splits))[0]
        test_tracks = [tracks[i] for i in test_idx]
        test_accum = np.zeros((len(test_idx), 512), dtype=np.float32)

        print(f"\n4-Fold CV Extraction: Total Samples = {len(df)}, Test Samples [8, 9] = {len(test_idx)}")

        for k in range(4):
            val_splits = fold_val_splits[k]
            val_idx = np.where(np.isin(sample_splits, val_splits))[0]
            val_tracks = [tracks[i] for i in val_idx]

            fold_model_dir = fold_subdirs[k]
            print(f"\n--- [Fold {k}] Loading model from: {fold_model_dir} ---")
            fold_model = load_model_from_checkpoint(fold_model_dir, device)

            print(f"[Fold {k}] Extracting Out-of-Fold Val Embeddings (Splits {val_splits}, {len(val_idx)} samples)...")
            val_emb = extract_embeddings_for_tracks(
                val_tracks, fold_model, device, batch_size=args.batch_size, max_length=args.max_length, desc=f"Fold {k} Val"
            )
            all_embeddings[val_idx] = val_emb
            for i in val_idx:
                fold_assignment[i] = f"fold_{k}"

            print(f"[Fold {k}] Extracting Test Set Embeddings (Splits {test_splits}, {len(test_idx)} samples)...")
            fold_test_emb = extract_embeddings_for_tracks(
                test_tracks, fold_model, device, batch_size=args.batch_size, max_length=args.max_length, desc=f"Fold {k} Test"
            )
            test_accum += fold_test_emb

            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average test set embeddings across all 4 fold models
        print("\nComputing Ensemble Mean Embedding for Test Set [8, 9] across all 4 folds...")
        all_embeddings[test_idx] = test_accum / 4.0
        for i in test_idx:
            fold_assignment[i] = "ensemble_mean"

        # Handle any unmatched samples (fallback)
        unmatched_idx = np.where(sample_splits == -1)[0]
        if len(unmatched_idx) > 0:
            print(f"[Notice] {len(unmatched_idx)} samples had no split assignment in lookup table.")
            for i in unmatched_idx:
                fold_assignment[i] = "unassigned"

    else:
        # Single model mode
        model = load_model_from_checkpoint(model_path, device)
        print(f"\nExtracting representations for all {len(tracks)} samples using single model...")
        all_embeddings = extract_embeddings_for_tracks(
            tracks, model, device, batch_size=args.batch_size, max_length=args.max_length, desc="Extracting embeddings"
        )
        fold_assignment = np.array(["single_model"] * len(df), dtype=object)

    print(f"\nSaving embeddings to: {save_file}")
    np.savez_compressed(
        save_file,
        embeddings=all_embeddings,
        fold_assignment=fold_assignment,
        half_life=half_lives,
        half_life_transformed=half_lives_transformed,
        rate=rates,
        ensembl_transcript_id=transcript_ids,
        ensembl_gene_id=gene_ids,
        hgnc_symbol=gene_symbols,
        transcript_biotype=biotypes,
        seq_lens=seq_lens,
        model_checkpoint=str(model_path),
        is_finetuned=is_finetuned,
        is_4fold_cv=is_4fold,
    )

    print("Successfully saved!")
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Saved to:        {save_file}")


if __name__ == "__main__":
    main()
