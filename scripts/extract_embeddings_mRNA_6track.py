#!/usr/bin/env python3
"""
Extraction of Orthrus 6-track embeddings for mrna-bench (half-life datasets).
Runs on GPU cluster.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel
import mrna_bench as mb


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
    seq_bytes = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    oh = np.zeros((len(seq_bytes), 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0  # 'A'
    oh[seq_bytes == 67, 1] = 1.0  # 'C'
    oh[seq_bytes == 71, 2] = 1.0  # 'G'
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # 'T' (84) or 'U' (85)
    return oh


def parse_binary_track(track, length: int) -> np.ndarray:
    """
    Converts track columns (CDS or Splice) into a 1D float32 array of length `length`.
    Supports strings ("00100..."), lists, and existing NumPy arrays.
    """
    if isinstance(track, str):
        arr = np.array([float(c) for c in track], dtype=np.float32)
    elif isinstance(track, (list, np.ndarray)):
        arr = np.asarray(track, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported track type: {type(track)}")

    if len(arr) != length:
        # If length differs, adapt / pad / trim
        if len(arr) < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[:len(arr)] = arr
            arr = padded
        else:
            arr = arr[:length]
            
    return arr.reshape(-1, 1)


def build_six_track(seq: str, cds, splice) -> np.ndarray:
    """
    Creates the complete 6-track array (L, 6):
      Tracks 0-3: A, C, G, T/U (One-Hot)
      Track 4:    CDS (binary)
      Track 5:    Splice-Site (binary)
    """
    seq_oh = seq_to_one_hot(seq)
    l = len(seq)
    cds_track = parse_binary_track(cds, l)
    splice_track = parse_binary_track(splice, l)
    
    # Concatenate to (L, 6)
    six_track = np.concatenate([seq_oh, cds_track, splice_track], axis=1)
    return six_track


def extract_embeddings_for_dataset(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288
) -> dict:
    """
    Extracts embeddings for a DataFrame using dynamic length batching.
    """
    print(f"Processing dataset with {len(df)} entries...")
    
    # Filter / clamp to max_length according to paper (Orthrus excludes >12288)
    sample_data = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        seq = str(row["sequence"])
        if len(seq) > max_length:
            skipped_count += 1
            seq = seq[:max_length]
            cds = row["cds"][:max_length] if hasattr(row["cds"], "__getitem__") else row["cds"]
            splice = row["splice"][:max_length] if hasattr(row["splice"], "__getitem__") else row["splice"]
        else:
            cds = row["cds"]
            splice = row["splice"]

        six_track = build_six_track(seq, cds, splice)
        sample_data.append({
            "orig_idx": idx,
            "track": six_track,
            "length": six_track.shape[0],
            "gene": str(row.get("gene", "")),
            "chromosome": str(row.get("chromosome", "")),
            "target": float(row.get("target", np.nan))
        })
        
    if skipped_count > 0:
        print(f"Notice: {skipped_count} sequences were truncated to {max_length} bp.")

    # Sort by length to minimize padding overhead per batch
    sorted_samples = sorted(sample_data, key=lambda x: x["length"])
    
    embeddings_list = [None] * len(sample_data)
    
    print(f"Starting embedding extraction with batch size {batch_size}...")
    for i in tqdm(range(0, len(sorted_samples), batch_size), desc="Extracting embeddings"):
        batch = sorted_samples[i : i + batch_size]
        b_lens = [s["length"] for s in batch]
        max_b_len = max(b_lens)
        
        # Padded batch tensor (batch, max_b_len, 6)
        batch_arr = np.zeros((len(batch), max_b_len, 6), dtype=np.float32)
        for b_idx, s in enumerate(batch):
            l = s["length"]
            batch_arr[b_idx, :l, :] = s["track"]
            
        x_tensor = torch.from_numpy(batch_arr).to(device)
        lengths_tensor = torch.tensor(b_lens, dtype=torch.long, device=device)
        
        with torch.no_grad():
            # channel_last=True expects (B, L, C) with C=6
            batch_emb = model.representation(x_tensor, lengths_tensor, channel_last=True)
            batch_emb_np = batch_emb.cpu().numpy()
            
        for b_idx, s in enumerate(batch):
            orig_i = s["orig_idx"]
            embeddings_list[orig_i] = batch_emb_np[b_idx]
            
        all_embeddings = np.stack(embeddings_list, axis=0)
    all_targets = df["target"].values.astype(np.float32)
    all_genes = df["gene"].astype(str).values
    all_chromosomes = df["chromosome"].astype(str).values
    seq_lens = df["sequence"].str.len().values
    
    return {
        "embeddings": all_embeddings,
        "targets": all_targets,
        "genes": all_genes,
        "chromosomes": all_chromosomes,
        "seq_lens": seq_lens
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Orthrus 6-track embeddings")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench",
        help="Path to mrna_bench data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional central output directory. If None (default), saved directly in <data_dir>/<dataset_key>/embeddings/."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Hugging Face model identifier (e.g. quietflamingo/orthrus-large-6-track or antichronology/orthrus-6-track)"
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["human", "mouse", "both"],
        default="both",
        help="Which species to extract (human, mouse, or both)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length (up to 12288 bp according to paper)"
    )
    args = parser.parse_args()

    # Prepare paths
    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Registering mrna-bench path: {data_path}")
    mb.update_data_path(str(data_path))

    # Initialize device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading Orthrus 6-track model '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    species_to_process = []
    if args.species in ["human", "both"]:
        species_to_process.append(("human", "rnahl-human"))
    if args.species in ["mouse", "both"]:
        species_to_process.append(("mouse", "rnahl-mouse"))

    for spec_name, dataset_key in species_to_process:
        print(f"\n==================== {spec_name.upper()} DATASET ====================")
        print(f"Loading {dataset_key} via mrna-bench...")
        df = mb.load_dataset(dataset_key).data_df
        print(f"Loaded rows: {len(df)}")
        print(f"Available columns: {list(df.columns)}")

        result = extract_embeddings_for_dataset(
            df=df,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        if args.output_dir is not None:
            spec_out_dir = Path(args.output_dir)
            save_file = spec_out_dir / f"orthrus_6track_embeddings_{spec_name}.npz"
        else:
            # Default: Directly in <data_dir>/<dataset_key>/embeddings/
            spec_out_dir = data_path / dataset_key / "embeddings"
            save_file = spec_out_dir / "orthrus_6track_embeddings.npz"

        spec_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving embeddings to: {save_file}")
        np.savez_compressed(
            save_file,
            embeddings=result["embeddings"],
            targets=result["targets"],
            genes=result["genes"],
            chromosomes=result["chromosomes"],
            seq_lens=result["seq_lens"]
        )
        print(f"Successfully saved! Embedding shape: {result['embeddings'].shape}")

    print("\nAll embeddings were successfully extracted and saved.")


if __name__ == "__main__":
    main()
