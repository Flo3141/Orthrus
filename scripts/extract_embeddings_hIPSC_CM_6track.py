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

    # Sort by length to minimize padding within batches
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

    # Metadata in original DataFrame order
    transcript_ids = np.array([s["transcript_id"] for s in sample_data])
    gene_ids = np.array([s["gene_id"] for s in sample_data])
    gene_symbols = np.array([s["gene_symbol"] for s in sample_data])
    biotypes = np.array([s["biotype"] for s in sample_data])
    half_lives = np.array([s["half_life"] for s in sample_data], dtype=np.float32)
    half_lives_transformed = np.array([s["half_life_transformed"] for s in sample_data], dtype=np.float32)
    rates = np.array([s["rate"] for s in sample_data], dtype=np.float32)
    seq_lens = np.array([s["length"] for s in sample_data], dtype=np.int32)

    return {
        "embeddings": all_embeddings,
        "half_life": half_lives,
        "half_life_transformed": half_lives_transformed,
        "rate": rates,
        "ensembl_transcript_id": transcript_ids,
        "ensembl_gene_id": gene_ids,
        "hgnc_symbol": gene_symbols,
        "transcript_biotype": biotypes,
        "seq_lens": seq_lens,
    }


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
        help="Filename for the saved NPZ archive",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Hugging Face model identifier",
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
    save_file = output_dir / args.output_filename

    print(f"Loading hIPSC_CM dataset from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Ensure half_life_transformed exists (if called with raw hIPSC_CM_ej_cds.txt)
    if "half_life_transformed" not in df.columns and "half_life" in df.columns:
        print("Column 'half_life_transformed' not found - computing from 'half_life' (Log + Z-Score)...")
        y_raw = df["half_life"].astype(float)
        y_log = np.log(y_raw + 0.1)
        mu_log = float(y_log.mean())
        sigma_log = float(y_log.std(ddof=1))
        df["half_life_transformed"] = (y_log - mu_log) / sigma_log
        print(f"Transformation computed: mu={mu_log:.4f}, sigma={sigma_log:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading Orthrus 6-track model '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    result = extract_embeddings_for_hIPSC_CM(
        df=df,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"\nSaving embeddings to: {save_file}")
    np.savez_compressed(
        save_file,
        embeddings=result["embeddings"],
        half_life=result["half_life"],
        half_life_transformed=result["half_life_transformed"],
        rate=result["rate"],
        ensembl_transcript_id=result["ensembl_transcript_id"],
        ensembl_gene_id=result["ensembl_gene_id"],
        hgnc_symbol=result["hgnc_symbol"],
        transcript_biotype=result["transcript_biotype"],
        seq_lens=result["seq_lens"],
    )

    print("Successfully saved!")
    print(f"Embedding shape: {result['embeddings'].shape}")


if __name__ == "__main__":
    main()
