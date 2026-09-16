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


def main():
    parser = argparse.ArgumentParser(description="Extract Orthrus 8-track embeddings for hIPSC_CM")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_multitrack_with_trans_factors_minmax.npz",
        help="Path to 8-track augmented NPZ file (from generate_trans_factor_tracks.py)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus_8track_finetuned_hIPSC_CM/best_finetuned_backbone",
        help="Path to 8-track model directory (warm-started from convert_6track_to_8track.py or fine-tuned backbone)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM",
        help="Directory to save the extracted embeddings",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="orthrus_8track_embeddings_hIPSC_CM.npz",
        help="Filename for the saved embeddings NPZ archive",
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
    args = parser.parse_args()

    data_file = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / args.output_filename

    print("=" * 70)
    print("         Orthrus 8-Track Embedding Extraction Pipeline          ")
    print("=" * 70)
    print(f"Data file:         {data_file}")
    print(f"Model checkpoint:  {args.model_checkpoint}")
    print(f"Output file:       {save_file}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Max sequence len:  {args.max_length}")

    # 1. Load multi-track data
    print(f"\nLoading NPZ archive: {data_file}...")
    npz_data = np.load(data_file, allow_pickle=True)

    tracks = list(npz_data["tracks"])
    n_samples = len(tracks)
    print(f"Loaded {n_samples} transcripts.")

    # 2. Extract metadata
    metadata_list = []
    for i in range(n_samples):
        metadata_list.append({
            "tx_id": str(npz_data["ensembl_transcript_id"][i]) if "ensembl_transcript_id" in npz_data else f"tx_{i}",
            "gene_id": str(npz_data["ensembl_gene_id"][i]) if "ensembl_gene_id" in npz_data else "",
            "gene_symbol": str(npz_data["hgnc_symbol"][i]) if "hgnc_symbol" in npz_data else "",
            "half_life_transformed": float(npz_data["half_life_transformed"][i]) if "half_life_transformed" in npz_data else np.nan,
            "half_life": float(npz_data["half_life"][i]) if "half_life" in npz_data else np.nan,
            "rate": float(npz_data["rate"][i]) if "rate" in npz_data else np.nan,
            "has_mirna": bool(npz_data["has_mirna"][i]) if "has_mirna" in npz_data else False,
            "has_eclip": bool(npz_data["has_eclip"][i]) if "has_eclip" in npz_data else False,
            "has_gtf": bool(npz_data["has_gtf"][i]) if "has_gtf" in npz_data else False,
        })

    # 3. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading 8-track model from '{args.model_checkpoint}' on {device}...")
    model = AutoModel.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Extract embeddings
    embeddings = extract_embeddings_for_8track(
        tracks=tracks,
        metadata_list=metadata_list,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 5. Save in standardized NPZ format
    print(f"\nSaving embeddings to: {save_file}")
    np.savez_compressed(
        save_file,
        embeddings=embeddings,
        half_life_transformed=np.array([m["half_life_transformed"] for m in metadata_list], dtype=np.float32),
        half_life=np.array([m["half_life"] for m in metadata_list], dtype=np.float32),
        rate=np.array([m["rate"] for m in metadata_list], dtype=np.float32),
        ensembl_transcript_id=np.array([m["tx_id"] for m in metadata_list]),
        ensembl_gene_id=np.array([m["gene_id"] for m in metadata_list]),
        hgnc_symbol=np.array([m["gene_symbol"] for m in metadata_list]),
        has_mirna=np.array([m["has_mirna"] for m in metadata_list], dtype=bool),
        has_eclip=np.array([m["has_eclip"] for m in metadata_list], dtype=bool),
        has_gtf=np.array([m["has_gtf"] for m in metadata_list], dtype=bool),
        seq_lens=np.array([tr.shape[0] for tr in tracks], dtype=np.int32),
        normalization=str(npz_data.get("normalization", "none")),
    )

    print(f"Successfully saved! Embedding array shape: {embeddings.shape}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
