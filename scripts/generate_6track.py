#!/usr/bin/env python3
"""
Generation of Standardized 6-Track NPZ Archives for Downstream Orthrus Fine-Tuning.

Produces unified NPZ archives containing the 6 default canonical channels:
  Channel 0: Adenine (A)
  Channel 1: Cytosine (C)
  Channel 2: Guanine (G)
  Channel 3: Uracil / Thymine (U)
  Channel 4: Coding Sequence (CDS)
  Channel 5: Splice Sites / Exon Junctions

Features:
- Standardized loading via SixTrackDatasetLoader (hIPSC_CM, mRNABench Human & Mouse)
- Verified official Ensembl Transcript IDs (ENST... / ENSMUST...) with no silent fallbacks
- Incremental storage in chunks (resumable upon interruption)
- Compatible input format for finetune_ntrack.py and extract_embeddings.py
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from six_track_dataset import SixTrackDatasetLoader
from util import save_metadata


# =============================================================================
# Default Cluster Output Paths
# =============================================================================
DEFAULT_OUTPUT_PATHS = {
    "hipsc_cm": Path(
        "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/6track/hipsc_cm_6tracks.npz"
    ),
    "mrna_bench_human": Path(
        "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/orthrus/6track/mrna_bench_human_6tracks.npz"
    ),
    "mrna_bench_mouse": Path(
        "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/orthrus/6track/mrna_bench_mouse_6tracks.npz"
    ),
}

DEFAULT_CHANNEL_NAMES = ["A", "C", "G", "U", "CDS", "Splice"]


# =============================================================================
# 1. Chunk Management & Storage
# =============================================================================

def save_chunk(
    chunk_idx: int,
    chunk_items: list,
    chunks_dir: Path,
    dataset: str = "unknown",
):
    """Saves a batch of processed transcripts to a temporary chunk NPZ file."""
    chunk_file = chunks_dir / f"chunk_{chunk_idx:05d}.npz"

    tracks_arr = np.array([item["track"] for item in chunk_items], dtype=object)
    targets_arr = np.array([item["target"] for item in chunk_items], dtype=np.float32)
    targets_raw_arr = np.array([item["target_raw"] for item in chunk_items], dtype=np.float32)
    tx_arr = np.array([item["ensembl_transcript_id"] for item in chunk_items])
    gene_id_arr = np.array([item["ensembl_gene_id"] for item in chunk_items])
    gene_arr = np.array([item["gene"] for item in chunk_items])
    lens_arr = np.array([item["length"] for item in chunk_items], dtype=np.int32)

    np.savez_compressed(
        chunk_file,
        tracks=tracks_arr,
        target=targets_arr,
        target_raw=targets_raw_arr,
        ensembl_transcript_id=tx_arr,
        ensembl_gene_id=gene_id_arr,
        gene=gene_arr,
        seq_lens=lens_arr,
        dataset=str(dataset),
        track_name="6track",
        channel_names=np.array(DEFAULT_CHANNEL_NAMES),
    )


def merge_all_chunks(
    chunks_dir: Path,
    output_file: Path,
    dataset: str = "unknown",
    species: str = None,
    max_length: int = 12288,
):
    """Merges all generated chunks into the final master NPZ file."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"[Error] No chunk files found to merge in {chunks_dir}!")

    print(f"\nMerging {len(chunk_files)} chunks into {output_file}...")
    all_tracks = []
    targets = []
    targets_raw = []
    tx_ids = []
    gene_ids = []
    genes = []
    lens = []

    for cf in tqdm(chunk_files, desc="Merging chunks"):
        data = np.load(cf, allow_pickle=True)
        all_tracks.extend(data["tracks"])
        targets.extend(data["target"])
        targets_raw.extend(data["target_raw"])
        tx_ids.extend(data["ensembl_transcript_id"])
        gene_ids.extend(data["ensembl_gene_id"])
        genes.extend(data["gene"])
        lens.extend(data["seq_lens"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        tracks=np.array(all_tracks, dtype=object),
        target=np.array(targets, dtype=np.float32),
        target_raw=np.array(targets_raw, dtype=np.float32),
        ensembl_transcript_id=np.array(tx_ids),
        ensembl_gene_id=np.array(gene_ids),
        gene=np.array(genes),
        seq_lens=np.array(lens, dtype=np.int32),
        dataset=str(dataset),
        track_name="6track",
        channel_names=np.array(DEFAULT_CHANNEL_NAMES),
    )

    n = len(tx_ids)

    # Save standardized .meta.json companion file
    inferred_species = str(species) if species else ("human" if ("hipsc" in str(dataset).lower() or "human" in str(dataset).lower()) else "mouse")
    meta_dict = {
        "track_name": "6track",
        "dataset": str(dataset),
        "species": inferred_species,
        "normalization": "none",
        "num_channels": 6,
        "channel_names": list(DEFAULT_CHANNEL_NAMES),
        "max_length": int(max_length),
        "total_transcripts": int(n),
        "unique_genes": int(len(set(genes))),
        "seq_len_min": int(min(lens)) if lens else 0,
        "seq_len_max": int(max(lens)) if lens else 0,
        "seq_len_mean": float(np.mean(lens)) if lens else 0.0,
        "npz_file": output_file.name,
        "created_at": datetime.now().isoformat(),
    }
    save_metadata(output_file, meta_dict)

    print("\n" + "=" * 65)
    print("             6-TRACK ARCHIVE GENERATION REPORT             ")
    print("=" * 65)
    print(f"Dataset:                             {dataset}")
    print(f"Track name:                          6track")
    print(f"Total transcripts:                   {n:,}")
    print(f"Unique genes:                        {len(set(genes)):,}")
    print(f"Sequence length range:               {min(lens)} - {max(lens)} bp (mean: {np.mean(lens):.1f} bp)")
    print(f"Track dimensions per transcript:     (L, 6)")
    print(f"Channel configuration:               {DEFAULT_CHANNEL_NAMES}")
    print(f"Final file saved:                    {output_file}")
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Archive file size:                   {file_size_mb:.2f} MB")
    print("=" * 65)


# =============================================================================
# 2. Main Generation Pipeline
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate standardized 6-track NPZ archives for Orthrus model fine-tuning and evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hipsc_cm", "mrna_bench"],
        help="Dataset to process: 'hipsc_cm' or 'mrna_bench' (required)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        choices=["human", "mouse"],
        help="Species for mRNABench: 'human' or 'mouse' (required if --dataset is 'mrna_bench')",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of transcripts per incremental storage chunk (default: 1000)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length (default: 12288 bp)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Forces regeneration of all chunks and overwrites existing chunk directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Enforce species requirement for mrna_bench
    if args.dataset == "mrna_bench" and not args.species:
        raise ValueError("[Error] --species ('human' or 'mouse') must be specified when --dataset is 'mrna_bench'!")

    dataset_key = f"{args.dataset}_{args.species}" if args.species else args.dataset
    if dataset_key not in DEFAULT_OUTPUT_PATHS:
        raise ValueError(f"[Error] Unknown dataset key '{dataset_key}'! Supported datasets: {list(DEFAULT_OUTPUT_PATHS.keys())}")

    out_file = DEFAULT_OUTPUT_PATHS[dataset_key]


    out_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("        ORTHRUS STANDARDIZED 6-TRACK NPZ ARCHIVE GENERATOR        ")
    print("=" * 75)
    print(f"Dataset:            {args.dataset}" + (f" ({args.species})" if args.species else ""))
    print(f"Output file:        {out_file}")
    print(f"Chunk directory:    {chunks_dir} (Chunk size: {args.chunk_size})")
    print(f"Max sequence len:   {args.max_length} bp")

    # Check existing chunks for resuming or recreation
    completed_tx_ids = set()
    existing_chunks = sorted(chunks_dir.glob("chunk_*.npz"))

    if args.recreate:
        print(f"[Recreate] Flag --recreate active: Removing {len(existing_chunks)} existing chunks for a fresh restart...")
        for cf in existing_chunks:
            try:
                cf.unlink()
            except Exception as e:
                print(f"[Warning] Could not delete {cf.name}: {e}")
        existing_chunks = []
    else:
        for cf in existing_chunks:
            try:
                c_data = np.load(cf, allow_pickle=True)
                completed_tx_ids.update(c_data["ensembl_transcript_id"])
            except Exception:
                continue
        if completed_tx_ids:
            print(f"[Resume] Found {len(completed_tx_ids):,} already processed transcripts across {len(existing_chunks)} chunks.")

    # 1. Load Dataset via SixTrackDatasetLoader
    print(f"\n[Loader] Loading dataset '{args.dataset}'" + (f" ({args.species})" if args.species else "") + "...")
    loader = SixTrackDatasetLoader(dataset=args.dataset, species=args.species)
    df = loader.get_data()
    total_samples = len(df)
    print(f"[Loader] Successfully loaded {total_samples:,} canonical transcripts.")

    # 2. Process transcripts incrementally
    chunk_idx = len(existing_chunks)
    current_chunk_items = []

    print("\nProcessing transcripts and writing 6-track chunks...")
    with tqdm(total=total_samples, desc="Progress", initial=len(completed_tx_ids)) as pbar:
        for idx, row in df.iterrows():
            tx_id = str(row["ensembl_transcript_id"]).strip()

            if tx_id in completed_tx_ids:
                continue

            six_track = row["track"].copy()
            l = int(row["length"])

            if l > args.max_length:
                six_track = six_track[:args.max_length, :]
                l = args.max_length

            target_val = float(row["target"])
            target_raw_val = float(row["target_raw"])
            gene_name = str(row["gene"]).strip()
            gene_id_val = str(row["ensembl_gene_id"]).strip()

            current_chunk_items.append({
                "track": six_track,
                "ensembl_transcript_id": tx_id,
                "ensembl_gene_id": gene_id_val,
                "gene": gene_name,
                "target": target_val,
                "target_raw": target_raw_val,
                "length": l,
            })
            pbar.update(1)

            if len(current_chunk_items) >= args.chunk_size:
                save_chunk(
                    chunk_idx,
                    current_chunk_items,
                    chunks_dir,
                    dataset=dataset_key,
                )
                chunk_idx += 1
                current_chunk_items = []

        # Save remaining chunk
        if current_chunk_items:
            save_chunk(
                chunk_idx,
                current_chunk_items,
                chunks_dir,
                dataset=dataset_key,
            )

    # 3. Merge all chunks into the final NPZ file
    merge_all_chunks(
        chunks_dir=chunks_dir,
        output_file=out_file,
        dataset=dataset_key,
        species=args.species,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
