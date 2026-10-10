#!/usr/bin/env python3
"""
generate_m6a_tracks.py
=============================================================================
Generation of 7-track RNA representations (Channels 0-5 + m6A modification track)
for the Orthrus model from experimental m6A validation outputs.

Takes the verified m6A sites produced by `validate_and_filter_m6a.py`
(strictly requiring 'validated_m6a_sites.tsv') and constructs an augmented
7-track NPZ file (e.g. hIPSC_CM_7track_m6a.npz) compatible with Orthrus fine-tuning.

Channel Configuration (Total: 7 channels):
  - Channels 0-3: One-Hot RNA Sequence (A, C, G, U)
  - Channel 4:    CDS marker (1.0 in CDS, 0.0 in UTRs)
  - Channel 5:    Splice-site / Exon-Junction marker (1.0 at 'ej', 0.0 elsewhere)
  - Channel 6:    m6A Modification Track (quantitative modification frequency or binary flag)

Mapping Strategy:
  - Exclusively cDNA coordinate mapping: Uses 'cDNAstart' and 'cDNAend' directly
    from 'validated_m6a_sites.tsv'.
  - Strict validation: If 'validated_m6a_sites.tsv' does not exist or if required
    columns ('TxId', 'cDNAstart', 'cDNAend', score column) are missing, an error
    is raised immediately (no fallback).

Features:
  - Multiple scoring metrics:
      * 'mean_score': Mean modification percentage across biological replicates (0-100%)
      * 'effect_size': Drop in modification rate under METTL3 inhibitor (METTL3 dependency)
      * 'a_pct_modified': Baseline DMR modification frequency in Control
      * 'binary': Discrete flag (1.0 at validated m6A site, 0.0 elsewhere)
  - Flexible normalization methods:
      * 'unit': Scales 0-100% modification rate directly to [0.0, 1.0]
      * 'binary': Binary indicator [0.0, 1.0]
      * 'log': np.log1p(score)
      * 'minmax': Robust quantile scaling to [0.0, 1.0] using dynamically estimated q99
      * 'none': Raw values without scaling
  - Incremental chunk saving and automatic resume (avoids loss of progress on cluster preemptions)
  - Automatic merge into final NPZ file with complete metadata for Orthrus training
=============================================================================
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. Base Saluki 6-Track Parser
# =============================================================================

def parse_saluki_base_tracks(raw_seq: str) -> Tuple[np.ndarray, int, int, List[int]]:
    """
    Parses comma-separated tokens from Saluki/Orthrus dataset representation:
      - Tokens: e.g. "a", "c", "G", "ejA", "T", etc.
      - Uppercase characters indicate CDS.
      - 'ej' indicates exon junction / splice site.
    Returns:
      six_track: array of shape (L, 6)
      l: sequence length
      utr3_start: index where 3' UTR starts
      upper_indices: list of CDS token indices
    """
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    l = len(tokens)
    if l == 0:
        return np.zeros((0, 6), dtype=np.float32), 0, 0, []

    clean_seq = "".join(tok[0] for tok in tokens)

    seq_bytes = np.frombuffer(clean_seq.upper().encode("ascii"), dtype=np.uint8)
    oh = np.zeros((l, 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0  # A
    oh[seq_bytes == 67, 1] = 1.0  # C
    oh[seq_bytes == 71, 2] = 1.0  # G
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # T or U

    cds_track = np.array([1.0 if tok[0].isupper() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)
    splice_track = np.array([1.0 if "ej" in tok.lower() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    six_track = np.concatenate([oh, cds_track, splice_track], axis=1)

    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    utr3_start = (max(upper_indices) + 3) if upper_indices else 0
    utr3_start = min(utr3_start, l)

    return six_track, l, utr3_start, upper_indices


# =============================================================================
# 2. Loading & Indexing Validated m6A Sites (Strict: validated_m6a_sites.tsv only)
# =============================================================================

def load_validated_m6a_data(
    m6a_input_path: str,
    score_col: str = "mean_score",
    filter_drach: bool = False
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """
    Strictly loads validated m6A site table 'validated_m6a_sites.tsv'.
    Raises FileNotFoundError if the file does not exist.
    Raises KeyError if required columns are missing. No fallback.

    Returns:
      cdna_lookup: dict mapping transcript_id (full and unversioned) -> list of site records
      stats: summary statistics of loaded sites
    """
    p = Path(m6a_input_path)
    if p.is_dir():
        p = p / "validated_m6a_sites.tsv"

    # 1. Strict existence check for the file
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(
            f"[ERROR] The required file 'validated_m6a_sites.tsv' does not exist at: {p}\n"
            f"Please run 'validate_and_filter_m6a.py' first to generate this file."
        )

    print(f"\n[m6A Loader] Loading validated m6A positions from: {p}")
    df = pd.read_csv(p, sep="\t", low_memory=False)

    # 2. Strict column validation
    required_cols = ["TxId", "cDNAstart", "cDNAend"]
    if score_col != "binary":
        required_cols.append(score_col)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"[ERROR] Required column(s) missing in '{p.name}': {missing_cols}!\n"
            f"Available columns in file: {list(df.columns)}"
        )

    # Optional: Filter for canonical DRACH motif if requested
    if filter_drach:
        if "is_drach" not in df.columns:
            raise KeyError(
                "[ERROR] Flag '--filter_drach' passed, but column 'is_drach' is missing in 'validated_m6a_sites.tsv'!"
            )
        n_before = len(df)
        df = df[df["is_drach"] == True].copy()
        print(f"             Filtered for is_drach == True: {len(df):,} of {n_before:,} positions retained.")

    total_sites = len(df)
    print(f"             Total validated m6A records: {total_sites:,}")

    cdna_lookup: Dict[str, List[dict]] = {}

    for row_idx, row in df.iterrows():
        tx_id = str(row["TxId"]).strip()
        if not tx_id or tx_id == "nan":
            continue

        clean_tx = tx_id.split(".")[0]

        # Validate coordinates
        try:
            c_start = int(row["cDNAstart"])
            c_end = int(row["cDNAend"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"[ERROR] Invalid cDNA coordinates in row {row_idx}: "
                f"cDNAstart={row['cDNAstart']}, cDNAend={row['cDNAend']}"
            ) from e

        # Determine score
        if score_col == "binary":
            score_val = 1.0
        else:
            try:
                score_val = float(row[score_col])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"[ERROR] Invalid score in row {row_idx} in column '{score_col}': {row[score_col]}"
                ) from e

        site_entry = {
            "cDNAstart": c_start,
            "cDNAend": c_end,
            "score": score_val,
            "is_drach": bool(row.get("is_drach", False)) if "is_drach" in df.columns else False,
            "motif": str(row.get("motif_5mer", "")) if "motif_5mer" in df.columns else "",
        }

        # Index under both full and unversioned transcript ID
        for key in {tx_id, clean_tx}:
            cdna_lookup.setdefault(key, []).append(site_entry)

    unique_tx = len(set(df["TxId"].astype(str)))
    print(f"             Indexed {total_sites:,} m6A sites across {unique_tx:,} unique transcripts.")

    stats = {
        "total_records": total_sites,
        "unique_transcripts": unique_tx,
        "score_column": score_col,
    }
    return cdna_lookup, stats


# =============================================================================
# 3. Normalization for m6A Modification Track (Channel 6)
# =============================================================================

def compute_m6a_dynamic_quantile(
    df: pd.DataFrame,
    cdna_lookup: dict,
    max_length: int = 12288,
    quantile: float = 0.99,
    sample_size: int = 3000,
) -> float:
    """
    Dynamically computes high percentiles (default: 99th percentile, q99)
    of non-zero m6A modification values across the active dataset.
    """
    q_pct = quantile * 100.0 if quantile <= 1.0 else quantile
    print(f"\n[Quantile Estimation] Dynamically calculating {q_pct:.1f}th percentile for m6A channel...")

    if sample_size is not None and 0 < sample_size < len(df):
        eval_df = df.sample(n=sample_size, random_state=42)
    else:
        eval_df = df

    all_vals = []
    for _, row in eval_df.iterrows():
        tx_id = str(row.get("ensembl_transcript_id", ""))
        clean_tx = tx_id.split(".")[0]

        sites = cdna_lookup.get(tx_id) or cdna_lookup.get(clean_tx)
        if sites:
            for s in sites:
                val = float(s["score"])
                if val > 0:
                    all_vals.append(val)

    if all_vals:
        q_val = float(np.percentile(all_vals, q_pct))
        print(f"      Calculated m6A q{int(q_pct)}: {q_val:.4f} (from {len(all_vals):,} active positions)")
        return q_val
    else:
        print("      [Warning] No positive m6A values found during sampling. Defaulting to 100.0.")
        return 100.0


def normalize_m6a_track(
    m6a_track: np.ndarray,
    norm_method: str,
    m6a_q99: float = 100.0,
) -> np.ndarray:
    """
    Normalizes the 1D m6A track values.

    Methods:
      - 'none': Keep raw values unchanged
      - 'unit': Scales 0-100% modification rate directly to [0.0, 1.0] (divided by 100.0)
      - 'binary': Discretizes to binary flags (1.0 for positive, 0.0 otherwise)
      - 'log': np.log1p(x) -> log(1 + x)
      - 'minmax': Log-transformation with Robust Quantile Scaling:
                  x_scaled = min(1.0, log(1 + x) / log(1 + q99))
    """
    m6a_track = np.maximum(0.0, m6a_track)

    if norm_method == "none":
        return m6a_track

    if norm_method == "unit":
        # Score is typically 0-100% from modkit/Bambu
        max_val = max(100.0, float(np.max(m6a_track))) if np.any(m6a_track > 1.0) else 1.0
        return np.clip(m6a_track / max_val, 0.0, 1.0)

    if norm_method == "binary":
        return (m6a_track > 0.0).astype(np.float32)

    if norm_method == "log":
        return np.log1p(m6a_track)

    if norm_method in ["minmax", "min_max"]:
        denom = np.log1p(float(m6a_q99))
        if denom > 0:
            return np.clip(np.log1p(m6a_track) / denom, 0.0, 1.0)
        return m6a_track

    raise ValueError(
        f"Unknown normalization method: '{norm_method}'. "
        "(Allowed: 'none', 'unit', 'binary', 'log', 'minmax')"
    )


# =============================================================================
# 4. Incremental Chunk Storage & Merging
# =============================================================================

def save_chunk(
    chunk_idx: int,
    chunk_items: list,
    chunks_dir: Path,
    normalization: str,
    score_column: str,
    m6a_q99: Optional[float] = None,
):
    """Saves a block of processed transcripts incrementally as an NPZ archive."""
    chunk_file = chunks_dir / f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(
        chunk_file,
        tracks=np.array([item["track"] for item in chunk_items], dtype=object),
        ensembl_transcript_id=np.array([item["transcript_id"] for item in chunk_items]),
        ensembl_gene_id=np.array([item["gene_id"] for item in chunk_items]),
        hgnc_symbol=np.array([item["gene_symbol"] for item in chunk_items]),
        half_life_transformed=np.array([item["half_life_transformed"] for item in chunk_items], dtype=np.float32),
        half_life=np.array([item["half_life"] for item in chunk_items], dtype=np.float32),
        rate=np.array([item["rate"] for item in chunk_items], dtype=np.float32),
        seq_lens=np.array([item["length"] for item in chunk_items], dtype=np.int32),
        has_m6a=np.array([item["has_m6a"] for item in chunk_items], dtype=bool),
        num_m6a_sites=np.array([item["num_m6a_sites"] for item in chunk_items], dtype=np.int32),
        mean_m6a_score=np.array([item["mean_m6a_score"] for item in chunk_items], dtype=np.float32),
        normalization=str(normalization),
        score_column=str(score_column),
        m6a_q99=np.float32(m6a_q99) if m6a_q99 is not None else np.nan,
    )


def merge_all_chunks(
    chunks_dir: Path,
    output_file: Path,
    normalization: str,
    score_column: str,
    m6a_q99: Optional[float] = None,
):
    """Merges all generated chunk archives into the final master NPZ."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise RuntimeError(f"No chunks found in directory {chunks_dir} to merge!")

    print(f"\nMerging {len(chunk_files)} chunks into {output_file}...")
    all_tracks = []
    tx_ids = []
    gene_ids = []
    gene_symbols = []
    hl_trans = []
    hl_raw = []
    rates = []
    lens = []
    has_m6a_flags = []
    site_counts = []
    mean_scores = []

    for cf in tqdm(chunk_files, desc="Merging chunks"):
        data = np.load(cf, allow_pickle=True)
        all_tracks.extend(data["tracks"])
        tx_ids.extend(data["ensembl_transcript_id"])
        gene_ids.extend(data["ensembl_gene_id"])
        gene_symbols.extend(data["hgnc_symbol"])
        hl_trans.extend(data["half_life_transformed"])
        hl_raw.extend(data["half_life"])
        rates.extend(data["rate"])
        lens.extend(data["seq_lens"])
        has_m6a_flags.extend(data["has_m6a"])
        site_counts.extend(data["num_m6a_sites"])
        mean_scores.extend(data["mean_m6a_score"])

    np.savez_compressed(
        output_file,
        tracks=np.array(all_tracks, dtype=object),
        ensembl_transcript_id=np.array(tx_ids),
        ensembl_gene_id=np.array(gene_ids),
        hgnc_symbol=np.array(gene_symbols),
        half_life_transformed=np.array(hl_trans, dtype=np.float32),
        half_life=np.array(hl_raw, dtype=np.float32),
        rate=np.array(rates, dtype=np.float32),
        seq_lens=np.array(lens, dtype=np.int32),
        has_m6a=np.array(has_m6a_flags, dtype=bool),
        num_m6a_sites=np.array(site_counts, dtype=np.int32),
        mean_m6a_score=np.array(mean_scores, dtype=np.float32),
        normalization=str(normalization),
        score_column=str(score_column),
        m6a_q99=np.float32(m6a_q99) if m6a_q99 is not None else np.nan,
    )

    n_total = len(tx_ids)
    n_with_m6a = int(sum(has_m6a_flags))
    total_sites = int(sum(site_counts))

    print("\n" + "=" * 70)
    print("             m6A 7-TRACK DATASET GENERATION REPORT            ")
    print("=" * 70)
    print(f"Total transcripts processed:         {n_total:,}")
    print(f"Transcripts with validated m6A:      {n_with_m6a:,} ({n_with_m6a / max(1, n_total) * 100:.2f} %)")
    print(f"Total m6A modification sites mapped: {total_sites:,}")
    if n_with_m6a > 0:
        print(f"Mean sites per modified transcript:  {total_sites / n_with_m6a:.2f}")
    print(f"Channel configuration (7 channels):  [A, C, G, U, CDS, Splice, m6A]")
    print(f"Track dimensions per transcript:     (L, 7)")
    print(f"Coordinate mapping:                  cDNA (direct)")
    print(f"Normalization method:                {normalization.upper()}")
    print(f"Score metric:                        {score_column}")
    if m6a_q99 is not None and not np.isnan(m6a_q99):
        print(f"m6A q99 reference bound:             {m6a_q99:.4f}")
    print(f"Final output file saved:             {output_file}")
    print("=" * 70 + "\n")


# =============================================================================
# 5. Main Pipeline Routine
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate continuous/discrete m6A modification tracks (7-track) for the Orthrus model using cDNA coordinates."
    )
    parser.add_argument(
        "--saluki_data",
        "--data_path",
        dest="saluki_data",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt",
        help="Path to hIPSC_CM base dataset file (tab-separated)",
    )
    parser.add_argument(
        "--m6a_file",
        "--m6a_dir",
        dest="m6a_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/m6a/validated_m6a_sites.tsv",
        help="Path to 'validated_m6a_sites.tsv' or the directory containing it",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/hIPSC_CM_7track_m6a.npz",
        help="Destination NPZ archive path (automatically suffixed with normalization method)",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="mean_score",
        choices=["mean_score", "effect_size", "a_pct_modified", "max_score", "binary"],
        help="Metric to map to channel 6: 'mean_score' (modification %% in Ctrl), 'effect_size' (drop upon M3inh), 'a_pct_modified', 'binary'",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="unit",
        choices=["unit", "minmax", "binary", "log", "none"],
        help="Normalization method for m6A track (channel 6): 'unit' (0-100%% -> [0, 1]), 'minmax' (robust q99 log-scaling), 'binary', 'log', 'none'",
    )
    parser.add_argument(
        "--m6a_q99",
        type=float,
        default=None,
        help="Upper reference bound for minmax quantile scaling (default: dynamically calculated from data)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Percentile to compute dynamically when using minmax (default: 0.99)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Number of transcripts per incremental storage chunk (default: 500)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length for Orthrus (default: 12288 bp)",
    )
    parser.add_argument(
        "--filter_drach",
        action="store_true",
        help="Only include sites matching the canonical DRACH consensus motif",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Forces complete regeneration and deletes existing chunks",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    norm_method = args.normalization.lower()
    out_file = Path(args.output_file)

    # Append normalization suffix if not already present
    if norm_method != "none":
        stem = out_file.stem
        if not stem.endswith(f"_{norm_method}"):
            out_file = out_file.parent / f"{stem}_{norm_method}{out_file.suffix}"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("   ORTHRUS 7-TRACK m6A DATASET GENERATOR (cDNA Mapping)   ")
    print("=" * 75)
    print(f"Mapping:            cDNA Coordinates (direct)")
    print(f"Score metric:       {args.score_col}")
    print(f"Normalization:      {norm_method.upper()}")
    print(f"Output archive:     {out_file}")
    print(f"Chunks directory:   {chunks_dir}")
    print(f"Chunk size:         {args.chunk_size}")

    # Check for existing chunks for resuming
    completed_tx_ids = set()
    existing_chunks = sorted(chunks_dir.glob("chunk_*.npz"))

    if args.recreate:
        print(f"[Recreate] Removing {len(existing_chunks)} existing chunks...")
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
            print(f"[Resume] Found {len(completed_tx_ids)} already processed transcripts in {len(existing_chunks)} chunks.")

    # 1. Strict loading of validated m6A data (validated_m6a_sites.tsv only, no fallback)
    cdna_lookup, m6a_stats = load_validated_m6a_data(
        m6a_input_path=args.m6a_file,
        score_col=args.score_col,
        filter_drach=args.filter_drach
    )

    # 2. Load base dataset
    saluki_path = Path(args.saluki_data)
    print(f"\n[Dataset] Loading base transcript sequences from: {saluki_path}...")
    if not saluki_path.exists():
        raise FileNotFoundError(f"[ERROR] Base dataset file not found at: {saluki_path}")

    df_base = pd.read_csv(saluki_path, sep="\t")

    # Verify base columns
    required_base_cols = ["ensembl_transcript_id", "sequence"]
    missing_base_cols = [c for c in required_base_cols if c not in df_base.columns]
    if missing_base_cols:
        raise KeyError(
            f"[ERROR] Required column(s) missing in base dataset '{saluki_path}': {missing_base_cols}!"
        )

    total_transcripts = len(df_base)
    print(f"          Total entries in base dataset: {total_transcripts:,}")

    # 3. Dynamic quantile estimation for minmax normalization
    m6a_q99 = args.m6a_q99
    if norm_method in ["minmax", "min_max"]:
        if not args.recreate and existing_chunks:
            for cf in existing_chunks:
                try:
                    c_data = np.load(cf, allow_pickle=True)
                    if m6a_q99 is None and "m6a_q99" in c_data and not np.isnan(c_data["m6a_q99"]):
                        m6a_q99 = float(c_data["m6a_q99"])
                        print(f"[Resume] Reusing m6A q99 from {cf.name}: {m6a_q99:.4f}")
                        break
                except Exception:
                    continue

        if m6a_q99 is None:
            m6a_q99 = compute_m6a_dynamic_quantile(
                df=df_base,
                cdna_lookup=cdna_lookup,
                max_length=args.max_length,
                quantile=args.quantile,
            )

        print(f"[Scaling] Active m6A reference bound: q99 = {m6a_q99:.4f}\n")

    # 4. Process transcripts (exclusively cDNA mapping)
    chunk_idx = len(existing_chunks)
    current_chunk_items = []

    print("\nProcessing transcripts into 7-track arrays (cDNA mapping)...")
    with tqdm(total=total_transcripts, desc="Progress", initial=len(completed_tx_ids)) as pbar:
        for idx, row in df_base.iterrows():
            tx_id = str(row["ensembl_transcript_id"])
            clean_tx = tx_id.split(".")[0]

            if tx_id in completed_tx_ids or clean_tx in completed_tx_ids:
                continue

            raw_seq = str(row["sequence"])

            # Channels 0-5 (A, C, G, U, CDS, Splice)
            six_track, l, utr3_start, upper_indices = parse_saluki_base_tracks(raw_seq)
            if l > args.max_length:
                six_track = six_track[:args.max_length, :]
                l = args.max_length

            # Channel 6: m6A track via cDNA coordinates
            m6a_1d = np.zeros(l, dtype=np.float32)
            has_m6a = False
            num_sites = 0

            # Direct cDNA mapping
            sites = cdna_lookup.get(tx_id) or cdna_lookup.get(clean_tx)
            if sites:
                for s in sites:
                    c_start = s["cDNAstart"]
                    c_end = s["cDNAend"]
                    score_val = s["score"]

                    if 0 <= c_start < l:
                        # 0-based half-open interval [c_start, c_end)
                        end_pos = min(l, max(c_start + 1, c_end))
                        m6a_1d[c_start:end_pos] = np.maximum(m6a_1d[c_start:end_pos], score_val)
                        has_m6a = True
                        num_sites += 1

            # Apply normalization to channel 6
            m6a_norm = normalize_m6a_track(
                m6a_1d,
                norm_method=norm_method,
                m6a_q99=m6a_q99 if m6a_q99 is not None else 100.0
            )

            # Concatenate to (L, 7)
            m6a_channel = m6a_norm.reshape(-1, 1)
            multi_track = np.concatenate([six_track, m6a_channel], axis=1)

            mean_score_val = float(np.mean(m6a_1d[m6a_1d > 0])) if has_m6a else 0.0

            current_chunk_items.append({
                "track": multi_track,
                "transcript_id": tx_id,
                "gene_id": str(row.get("ensembl_gene_id", "")),
                "gene_symbol": str(row.get("hgnc_symbol", "")),
                "half_life_transformed": float(row.get("half_life_transformed", np.nan)),
                "half_life": float(row.get("half_life", np.nan)),
                "rate": float(row.get("rate", np.nan)),
                "length": l,
                "has_m6a": has_m6a,
                "num_m6a_sites": num_sites,
                "mean_m6a_score": mean_score_val,
            })
            pbar.update(1)

            # Save incrementally
            if len(current_chunk_items) >= args.chunk_size:
                save_chunk(
                    chunk_idx=chunk_idx,
                    chunk_items=current_chunk_items,
                    chunks_dir=chunks_dir,
                    normalization=norm_method,
                    score_column=args.score_col,
                    m6a_q99=m6a_q99,
                )
                chunk_idx += 1
                current_chunk_items = []

        # Save any remaining transcripts in the final chunk
        if current_chunk_items:
            save_chunk(
                chunk_idx=chunk_idx,
                chunk_items=current_chunk_items,
                chunks_dir=chunks_dir,
                normalization=norm_method,
                score_column=args.score_col,
                m6a_q99=m6a_q99,
            )

    # 5. Merge all chunks into final NPZ archive
    merge_all_chunks(
        chunks_dir=chunks_dir,
        output_file=out_file,
        normalization=norm_method,
        score_column=args.score_col,
        m6a_q99=m6a_q99,
    )


if __name__ == "__main__":
    main()
