#!/usr/bin/env python3
"""
Generation of continuous directional RBP density tracks (2 RBP Channels: Stabilizers & Destabilizers)
for the hIPSC_CM dataset with incremental saving and O(log N) vectorization.

Channels (Total: 8):
  0-3: One-Hot Sequence (A, C, G, U)
  4:   CDS Annotation (1.0 in CDS, 0.0 in UTRs)
  5:   Exon-Junction / Splice Sites (1.0 at splice sites, 0.0 elsewhere)
  6:   Stabilizer RBPs (ENCODE eCLIP peaks of curated stabilizing factors)
  7:   Destabilizer RBPs (ENCODE eCLIP peaks of curated destabilizing factors)

Note:
  TargetScan miRNA channel was retired due to low dataset coverage (74% zero-inflation).
  The two separate RBP channels provide directional regulatory signals without mutual cancellation.

Features:
- Fast binary search (np.searchsorted): Drastically reduces runtime
- Incremental storage in chunks (resumable upon interruption)
- Dynamic robust quantile scaling (MinMax / Log)
- Automatic merging into the final NPZ file
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict
import gffutils
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. Fast Interval Index (O(log N) search instead of O(N) loop)
# =============================================================================

class FastIntervalIndex:
    """
    Indexes genomic peaks of a chromosome/strand for extremely fast
    overlap queries using sorted NumPy arrays and np.searchsorted.
    """
    def __init__(self, intervals: list):
        if not intervals:
            self.empty = True
            return
        self.empty = False
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([x[0] for x in intervals_sorted], dtype=np.int64)
        self.ends = np.array([x[1] for x in intervals_sorted], dtype=np.int64)
        self.scores = np.array([x[2] for x in intervals_sorted], dtype=np.float32)
        self.max_peak_len = int(np.max(self.ends - self.starts)) if len(self.ends) > 0 else 500

    def get_overlaps(self, ex_start: int, ex_end: int):
        if self.empty:
            return None, None, None

        # Only check peaks whose start <= ex_end and >= ex_start - max_peak_len
        left_idx = np.searchsorted(self.starts, ex_start - self.max_peak_len, side="left")
        right_idx = np.searchsorted(self.starts, ex_end, side="right")

        if left_idx >= right_idx:
            return None, None, None

        sub_starts = self.starts[left_idx:right_idx]
        sub_ends = self.ends[left_idx:right_idx]
        sub_scores = self.scores[left_idx:right_idx]

        # Exact intersection: peak ends after ex_start and begins before ex_end
        mask = (sub_ends >= ex_start) & (sub_starts <= ex_end)
        if not np.any(mask):
            return None, None, None

        return sub_starts[mask], sub_ends[mask], sub_scores[mask]


# =============================================================================
# 2. GTF SQLite Database Access & Coordinate Mapping (via gffutils)
# =============================================================================

class GtfDbHelper:
    """
    Encapsulates access to the GTF FeatureDB via gffutils.
    """
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"[Warning] GTF-DB file '{self.db_path}' not found locally (runs on cluster).")

        print(f"[GTF-DB] Loading gffutils.FeatureDB: {self.db_path.name}")
        self.db = gffutils.FeatureDB(str(self.db_path))

    def get_transcript_exons(self, transcript_id: str) -> dict:
        """
        Returns chromosome, strand, and a list of exon intervals (start, end)
        ordered in transcription direction (5' -> 3').
        """
        clean_id = transcript_id.split(".")[0]

        tx = None
        for candidate in [transcript_id, clean_id]:
            try:
                tx = self.db[candidate]
                break
            except Exception:
                continue

        if tx is None:
            return None

        strand = tx.strand
        chrom = tx.chrom
        exons = [
            (exon.start, exon.end)
            for exon in self.db.children(tx, featuretype="exon", order_by="start")
        ]

        if strand == "-":
            exons.reverse()

        return {"chrom": chrom, "strand": strand, "exons": exons}


def map_genomic_intervals_to_transcript(
    exons: list,
    strand: str,
    peak_index: FastIntervalIndex,
    transcript_len: int,
) -> np.ndarray:
    """
    Maps genomic peaks to mature mRNA with high performance via binary search.
    """
    track = np.zeros(transcript_len, dtype=np.float32)
    if not exons or peak_index.empty:
        return track

    curr_tx_pos = 0
    for ex_start, ex_end in exons:
        ex_len = ex_end - ex_start + 1
        p_starts, p_ends, p_scores = peak_index.get_overlaps(ex_start, ex_end)

        if p_starts is not None:
            for p_start, p_end, score in zip(p_starts, p_ends, p_scores):
                overlap_start = max(ex_start, int(p_start))
                overlap_end = min(ex_end, int(p_end))

                if strand == "+":
                    rel_start = curr_tx_pos + (overlap_start - ex_start)
                    rel_end = curr_tx_pos + (overlap_end - ex_start) + 1
                else:
                    rel_start = curr_tx_pos + (ex_end - overlap_end)
                    rel_end = curr_tx_pos + (ex_end - overlap_start) + 1

                rel_start = max(0, min(rel_start, transcript_len))
                rel_end = max(0, min(rel_end, transcript_len))

                if rel_start < rel_end:
                    track[rel_start:rel_end] += float(score)

        curr_tx_pos += ex_len

    return track


# =============================================================================
# 3. Parser for BED Files (ENCODE eCLIP Stabilizers & Destabilizers)
# =============================================================================

def load_bed_indexed(bed_path: Path, label: str = "BED") -> dict:
    """
    Loads eCLIP peaks from a BED file and builds a FastIntervalIndex for each (chrom, strand).
    """
    if not bed_path.exists():
        raise FileNotFoundError(f"{label} file '{bed_path}' not found.")

    print(f"Loading {label} intervals from: {bed_path}...")
    raw_data = {}
    with open(bed_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or line.startswith("track") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            chrom = parts[0].replace("chr", "")
            try:
                start = int(parts[1]) + 1   # BED 0-based -> 1-based (like GTF)
                end = int(parts[2])         # BED end is exclusive, matches 1-based inclusive
            except ValueError:
                continue

            if start >= end:
                continue

            strand = parts[5]
            if strand not in ["+", "-"]:
                continue

            # SignalValue (column 6 in narrowPeak / BED): default to 1.0 if missing
            score = 1.0
            if len(parts) >= 7 and parts[6] not in [".", "-1", "nan", "NaN", ""]:
                try:
                    s_val = float(parts[6])
                    if s_val > 0.0:
                        score = s_val
                    else:
                        continue
                except ValueError:
                    score = 1.0

            key = (chrom, strand)
            raw_data.setdefault(key, []).append((start, end, score))

    # Create FastIntervalIndex per chromosome/strand
    indexed_data = {}
    for key, intervals in raw_data.items():
        indexed_data[key] = FastIntervalIndex(intervals)

    total_peaks = sum(len(v) for v in raw_data.values())
    print(f"{label}: Indexed {total_peaks:,} peaks across {len(indexed_data)} (chrom, strand) contigs.")
    return indexed_data


# =============================================================================
# 4. Saluki 6-Track Base Parser
# =============================================================================

def parse_saluki_base_tracks(raw_seq: str) -> tuple:
    """
    Parses comma-separated token sequence into Saluki 6-track format:
    Channels 0-3: One-hot nucleotides (A, C, G, U)
    Channel 4:    CDS (1.0 if token is uppercase, 0.0 otherwise)
    Channel 5:    Splice site / Exon junction (1.0 if 'ej' in token)
    """
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    l = len(tokens)
    if l == 0:
        return np.zeros((0, 6), dtype=np.float32), 0

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
    return six_track, l


# =============================================================================
# 5. Normalization for RBP Tracks (Channels 6 & 7)
# =============================================================================

def compute_dynamic_quantiles(
    df: pd.DataFrame,
    stab_data: dict,
    destab_data: dict,
    gtf_helper: GtfDbHelper,
    max_length: int = 12288,
    quantile: float = 0.99,
    sample_size: int = 2000,
) -> tuple:
    """
    Dynamically determines high percentiles (default: 99th percentile, q99)
    of non-zero RBP signal values across the dataset for Stabilizer and Destabilizer channels.
    """
    q_pct = quantile * 100.0 if quantile <= 1.0 else quantile
    print(f"\n[Quantile Estimation] Dynamically calculating {q_pct:.1f}th percentile (q{int(q_pct)}) across dataset...")

    if sample_size is not None and sample_size > 0 and len(df) > sample_size:
        eval_df = df.sample(n=sample_size, random_state=42)
        print(f"[Quantile Estimation] Subsampling {sample_size} transcripts for fast percentile estimation...")
    else:
        eval_df = df
        print(f"[Quantile Estimation] Scanning all {len(eval_df)} transcripts...")

    all_stab_vals = []
    all_destab_vals = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Estimating quantiles"):
        tx_id = str(row.get("ensembl_transcript_id", ""))
        clean_tx = tx_id.split(".")[0]
        raw_seq = str(row["sequence"])
        l = min(len(raw_seq.split(",")), max_length)

        tx_info = gtf_helper.get_transcript_exons(clean_tx)
        if tx_info is not None:
            chrom = str(tx_info["chrom"]).replace("chr", "")
            strand = tx_info["strand"]
            exons = tx_info["exons"]
            key = (chrom, strand)

            if key in stab_data:
                stab_1d = map_genomic_intervals_to_transcript(exons, strand, stab_data[key], l)
                nz_stab = stab_1d[stab_1d > 0]
                if len(nz_stab) > 0:
                    all_stab_vals.append(nz_stab)

            if key in destab_data:
                destab_1d = map_genomic_intervals_to_transcript(exons, strand, destab_data[key], l)
                nz_destab = destab_1d[destab_1d > 0]
                if len(nz_destab) > 0:
                    all_destab_vals.append(nz_destab)

    if all_stab_vals:
        flat_stab = np.concatenate(all_stab_vals)
        stab_q = float(np.percentile(flat_stab, q_pct))
        print(f"[Quantile Estimation] Stabilizer RBP {q_pct:.1f}% percentile: {stab_q:.4f} (from {len(flat_stab):,} active positions)")
    else:
        stab_q = 600.0
        print(f"[Quantile Estimation] Warning: No positive Stabilizer RBP values found, fallback to {stab_q:.4f}")

    if all_destab_vals:
        flat_destab = np.concatenate(all_destab_vals)
        destab_q = float(np.percentile(flat_destab, q_pct))
        print(f"[Quantile Estimation] Destabilizer RBP {q_pct:.1f}% percentile: {destab_q:.4f} (from {len(flat_destab):,} active positions)")
    else:
        destab_q = 600.0
        print(f"[Quantile Estimation] Warning: No positive Destabilizer RBP values found, fallback to {destab_q:.4f}")

    return stab_q, destab_q


def normalize_rbp_tracks(
    stab_track: np.ndarray,
    destab_track: np.ndarray,
    norm_method: str,
    stab_q99: float = 600.0,
    destab_q99: float = 600.0,
) -> tuple:
    """
    Normalizes continuous channels 6 (Stabilizer RBP) and 7 (Destabilizer RBP) using global scaling.
    
    Methods:
      - 'none': Keep raw values (>= 0)
      - 'log': np.log1p(x) -> log(1 + x)
      - 'minmax': Log-transformation with Robust Quantile Scaling:
                  x_scaled = min(1.0, log(1 + x) / log(1 + q_99))
    """
    # Clamp negative noise cleanly to 0
    stab_track = np.maximum(0.0, stab_track)
    destab_track = np.maximum(0.0, destab_track)

    if norm_method == "none":
        return stab_track, destab_track

    if norm_method == "log":
        return np.log1p(stab_track), np.log1p(destab_track)

    if norm_method in ["minmax", "min_max"]:
        denom_stab = np.log1p(float(stab_q99))
        denom_destab = np.log1p(float(destab_q99))
        norm_stab = (
            np.clip(np.log1p(stab_track) / denom_stab, 0.0, 1.0)
            if denom_stab > 0
            else stab_track
        )
        norm_destab = (
            np.clip(np.log1p(destab_track) / denom_destab, 0.0, 1.0)
            if denom_destab > 0
            else destab_track
        )
        return norm_stab, norm_destab

    raise ValueError(
        f"Unknown normalization method: '{norm_method}' (allowed: 'none', 'log', 'minmax')"
    )


# =============================================================================
# 6. Incremental Chunking & Merging
# =============================================================================

def save_chunk(
    chunk_idx: int,
    chunk_items: list,
    chunks_dir: Path,
    normalization: str = "none",
    stab_q99: float = None,
    destab_q99: float = None,
):
    """Saves a block of transcripts incrementally as NPZ."""
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
        has_stab_rbp=np.array([item["has_stab_rbp"] for item in chunk_items], dtype=bool),
        has_destab_rbp=np.array([item["has_destab_rbp"] for item in chunk_items], dtype=bool),
        has_gtf=np.array([item["has_gtf"] for item in chunk_items], dtype=bool),
        normalization=str(normalization),
        stab_q99=np.float32(stab_q99) if stab_q99 is not None else np.nan,
        destab_q99=np.float32(destab_q99) if destab_q99 is not None else np.nan,
    )


def merge_all_chunks(
    chunks_dir: Path,
    output_file: Path,
    normalization: str = "none",
    stab_q99: float = None,
    destab_q99: float = None,
):
    """Merges all generated chunks into the final master NPZ."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        print("[Error] No chunks found to merge!")
        return

    print(f"\nMerging {len(chunk_files)} chunks into {output_file}...")
    all_tracks = []
    tx_ids = []
    gene_ids = []
    gene_symbols = []
    hl_trans = []
    hl_raw = []
    rates = []
    lens = []
    stab_flags = []
    destab_flags = []
    gtf_flags = []

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
        stab_flags.extend(data["has_stab_rbp"])
        destab_flags.extend(data["has_destab_rbp"])
        gtf_flags.extend(data["has_gtf"])

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
        has_stab_rbp=np.array(stab_flags, dtype=bool),
        has_destab_rbp=np.array(destab_flags, dtype=bool),
        has_gtf=np.array(gtf_flags, dtype=bool),
        normalization=str(normalization),
        stab_q99=np.float32(stab_q99) if stab_q99 is not None else np.nan,
        destab_q99=np.float32(destab_q99) if destab_q99 is not None else np.nan,
    )

    n = len(tx_ids)
    print("\n" + "=" * 70)
    print("             COVERAGE & STATISTICS REPORT (2 RBP TRACKS)             ")
    print("=" * 70)
    print(f"Total transcripts:                     {n}")
    print(f"Successfully mapped in GTF DB:         {sum(gtf_flags)} ({sum(gtf_flags)/n*100:.2f} %)")
    print(f"Transcripts with Stabilizer RBP peaks: {sum(stab_flags)} ({sum(stab_flags)/n*100:.2f} %)")
    print(f"Transcripts with Destabilizer peaks:   {sum(destab_flags)} ({sum(destab_flags)/n*100:.2f} %)")
    print(f"RBP Normalization:                     {normalization.upper()} (Global Scaling)")
    if stab_q99 is not None and not np.isnan(stab_q99):
        print(f"Global Stabilizer q99 bound:           {stab_q99:.4f}")
    if destab_q99 is not None and not np.isnan(destab_q99):
        print(f"Global Destabilizer q99 bound:         {destab_q99:.4f}")
    print(f"Track dimensions per transcript:       (L, 8)")
    print(f"Channel configuration:                 [A, C, G, U, CDS, Splice, RBP_Stab, RBP_Destab]")
    print(f"Final file saved:                      {output_file}")
    print("=" * 70)


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate continuous 2-channel RBP density tracks (Stabilizers & Destabilizers) for hIPSC_CM"
    )
    parser.add_argument(
        "--saluki_data",
        "--data_path",
        dest="saluki_data",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt",
        help="Path to hIPSC_CM data file (tab-separated)",
    )
    parser.add_argument(
        "--gtf_db",
        type=str,
        default="/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db",
        help="Path to GTF SQLite DB",
    )
    parser.add_argument(
        "--stabilizers_bed",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/curated_candidates/stabilizers_peaks.bed",
        help="Path to Stabilizers eCLIP BED file (Channel 6)",
    )
    parser.add_argument(
        "--destabilizers_bed",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/curated_candidates/destabilizers_peaks.bed",
        help="Path to Destabilizers eCLIP BED file (Channel 7)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/hIPSC_CM_8track_2rbp.npz",
        help="Output file for the 8-track NPZ archive (automatically appended with normalization suffix)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["none", "log", "minmax"],
        help="Normalization method for RBP tracks (channels 6 & 7): 'none', 'log', or 'minmax' (default: minmax)",
    )
    parser.add_argument(
        "--stab_q99",
        type=float,
        default=None,
        help="99th percentile (q99) reference bound for Stabilizer RBP channel (default: None, dynamically estimated)",
    )
    parser.add_argument(
        "--destab_q99",
        type=float,
        default=None,
        help="99th percentile (q99) reference bound for Destabilizer RBP channel (default: None, dynamically estimated)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Percentile to compute dynamically (default: 0.99 for 99th percentile)",
    )
    parser.add_argument(
        "--quantile_sample_size",
        type=int,
        default=2000,
        help="Number of transcripts to sample for dynamic quantile estimation (default: 2000, set to 0 for all transcripts)",
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
        help="Maximum sequence length (default: 12288 bp)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Forces regeneration of all chunks and overwrites/deletes existing chunks.",
    )
    args = parser.parse_args()

    norm_method = args.normalization.lower()
    out_file = Path(args.output_file)

    # Dynamically adjust filename and chunk folder according to normalization
    if norm_method != "none":
        stem = out_file.stem
        if not stem.endswith(f"_{norm_method}"):
            out_file = out_file.parent / f"{stem}_{norm_method}{out_file.suffix}"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   Generation of 2-Channel Directional RBP Tracks (Stabilizers vs. Destabilizers)   ")
    print("=" * 80)
    print(f"Normalization:      {norm_method.upper()} (Global Robust Quantile Scaling)")
    print(f"Output file:        {out_file}")
    print(f"Chunk directory:    {chunks_dir} (Chunk size: {args.chunk_size})")

    # Check for already processed transcripts for seamless resume
    completed_tx_ids = set()
    existing_chunks = sorted(chunks_dir.glob("chunk_*.npz"))

    if args.recreate:
        print(f"[Recreate] Flag --recreate active: Removing {len(existing_chunks)} existing chunks for a complete restart...")
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
            print(f"[Resume] Found {len(completed_tx_ids)} already processed transcripts in {len(existing_chunks)} chunks!")

    # 1. Load GTF database & indexed RBP BED files
    gtf_helper = GtfDbHelper(Path(args.gtf_db))
    stab_data = load_bed_indexed(Path(args.stabilizers_bed), label="Stabilizers RBP")
    destab_data = load_bed_indexed(Path(args.destabilizers_bed), label="Destabilizers RBP")

    # 2. Load hIPSC_CM dataset
    data_path = Path(args.saluki_data)
    print(f"\nLoading hIPSC_CM dataset: {data_path}...")
    df = pd.read_csv(data_path, sep="\t")

    total_samples = len(df)
    print(f"Total entries in hIPSC_CM: {total_samples}")

    # Dynamic quantile determination for global scaling
    stab_q99 = args.stab_q99
    destab_q99 = args.destab_q99

    if norm_method in ["minmax", "min_max"]:
        # Check existing chunks for saved quantiles (to ensure consistency across resume)
        if not args.recreate and existing_chunks:
            for cf in existing_chunks:
                try:
                    c_data = np.load(cf, allow_pickle=True)
                    if stab_q99 is None and "stab_q99" in c_data and not np.isnan(c_data["stab_q99"]):
                        stab_q99 = float(c_data["stab_q99"])
                    if destab_q99 is None and "destab_q99" in c_data and not np.isnan(c_data["destab_q99"]):
                        destab_q99 = float(c_data["destab_q99"])
                    if stab_q99 is not None and destab_q99 is not None:
                        print(f"[Resume] Reusing dynamic quantiles from {cf.name}: Stabilizer q99={stab_q99:.4f}, Destabilizer q99={destab_q99:.4f}")
                        break
                except Exception:
                    continue

        if stab_q99 is None or destab_q99 is None:
            dyn_stab, dyn_destab = compute_dynamic_quantiles(
                df=df,
                stab_data=stab_data,
                destab_data=destab_data,
                gtf_helper=gtf_helper,
                max_length=args.max_length,
                quantile=args.quantile,
                sample_size=args.quantile_sample_size,
            )
            if stab_q99 is None:
                stab_q99 = dyn_stab
            if destab_q99 is None:
                destab_q99 = dyn_destab

        print(f"[Scaling Bounds] Active global bounds: Stabilizer q99 = {stab_q99:.4f}, Destabilizer q99 = {destab_q99:.4f}\n")

    # Determine next chunk index
    chunk_idx = len(existing_chunks)
    current_chunk_items = []

    print("\nProcessing transcripts (with fast O(log N) lookup & incremental saving)...")
    with tqdm(total=total_samples, desc="Progress", initial=len(completed_tx_ids)) as pbar:
        for idx, row in df.iterrows():
            tx_id = str(row.get("ensembl_transcript_id", ""))
            clean_tx = tx_id.split(".")[0]

            # Skip already completed transcripts
            if tx_id in completed_tx_ids or clean_tx in completed_tx_ids:
                continue

            raw_seq = str(row["sequence"])

            # Base 6-tracks (A, C, G, U, CDS, Splice)
            six_track, l = parse_saluki_base_tracks(raw_seq)
            if l > args.max_length:
                six_track = six_track[:args.max_length, :]
                l = args.max_length

            # Channel 6: Stabilizers RBP
            # Channel 7: Destabilizers RBP
            stab_track = np.zeros((l, 1), dtype=np.float32)
            destab_track = np.zeros((l, 1), dtype=np.float32)
            has_stab_rbp = False
            has_destab_rbp = False
            has_gtf = False

            tx_info = gtf_helper.get_transcript_exons(clean_tx)
            if tx_info is not None:
                has_gtf = True
                chrom = str(tx_info["chrom"]).replace("chr", "")
                strand = tx_info["strand"]
                exons = tx_info["exons"]
                key = (chrom, strand)

                # Query Stabilizers Index
                if key in stab_data:
                    stab_1d = map_genomic_intervals_to_transcript(exons, strand, stab_data[key], l)
                    if np.any(stab_1d > 0):
                        has_stab_rbp = True
                        stab_track[:, 0] = stab_1d

                # Query Destabilizers Index
                if key in destab_data:
                    destab_1d = map_genomic_intervals_to_transcript(exons, strand, destab_data[key], l)
                    if np.any(destab_1d > 0):
                        has_destab_rbp = True
                        destab_track[:, 0] = destab_1d

            # Apply robust scaling / normalization (channels 6 & 7)
            stab_track, destab_track = normalize_rbp_tracks(
                stab_track,
                destab_track,
                norm_method=norm_method,
                stab_q99=stab_q99,
                destab_q99=destab_q99,
            )

            # Concatenate to (L, 8)
            multi_track = np.concatenate([six_track, stab_track, destab_track], axis=1)

            current_chunk_items.append({
                "track": multi_track,
                "transcript_id": tx_id,
                "gene_id": str(row.get("ensembl_gene_id", "")),
                "gene_symbol": str(row.get("hgnc_symbol", "")),
                "half_life_transformed": float(row.get("half_life_transformed", np.nan)),
                "half_life": float(row.get("half_life", np.nan)),
                "rate": float(row.get("rate", np.nan)),
                "length": l,
                "has_stab_rbp": has_stab_rbp,
                "has_destab_rbp": has_destab_rbp,
                "has_gtf": has_gtf,
            })
            pbar.update(1)

            # Save incrementally after every chunk_size transcripts
            if len(current_chunk_items) >= args.chunk_size:
                save_chunk(
                    chunk_idx,
                    current_chunk_items,
                    chunks_dir,
                    normalization=norm_method,
                    stab_q99=stab_q99,
                    destab_q99=destab_q99,
                )
                chunk_idx += 1
                current_chunk_items = []

        # Save last incomplete chunk
        if current_chunk_items:
            save_chunk(
                chunk_idx,
                current_chunk_items,
                chunks_dir,
                normalization=norm_method,
                stab_q99=stab_q99,
                destab_q99=destab_q99,
            )

    # 3. Merge all chunks into final file
    merge_all_chunks(
        chunks_dir,
        out_file,
        normalization=norm_method,
        stab_q99=stab_q99,
        destab_q99=destab_q99,
    )


if __name__ == "__main__":
    main()
