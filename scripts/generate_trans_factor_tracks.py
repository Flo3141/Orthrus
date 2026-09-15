#!/usr/bin/env python3
"""
Generation of continuous trans-factor density and affinity tracks (Strategy B)
for the hIPSC_CM dataset with incremental saving and O(log N) vectorization.

Combines:
1. GTF SQLite database (Homo_sapiens.GRCh38.108.gtf.db via gffutils)
2. TargetScan: miRNA binding affinities / context++ scores
3. ENCODE eCLIP: Experimental RBP peak signal values

Features:
- Fast binary search (np.searchsorted): Drastically reduces runtime
- Incremental storage in chunks (resumable upon interruption)
- Automatic merging into the final NPZ file
"""

import argparse
import os
from pathlib import Path
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
# 3. Parser for TargetScan and ENCODE eCLIP
# =============================================================================

def load_targetscan_data(targetscan_path: Path) -> dict:
    if not targetscan_path.exists():
        raise FileNotFoundError(f"TargetScan file '{targetscan_path}' not found.")

    print(f"Loading TargetScan data from: {targetscan_path}...")
    df_ts = pd.read_csv(targetscan_path, sep="\t", low_memory=False)
    tx_col = "Transcript ID"
    score_col = "weighted context++ score"
    # The two column names have different naming conventions
    start_col = "UTR_start"
    end_col = "UTR end"
    # Remove invalid rows without start, end, or score
    df_ts = df_ts.dropna(subset=[tx_col, start_col, end_col, score_col])

    mapping = {}
    for _, row in df_ts.iterrows():
        raw_tx = str(row[tx_col]).split(".")[0]
        score = abs(float(row[score_col]))

        start = int(row[start_col])
        end = int(row[end_col])

        if start >= end:
            continue

        mapping.setdefault(raw_tx, []).append((start, end, score))

    print(f"TargetScan: Loaded binding sites for {len(mapping)} unique transcripts.")
    return mapping


def load_eclip_indexed(bed_path: Path) -> dict:
    """
    Loads ENCODE eCLIP peaks and builds a FastIntervalIndex for each (chrom, strand).
    """
    if not bed_path.exists():
        raise FileNotFoundError(f"ENCODE eCLIP file '{bed_path}' not found.")

    print(f"Loading ENCODE eCLIP intervals from: {bed_path}...")
    raw_data = {}
    with open(bed_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or line.startswith("track") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue

            chrom = parts[0].replace("chr", "")
            try:
                start = int(parts[1]) + 1   # BED 0-based -> 1-based (like GTF)
                end = int(parts[2])         # BED end is already exclusive, thus corresponds to 1-based inclusive
            except ValueError:
                continue

            if start >= end:
                continue

            strand = parts[5]
            if strand not in ["+", "-"]:
                continue

            # SignalValue (parts[6]) must be present, positive, and a valid number
            if parts[6] in [".", "-1", "nan", "NaN", ""]:
                continue
            try:
                score = float(parts[6])
                # Negative signal values indicate de-enrichment (less signal than input control)
                # and do not represent RBP binding
                if score <= 0.0:
                    continue
            except ValueError:
                continue

            key = (chrom, strand)
            raw_data.setdefault(key, []).append((start, end, score))

    # Create FastIntervalIndex per chromosome/strand
    indexed_data = {}
    for key, intervals in raw_data.items():
        indexed_data[key] = FastIntervalIndex(intervals)

    total_peaks = sum(len(v) for v in raw_data.values())
    print(f"eCLIP: Indexed {total_peaks} peaks across {len(indexed_data)} (chrom, strand) combinations.")
    return indexed_data


# =============================================================================
# 4. Saluki 6-Track Base Parser
# =============================================================================

def parse_saluki_base_tracks(raw_seq: str) -> tuple:
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    l = len(tokens)
    if l == 0:
        return np.zeros((0, 6), dtype=np.float32), 0, 0, []

    clean_seq = "".join(tok[0] for tok in tokens)

    seq_bytes = np.frombuffer(clean_seq.upper().encode("ascii"), dtype=np.uint8)
    oh = np.zeros((l, 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0
    oh[seq_bytes == 67, 1] = 1.0
    oh[seq_bytes == 71, 2] = 1.0
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0

    cds_track = np.array([1.0 if tok[0].isupper() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)
    splice_track = np.array([1.0 if "ej" in tok.lower() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    six_track = np.concatenate([oh, cds_track, splice_track], axis=1)

    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    utr3_start = (max(upper_indices) + 3) if upper_indices else 0
    utr3_start = min(utr3_start, l)

    return six_track, l, utr3_start, upper_indices


# =============================================================================
# 5. Normalization for Continuous Trans-Factor Tracks (Channels 6 & 7)
# =============================================================================

def normalize_trans_factor_tracks(
    mirna_track: np.ndarray,
    eclip_track: np.ndarray,
    norm_method: str,
    mirna_max: float = 5.0,
    eclip_max: float = 600.0,
    minmax_mode: str = "global",
) -> tuple:
    """
    Normalizes the continuous channels 6 (miRNA) and 7 (eCLIP).
    
    Methods:
      - 'none': Keep unchanged (raw values)
      - 'log': np.log1p(x) -> log(1 + x)
      - 'minmax':
          - With minmax_mode='global': x / max_val (clipped to [0, 1])
          - With minmax_mode='sample': x / (max(x) if max(x) > 0 else 1.0)
    """
    # Clamp negative values (de-enrichment / noise) cleanly to 0 before transformation
    mirna_track = np.maximum(0.0, mirna_track)
    eclip_track = np.maximum(0.0, eclip_track)

    if norm_method == "none":
        return mirna_track, eclip_track

    if norm_method == "log":
        # log1p preserves sparsity (0.0 -> 0.0) and compresses extreme values
        norm_mirna = np.log1p(mirna_track)
        norm_eclip = np.log1p(eclip_track)
        return norm_mirna, norm_eclip

    if norm_method in ["minmax", "min_max"]:
        if minmax_mode == "sample":
            # Per-transcript min-max scaling
            m_max = float(np.max(mirna_track))
            e_max = float(np.max(eclip_track))
            norm_mirna = (mirna_track / m_max) if m_max > 0 else mirna_track
            norm_eclip = (eclip_track / e_max) if e_max > 0 else eclip_track
        else:
            # Global reference scaling to [0, 1]
            norm_mirna = np.clip(mirna_track / float(mirna_max), 0.0, 1.0)
            norm_eclip = np.clip(eclip_track / float(eclip_max), 0.0, 1.0)
        return norm_mirna, norm_eclip

    raise ValueError(f"Unknown normalization method: '{norm_method}' (allowed: 'none', 'log', 'minmax')")


# =============================================================================
# 6. Incremental Chunking & Merging
# =============================================================================

def save_chunk(chunk_idx: int, chunk_items: list, chunks_dir: Path, normalization: str = "none"):
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
        has_mirna=np.array([item["has_mirna"] for item in chunk_items], dtype=bool),
        has_eclip=np.array([item["has_eclip"] for item in chunk_items], dtype=bool),
        has_gtf=np.array([item["has_gtf"] for item in chunk_items], dtype=bool),
        normalization=str(normalization),
    )


def merge_all_chunks(chunks_dir: Path, output_file: Path, normalization: str = "none"):
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
    mirna_flags = []
    eclip_flags = []
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
        mirna_flags.extend(data["has_mirna"])
        eclip_flags.extend(data["has_eclip"])
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
        has_mirna=np.array(mirna_flags, dtype=bool),
        has_eclip=np.array(eclip_flags, dtype=bool),
        has_gtf=np.array(gtf_flags, dtype=bool),
        normalization=str(normalization),
    )

    n = len(tx_ids)
    print("\n" + "=" * 65)
    print("             COVERAGE & STATISTICS REPORT             ")
    print("=" * 65)
    print(f"Total transcripts:                   {n}")
    print(f"Successfully mapped in GTF DB:       {sum(gtf_flags)} ({sum(gtf_flags)/n*100:.2f} %)")
    print(f"Transcripts with TargetScan miRNAs:  {sum(mirna_flags)} ({sum(mirna_flags)/n*100:.2f} %)")
    print(f"Transcripts with ENCODE eCLIP peaks: {sum(eclip_flags)} ({sum(eclip_flags)/n*100:.2f} %)")
    print(f"Trans-factor normalization:          {normalization.upper()}")
    print(f"Track dimensions per transcript:     (L, 8)")
    print(f"Channel configuration:               [A, C, G, U, CDS, Splice, TargetScan, eCLIP]")
    print(f"Final file saved:                    {output_file}")
    print("=" * 65)


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate continuous trans-factor density tracks for hIPSC_CM (with incremental saving)"
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
        "--targetscan_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt",
        help="Path to TargetScan prediction table",
    )
    parser.add_argument(
        "--encode_eclip_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/all_rbp_peaks_merged.bed",
        help="Path to ENCODE eCLIP BED/narrowPeak file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_multitrack_with_trans_factors.npz",
        help="Output file for the augmented NPZ archive (automatically appended with normalization suffix)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "log", "minmax"],
        help="Normalization method for trans-factor tracks (channels 6 & 7): 'none', 'log' (np.log1p), or 'minmax' (scaled to [0, 1])",
    )
    parser.add_argument(
        "--minmax_mode",
        type=str,
        default="global",
        choices=["global", "sample"],
        help="Mode for 'minmax': 'global' (uses reference maxima mirna_max/eclip_max) or 'sample' (separately per transcript)",
    )
    parser.add_argument(
        "--mirna_max",
        type=float,
        default=5.0,
        help="Reference maximum for miRNA channel with global minmax (default: 5.0)",
    )
    parser.add_argument(
        "--eclip_max",
        type=float,
        default=600.0,
        help="Reference maximum for eCLIP channel with global minmax (default: 600.0)",
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
        # Only append if suffix is not already in the stem
        if not stem.endswith(f"_{norm_method}"):
            out_file = out_file.parent / f"{stem}_{norm_method}{out_file.suffix}"

    out_file.parent.mkdir(parents=True, exist_ok=True)

    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("   Generation of Trans-Factor Density Tracks (TargetScan, ENCODE eCLIP)   ")
    print("=" * 75)
    print(f"Normalization:      {norm_method.upper()} " + (f"(Mode: {args.minmax_mode})" if norm_method == "minmax" else ""))
    print(f"Output file:        {out_file}")
    print(f"Chunk directory:    {chunks_dir} (Chunk size: {args.chunk_size})")

    # Check for already processed transcripts for seamless resume or recreate
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

    # 1. Load databases & lookup tables
    gtf_helper = GtfDbHelper(Path(args.gtf_db))
    ts_data = load_targetscan_data(Path(args.targetscan_file))
    eclip_data = load_eclip_indexed(Path(args.encode_eclip_file))

    # 2. Load hIPSC_CM dataset
    data_path = Path(args.saluki_data)
    print(f"\nLoading hIPSC_CM dataset: {data_path}...")
    df = pd.read_csv(data_path, sep="\t")

    total_samples = len(df)
    print(f"Total entries in hIPSC_CM: {total_samples}")

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

            # Base 6-tracks
            six_track, l, utr3_start, upper_indices = parse_saluki_base_tracks(raw_seq)
            if l > args.max_length:
                six_track = six_track[:args.max_length, :]
                l = args.max_length

            # Channel 6: TargetScan miRNA
            mirna_track = np.zeros((l, 1), dtype=np.float32)
            has_mirna = False
            if clean_tx in ts_data and upper_indices:
                for start, end, score in ts_data[clean_tx]:
                    abs_start = utr3_start + (start - 1)
                    abs_end = utr3_start + end
                    if abs_start < l:
                        clamped_end = min(abs_end, l)
                        mirna_track[abs_start:clamped_end, 0] += score
                        has_mirna = True

            # Channel 7: ENCODE eCLIP via fast index
            eclip_track = np.zeros((l, 1), dtype=np.float32)
            has_eclip = False
            has_gtf = False

            tx_info = gtf_helper.get_transcript_exons(clean_tx)
            if tx_info is not None:
                has_gtf = True
                chrom = str(tx_info["chrom"]).replace("chr", "")
                strand = tx_info["strand"]
                exons = tx_info["exons"]
                key = (chrom, strand)

                if key in eclip_data:
                    eclip_1d = map_genomic_intervals_to_transcript(exons, strand, eclip_data[key], l)
                    if np.any(eclip_1d > 0):
                        has_eclip = True
                        eclip_track[:, 0] = eclip_1d

            # Apply trans-factor normalization (channels 6 & 7)
            mirna_track, eclip_track = normalize_trans_factor_tracks(
                mirna_track,
                eclip_track,
                norm_method=norm_method,
                mirna_max=args.mirna_max,
                eclip_max=args.eclip_max,
                minmax_mode=args.minmax_mode,
            )

            # Concatenate to (L, 8)
            multi_track = np.concatenate([six_track, mirna_track, eclip_track], axis=1)

            # Add to current chunk
            current_chunk_items.append({
                "track": multi_track,
                "transcript_id": tx_id,
                "gene_id": str(row.get("ensembl_gene_id", "")),
                "gene_symbol": str(row.get("hgnc_symbol", "")),
                "half_life_transformed": float(row.get("half_life_transformed", np.nan)),
                "half_life": float(row.get("half_life", np.nan)),
                "rate": float(row.get("rate", np.nan)),
                "length": l,
                "has_mirna": has_mirna,
                "has_eclip": has_eclip,
                "has_gtf": has_gtf,
            })
            pbar.update(1)

            # Save incrementally after every chunk_size transcripts
            if len(current_chunk_items) >= args.chunk_size:
                save_chunk(chunk_idx, current_chunk_items, chunks_dir, normalization=norm_method)
                chunk_idx += 1
                current_chunk_items = []

        # Save last incomplete chunk
        if current_chunk_items:
            save_chunk(chunk_idx, current_chunk_items, chunks_dir, normalization=norm_method)

    # 3. Merge all chunks into final file
    merge_all_chunks(chunks_dir, out_file, normalization=norm_method)


if __name__ == "__main__":
    main()
