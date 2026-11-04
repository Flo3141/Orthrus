#!/usr/bin/env python3
"""
Unified Track Generator for Orthrus (6-track base archive + trans-factor extension tracks).

Tracks (--track):
  6track            [A, C, G, U, CDS, Splice], parsed from the raw dataset via SixTrackDatasetLoader
  7track_m6a        6track + m6A             (validated m6A sites, cDNA coordinates)
  8track_2rbp       6track + RBP_Stabilizer + RBP_Destabilizer (ENCODE eCLIP peaks of curated RBPs)
  8track_miRNA_rbp  6track + miRNA + eCLIP   (TargetScan context++ scores, all ENCODE eCLIP peaks)

Extension tracks are built on the 6-track archive (generate it first). This guarantees identical transcripts,
order and 3' truncation across all tracks. Each extra channel is computed on the full transcript (using the stored
untruncated length and 3' UTR start), restricted to the kept 3' window and normalized with a global bound q99
(the 99th percentile of the non-zero signal in the kept windows of a transcript sample):
  minmax: min(1, log1p(x) / log1p(q99))   (not classic min-max scaling; name kept for existing file names)
  log:    log1p(x)
  none:   raw values

A new track only needs a new entry in TRACK_PRESETS (and, for a new data source, a new Channel class).

Features:
- Fast binary search (np.searchsorted) for genomic peak overlaps
- Incremental storage in chunks (resumable upon interruption, guarded by run_config.json)
- Automatic merging into the final NPZ file with companion .meta.json
"""

import argparse
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from six_track_dataset import SixTrackDatasetLoader
from util import (
    save_metadata, load_metadata, ensure_run_config, to_object_array, RUN_CONFIG_FILENAME, TRUNCATION_MODE,
    truncate_to_max_length, require_truncation_mode, DATA_DIR, DATASET_DIRS, GTF_DB_PATH,
)

BASE_TRACK = "6track"
BASE_CHANNEL_NAMES = ["A", "C", "G", "U", "CDS", "Splice"]
QUANTILES_FILENAME = "quantiles.json"


# =============================================================================
# Fixed Reference File Paths (Cluster Constants)
# =============================================================================
VALIDATED_M6A_PATH = DATA_DIR / "m6a" / "validated_m6a_sites.tsv"
STABILIZERS_BED_PATH = DATA_DIR / "eclip" / "curated_candidates" / "stabilizers_peaks.bed"
DESTABILIZERS_BED_PATH = DATA_DIR / "eclip" / "curated_candidates" / "destabilizers_peaks.bed"
TARGETSCAN_FILE_PATH = DATA_DIR / "targetscan" / "Predicted_Targets_Context_Scores.default_predictions.txt"
ENCODE_ECLIP_FILE_PATH = DATA_DIR / "eclip" / "all_rbp_peaks_merged.bed"

# The TargetScan context score file lists sites for all species of the UTR alignment
# (column 'Gene Tax ID'). Only rows of the target species refer to the transcript's own UTR.
TARGETSCAN_HUMAN_TAX_ID = 9606

# cDNAstart/cDNAend in validated_m6a_sites.tsv are 1-based and inclusive (cDNAstart == cDNAend for single sites).
# Part of the run configuration to prevent resuming from chunks built with the former 0-based interpretation.
CDNA_COORDINATE_CONVENTION = "1-based-inclusive"

# Minimum fraction of mapped m6A sites that must fall on an 'A' in the transcript sequence
MIN_A_BASE_FRACTION = 0.95


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
        import gffutils

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"[Warning] GTF-DB file '{self.db_path}' not found locally (runs on cluster).")

        print(f"[GTF-DB] Loading gffutils.FeatureDB: {self.db_path.name}")
        self.db = gffutils.FeatureDB(str(self.db_path))
        self._cache_key = None
        self._cache_value = None

    def get_transcript_exons(self, transcript_id: str) -> dict:
        """
        Returns chromosome, strand, and a list of exon intervals (start, end)
        ordered in transcription direction (5' -> 3').
        """
        # Channels of the same track query the same transcript consecutively
        if transcript_id == self._cache_key:
            return self._cache_value

        clean_id = transcript_id.split(".")[0]

        tx = None
        for candidate in [transcript_id, clean_id]:
            try:
                tx = self.db[candidate]
                break
            except Exception:
                continue

        info = None
        if tx is not None:
            strand = tx.strand
            exons = [
                (exon.start, exon.end)
                for exon in self.db.children(tx, featuretype="exon", order_by="start")
            ]
            if strand == "-":
                exons.reverse()
            exon_length = sum(end - start + 1 for start, end in exons)
            info = {"chrom": str(tx.chrom).replace("chr", ""), "strand": strand, "exons": exons, "exon_length": exon_length}

        self._cache_key, self._cache_value = transcript_id, info
        return info


@lru_cache(maxsize=1)
def gtf_db() -> GtfDbHelper:
    """Shared GTF database of all channels (opened once)."""
    return GtfDbHelper(GTF_DB_PATH)


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
# 3. Reference Parsers (BED peaks, TargetScan, validated m6A sites)
# =============================================================================

def load_bed_indexed(bed_path: Path, label: str = "BED") -> dict:
    """
    Loads eCLIP peaks from a BED file and builds a FastIntervalIndex for each (chrom, strand).
    """
    if not bed_path.exists():
        raise FileNotFoundError(f"{label} file '{bed_path}' not found.")

    print(f"Loading {label} intervals from: {bed_path}...")
    raw_data = {}
    n_no_score = 0
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

            # SignalValue (column 6 in narrowPeak / BED) must be present, positive, and a valid number.
            # Peaks without a score are discarded: a default value would mix an arbitrary scale with log2FC values.
            # Negative signal values indicate de-enrichment (less signal than input control) and do not represent binding.
            if len(parts) < 7 or parts[6] in [".", "-1", "nan", "NaN", ""]:
                n_no_score += 1
                continue
            try:
                score = float(parts[6])
            except ValueError:
                n_no_score += 1
                continue
            if score <= 0.0:
                continue

            key = (chrom, strand)
            raw_data.setdefault(key, []).append((start, end, score))

    # Create FastIntervalIndex per chromosome/strand
    indexed_data = {}
    for key, intervals in raw_data.items():
        indexed_data[key] = FastIntervalIndex(intervals)

    total_peaks = sum(len(v) for v in raw_data.values())
    print(f"{label}: Indexed {total_peaks:,} peaks across {len(indexed_data)} (chrom, strand) contigs "
          f"(discarded {n_no_score:,} peaks without valid signalValue).")
    return indexed_data


def load_targetscan_data(targetscan_path: Path, tax_id: int = TARGETSCAN_HUMAN_TAX_ID) -> dict:
    if not targetscan_path.exists():
        raise FileNotFoundError(f"TargetScan file '{targetscan_path}' not found.")

    print(f"Loading TargetScan data from: {targetscan_path}...")
    df_ts = pd.read_csv(targetscan_path, sep="\t", low_memory=False)

    # Keep only sites of the target species. Rows of other species carry the same (human)
    # transcript ID, but their UTR coordinates refer to the orthologous UTR.
    tax_col = "Gene Tax ID"
    if tax_col not in df_ts.columns:
        raise KeyError(f"[Error] Column '{tax_col}' not found in TargetScan file! Columns: {list(df_ts.columns)}")
    n_before = len(df_ts)
    df_ts = df_ts[pd.to_numeric(df_ts[tax_col], errors="coerce") == tax_id]
    print(f"TargetScan: Retained {len(df_ts):,} of {n_before:,} rows with {tax_col} == {tax_id}.")
    if df_ts.empty:
        raise RuntimeError(f"[Error] No TargetScan rows with {tax_col} == {tax_id} found in {targetscan_path}!")

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


def load_validated_m6a_data(m6a_input_path: Path, score_col: str = "mean_score", filter_drach: bool = False) -> dict:
    """
    Strictly loads the validated m6A site table 'validated_m6a_sites.tsv' (written by validate_and_filter_m6a.py).
    Returns a dict mapping transcript ID (full and unversioned) -> list of (cDNAstart, cDNAend, score).
    """
    p = Path(m6a_input_path)
    if p.is_dir():
        p = p / "validated_m6a_sites.tsv"
    if not p.is_file():
        raise FileNotFoundError(
            f"[ERROR] The required file 'validated_m6a_sites.tsv' does not exist at: {p}\n"
            f"Please run 'validate_and_filter_m6a.py' first to generate this file."
        )

    print(f"\n[m6A Loader] Loading validated m6A positions from: {p}")
    df = pd.read_csv(p, sep="\t", low_memory=False)

    required_cols = ["TxId", "cDNAstart", "cDNAend"] + ([score_col] if score_col != "binary" else [])
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"[ERROR] Required column(s) missing in '{p.name}': {missing_cols}!\n"
            f"Available columns in file: {list(df.columns)}"
        )

    # Optional: Filter for canonical DRACH motif if requested
    if filter_drach:
        if "is_drach" not in df.columns:
            raise KeyError("[ERROR] Flag '--filter_drach' passed, but column 'is_drach' is missing in 'validated_m6a_sites.tsv'!")
        n_before = len(df)
        df = df[df["is_drach"] == True].copy()
        print(f"             Filtered for is_drach == True: {len(df):,} of {n_before:,} positions retained.")

    cdna_lookup = {}
    for row_idx, row in df.iterrows():
        tx_id = str(row["TxId"]).strip()
        if not tx_id or tx_id == "nan":
            continue
        try:
            c_start = int(row["cDNAstart"])
            c_end = int(row["cDNAend"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"[ERROR] Invalid cDNA coordinates in row {row_idx}: cDNAstart={row['cDNAstart']}, cDNAend={row['cDNAend']}"
            ) from e
        try:
            score_val = 1.0 if score_col == "binary" else float(row[score_col])
        except (ValueError, TypeError) as e:
            raise ValueError(f"[ERROR] Invalid score in row {row_idx} in column '{score_col}': {row[score_col]}") from e

        # Index under both full and unversioned transcript ID
        for key in {tx_id, tx_id.split(".")[0]}:
            cdna_lookup.setdefault(key, []).append((c_start, c_end, score_val))

    print(f"             Indexed {len(df):,} m6A sites across {df['TxId'].astype(str).nunique():,} unique transcripts.")
    return cdna_lookup


# =============================================================================
# 4. Channel Providers (one extra input channel each)
# =============================================================================

class Channel:
    """
    One extra input channel. signal() returns the raw 1D signal over the full transcript (5' -> 3', length
    full_length); the generator keeps the 3' window of the 6-track archive and normalizes it.
    Constructors only store parameters; reference data is read in load() (after the run configuration check).
    """
    name: str   # entry in channel_names
    key: str    # short ID for per-channel NPZ fields and metadata (has_<key>, <key>_q99, transcripts_with_<key>)

    def config(self) -> dict:
        """Parameters and reference files that define the channel (run configuration & metadata)."""
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def signal(self, row) -> np.ndarray:
        raise NotImplementedError

    def transcript_stats(self, row, window: np.ndarray) -> dict:
        """Additional per-transcript fields stored in the NPZ (NumPy scalars define the dtype)."""
        return {}

    def summarize(self, fields: dict) -> dict:
        """Dataset-level statistics for the metadata, computed from the merged per-transcript fields."""
        return {}

    def finalize(self) -> None:
        """Sanity checks after all transcripts were processed."""


class BedPeakChannel(Channel):
    """Genomic BED peaks (signalValue) mapped onto the transcript via its GTF exon structure."""

    def __init__(self, name: str, key: str, bed_path: Path):
        self.name, self.key, self.bed_path = name, key, bed_path

    def config(self) -> dict:
        return {"bed": self.bed_path, "gtf_db": GTF_DB_PATH, "peaks_without_score": "discarded", "gtf_exon_length_check": True}

    def load(self) -> None:
        self.gtf = gtf_db()
        self.peaks = load_bed_indexed(self.bed_path, label=self.name)

    def _exons(self, row):
        """GTF exon structure, or None if the transcript is missing in the GTF or its exon length does not match."""
        info = self.gtf.get_transcript_exons(row.clean_tx)
        # Exon structure must match the dataset sequence, otherwise peaks would be mapped to wrong positions
        return info if info is not None and info["exon_length"] == row.full_length else None

    def signal(self, row) -> np.ndarray:
        info = self._exons(row)
        if info is None or (info["chrom"], info["strand"]) not in self.peaks:
            return np.zeros(row.full_length, dtype=np.float32)
        return map_genomic_intervals_to_transcript(
            info["exons"], info["strand"], self.peaks[(info["chrom"], info["strand"])], row.full_length
        )

    def transcript_stats(self, row, window: np.ndarray) -> dict:
        info = self.gtf.get_transcript_exons(row.clean_tx)
        return {
            "has_gtf": info is not None and info["exon_length"] == row.full_length,
            "gtf_length_mismatch": info is not None and info["exon_length"] != row.full_length,
        }

    def summarize(self, fields: dict) -> dict:
        return {
            "transcripts_in_gtf": int(np.sum(fields["has_gtf"])),
            "transcripts_gtf_length_mismatch": int(np.sum(fields["gtf_length_mismatch"])),
        }


class TargetScanChannel(Channel):
    """TargetScan miRNA sites (|weighted context++ score|) on 3' UTR coordinates of the target species."""

    def __init__(self, name: str, key: str, ts_path: Path, tax_id: int = TARGETSCAN_HUMAN_TAX_ID):
        self.name, self.key, self.ts_path, self.tax_id = name, key, ts_path, tax_id

    def config(self) -> dict:
        return {"targetscan_file": self.ts_path, "targetscan_tax_id": self.tax_id}

    def load(self) -> None:
        self.sites = load_targetscan_data(self.ts_path, self.tax_id)

    def signal(self, row) -> np.ndarray:
        l = row.full_length
        track = np.zeros(l, dtype=np.float32)
        if row.utr3_start > 0:
            for start, end, score in self.sites.get(row.clean_tx, []):
                abs_start = row.utr3_start + (start - 1)
                abs_end = row.utr3_start + end
                if abs_start < l:
                    track[abs_start:min(abs_end, l)] += score
        return track


class M6AChannel(Channel):
    """Validated m6A sites on cDNA coordinates (1-based inclusive); overlapping sites keep the maximum score."""

    def __init__(self, name: str, key: str, tsv_path: Path, score_col: str, filter_drach: bool):
        self.name, self.key, self.tsv_path = name, key, tsv_path
        self.score_col, self.filter_drach = score_col, filter_drach
        self.n_sites_mapped = 0
        self.n_sites_on_a = 0

    def config(self) -> dict:
        return {
            "m6a_file": self.tsv_path, "score_column": self.score_col, "filter_drach": self.filter_drach,
            "cdna_coordinate_convention": CDNA_COORDINATE_CONVENTION,
        }

    def load(self) -> None:
        self.sites = load_validated_m6a_data(self.tsv_path, self.score_col, self.filter_drach)

    def _sites(self, row) -> list:
        """Sites within the full transcript as 0-based half-open (start, end, score)."""
        sites = self.sites.get(row.tx_id) or self.sites.get(row.clean_tx) or []
        return [(c_start - 1, c_end, score) for c_start, c_end, score in sites if 0 <= c_start - 1 < row.full_length]

    def signal(self, row) -> np.ndarray:
        track = np.zeros(row.full_length, dtype=np.float32)
        for start, end, score in self._sites(row):
            end = min(row.full_length, max(start + 1, end))
            track[start:end] = np.maximum(track[start:end], score)
        return track

    def transcript_stats(self, row, window: np.ndarray) -> dict:
        # Sites in the kept 3' window; m6A must fall on an adenosine (channel A of the 6-track window)
        offset = row.full_length - row.seq_lens
        window_starts = [start - offset for start, _, _ in self._sites(row) if start >= offset]
        self.n_sites_mapped += len(window_starts)
        self.n_sites_on_a += int(np.sum(row.track[window_starts, 0] == 1.0)) if window_starts else 0
        active = window[window > 0]
        return {
            "num_m6a_sites": np.int32(len(window_starts)),
            "mean_m6a_score": np.float32(active.mean() if active.size else 0.0),
        }

    def summarize(self, fields: dict) -> dict:
        return {"total_m6a_sites": int(np.sum(fields["num_m6a_sites"]))}

    def finalize(self) -> None:
        if self.n_sites_mapped == 0:
            return
        a_fraction = self.n_sites_on_a / self.n_sites_mapped
        print(f"[Sanity Check] {self.n_sites_on_a:,} of {self.n_sites_mapped:,} mapped m6A sites fall on 'A' ({a_fraction * 100:.2f} %).")
        if a_fraction < MIN_A_BASE_FRACTION:
            raise RuntimeError(
                f"[Error] Only {a_fraction * 100:.2f} % of mapped m6A sites fall on 'A' "
                f"(expected >= {MIN_A_BASE_FRACTION * 100:.0f} %)! cDNA coordinates do not match the transcript sequences."
            )


# Extension tracks: track name -> extra channels (appended to the 6 base channels in this order)
TRACK_PRESETS = {
    "7track_m6a": lambda args: [
        M6AChannel("m6A", "m6a", VALIDATED_M6A_PATH, args.score_col, args.filter_drach),
    ],
    "8track_2rbp": lambda args: [
        BedPeakChannel("RBP_Stabilizer", "stab", STABILIZERS_BED_PATH),
        BedPeakChannel("RBP_Destabilizer", "destab", DESTABILIZERS_BED_PATH),
    ],
    "8track_miRNA_rbp": lambda args: [
        TargetScanChannel("miRNA", "mirna", TARGETSCAN_FILE_PATH),
        BedPeakChannel("eCLIP", "eclip", ENCODE_ECLIP_FILE_PATH),
    ],
}


# =============================================================================
# 5. Normalization & Dynamic Quantiles
# =============================================================================

def normalize_signal(x: np.ndarray, norm_method: str, q99: float = None) -> np.ndarray:
    """
    Normalizes a raw channel signal (negative values are clamped to 0):
      - 'none':   raw values
      - 'log':    log(1 + x)
      - 'minmax': min(1, log(1 + x) / log(1 + q99)), robust quantile scaling with a global bound
    """
    x = np.maximum(0.0, x)
    if norm_method == "none":
        return x
    if norm_method == "log":
        return np.log1p(x)
    if norm_method == "minmax":
        denom = np.log1p(float(q99))
        return np.clip(np.log1p(x) / denom, 0.0, 1.0) if denom > 0 else x
    raise ValueError(f"Unknown normalization method: '{norm_method}' (allowed: 'none', 'log', 'minmax')")


def window_signal(channel: Channel, row) -> np.ndarray:
    """Raw channel signal restricted to the kept 3' window of the 6-track archive."""
    return channel.signal(row)[row.full_length - row.seq_lens:]


def compute_dynamic_quantiles(channels: list, base_df: pd.DataFrame, quantile: float, sample_size: int) -> dict:
    """
    Dynamically determines a high percentile (default: q99) of the non-zero signal values
    within the kept 3' windows of a transcript sample, separately for each channel.
    """
    q_pct = quantile * 100.0 if quantile <= 1.0 else quantile
    print(f"\n[Quantile Estimation] Dynamically calculating {q_pct:.1f}th percentile (q{int(q_pct)}) per channel...")

    if sample_size is not None and 0 < sample_size < len(base_df):
        eval_df = base_df.sample(n=sample_size, random_state=42)
        print(f"[Quantile Estimation] Subsampling {sample_size} transcripts for fast percentile estimation...")
    else:
        eval_df = base_df
        print(f"[Quantile Estimation] Scanning all {len(eval_df)} transcripts...")

    values = {ch.key: [] for ch in channels}
    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df), desc="Estimating quantiles"):
        for ch in channels:
            w = window_signal(ch, row)
            if np.any(w > 0):
                values[ch.key].append(w[w > 0])

    q99 = {}
    for ch in channels:
        if not values[ch.key]:
            raise RuntimeError(
                f"[Quantile Estimation] No positive {ch.name} signal found in the dataset sample! "
                f"Please verify the reference files, transcript ID mapping and coordinates ({ch.config()})."
            )
        flat = np.concatenate(values[ch.key])
        q99[ch.key] = float(np.percentile(flat, q_pct))
        print(f"[Quantile Estimation] {ch.name} {q_pct:.1f}% percentile: {q99[ch.key]:.4f} (from {len(flat):,} active positions)")
    return q99


# =============================================================================
# 6. Incremental Chunking & Merging
# =============================================================================

def track_output_path(dataset_key: str, track: str, normalization: str = "none") -> Path:
    if track == BASE_TRACK:
        stem = f"{dataset_key}_6tracks"
    else:
        stem = f"{dataset_key}_{track}" + (f"_{normalization}" if normalization != "none" else "")
    return DATASET_DIRS[dataset_key] / "orthrus" / track / f"{stem}.npz"


def prepare_chunks_dir(chunks_dir: Path, recreate: bool, run_config: dict) -> int:
    """Validates/initializes the chunk directory and returns the number of transcripts already processed."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = sorted(chunks_dir.glob("chunk_*.npz"))

    if recreate:
        print(f"[Recreate] Flag --recreate active: Removing {len(existing_chunks)} existing chunks for a complete restart...")
        for cf in existing_chunks:
            cf.unlink()
        existing_chunks = []
        (chunks_dir / RUN_CONFIG_FILENAME).unlink(missing_ok=True)
        (chunks_dir / QUANTILES_FILENAME).unlink(missing_ok=True)

    # Existing chunks may only be resumed if they were generated with identical parameters
    ensure_run_config(chunks_dir, run_config, has_existing_state=bool(existing_chunks), reset_hint="Please rerun with --recreate.")

    # Chunks hold consecutive transcripts in input order, so a resume continues after the last saved transcript
    n_done = 0
    for cf in existing_chunks:
        with np.load(cf, allow_pickle=True) as c_data:
            n_done += len(c_data["seq_lens"])
    if n_done:
        print(f"[Resume] Found {n_done:,} already processed transcripts in {len(existing_chunks)} chunks!")
    return n_done


def save_chunk(chunks_dir: Path, chunk_idx: int, items: list) -> None:
    """Saves a block of processed transcripts incrementally as NPZ (one array per item field)."""
    fields = {k: np.array([item[k] for item in items]) for k in items[0] if k != "track"}
    np.savez_compressed(chunks_dir / f"chunk_{chunk_idx:05d}.npz", tracks=to_object_array(item["track"] for item in items), **fields)


def process_transcripts(rows, total: int, n_done: int, chunks_dir: Path, chunk_size: int, build_item) -> None:
    """Builds one item per transcript (skipping the first n_done) and saves them in chunks."""
    chunk_idx = len(list(chunks_dir.glob("chunk_*.npz")))
    items = []
    with tqdm(total=total, desc="Progress", initial=n_done) as pbar:
        for i, row in enumerate(rows):
            if i < n_done:
                continue
            items.append(build_item(row))
            pbar.update(1)
            if len(items) >= chunk_size:
                save_chunk(chunks_dir, chunk_idx, items)
                chunk_idx += 1
                items = []
        if items:
            save_chunk(chunks_dir, chunk_idx, items)


def merge_all_chunks(chunks_dir: Path) -> dict:
    """Merges all chunks into one dict of arrays (tracks as 1D object array)."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"[Error] No chunk files found to merge in {chunks_dir}!")

    print(f"\nMerging {len(chunk_files)} chunks...")
    parts = {}
    for cf in tqdm(chunk_files, desc="Merging chunks"):
        with np.load(cf, allow_pickle=True) as data:
            for k in data.files:
                parts.setdefault(k, []).append(data[k])
    merged = {k: np.concatenate(v) for k, v in parts.items() if k != "tracks"}
    merged["tracks"] = to_object_array(t for chunk in parts["tracks"] for t in chunk)
    return merged


def write_track_archive(out_file: Path, merged: dict, meta: dict, npz_extra: dict) -> None:
    """Writes the final NPZ archive and its companion .meta.json."""
    lens = merged["seq_lens"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        **merged,
        dataset=meta["dataset"],
        track_name=meta["track_name"],
        channel_names=np.array(meta["channel_names"]),
        **npz_extra,
    )
    meta.update({
        "num_channels": len(meta["channel_names"]),
        "total_transcripts": int(len(lens)),
        "unique_genes": int(len(set(merged["gene"].tolist()))),
        "seq_len_min": int(lens.min()),
        "seq_len_max": int(lens.max()),
        "seq_len_mean": float(lens.mean()),
        "npz_file": out_file.name,
        "created_at": datetime.now().isoformat(),
    })
    save_metadata(out_file, meta)

    print("\n" + "=" * 70)
    print(f"             {meta['track_name'].upper()} ARCHIVE GENERATION REPORT             ")
    print("=" * 70)
    print(f"Dataset:                         {meta['dataset']}")
    print(f"Total transcripts:               {meta['total_transcripts']:,}")
    print(f"Unique genes:                    {meta['unique_genes']:,}")
    print(f"Sequence length range:           {meta['seq_len_min']} - {meta['seq_len_max']} bp (mean: {meta['seq_len_mean']:.1f} bp)")
    print(f"Channel configuration:           {meta['channel_names']}")
    for k, v in meta.get("parameters", {}).items():
        if k.endswith("_q99") or k.startswith("transcripts_") or k.startswith("total_"):
            print(f"{k + ':':<33}{v}")
    print(f"Final file saved:                {out_file}")
    print("=" * 70)


# =============================================================================
# 7. Track Generation
# =============================================================================

def generate_base_track(args, dataset_key: str, species: str) -> None:
    """Step 1: 6-track archive from the raw dataset (input of all extension tracks)."""
    out_file = track_output_path(dataset_key, BASE_TRACK)
    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"
    max_length = args.max_length or 12288

    print(f"Output file:        {out_file}")
    print(f"Chunk directory:    {chunks_dir} (Chunk size: {args.chunk_size})")
    print(f"Max sequence len:   {max_length} bp")

    n_done = prepare_chunks_dir(chunks_dir, args.recreate, {
        "script": Path(__file__).name,
        "track": BASE_TRACK,
        "dataset": dataset_key,
        "max_length": max_length,
        "truncation": TRUNCATION_MODE,
    })

    print(f"\n[Loader] Loading dataset '{args.dataset}'" + (f" ({args.species})" if args.species else "") + "...")
    df = SixTrackDatasetLoader(dataset=args.dataset, species=args.species).get_data()
    print(f"[Loader] Successfully loaded {len(df):,} canonical transcripts.")

    def build_item(row) -> dict:
        full_length = int(row.length)
        return {
            "track": truncate_to_max_length(row.track, max_length),  # keeps the 3' end
            "target": np.float32(row.target),
            "target_raw": np.float32(row.target_raw),
            "ensembl_transcript_id": str(row.ensembl_transcript_id).strip(),
            "ensembl_gene_id": str(row.ensembl_gene_id).strip(),
            "gene": str(row.gene).strip(),
            "seq_lens": np.int32(min(full_length, max_length)),
            # Untruncated length and 3' UTR start: extension channels are computed on the full transcript
            "full_length": np.int32(full_length),
            "utr3_start": np.int32(row.utr3_start),
        }

    print("\nProcessing transcripts and writing 6-track chunks...")
    process_transcripts(df.itertuples(index=False), len(df), n_done, chunks_dir, args.chunk_size, build_item)

    write_track_archive(out_file, merge_all_chunks(chunks_dir), {
        "track_name": BASE_TRACK,
        "dataset": dataset_key,
        "species": species,
        "normalization": "none",
        "channel_names": list(BASE_CHANNEL_NAMES),
        "max_length": int(max_length),
        "truncation": TRUNCATION_MODE,
    }, npz_extra={})


def load_base_tracks(dataset_key: str):
    """Loads the 6-track archive as the common basis (transcripts, order, truncation) of all extension tracks."""
    base_file = track_output_path(dataset_key, BASE_TRACK)
    if not base_file.exists():
        raise FileNotFoundError(
            f"[Error] 6-track archive not found: {base_file}! Generate it first: generate_tracks.py --track {BASE_TRACK}"
        )
    base_meta = load_metadata(base_file, required=True)
    require_truncation_mode(base_meta, base_file)

    print(f"\n[Base] Loading 6-track archive: {base_file}")
    with np.load(base_file, allow_pickle=True) as npz:
        missing = {"full_length", "utr3_start"} - set(npz.files)
        if missing:
            raise KeyError(
                f"[Error] 6-track archive {base_file} has no fields {sorted(missing)} (created by an older generator)! "
                f"Please regenerate it: generate_tracks.py --track {BASE_TRACK} --recreate"
            )
        base_df = pd.DataFrame({
            "track": list(npz["tracks"]),
            "tx_id": npz["ensembl_transcript_id"].astype(str),
            "target": npz["target"],
            "target_raw": npz["target_raw"],
            "ensembl_gene_id": npz["ensembl_gene_id"].astype(str),
            "gene": npz["gene"].astype(str),
            "seq_lens": npz["seq_lens"].astype(int),
            "full_length": npz["full_length"].astype(int),
            "utr3_start": npz["utr3_start"].astype(int),
        })
    base_df["clean_tx"] = base_df["tx_id"].str.split(".").str[0]
    print(f"[Base] {len(base_df):,} transcripts (max_length={base_meta['max_length']}, created {base_meta['created_at']}).")
    return base_df, base_meta, base_file


def parse_q99_overrides(entries: list, channels: list) -> dict:
    """Parses '--q99 KEY=VALUE' entries (KEY: channel key, e.g. mirna=15.0)."""
    valid = {ch.key for ch in channels}
    overrides = {}
    for entry in entries or []:
        key, sep, value = entry.partition("=")
        if not sep or key not in valid:
            raise ValueError(f"[Error] Invalid --q99 entry '{entry}'! Expected KEY=VALUE with KEY in {sorted(valid)}.")
        overrides[key] = float(value)
    return overrides


def generate_extension_track(args, dataset_key: str, species: str) -> None:
    """Step 2: 6-track archive + extra channels of the track preset."""
    if species != "human":
        raise ValueError(
            f"[Error] Track '{args.track}' supports only human data: all trans-factor references (GRCh38 GTF, ENCODE eCLIP, "
            f"TargetScanHuman, validated m6A sites) are human. Use the 6-track archive for {species}."
        )
    if not args.normalization:
        raise ValueError(f"[Error] --normalization is required for track '{args.track}'!")

    channels = TRACK_PRESETS[args.track](args)
    q99_overrides = parse_q99_overrides(args.q99, channels)
    out_file = track_output_path(dataset_key, args.track, args.normalization)
    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"

    print(f"Extra channels:     {[ch.name for ch in channels]}")
    print(f"Normalization:      {args.normalization.upper()} (Global Robust Quantile Scaling)")
    print(f"Output file:        {out_file}")
    print(f"Chunk directory:    {chunks_dir} (Chunk size: {args.chunk_size})")

    base_df, base_meta, base_file = load_base_tracks(dataset_key)
    max_length = int(base_meta["max_length"])
    if args.max_length is not None and args.max_length != max_length:
        raise ValueError(
            f"[Error] --max_length {args.max_length} differs from the 6-track archive ({max_length})! "
            f"Extension tracks inherit the truncation of the 6-track archive."
        )

    n_done = prepare_chunks_dir(chunks_dir, args.recreate, {
        "script": Path(__file__).name,
        "track": args.track,
        "dataset": dataset_key,
        "normalization": args.normalization,
        "q99_overrides": q99_overrides,
        "quantile": args.quantile,
        "quantile_sample_size": args.quantile_sample_size,
        "max_length": max_length,
        "truncation": TRUNCATION_MODE,
        "base_tracks": base_file,
        "base_created_at": base_meta["created_at"],
        "channels": {ch.key: {"name": ch.name, **ch.config()} for ch in channels},
    })

    for ch in channels:
        ch.load()

    # Global q99 bounds for minmax: CLI override > saved bounds of the interrupted run > dynamic estimation
    q99 = {}
    if args.normalization == "minmax":
        quantiles_file = chunks_dir / QUANTILES_FILENAME
        saved = json.loads(quantiles_file.read_text()) if quantiles_file.exists() else {}
        missing = [ch for ch in channels if ch.key not in q99_overrides and ch.key not in saved]
        dynamic = compute_dynamic_quantiles(missing, base_df, args.quantile, args.quantile_sample_size) if missing else {}
        q99 = {ch.key: q99_overrides.get(ch.key, saved.get(ch.key, dynamic.get(ch.key))) for ch in channels}
        quantiles_file.write_text(json.dumps(q99, indent=2))
        print("[Scaling Bounds] Active global q99 bounds: " + ", ".join(f"{ch.name} = {q99[ch.key]:.4f}" for ch in channels) + "\n")

    def build_item(row) -> dict:
        extra, stats = [], {}
        for ch in channels:
            window = window_signal(ch, row)
            extra.append(normalize_signal(window, args.normalization, q99.get(ch.key)))
            stats[f"has_{ch.key}"] = bool(np.any(window > 0))
            stats.update(ch.transcript_stats(row, window))
        return {
            "track": np.concatenate([row.track, np.stack(extra, axis=1)], axis=1).astype(np.float32),
            "target": np.float32(row.target),
            "target_raw": np.float32(row.target_raw),
            "ensembl_transcript_id": row.tx_id,
            "ensembl_gene_id": row.ensembl_gene_id,
            "gene": row.gene,
            "seq_lens": np.int32(row.seq_lens),
            "full_length": np.int32(row.full_length),
            "utr3_start": np.int32(row.utr3_start),
            **stats,
        }

    print("\nProcessing transcripts (with fast O(log N) lookup & incremental saving)...")
    process_transcripts(base_df.itertuples(index=False), len(base_df), n_done, chunks_dir, args.chunk_size, build_item)
    for ch in channels:
        ch.finalize()

    merged = merge_all_chunks(chunks_dir)
    parameters = {
        "normalization": args.normalization,
        "quantile": float(args.quantile),
        "quantile_sample_size": args.quantile_sample_size,
        "channels": {ch.key: {"name": ch.name, **ch.config()} for ch in channels},
    }
    for ch in channels:
        parameters[f"{ch.key}_q99"] = q99.get(ch.key)
    for ch in channels:
        parameters[f"transcripts_with_{ch.key}"] = int(np.sum(merged[f"has_{ch.key}"]))
        parameters.update(ch.summarize(merged))

    write_track_archive(out_file, merged, {
        "track_name": args.track,
        "dataset": dataset_key,
        "species": species,
        "normalization": args.normalization,
        "channel_names": list(BASE_CHANNEL_NAMES) + [ch.name for ch in channels],
        "max_length": max_length,
        "truncation": TRUNCATION_MODE,
        "base_tracks": {"npz_file": base_file.name, "created_at": base_meta["created_at"]},
        "parameters": parameters,
    }, npz_extra={
        "normalization": args.normalization,
        **{f"{ch.key}_q99": np.float32(q99[ch.key]) if ch.key in q99 else np.nan for ch in channels},
    })


# =============================================================================
# 8. Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Orthrus track archives: the 6-track base archive or a trans-factor extension track built on it."
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["hipsc_cm", "mrna_bench"],
                        help="Dataset to process: 'hipsc_cm' or 'mrna_bench' (required)")
    parser.add_argument("--species", type=str, default=None, choices=["human", "mouse"],
                        help="Species for mRNABench: 'human' or 'mouse' (required if --dataset is 'mrna_bench')")
    parser.add_argument("--track", type=str, required=True, choices=[BASE_TRACK] + list(TRACK_PRESETS),
                        help="Track to generate. Extension tracks require the 6-track archive of the same dataset.")
    parser.add_argument("--normalization", type=str, default=None, choices=["minmax", "log", "none"],
                        help="Normalization of the extra channels (required for extension tracks): 'minmax' "
                             "(log1p with robust q99 scaling to [0, 1]), 'log' (log1p), 'none' (raw values)")
    parser.add_argument("--q99", type=str, nargs="*", default=None, metavar="KEY=VALUE",
                        help="Fixed q99 bounds per channel key for minmax (e.g. mirna=15.0 eclip=50.0; "
                             "default: dynamically estimated). Keys: m6a, stab, destab, mirna, eclip")
    parser.add_argument("--quantile", type=float, default=0.99,
                        help="Percentile to compute dynamically (default: 0.99 for 99th percentile)")
    parser.add_argument("--quantile_sample_size", type=int, default=2000,
                        help="Number of transcripts to sample for dynamic quantile estimation (default: 2000, 0 = all)")
    parser.add_argument("--score_col", type=str, default="a_pct_modified",
                        choices=["mean_score", "effect_size", "a_pct_modified", "max_score", "binary"],
                        help="7track_m6a: metric mapped to the m6A channel (default: a_pct_modified)")
    parser.add_argument("--filter_drach", action="store_true",
                        help="7track_m6a: only include sites matching the canonical DRACH consensus motif")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Number of transcripts per incremental storage chunk (default: 500)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="6track: maximum sequence length, longer transcripts keep their 3' end (default: 12288). "
                             "Extension tracks inherit it from the 6-track archive.")
    parser.add_argument("--recreate", action="store_true",
                        help="Forces regeneration of all chunks and overwrites/deletes existing chunks.")
    args = parser.parse_args()

    if args.dataset == "mrna_bench" and not args.species:
        parser.error("--species ('human' or 'mouse') must be specified when --dataset is 'mrna_bench'!")
    if args.dataset == "hipsc_cm" and args.species:
        parser.error("--species is only valid for --dataset mrna_bench (hipsc_cm is human).")
    return args


def main():
    args = parse_args()
    dataset_key = f"{args.dataset}_{args.species}" if args.species else args.dataset
    species = args.species or "human"

    print("=" * 75)
    print(f"   ORTHRUS TRACK GENERATOR: {args.track}   ")
    print("=" * 75)
    print(f"Dataset:            {dataset_key}")

    if args.track == BASE_TRACK:
        generate_base_track(args, dataset_key, species)
    else:
        generate_extension_track(args, dataset_key, species)


if __name__ == "__main__":
    main()
