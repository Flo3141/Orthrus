#!/usr/bin/env python3
"""
SixTrackDatasetLoader
=====================
Modular loader and transformer for mRNA datasets into the Orthrus 6-track format:
  Channel 0: A (Adenine)
  Channel 1: C (Cytosine)
  Channel 2: G (Guanine)
  Channel 3: U / T (Uracil / Thymine)
  Channel 4: CDS Mask (1.0 coding, 0.0 UTR)
  Channel 5: Splice Sites / Exon Junctions (1.0 at splice site, 0.0 elsewhere)

Supported datasets:
  - 'hipsc_cm': Human iPSC-derived cardiomyocytes (Saluki comma-separated tokens)
  - 'mrna_bench': mRNABench consensus half-life dataset (species: 'human' or 'mouse')

Sequences are preserved in their native, unpadded original length.
Model-specific truncation or padding is deferred to downstream scripts.
"""

import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm


class SixTrackDatasetLoader:
    """
    Modular loader converting hIPSC_CM or mRNABench into a standardized
    6-track pandas DataFrame.
    """

    # Default Cluster Paths
    HIPSC_CM_DEFAULT_PATH = Path(
        "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt"
    )
    MRNA_BENCH_DEFAULT_DIR = Path(
        "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench"
    )

    def __init__(
        self,
        dataset: str,
        species: Optional[str] = None,
        data_path: Optional[Union[str, Path]] = None,
        require_transcript_id: bool = False,
        show_progress: bool = True,
    ):
        """
        Parameters
        ----------
        dataset : str
            Dataset name. Must be either 'hipsc_cm' or 'mrna_bench'.
        species : str, optional
            Required if dataset is 'mrna_bench'. Options: 'human' or 'mouse'.
        data_path : str or Path, optional
            Override path to dataset file or mrna_bench data directory.
            If None, uses hardcoded cluster defaults.
        require_transcript_id : bool, default False
            If True, discards any entries that do not have a verified ENST/ENSMUST transcript ID.
            Essential when preparing data for downstream 8-track (miRNA/eCLIP) pipelines.
        show_progress : bool, default True
            Whether to display a tqdm progress bar during processing.
        """
        self.dataset = dataset.lower().strip()
        self.require_transcript_id = require_transcript_id
        if self.dataset not in ["hipsc_cm", "mrna_bench"]:
            raise ValueError(
                f"Invalid dataset '{dataset}'. Supported: 'hipsc_cm', 'mrna_bench'"
            )

        self.species = species.lower().strip() if species else None
        if self.dataset == "mrna_bench":
            if self.species not in ["human", "mouse"]:
                raise ValueError(
                    "When dataset='mrna_bench', 'species' must be specified as 'human' or 'mouse'."
                )

        self.data_path = Path(data_path) if data_path else None
        self.show_progress = show_progress

    # -------------------------------------------------------------------------
    # Fast 1D / 2D Track Encoders
    # -------------------------------------------------------------------------

    @staticmethod
    def seq_to_one_hot(seq: str) -> np.ndarray:
        """
        Encodes an RNA/DNA sequence into (L, 4) one-hot array (float32).
        Channels: 0: A, 1: C, 2: G, 3: U/T. Unknown / 'N' -> [0, 0, 0, 0].
        """
        seq_bytes = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
        l = len(seq_bytes)
        oh = np.zeros((l, 4), dtype=np.float32)
        oh[seq_bytes == 65, 0] = 1.0  # 'A'
        oh[seq_bytes == 67, 1] = 1.0  # 'C'
        oh[seq_bytes == 71, 2] = 1.0  # 'G'
        oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # 'T' (84) or 'U' (85)
        return oh

    @staticmethod
    def parse_binary_track(track, length: int) -> np.ndarray:
        """
        Converts a track (binary string, list, ndarray, or numeric) into an (L, 1) float32 array.
        """
        if isinstance(track, (int, np.integer)):
            track = str(track).zfill(length)
        elif isinstance(track, (float, np.floating)):
            if np.isnan(track):
                return np.zeros((length, 1), dtype=np.float32)
            track = str(int(track)).zfill(length)

        if isinstance(track, str):
            arr = np.array([float(c) for c in track], dtype=np.float32)
        elif isinstance(track, (list, np.ndarray, pd.Series)):
            arr = np.asarray(track, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported track type: {type(track)}")

        if len(arr) != length:
            if len(arr) < length:
                padded = np.zeros(length, dtype=np.float32)
                padded[:len(arr)] = arr
                arr = padded
            else:
                arr = arr[:length]

        return arr.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # Internal Dataset Processors
    # -------------------------------------------------------------------------

    def _load_hipsc_cm(self) -> pd.DataFrame:
        """
        Loads and parses hIPSC_CM dataset from Saluki comma-separated token sequences.
        """
        file_path = self.data_path or self.HIPSC_CM_DEFAULT_PATH
        if not file_path.exists():
            raise FileNotFoundError(f"hIPSC_CM dataset file not found at: {file_path}")

        print(f"[SixTrackLoader] Loading hIPSC_CM data from: {file_path}")
        raw_df = pd.read_csv(file_path, sep="\t")
        total_rows = len(raw_df)

        records = []
        iterator = raw_df.iterrows()
        if self.show_progress:
            iterator = tqdm(iterator, total=total_rows, desc="Processing hIPSC_CM 6-tracks")

        for _, row in iterator:
            raw_seq = str(row["sequence"])
            tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
            l = len(tokens)
            if l == 0:
                continue

            # Extract nucleotide characters
            clean_seq = "".join(tok[0] for tok in tokens)

            # 1. Channels 0-3: One-Hot
            oh = self.seq_to_one_hot(clean_seq)

            # 2. Channel 4: CDS (uppercase letters denote CDS)
            cds_track = np.array(
                [1.0 if tok[0].isupper() else 0.0 for tok in tokens],
                dtype=np.float32,
            ).reshape(-1, 1)

            # 3. Channel 5: Splice Sites (tokens with '_ej' denote exon junctions)
            splice_track = np.array(
                [1.0 if "ej" in tok.lower() else 0.0 for tok in tokens],
                dtype=np.float32,
            ).reshape(-1, 1)

            # Concatenate to (L, 6)
            six_track = np.concatenate([oh, cds_track, splice_track], axis=1)

            # Calculate 3' UTR start index (informative metadata for downstream miRNA)
            upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
            utr3_start = (max(upper_indices) + 3) if upper_indices else 0
            utr3_start = min(utr3_start, l)

            records.append({
                "ensembl_transcript_id": str(row.get("ensembl_transcript_id", "")),
                "ensembl_gene_id": str(row.get("ensembl_gene_id", "")),
                "gene": str(row.get("hgnc_symbol", "")),
                "sequence": clean_seq,
                "track": six_track,
                "length": l,
                "target": float(row.get("half_life_transformed", np.nan)),
                "target_raw": float(row.get("half_life", np.nan)),
                "utr3_start": utr3_start,
            })

        return pd.DataFrame(records)

    def _load_mrna_bench(self) -> pd.DataFrame:
        """
        Loads and parses mRNABench dataset ('rnahl-human' or 'rnahl-mouse').
        """
        dataset_key = f"rnahl-{self.species}"
        data_dir = self.data_path or self.MRNA_BENCH_DEFAULT_DIR

        print(f"[SixTrackLoader] Loading mRNABench ({dataset_key}) from: {data_dir}")

        # Check if mrna_bench package is available or load exported TSV/Parquet
        df_raw = None
        if data_dir.is_file():
            print(f"[SixTrackLoader] Loading mRNABench from direct file: {data_dir.name}")
            sep = "\t" if data_dir.suffix in [".tsv", ".txt"] else ","
            df_raw = pd.read_csv(data_dir, sep=sep)
        else:
            try:
                import mrna_bench as mb
                mb.update_data_path(str(data_dir))
                df_raw = mb.load_dataset(dataset_key).data_df
            except ImportError:
                # Fallback to local table if package is not installed in current environment
                candidate_files = [
                    data_dir / dataset_key / "mrna_protein.tsv",
                    data_dir / dataset_key / f"{self.species}_mrna_protein.tsv",
                    data_dir / f"{dataset_key}.tsv",
                    data_dir / f"{dataset_key}.csv",
                ]
                for c in candidate_files:
                    if c.exists():
                        print(f"[SixTrackLoader] Found exported table fallback: {c.name}")
                        sep = "\t" if c.suffix == ".tsv" else ","
                        df_raw = pd.read_csv(c, sep=sep)
                        break

                if df_raw is None:
                    raise ImportError(
                        "mrna_bench Python package is not installed and no fallback tabular file was found. "
                        f"Please install mrna_bench or provide data in {data_dir}."
                    )

        total_rows = len(df_raw)
        records = []
        iterator = df_raw.iterrows()
        if self.show_progress:
            iterator = tqdm(iterator, total=total_rows, desc=f"Processing mRNABench ({self.species}) 6-tracks")

        for idx, row in iterator:
            seq = str(row["sequence"]).strip()
            l = len(seq)
            if l == 0:
                continue

            # 1. Channels 0-3: One-Hot
            oh = self.seq_to_one_hot(seq)

            # 2. Channel 4: CDS track
            cds_raw = row["cds"]
            cds_track = self.parse_binary_track(cds_raw, l)

            # 3. Channel 5: Splice track
            splice_raw = row["splice"]
            splice_track = self.parse_binary_track(splice_raw, l)

            # Concatenate to (L, 6)
            six_track = np.concatenate([oh, cds_track, splice_track], axis=1)

            # Calculate 3' UTR start index from CDS track (informative metadata)
            # Find the last position where cds == 1, then +3 for stop codon
            cds_indices = np.where(cds_track.flatten() > 0.5)[0]
            if len(cds_indices) > 0:
                utr3_start = min(int(cds_indices[-1]) + 3, l)
            else:
                utr3_start = 0

            gene_symbol = str(row.get("gene", "")).strip()

            # Transcript ID: look for mapped GENCODE/Ensembl transcript ID if available
            tx_id = ""
            for candidate_col in ["transcript_id", "gencode_transcript_id", "ensembl_transcript_id"]:
                if candidate_col in row and pd.notna(row[candidate_col]) and str(row[candidate_col]).strip():
                    val = str(row[candidate_col]).strip()
                    # Only accept real transcript IDs (ENST... / ENSMUST...)
                    if val.startswith("ENST") or val.startswith("ENSMUST"):
                        tx_id = val
                        break

            if not tx_id and isinstance(idx, str) and (idx.startswith("ENST") or idx.startswith("ENSMUST")):
                tx_id = idx.strip()

            # No fallback to gene symbol! Transcripts without verified ENST ID remain empty.

            gene_id = ""
            for candidate_gene_col in ["ensembl_gene_id", "gene_id", "gencode_gene_id"]:
                if candidate_gene_col in row and pd.notna(row[candidate_gene_col]) and str(row[candidate_gene_col]).strip():
                    gene_id = str(row[candidate_gene_col]).strip()
                    break

            records.append({
                "ensembl_transcript_id": tx_id,
                "ensembl_gene_id": gene_id,
                "gene": gene_symbol,
                "sequence": seq,
                "track": six_track,
                "length": l,
                "target": float(row.get("target", np.nan)),
                "target_raw": float(row.get("target_raw", row.get("half_life", np.nan))),
                "utr3_start": utr3_start,
            })

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_data(self) -> pd.DataFrame:
        """
        Processes and returns the transformed dataset as a standardized pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with standardized columns:
              - 'track': np.ndarray of shape (L, 6) with dtype float32
              - 'length': int (unpadded sequence length L)
              - 'sequence': str (nucleotide sequence)
              - 'ensembl_transcript_id': str (Ensembl transcript ID)
              - 'ensembl_gene_id': str (Ensembl gene ID, or empty string if unmapped)
              - 'gene': str (HGNC / MGI gene symbol)
              - 'target': float (transformed half-life / target value)
              - 'target_raw': float (raw half-life in hours / raw target value)
              - 'utr3_start': int (3' UTR start index, 0 if not present)
        """
        if self.dataset == "hipsc_cm":
            df = self._load_hipsc_cm()
        elif self.dataset == "mrna_bench":
            df = self._load_mrna_bench()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        if self.require_transcript_id:
            initial_count = len(df)
            valid_mask = df["ensembl_transcript_id"].astype(str).str.strip().ne("")
            df = df[valid_mask].reset_index(drop=True)
            dropped = initial_count - len(df)
            print(
                f"[SixTrackLoader] Filtered by verified ENST ID: Retained {len(df)} entries, "
                f"dropped {dropped} unmapped genes ({dropped / initial_count * 100:.2f}%)."
            )

        print(
            f"[SixTrackLoader] Finished processing {len(df)} entries for '{self.dataset}'"
            + (f" ({self.species})" if self.species else "")
            + f". Track shape: (L, 6), lengths ranging from {df['length'].min()} to {df['length'].max()} bp."
        )
        return df


# =============================================================================
# Quick Test / Verification
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SixTrackDatasetLoader")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hipsc_cm",
        choices=["hipsc_cm", "mrna_bench"],
        help="Dataset to load",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="human",
        choices=["human", "mouse"],
        help="Species if dataset is mrna_bench",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional path override",
    )
    args = parser.parse_args()

    loader = SixTrackDatasetLoader(
        dataset=args.dataset,
        species=args.species,
        data_path=args.data_path,
    )
    df = loader.get_data()
    print("\nDataFrame preview:")
    print(df[["ensembl_transcript_id", "ensembl_gene_id", "gene", "length", "target", "target_raw", "utr3_start"]].head())
    first_track = df.iloc[0]["track"]
    print(f"\nFirst track shape: {first_track.shape}, dtype: {first_track.dtype}")
    print(f"Channels: [A, C, G, U, CDS, Splice]")
