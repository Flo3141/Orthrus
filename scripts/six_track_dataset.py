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


# Standard genetic code for CDS translation
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGA": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

GENCODE_REFERENCE_URLS = {
    "human": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/gencode.v29.pc_translations.fa.gz",
    "mouse": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M21/gencode.vM21.pc_translations.fa.gz",
}


def translate_codon(codon: str) -> str:
    """Translates a 3-nucleotide codon into an amino acid."""
    codon = codon.upper().replace("U", "T")
    return GENETIC_CODE.get(codon, "X")


def extract_and_translate_cds(sequence: str, cds_track) -> str:
    """Extracts CDS from sequence and translates into protein sequence."""
    if cds_track is None or len(sequence) == 0:
        return ""
    if isinstance(cds_track, (np.ndarray, list, pd.Series)):
        cds_positions = np.where(np.asarray(cds_track) > 0.5)[0]
    else:
        cds_positions = [i for i, val in enumerate(cds_track) if str(val) == "1"]
    if len(cds_positions) == 0:
        return ""
    seq_len = len(sequence)
    codons = [sequence[pos : pos + 3] for pos in cds_positions if pos + 3 <= seq_len]
    prot = "".join(translate_codon(c) for c in codons)
    return prot[:-1] if prot.endswith("*") else prot


def load_hgnc_alias_mapping(gencode_dir: Path) -> dict[str, str]:
    """
    Downloads and caches the official HGNC complete set table, returning
    a mapping from gene symbol/alias/previous symbol (uppercase) to Ensembl Gene ID.
    Source: https://www.genenames.org/
    """
    import urllib.request
    hgnc_url = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
    hgnc_path = gencode_dir / "hgnc_complete_set.txt"
    if not (hgnc_path.exists() and hgnc_path.stat().st_size > 0):
        print(f"[HGNC] Downloading official HGNC complete set from {hgnc_url}...")
        temp_file = gencode_dir / "hgnc_complete_set.txt.part"
        req = urllib.request.Request(hgnc_url, headers={"User-Agent": "MScThesis/1.0"})
        with urllib.request.urlopen(req) as resp, open(temp_file, "wb") as f_out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f_out.write(chunk)
        temp_file.rename(hgnc_path)

    print(f"[HGNC] Loading official HGNC gene alias mappings from '{hgnc_path.name}'...")
    alias_to_ensg: dict[str, str] = {}
    with open(hgnc_path, "rt", encoding="utf-8") as f_in:
        header = f_in.readline().strip().split("\t")
        sym_idx = header.index("symbol") if "symbol" in header else -1
        ensg_idx = header.index("ensembl_gene_id") if "ensembl_gene_id" in header else -1
        alias_idx = header.index("alias_symbol") if "alias_symbol" in header else -1
        prev_idx = header.index("prev_symbol") if "prev_symbol" in header else -1

        for line in f_in:
            parts = line.strip().split("\t")
            ensg = parts[ensg_idx].strip() if ensg_idx != -1 and len(parts) > ensg_idx else ""
            if not ensg:
                continue
            clean_ensg = ensg.split(".")[0].strip().upper()
            if sym_idx != -1 and len(parts) > sym_idx:
                sym = parts[sym_idx].strip().upper()
                if sym:
                    alias_to_ensg[sym] = clean_ensg
            if alias_idx != -1 and len(parts) > alias_idx:
                for a in parts[alias_idx].replace('"', '').split("|"):
                    a_clean = a.strip().upper()
                    if a_clean and a_clean not in alias_to_ensg:
                        alias_to_ensg[a_clean] = clean_ensg
            if prev_idx != -1 and len(parts) > prev_idx:
                for p in parts[prev_idx].replace('"', '').split("|"):
                    p_clean = p.strip().upper()
                    if p_clean and p_clean not in alias_to_ensg:
                        alias_to_ensg[p_clean] = clean_ensg
    return alias_to_ensg


def load_mgi_alias_mapping(gencode_dir: Path) -> dict[str, str]:
    """
    Downloads and caches official MGI gene model coordinates mapping mouse symbols
    to Ensembl Gene ID (ENSMUSG...).
    Source: Mouse Genome Informatics (The Jackson Laboratory)
    """
    import urllib.request
    mgi_url = "https://www.informatics.jax.org/downloads/reports/MGI_Gene_Model_Coord.rpt"
    mgi_path = gencode_dir / "MGI_Gene_Model_Coord.rpt"
    if not (mgi_path.exists() and mgi_path.stat().st_size > 0):
        print(f"[MGI] Downloading official MGI gene model mapping from {mgi_url}...")
        temp_file = gencode_dir / "MGI_Gene_Model_Coord.rpt.part"
        req = urllib.request.Request(mgi_url, headers={"User-Agent": "MScThesis/1.0"})
        with urllib.request.urlopen(req) as resp, open(temp_file, "wb") as f_out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f_out.write(chunk)
        temp_file.rename(mgi_path)

    print(f"[MGI] Loading official MGI mouse gene mappings from '{mgi_path.name}'...")
    alias_to_ensmusg: dict[str, str] = {}
    with open(mgi_path, "rt", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) > 10:
                sym = parts[2].strip().upper()
                ensg = parts[10].strip().split(".")[0].upper()
                if sym and ensg.startswith("ENSMUSG"):
                    alias_to_ensmusg[sym] = ensg
    return alias_to_ensmusg


def map_mrna_bench_to_gencode(df_raw: pd.DataFrame, species: str, gencode_dir: Path) -> pd.DataFrame:
    """
    Maps mRNABench genes to official GENCODE Ensembl transcript IDs (ENST.../ENSMUST...)
    using the exact reference releases matching the dataset:
      - Human: GENCODE v29 (Ensembl 94 / GRCh38)
      - Mouse: GENCODE vM21 (Ensembl 96 / GRCm38)
    Resolution Strategy:
      1. Exact Symbol Match in GENCODE header.
      2. 100% Exact Protein Sequence Match against all GENCODE translations (resolves clone/HAVANA names like AC118549.1 for ZZZ3).
      3. Official HGNC / MGI Alias & Ensembl Gene ID lookup.
    Raises RuntimeError if any transcript cannot be mapped (strictly no silent fallbacks).
    """
    import difflib
    import gzip
    import urllib.request

    species_key = species.lower()
    if species_key not in GENCODE_REFERENCE_URLS:
        raise ValueError(f"Unknown species '{species}'. Supported: {list(GENCODE_REFERENCE_URLS.keys())}")

    url = GENCODE_REFERENCE_URLS[species_key]
    filename = url.split("/")[-1]
    gencode_dir.mkdir(parents=True, exist_ok=True)
    local_file = gencode_dir / filename

    if not (local_file.exists() and local_file.stat().st_size > 0):
        print(f"[GENCODE] Reference not found locally. Downloading {species.upper()} reference from {url}...")
        temp_file = gencode_dir / f"{filename}.part"
        req = urllib.request.Request(url, headers={"User-Agent": "MScThesis/1.0"})
        with urllib.request.urlopen(req) as resp, open(temp_file, "wb") as f_out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f_out.write(chunk)
        temp_file.rename(local_file)

    print(f"[GENCODE] Loading official translations from '{local_file.name}'...")
    gene_dict: dict[str, list[dict]] = {}
    gene_id_dict: dict[str, list[dict]] = {}
    seq_dict: dict[str, list[dict]] = {}

    with gzip.open(local_file, "rt", encoding="utf-8") as f_in:
        current_entry = None
        seq_chunks = []
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_entry and seq_chunks:
                    seq_str = "".join(seq_chunks)
                    current_entry["seq"] = seq_str
                    gene_dict.setdefault(current_entry["gene_symbol"], []).append(current_entry)
                    if current_entry["gene_id_clean"]:
                        gene_id_dict.setdefault(current_entry["gene_id_clean"], []).append(current_entry)
                    if seq_str:
                        seq_dict.setdefault(seq_str, []).append(current_entry)
                parts = line[1:].split("|")
                # parts[0]: ENSP..., parts[1]: ENST..., parts[2]: ENSG..., parts[6]: Gene Symbol
                full_gene_id = parts[2] if len(parts) > 2 else ""
                clean_gene_id = full_gene_id.split(".")[0].strip().upper()
                current_entry = {
                    "protein_id": parts[0] if len(parts) > 0 else "",
                    "transcript_id": parts[1] if len(parts) > 1 else "",
                    "gene_id": full_gene_id,
                    "gene_id_clean": clean_gene_id,
                    "gene_symbol": parts[6].strip().upper() if len(parts) > 6 else "",
                }
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if current_entry and seq_chunks:
            seq_str = "".join(seq_chunks)
            current_entry["seq"] = seq_str
            gene_dict.setdefault(current_entry["gene_symbol"], []).append(current_entry)
            if current_entry["gene_id_clean"]:
                gene_id_dict.setdefault(current_entry["gene_id_clean"], []).append(current_entry)
            if seq_str:
                seq_dict.setdefault(seq_str, []).append(current_entry)

    df_mapped = df_raw.copy()
    mapped_tx_ids = []
    mapped_gene_ids = []
    alias_map: Optional[dict[str, str]] = None

    print(f"[GENCODE] Mapping {len(df_mapped):,} mRNABench ({species}) genes to Ensembl transcript IDs...")
    for idx, row in df_mapped.iterrows():
        gene = str(row.get("gene", "")).strip()
        gene_upper = gene.upper()
        seq = str(row.get("sequence", "")).strip()
        cds_raw = row.get("cds", None)
        prot = extract_and_translate_cds(seq, cds_raw)

        # Strategy 1: Exact symbol match in GENCODE
        entries = gene_dict.get(gene_upper, [])

        # Strategy 2: Exact protein sequence match against all GENCODE translations
        if not entries and prot:
            seq_matches = seq_dict.get(prot, [])
            if seq_matches:
                entries = seq_matches

        # Strategy 3: Official HGNC / MGI alias and Ensembl Gene ID resolution
        if not entries:
            if alias_map is None:
                if species_key == "human":
                    alias_map = load_hgnc_alias_mapping(gencode_dir)
                else:
                    alias_map = load_mgi_alias_mapping(gencode_dir)
            clean_gene_id = alias_map.get(gene_upper)
            if clean_gene_id and clean_gene_id in gene_id_dict:
                entries = gene_id_dict[clean_gene_id]

        if not entries:
            raise RuntimeError(
                f"[Error] Gene '{gene}' from mRNABench ({species}) could not be resolved in GENCODE reference "
                f"({local_file.name})! Checked: (1) exact symbol, (2) exact protein sequence, (3) HGNC/MGI aliases. "
                f"No fallbacks allowed. All transcripts must have verified Ensembl transcript IDs."
            )

        canonical = entries[0]
        if prot and prot == canonical["seq"]:
            best = canonical
        else:
            matched = None
            if prot:
                for e in entries[1:]:
                    if prot == e["seq"]:
                        matched = e
                        break
            if matched:
                best = matched
            else:
                if prot:
                    best = max(
                        entries,
                        key=lambda e: difflib.SequenceMatcher(None, prot, e["seq"], autojunk=False).ratio(),
                    )
                else:
                    best = canonical

        tx_id = best.get("transcript_id", "").strip()
        if not tx_id or not (tx_id.startswith("ENST") or tx_id.startswith("ENSMUST")):
            raise RuntimeError(
                f"[Error] Failed to resolve Ensembl transcript ID for gene '{gene}' in GENCODE! "
                f"No fallbacks allowed."
            )
        mapped_tx_ids.append(tx_id)
        mapped_gene_ids.append(best.get("gene_id", ""))

    df_mapped["transcript_id"] = mapped_tx_ids
    df_mapped["gencode_transcript_id"] = mapped_tx_ids
    df_mapped["ensembl_transcript_id"] = mapped_tx_ids
    df_mapped["ensembl_gene_id"] = mapped_gene_ids
    print(f"[GENCODE] Successfully mapped all {len(df_mapped):,} transcripts to official Ensembl IDs.")
    return df_mapped


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
        All transcripts are strictly mapped to official Ensembl transcript IDs (ENST... / ENSMUST...).
        """
        dataset_key = f"mrna_bench_{self.species}"
        data_dir = self.data_path or self.MRNA_BENCH_DEFAULT_DIR

        print(f"[SixTrackLoader] Loading mRNABench ({dataset_key}) from: {data_dir}")

        df_raw = None

        # Priority 1: Check if pre-mapped GENCODE table already exists AND is 100% complete
        candidate_subdirs = [
            dataset_key,
            f"rnahl-{self.species}",
            f"rnahl_{self.species}",
            self.species,
        ]
        mapped_candidates = []
        for sdir in candidate_subdirs:
            mapped_candidates.extend([
                data_dir / sdir / "mrna_protein.parquet",
                data_dir / sdir / f"{self.species}_mrna_protein.parquet",
                data_dir / sdir / "mrna_protein.tsv",
                data_dir / sdir / f"{self.species}_mrna_protein.tsv",
                data_dir / sdir / "mrna_protein.csv",
                data_dir / sdir / f"{self.species}_mrna_protein.csv",
            ])
        mapped_candidates.extend([
            data_dir / "mrna_protein.parquet",
            data_dir / f"{self.species}_mrna_protein.parquet",
            data_dir / "mrna_protein.tsv",
            data_dir / f"{self.species}_mrna_protein.tsv",
            data_dir / "mrna_protein.csv",
            data_dir / f"{self.species}_mrna_protein.csv",
        ])
        complete_mapped_df = None
        unmapped_candidate_df = None

        for cand in mapped_candidates:
            if cand.exists():
                if cand.suffix == ".parquet":
                    df_candidate = pd.read_parquet(cand)
                else:
                    sep = "\t" if cand.suffix == ".tsv" else ","
                    df_candidate = pd.read_csv(cand, sep=sep)

                valid_col = None
                for c in ["transcript_id", "gencode_transcript_id", "ensembl_transcript_id"]:
                    if c in df_candidate.columns:
                        vals = df_candidate[c].astype(str).str.strip()
                        if len(vals) > 0 and (vals.str.startswith(("ENST", "ENSMUST"))).all():
                            valid_col = c
                            break

                if valid_col:
                    print(f"[SixTrackLoader] Verified fully-mapped dataset table: {cand.name} ({len(df_candidate):,} transcripts)")
                    complete_mapped_df = df_candidate
                    break
                else:
                    if unmapped_candidate_df is None:
                        unmapped_candidate_df = df_candidate
                    print(f"[SixTrackLoader] Notice: '{cand.name}' exists but contains unverified/missing transcript IDs. Will run verified GENCODE mapping.")

        if complete_mapped_df is not None:
            df_raw = complete_mapped_df
        else:
            # Priority 2: Load raw dataset or unmapped candidate and map to GENCODE
            if unmapped_candidate_df is not None:
                df_raw = unmapped_candidate_df
            elif data_dir.is_file():
                print(f"[SixTrackLoader] Loading mRNABench from direct file: {data_dir.name}")
                sep = "\t" if data_dir.suffix in [".tsv", ".txt"] else ","
                df_raw = pd.read_csv(data_dir, sep=sep)
            else:
                try:
                    import mrna_bench as mb
                    mb.update_data_path(str(data_dir))
                    bench_key = f"rnahl-{self.species}"
                    df_raw = mb.load_dataset(bench_key).data_df
                except Exception:
                    # Tabular fallback candidates
                    fallback_files = []
                    for sdir in candidate_subdirs:
                        fallback_files.extend([
                            data_dir / f"{sdir}.tsv",
                            data_dir / f"{sdir}.csv",
                        ])
                    for c in fallback_files:
                        if c.exists():
                            print(f"[SixTrackLoader] Found tabular table fallback: {c.name}")
                            sep = "\t" if c.suffix == ".tsv" else ","
                            df_raw = pd.read_csv(c, sep=sep)
                            break

                    if df_raw is None:
                        raise ImportError(
                            "mrna_bench Python package is not installed and no fallback tabular file was found. "
                            f"Please install mrna_bench or provide data in {data_dir}."
                        )

            # Map all transcripts to GENCODE
            gencode_dir = data_dir / "gencode" if data_dir.is_dir() else Path("gencode")
            df_raw = map_mrna_bench_to_gencode(df_raw, self.species, gencode_dir)

            # Save the verified mapped table back so all future runs are instant
            try:
                target_cache = data_dir / dataset_key / "mrna_protein.parquet"
                target_cache.parent.mkdir(parents=True, exist_ok=True)
                df_raw.to_parquet(target_cache, index=False)
                print(f"[SixTrackLoader] Successfully cached fully-mapped table to: {target_cache}")
            except Exception as e:
                print(f"[SixTrackLoader] Note: Could not cache mapped parquet file ({e}).")

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
            cds_indices = np.where(cds_track.flatten() > 0.5)[0]
            if len(cds_indices) > 0:
                utr3_start = min(int(cds_indices[-1]) + 3, l)
            else:
                utr3_start = 0

            gene_symbol = str(row.get("gene", "")).strip()

            # Transcript ID: look for mapped GENCODE/Ensembl transcript ID
            tx_id = ""
            for candidate_col in ["transcript_id", "gencode_transcript_id", "ensembl_transcript_id"]:
                if candidate_col in row and pd.notna(row[candidate_col]) and str(row[candidate_col]).strip():
                    val = str(row[candidate_col]).strip()
                    # Only accept real transcript IDs (ENST... / ENSMUST...)
                    if val.startswith("ENST") or val.startswith("ENSMUST"):
                        tx_id = val
                        break

            # STRICT: No fallback! Raise RuntimeError if unmapped.
            if not tx_id or not (tx_id.startswith("ENST") or tx_id.startswith("ENSMUST")):
                raise RuntimeError(
                    f"[Error] Transcript for gene '{gene_symbol}' in {dataset_key} has no verified "
                    f"Ensembl transcript ID (ENST.../ENSMUST...)! No fallbacks allowed. "
                    f"All transcripts must have official Ensembl/GENCODE transcript IDs."
                )

            gene_id = ""
            for candidate_gene_col in ["ensembl_gene_id", "gene_id", "gencode_gene_id"]:
                if candidate_gene_col in row and pd.notna(row[candidate_gene_col]) and str(row[candidate_gene_col]).strip():
                    gene_id = str(row[candidate_gene_col]).strip()
                    break

            if not gene_id or not (gene_id.startswith("ENSG") or gene_id.startswith("ENSMUSG")):
                gene_id = gene_symbol

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
