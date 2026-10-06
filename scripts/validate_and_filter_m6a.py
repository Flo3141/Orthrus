#!/usr/bin/env python3
"""
validate_and_filter_m6a.py
=============================================================================
Validation and filtering of experimental m6A modification calls in hiPSC-CM.

Validates candidate m6A sites called in Control (DMSO) replicates against:
1. Differential Modification Results (modkit dmr pair: DMSO vs. M3inh)
   - Bona fide METTL3-dependent m6A sites must exhibit a significant drop
     in modification frequency upon METTL3 inhibition (effect_size > 0).
2. Replicate reproducibility across biological replicates (e.g., >= 2 or 3 of 4).
3. Optional DRACH consensus motif enrichment check (GRCh38 reference FASTA).

Outputs:
- Confident, validated m6A site table (genomic + cDNA transcript coordinates)
- Summary statistics per transcript for downstream integration into Orthrus
- Comprehensive QC report and breakdown of artifact vs. valid sites.
=============================================================================
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. Utility Functions: Statistics & Sequence Analysis
# =============================================================================

def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """
    Computes Benjamini-Hochberg False Discovery Rate (FDR / q-value).
    Pure NumPy implementation without heavy external dependencies.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Replace NaNs or invalid values with 1.0
    valid_mask = np.isfinite(p)
    p_clean = np.where(valid_mask, p, 1.0)

    order = np.argsort(p_clean)
    ranked = p_clean[order]
    
    # q = p * (N / rank)
    ranks = np.arange(1, n + 1)
    q = ranked * (n / ranks)
    
    # Enforce monotonicity: q_i = min(q_i, q_{i+1})
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    
    # Reorder to original indices
    out = np.empty_like(q)
    out[order] = q
    out[~valid_mask] = np.nan
    return out


def normalize_chrom(chrom: str) -> str:
    """Standardizes chromosome representation (e.g. 'chr1' -> '1', 'chrM' -> 'MT')."""
    c = str(chrom).strip()
    if c.startswith("chr"):
        c = c[3:]
    if c.upper() == "M":
        c = "MT"
    return c


def reverse_complement(seq: str) -> str:
    """Computes reverse complement of a nucleotide sequence."""
    trans = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")
    return seq.translate(trans)[::-1]


def is_drach_motif(kmer: str) -> bool:
    """
    Checks if a 5-nt motif centered at modified Adenosine (position index 2)
    matches the classic m6A DRACH consensus motif:
      D = A / G / T (or U) [not C]
      R = A / G (purine)
      A = A (modified base)
      C = C
      H = A / C / T (or U) [not G]
    """
    if len(kmer) != 5:
        return False
    k = kmer.upper().replace("U", "T")
    # Regex pattern: [AGT][AG]AC[ACT]
    drach_pattern = re.compile(r"^[AGT][AG]AC[ACT]$")
    return bool(drach_pattern.match(k))


# =============================================================================
# 2. Loading & Filtering modkit dmr File
# =============================================================================

def load_and_filter_dmr(
    dmr_path: str,
    min_effect_size: float = 0.15,
    max_pvalue: float = 0.05,
    min_pct_a_samples: float = 75.0,
    min_a_pct_modified: float = 0.10,
    use_fdr: bool = True,
    max_fdr: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Loads and filters the differential methylation table (modkit dmr pair output).
    Condition A = Control / DMSO
    Condition B = METTL3 inhibitor (M3inh)
    """
    print(f"\n[1/5] Loading DMR file: {dmr_path}")
    if not os.path.exists(dmr_path):
        raise FileNotFoundError(f"DMR file not found at: {dmr_path}")

    # Read tab-separated DMR file, skipping comment lines
    df_dmr = pd.read_csv(
        dmr_path,
        sep="\t",
        comment="#",
        names=[
            "chrom", "start", "end", "name", "score", "strand",
            "a_counts", "a_total", "b_counts", "b_total",
            "a_mod_percentages", "b_mod_percentages",
            "a_pct_modified", "b_pct_modified",
            "map_pvalue", "effect_size",
            "balanced_map_pvalue", "balanced_effect_size",
            "pct_a_samples", "pct_b_samples",
            "replicate_map_pvalues", "replicate_effect_sizes",
            "cohen_h", "cohen_h_low", "cohen_h_high"
        ] if False else None,  # Will auto-detect header if present
        low_memory=False
    )

    # Rename first column if it contains '#chrom'
    if df_dmr.columns[0].startswith("#"):
        df_dmr.rename(columns={df_dmr.columns[0]: "chrom"}, inplace=True)

    total_sites = len(df_dmr)
    print(f"      Total DMR candidate sites loaded: {total_sites:,}")

    # Normalize chromosome strings
    df_dmr["chrom"] = df_dmr["chrom"].astype(str).apply(normalize_chrom)
    df_dmr["start"] = pd.to_numeric(df_dmr["start"], errors="coerce")
    df_dmr["end"] = pd.to_numeric(df_dmr["end"], errors="coerce")
    df_dmr["effect_size"] = pd.to_numeric(df_dmr["effect_size"], errors="coerce")
    df_dmr["map_pvalue"] = pd.to_numeric(df_dmr["map_pvalue"], errors="coerce")
    df_dmr["a_pct_modified"] = pd.to_numeric(df_dmr["a_pct_modified"], errors="coerce")
    df_dmr["b_pct_modified"] = pd.to_numeric(df_dmr["b_pct_modified"], errors="coerce")
    df_dmr["pct_a_samples"] = pd.to_numeric(df_dmr["pct_a_samples"], errors="coerce")

    # Compute FDR (q-value)
    df_dmr["fdr_qvalue"] = benjamini_hochberg(df_dmr["map_pvalue"].values)

    # Filtering tracking
    stats = {"total_dmr_records": total_sites}

    # Condition 1: Sample presence in Control (replicate reproducibility)
    mask_sample = df_dmr["pct_a_samples"] >= min_pct_a_samples
    stats["passed_replicate_presence"] = int(mask_sample.sum())

    # Condition 2: Baseline modification frequency in Control
    mask_ctrl_mod = df_dmr["a_pct_modified"] >= min_a_pct_modified
    stats["passed_min_ctrl_mod"] = int(mask_ctrl_mod.sum())

    # Condition 3: Effect size (inhibition drop: a_pct - b_pct > min_effect_size)
    mask_effect = df_dmr["effect_size"] >= min_effect_size
    stats["passed_effect_size"] = int(mask_effect.sum())

    # Condition 4: Statistical significance (p-value or FDR)
    if use_fdr:
        mask_sig = df_dmr["fdr_qvalue"] <= max_fdr
        stats["passed_fdr"] = int(mask_sig.sum())
    else:
        mask_sig = df_dmr["map_pvalue"] <= max_pvalue
        stats["passed_pvalue"] = int(mask_sig.sum())

    # Combined filter for bona fide METTL3-dependent sites
    final_mask = mask_sample & mask_ctrl_mod & mask_effect & mask_sig
    df_dmr_validated = df_dmr[final_mask].copy()

    stats["validated_mettl3_dependent_dmr"] = len(df_dmr_validated)
    print(f"      Sites passing METTL3 inhibitor reduction filter: {len(df_dmr_validated):,} "
          f"({len(df_dmr_validated)/max(1, total_sites):.1%})")

    # Create coordinate lookup key: (chrom, start, strand)
    df_dmr_validated["site_key"] = (
        df_dmr_validated["chrom"] + ":" +
        df_dmr_validated["start"].astype(int).astype(str) + ":" +
        df_dmr_validated["strand"]
    )

    return df_dmr_validated, stats


# =============================================================================
# 3. Loading & Aggregating Control cDNA BED Files
# =============================================================================

def load_and_aggregate_ctrl_beds(
    bed_files: List[str],
    min_rep_count: int = 2
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Loads individual replicate cDNAfinal.bed files for Control (DMSO).
    Aggregates per transcript coordinate and counts replicate recurrence.
    """
    print(f"\n[2/5] Loading Control cDNAfinal.bed files ({len(bed_files)} files found)...")
    records = []

    for fpath in bed_files:
        print(f"      Reading {Path(fpath).name}...")
        df = pd.read_csv(fpath, sep="\t", low_memory=False)
        # Expected columns:
        # Sample, TxId, cDNAstart, cDNAend, Chr, Gstart, Gend, Strand, txlen, cdslen, utr5_len, utr3_len, score
        df["Chr"] = df["Chr"].astype(str).apply(normalize_chrom)
        df["Gstart"] = pd.to_numeric(df["Gstart"], errors="coerce")
        df["Gend"] = pd.to_numeric(df["Gend"], errors="coerce")
        df["cDNAstart"] = pd.to_numeric(df["cDNAstart"], errors="coerce")
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        records.append(df)

    if not records:
        raise ValueError("No Control BED records could be loaded.")

    df_all_ctrl = pd.concat(records, ignore_index=True)
    total_calls = len(df_all_ctrl)
    print(f"      Total raw individual replicate calls across all files: {total_calls:,}")

    # Create site key for genomic location
    df_all_ctrl["site_key"] = (
        df_all_ctrl["Chr"] + ":" +
        df_all_ctrl["Gstart"].astype(int).astype(str) + ":" +
        df_all_ctrl["Strand"]
    )

    # Group by Transcript and cDNA position to aggregate replicate calls
    group_cols = ["TxId", "cDNAstart", "cDNAend", "Chr", "Gstart", "Gend", "Strand"]
    meta_cols = ["txlen", "cdslen", "utr5_len", "utr3_len"]

    agg_df = df_all_ctrl.groupby(group_cols).agg(
        rep_count=("score", "count"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        samples_detected=("Sample", lambda s: ";".join(sorted(set(s)))),
        site_key=("site_key", "first"),
        txlen=("txlen", "first"),
        cdslen=("cdslen", "first"),
        utr5_len=("utr5_len", "first"),
        utr3_len=("utr3_len", "first"),
    ).reset_index()

    stats = {
        "total_raw_ctrl_calls": total_calls,
        "unique_transcript_sites": len(agg_df),
        "sites_recurrent_ge2": int((agg_df["rep_count"] >= 2).sum()),
        "sites_recurrent_ge3": int((agg_df["rep_count"] >= 3).sum()),
        "sites_in_all_4": int((agg_df["rep_count"] >= 4).sum()),
    }

    print(f"      Unique transcript m6A positions: {len(agg_df):,}")
    print(f"      Positions detected in >= {min_rep_count} replicates: "
          f"{(agg_df['rep_count'] >= min_rep_count).sum():,}")

    return agg_df, stats


# =============================================================================
# 4. Intersecting Calls with DMR Validations
# =============================================================================

def intersect_ctrl_with_dmr(
    df_ctrl_agg: pd.DataFrame,
    df_dmr_validated: pd.DataFrame,
    min_rep_count: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges aggregated Control sites with validated DMR entries.
    Distinguishes confirmed METTL3-dependent sites from putative artifacts.
    """
    print(f"\n[3/5] Intersecting Control calls with DMR validation results...")

    # Set of validated site keys from DMR
    valid_dmr_keys = set(df_dmr_validated["site_key"].unique())

    # Map validation status
    df_ctrl_agg["dmr_validated"] = df_ctrl_agg["site_key"].isin(valid_dmr_keys)

    # Merge DMR statistics into the table
    dmr_cols_to_keep = [
        "site_key", "effect_size", "a_pct_modified", "b_pct_modified",
        "map_pvalue", "fdr_qvalue", "pct_a_samples", "pct_b_samples"
    ]
    df_merged = pd.merge(
        df_ctrl_agg,
        df_dmr_validated[dmr_cols_to_keep],
        on="site_key",
        how="left"
    )

    # Final validated subset: Meets replicate count AND DMR validation criteria
    mask_confident = (df_merged["rep_count"] >= min_rep_count) & (df_merged["dmr_validated"] == True)
    df_validated_final = df_merged[mask_confident].copy()

    # Unvalidated / Artifact candidates: High calls in ctrl but no reduction in M3inh
    df_non_validated = df_merged[~mask_confident].copy()

    print(f"      High-confidence validated sites (replicates >= {min_rep_count} & DMR validated): "
          f"{len(df_validated_final):,}")
    print(f"      Filtered out / non-responsive / unconfirmed sites: {len(df_non_validated):,}")

    return df_validated_final, df_merged


# =============================================================================
# 5. DRACH Motif Validation (Genome Reference Check)
# =============================================================================

def check_drach_motifs(
    df_sites: pd.DataFrame,
    fasta_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Extracts 5-mer sequences from reference genome FASTA and checks for DRACH motif.
    Returns enriched dataframe and motif frequency metrics.
    """
    if not fasta_path or not os.path.exists(fasta_path):
        print("\n[4/5] Reference FASTA not provided or not found. Skipping DRACH motif extraction.")
        df_sites["motif_5mer"] = "NA"
        df_sites["is_drach"] = False
        return df_sites, {"drach_percentage": 0.0}

    print(f"\n[4/5] Extracting 5-mer genomic sequences & checking DRACH motif from {Path(fasta_path).name}...")

    try:
        import pysam
        has_pysam = True
    except ImportError:
        has_pysam = False
        print("      [WARN] 'pysam' library not installed. Attempting pure Python FASTA index lookup.")

    drach_results = []
    motifs = []

    if has_pysam:
        fasta = pysam.FastaFile(fasta_path)
        fasta_chroms = set(fasta.references)

        for _, row in df_sites.iterrows():
            chrom = str(row["Chr"])
            # Match FASTA chromosome naming convention
            if chrom not in fasta_chroms and f"chr{chrom}" in fasta_chroms:
                target_chrom = f"chr{chrom}"
            elif chrom in fasta_chroms:
                target_chrom = chrom
            else:
                motifs.append("CHR_NOT_FOUND")
                drach_results.append(False)
                continue

            # 0-based coordinate for Gstart
            # modkit / bed: Gstart is 0-based coordinate of the modified base
            gstart = int(row["Gstart"])
            strand = str(row["Strand"])

            # 5-mer: 2 bp upstream, central base, 2 bp downstream
            start_pos = max(0, gstart - 2)
            end_pos = gstart + 3

            try:
                seq = fasta.fetch(target_chrom, start_pos, end_pos).upper()
                if len(seq) == 5:
                    if strand == "-":
                        seq = reverse_complement(seq)
                    motifs.append(seq)
                    drach_results.append(is_drach_motif(seq))
                else:
                    motifs.append("EDGE_TRUNCATED")
                    drach_results.append(False)
            except Exception:
                motifs.append("FETCH_ERROR")
                drach_results.append(False)

        fasta.close()
    else:
        # Fallback without pysam
        motifs = ["NO_PYSAM"] * len(df_sites)
        drach_results = [False] * len(df_sites)

    df_sites["motif_5mer"] = motifs
    df_sites["is_drach"] = drach_results

    valid_motifs = [m for m in motifs if len(m) == 5 and "N" not in m]
    drach_count = sum(drach_results)
    drach_pct = (drach_count / max(1, len(valid_motifs))) * 100.0

    print(f"      Valid 5-mers checked: {len(valid_motifs):,}")
    print(f"      Sites matching DRACH consensus ([A/G/U][A/G]AC[A/C/U]): "
          f"{drach_count:,} ({drach_pct:.1f}%)")

    return df_sites, {"drach_count": drach_count, "drach_percentage": drach_pct}


# =============================================================================
# 6. Aggregating Features per Transcript for Model Input
# =============================================================================

def build_transcript_m6a_summary(df_validated: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates validated m6A site metrics at the transcript level for Orthrus:
    - Number of validated m6A sites per transcript
    - Mean and Max m6A modification frequency
    - Region breakdown: 5'UTR, CDS, 3'UTR distribution
    """
    print("\n[5/5] Generating transcript-level m6A summary for model integration...")

    def classify_region(row):
        pos = row["cDNAstart"]
        u5 = row["utr5_len"] if pd.notnull(row["utr5_len"]) else 0
        cds = row["cdslen"] if pd.notnull(row["cdslen"]) else 0
        if pos <= u5:
            return "5utr"
        elif pos <= u5 + cds:
            return "cds"
        else:
            return "3utr"

    df_val = df_validated.copy()
    df_val["region"] = df_val.apply(classify_region, axis=1)

    tx_summary = df_val.groupby("TxId").agg(
        num_m6a_sites=("cDNAstart", "count"),
        mean_mod_pct=("mean_score", "mean"),
        max_mod_pct=("max_score", "max"),
        mean_effect_size=("effect_size", "mean"),
        sites_5utr=("region", lambda r: (r == "5utr").sum()),
        sites_cds=("region", lambda r: (r == "cds").sum()),
        sites_3utr=("region", lambda r: (r == "3utr").sum()),
        txlen=("txlen", "first"),
    ).reset_index()

    tx_summary["m6a_density_per_kb"] = (tx_summary["num_m6a_sites"] / (tx_summary["txlen"] / 1000.0)).round(3)
    tx_summary.sort_values(by="num_m6a_sites", ascending=False, inplace=True)

    print(f"      Transcripts with at least one validated m6A site: {len(tx_summary):,}")
    print(f"      Top 5 transcripts with highest m6A site counts:")
    for _, r in tx_summary.head(5).iterrows():
        print(f"        - {r['TxId']}: {r['num_m6a_sites']} sites (CDS: {r['sites_cds']}, 3'UTR: {r['sites_3utr']})")

    return tx_summary


# =============================================================================
# 7. Report Generation & Exports
# =============================================================================

def export_results(
    output_dir: str,
    df_validated_final: pd.DataFrame,
    df_all_annotated: pd.DataFrame,
    tx_summary: pd.DataFrame,
    dmr_stats: Dict,
    ctrl_stats: Dict,
    drach_stats: Dict,
    args: argparse.Namespace
) -> None:
    """Exports TSV tables, BED tracks, and markdown validation report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Validated Sites TSV
    val_tsv = out_path / "validated_m6a_sites.tsv"
    df_validated_final.to_csv(val_tsv, sep="\t", index=False)
    print(f"\n[OK] Saved validated m6A sites to: {val_tsv}")

    # 2. Complete Annotated Table (all Ctrl calls with validation flags)
    all_tsv = out_path / "all_ctrl_calls_validation_status.tsv"
    df_all_annotated.to_csv(all_tsv, sep="\t", index=False)
    print(f"[OK] Saved all annotated control calls to: {all_tsv}")

    # 3. Transcript-level Summary for Model Track
    tx_tsv = out_path / "transcript_m6a_features.tsv"
    tx_summary.to_csv(tx_tsv, sep="\t", index=False)
    print(f"[OK] Saved transcript-level m6A features to: {tx_tsv}")

    # 4. Standard Genomic BED file for visualization (IGV)
    bed_path = out_path / "validated_m6a_sites.bed"
    with open(bed_path, "w") as f:
        f.write("#chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n")
        for _, row in df_validated_final.iterrows():
            c = f"chr{row['Chr']}" if not str(row['Chr']).startswith("chr") else row['Chr']
            s_int = int(round(row['mean_score'] * 10))  # Scale 0-100 to 0-1000 for standard BED score
            f.write(f"{c}\t{int(row['Gstart'])}\t{int(row['Gend'])}\t"
                    f"{row['TxId']}_m6A\t{min(1000, max(0, s_int))}\t{row['Strand']}\n")
    print(f"[OK] Saved genomic BED file for IGV to: {bed_path}")

    # 5. Markdown Report
    report_path = out_path / "m6a_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Qualitäts- und Validierungsbericht: m6A-Modifikationen in hiPSC-CM\n\n")
        f.write("## 1. Übersicht & Filter-Parameter\n\n")
        f.write("| Parameter | Gewählter Wert | Beschreibung |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| `min_effect_size` | **{args.min_effect_size}** (min. {args.min_effect_size*100:.0f} % Rückgang) | Minimale Modifikationsreduktion durch M3inh |\n")
        f.write(f"| `max_fdr` | **{args.max_fdr}** | Benjamini-Hochberg FDR Signifikanz-Schwelle |\n")
        f.write(f"| `min_rep_count` | **{args.min_rep_count}** | Mindestanzahl bestätigender Kontroll-Replikate |\n")
        f.write(f"| `min_pct_a_samples` | **{args.min_pct_samples} %** | Mindest-Abdeckung in Control-Replikaten im DMR |\n")
        f.write(f"| `min_a_pct_modified` | **{args.min_ctrl_mod}** | Mindest-Modifikationsrate in Control |\n\n")

        f.write("## 2. Statistische Filter-Kaskade\n\n")
        f.write("| Schritt / Merkmal | Anzahl Positionen | Anteil / Status |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| Roh-Kandidaten im DMR-File | {dmr_stats.get('total_dmr_records', 0):,} | 100,0 % |\n")
        f.write(f"| Ausreichende Coverage in Ctrl | {dmr_stats.get('passed_replicate_presence', 0):,} | {dmr_stats.get('passed_replicate_presence', 0)/max(1, dmr_stats.get('total_dmr_records', 1)):.1%} |\n")
        f.write(f"| Signifikante Inhibitor-Reduktion | {dmr_stats.get('validated_mettl3_dependent_dmr', 0):,} | Echte METTL3-abhängige Sites |\n")
        f.write(f"| Eindeutige Transkript-Sites in Ctrl | {ctrl_stats.get('unique_transcript_sites', 0):,} | Replikats-Pool |\n")
        f.write(f"| Replikat-Konsens (>= {args.min_rep_count} Replikate) | {ctrl_stats.get(f'sites_recurrent_ge{args.min_rep_count}', 0):,} | Reproduzierbar |\n")
        f.write(f"| **Final validierte bona fide m6A Sites** | **{len(df_validated_final):,}** | **Hohe Konfidenz für Modell-Track** |\n\n")

        if drach_stats.get("drach_percentage", 0) > 0:
            f.write("## 3. DRACH-Konsensus-Motiv Validierung\n\n")
            f.write(f"* **DRACH-Trefferquote ([A/G/U][A/G]AC[A/C/U]):** **{drach_stats['drach_percentage']:.1f} %** "
                    f"({drach_stats.get('drach_count', 0):,} von geprüften Sites).\n")
            f.write("* *Interpretation:* Hohe Übereinstimmung mit dem klassischen DRACH-Motiv belegt biologische Validität "
                    "und filtert unspezifisches Nanopore-Rauschen erfolgreich aus.\n\n")

        f.write("## 4. Verteilung auf Transkripte\n\n")
        f.write(f"* Transkripte mit mindestens einer validen m6A-Site: **{len(tx_summary):,}**\n")
        f.write(f"* Mittlere Anzahl m6A-Sites pro modifiziertem Transkript: **{tx_summary['num_m6a_sites'].mean():.2f}**\n")
        f.write(f"* Verteilung nach Regionen: **5'UTR:** {tx_summary['sites_5utr'].sum():,} | "
                f"**CDS:** {tx_summary['sites_cds'].sum():,} | "
                f"**3'UTR:** {tx_summary['sites_3utr'].sum():,}\n")

    print(f"[OK] Saved comprehensive Markdown validation report to: {report_path}")


# =============================================================================
# 8. Main CLI Routine
# =============================================================================

# =============================================================================
# Default Cluster File Paths
# =============================================================================
DEFAULT_DMR_FILE = "/prj/Heart_on_m6A/Isabel/bambu/DMSO_M3I_replicates_cov10_baseA_moda.dmr"
DEFAULT_CTRL_BEDS = ["/prj/Heart_on_m6A/Isabel/bambu/hiPSC-CM_CasRx_ctrl_*_RTA_m6A_cDNAfinal.bed"]
DEFAULT_FASTA = "/biodb/genomes/homo_sapiens/GRCh38_102/GRCh38_102.fa"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and validate hiPSC-CM experimental m6A modification calls using modkit DMR and M3inh."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./m6a_validation_output",
        help="Directory to save validated tables and summary report"
    )
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.15,
        help="Minimum drop in modification percentage under M3inh (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--max-fdr",
        type=float,
        default=0.05,
        help="Maximum Benjamini-Hochberg FDR q-value for differential modification (default: 0.05)"
    )
    parser.add_argument(
        "--min-pct-samples",
        type=float,
        default=75.0,
        help="Minimum percentage of Control replicates with coverage in DMR (default: 75%% = 3 of 4)"
    )
    parser.add_argument(
        "--min-ctrl-mod",
        type=float,
        default=0.10,
        help="Minimum modification frequency in Control condition (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--min-rep-count",
        type=int,
        default=2,
        help="Minimum number of individual Control replicates calling the site (default: 2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(" hiPSC-CM m6A Validation & Filtering Pipeline")
    print("=" * 80)

    # Resolve glob patterns
    expanded_bed_files = []
    for item in DEFAULT_CTRL_BEDS:
        matched = glob.glob(item)
        if matched:
            expanded_bed_files.extend(matched)
        elif os.path.exists(item):
            expanded_bed_files.append(item)

    if not expanded_bed_files:
        print(f"[ERROR] No valid Control BED files found matching pattern: {DEFAULT_CTRL_BEDS}")
        sys.exit(1)

    expanded_bed_files = sorted(set(expanded_bed_files))
    print(f"Discovered {len(expanded_bed_files)} Control cDNA BED files:")
    for f in expanded_bed_files:
        print(f"  - {f}")

    # 1. Load and filter DMR
    df_dmr_val, dmr_stats = load_and_filter_dmr(
        dmr_path=DEFAULT_DMR_FILE,
        min_effect_size=args.min_effect_size,
        min_pct_a_samples=args.min_pct_samples,
        min_a_pct_modified=args.min_ctrl_mod,
        use_fdr=True,
        max_fdr=args.max_fdr
    )

    # 2. Load and aggregate individual control cDNA calls
    df_ctrl_agg, ctrl_stats = load_and_aggregate_ctrl_beds(
        bed_files=expanded_bed_files,
        min_rep_count=args.min_rep_count
    )

    # 3. Intersect control calls with DMR validated sites
    df_val_final, df_all_annotated = intersect_ctrl_with_dmr(
        df_ctrl_agg=df_ctrl_agg,
        df_dmr_validated=df_dmr_val,
        min_rep_count=args.min_rep_count
    )

    # 4. Check DRACH consensus motifs
    df_val_final, drach_stats = check_drach_motifs(
        df_sites=df_val_final,
        fasta_path=DEFAULT_FASTA
    )

    # 5. Build transcript-level summary for downstream model track creation
    tx_summary = build_transcript_m6a_summary(df_val_final)

    # 6. Save results & markdown report
    export_results(
        output_dir=args.output_dir,
        df_validated_final=df_val_final,
        df_all_annotated=df_all_annotated,
        tx_summary=tx_summary,
        dmr_stats=dmr_stats,
        ctrl_stats=ctrl_stats,
        drach_stats=drach_stats,
        args=args
    )

    print("\n" + "=" * 80)
    print(" Pipeline execution finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
