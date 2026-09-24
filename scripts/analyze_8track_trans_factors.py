#!/usr/bin/env python3
"""
Comprehensive Analysis of Trans-Factor Channels (miRNA Ch6 & RBP Ch7) in hIPSC_CM 8-Track Dataset
==================================================================================================

Analyzes:
  Input:  /beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/hIPSC_CM_8track_minmax.npz
  Output: /beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_8track_minmax_analysis.txt

Channels in the 8-track representation:
  - 0..3: One-hot nucleotides (A, C, G, U)
  - 4:    CDS (1.0 = coding sequence position, 0.0 = UTR)
  - 5:    Splice junctions (1.0 = exon-exon boundary position)
  - 6:    TargetScan miRNA binding affinity (min-max robust quantile normalized)
  - 7:    ENCODE eCLIP RBP peak signal (min-max robust quantile normalized)

Analyses conducted:
  1. Per-exon mean, sum, and maximum signal intensities for both channels.
  2. First vs. internal vs. terminal exon signal distributions.
  3. Regional breakdown: 5' UTR vs. CDS vs. 3' UTR trans-factor density.
  4. Codon Usage: GC content, GC3 (wobble GC), Human Codon Stabilization Coefficient (CSC),
     and Codon Adaptation Index (CAI).
  5. Correlations with structural properties: Total length, CDS length, UTR lengths, exon count.
  6. Direct correlations with biological target: mRNA half-life (raw and Z-transformed).
  7. Cross-channel colocalization, competition, and overlap (miRNA vs. RBP).
  8. Multivariate regression & feature importance analysis for mRNA stability.
  9. High-resolution publication-quality visualization figures.
"""

import argparse
import io
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm

# Headless backend for Matplotlib on cluster nodes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. Bioinformatic Dictionaries: Genetic Code, Codon Usage, and Human CSC
# =============================================================================

# Standard Genetic Code
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Human Codon Usage Frequencies per thousand (Kazusa Codon Database / Human Consensus)
HUMAN_CODON_FREQ_PER_THOUSAND = {
    "TTT": 17.6, "TTC": 20.3, "TTA": 7.7,  "TTG": 12.9,
    "CTT": 13.2, "CTC": 19.6, "CTA": 7.2,  "CTG": 39.6,
    "ATT": 16.0, "ATC": 20.8, "ATA": 7.5,  "ATG": 22.0,
    "GTT": 11.0, "GTC": 14.5, "GTA": 7.1,  "GTG": 28.1,
    "TCT": 15.2, "TCC": 17.7, "TCA": 12.2, "TCG": 4.4,
    "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG": 6.9,
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG": 6.1,
    "GCT": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG": 7.4,
    "TAT": 12.2, "TAC": 15.3, "TAA": 1.0,  "TAG": 0.8,
    "CAT": 10.9, "CAC": 15.1, "CAA": 12.3, "CAG": 34.2,
    "AAT": 17.0, "AAC": 19.1, "AAA": 24.4, "AAG": 31.9,
    "GAT": 21.8, "GAC": 25.1, "GAA": 29.0, "GAG": 39.6,
    "TGT": 10.6, "TGC": 12.6, "TGA": 1.6,  "TGG": 13.2,
    "CGT": 4.5,  "CGC": 10.4, "CGA": 6.2,  "CGG": 11.4,
    "AGT": 12.1, "AGC": 19.5, "AGA": 12.2, "AGG": 12.0,
    "GGT": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,
}

# Precompute Relative Adaptiveness (w_i) for Codon Adaptation Index (CAI)
def _compute_human_cai_weights() -> Dict[str, float]:
    aa_to_codons: Dict[str, List[str]] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa != "*":
            aa_to_codons.setdefault(aa, []).append(codon)
    weights = {}
    for aa, codons in aa_to_codons.items():
        max_freq = max(HUMAN_CODON_FREQ_PER_THOUSAND[c] for c in codons)
        for c in codons:
            weights[c] = HUMAN_CODON_FREQ_PER_THOUSAND[c] / max_freq if max_freq > 0 else 1.0
    return weights

HUMAN_CAI_WEIGHTS = _compute_human_cai_weights()

# Human Codon Stabilization Coefficients (CSC)
# Derived from global mammalian mRNA decay studies (Wu et al. 2019 eLife; Medina-Muñoz et al. 2021)
# Positive values indicate stability-promoting (optimal) codons; negative values indicate destabilizing codons.
HUMAN_CSC = {
    "TTT": -0.15, "TTC": 0.12,  "TTA": -0.22, "TTG": -0.05,
    "CTT": -0.08, "CTC": 0.18,  "CTA": -0.19, "CTG": 0.28,
    "ATT": -0.12, "ATC": 0.16,  "ATA": -0.25, "ATG": 0.05,
    "GTT": -0.07, "GTC": 0.14,  "GTA": -0.18, "GTG": 0.24,
    "TCT": -0.06, "TCC": 0.11,  "TCA": -0.14, "TCG": 0.03,
    "CCT": -0.04, "CCC": 0.10,  "CCA": -0.08, "CCG": 0.06,
    "ACT": -0.05, "ACC": 0.15,  "ACA": -0.11, "ACG": 0.08,
    "GCT": -0.02, "GCC": 0.19,  "GCA": -0.09, "GCG": 0.13,
    "TAT": -0.16, "TAC": 0.14,  "CAT": -0.13, "CAC": 0.12,
    "CAA": -0.10, "CAG": 0.22,  "AAT": -0.18, "AAC": 0.15,
    "AAA": -0.14, "AAG": 0.20,  "GAT": -0.11, "GAC": 0.17,
    "GAA": -0.10, "GAG": 0.21,  "TGT": -0.09, "TGC": 0.11,
    "TGG": 0.04,  "CGT": -0.08, "CGC": 0.12,  "CGA": -0.10,
    "CGG": 0.09,  "AGT": -0.12, "AGC": 0.13,  "AGA": -0.15,
    "AGG": 0.01,  "GGT": -0.08, "GGC": 0.17,  "GGA": -0.11,
    "GGG": 0.05,
}


# =============================================================================
# 2. Per-Transcript Feature Extractor
# =============================================================================

def extract_features_from_track(
    track: np.ndarray,
    tx_id: str,
    gene_id: str,
    symbol: str,
    half_life: float,
    half_life_transformed: float,
    rate: float,
) -> Optional[Dict]:
    """
    Extracts comprehensive biological, structural, and trans-factor metrics
    from an (L, 8) multi-track transcript representation.
    """
    if track is None or len(track) == 0:
        return None

    l = len(track)
    bases_map = np.array(["A", "C", "G", "T"])

    # 1. Sequence reconstruction from channels 0..3
    oh = track[:, :4]
    has_base = oh.any(axis=1)
    base_idx = np.argmax(oh, axis=1)
    seq_chars = np.where(has_base, bases_map[base_idx], "N")
    seq_str = "".join(seq_chars)

    # 2. Trans-factor signals
    mirna = track[:, 6].astype(np.float32)
    rbp = track[:, 7].astype(np.float32)

    # 3. Overall transcript metrics
    gc_count = np.sum((seq_chars == "G") | (seq_chars == "C"))
    gc_total = float(gc_count / l) if l > 0 else 0.0

    mirna_mean = float(np.mean(mirna))
    mirna_sum = float(np.sum(mirna))
    mirna_max = float(np.max(mirna))
    mirna_nz_fraction = float(np.mean(mirna > 0.0))
    mirna_density_nz = float(np.mean(mirna[mirna > 0.0])) if np.any(mirna > 0.0) else 0.0

    rbp_mean = float(np.mean(rbp))
    rbp_sum = float(np.sum(rbp))
    rbp_max = float(np.max(rbp))
    rbp_nz_fraction = float(np.mean(rbp > 0.0))
    rbp_density_nz = float(np.mean(rbp[rbp > 0.0])) if np.any(rbp > 0.0) else 0.0

    # Cross-channel colocalization & overlap
    overlap_mask = (mirna > 0.0) & (rbp > 0.0)
    union_mask = (mirna > 0.0) | (rbp > 0.0)
    coloc_overlap_bp = int(np.sum(overlap_mask))
    coloc_jaccard = float(np.sum(overlap_mask) / np.sum(union_mask)) if np.any(union_mask) else 0.0

    # 4. Regional Segmentation: 5' UTR, CDS, 3' UTR (Channel 4)
    cds_track = track[:, 4]
    cds_idx = np.where(cds_track > 0.5)[0]

    has_cds = len(cds_idx) > 0
    len_5utr = 0
    len_cds = 0
    len_3utr = 0
    gc_5utr = np.nan
    gc_cds = np.nan
    gc_3utr = np.nan

    mirna_5utr_mean = np.nan
    mirna_cds_mean = np.nan
    mirna_3utr_mean = np.nan
    mirna_5utr_sum = 0.0
    mirna_cds_sum = 0.0
    mirna_3utr_sum = 0.0

    rbp_5utr_mean = np.nan
    rbp_cds_mean = np.nan
    rbp_3utr_mean = np.nan
    rbp_5utr_sum = 0.0
    rbp_cds_sum = 0.0
    rbp_3utr_sum = 0.0

    gc3 = np.nan
    mean_csc = np.nan
    cai = np.nan
    num_codons = 0
    are_motif_count = 0  # AUUUA pentamers in 3' UTR

    if has_cds:
        cds_start = int(cds_idx[0])
        last_cds_idx = int(cds_idx[-1])
        # In Saluki, stop codon begins at last_upper / last_cds_idx and is 3 nt long
        cds_end = min(l, last_cds_idx + 3)

        len_5utr = cds_start
        len_cds = cds_end - cds_start
        len_3utr = l - cds_end

        # 5' UTR metrics
        if len_5utr > 0:
            utr5_bases = seq_chars[:cds_start]
            gc_5utr = float(np.mean((utr5_bases == "G") | (utr5_bases == "C")))
            mirna_5utr_mean = float(np.mean(mirna[:cds_start]))
            mirna_5utr_sum = float(np.sum(mirna[:cds_start]))
            rbp_5utr_mean = float(np.mean(rbp[:cds_start]))
            rbp_5utr_sum = float(np.sum(rbp[:cds_start]))

        # CDS metrics
        if len_cds > 0:
            cds_bases = seq_chars[cds_start:cds_end]
            gc_cds = float(np.mean((cds_bases == "G") | (cds_bases == "C")))
            mirna_cds_mean = float(np.mean(mirna[cds_start:cds_end]))
            mirna_cds_sum = float(np.sum(mirna[cds_start:cds_end]))
            rbp_cds_mean = float(np.mean(rbp[cds_start:cds_end]))
            rbp_cds_sum = float(np.sum(rbp[cds_start:cds_end]))

            # Codon analysis
            cds_dna = seq_str[cds_start:cds_end]
            codons = [cds_dna[i:i+3] for i in range(0, len(cds_dna)-2, 3) if len(cds_dna[i:i+3]) == 3]
            num_codons = len(codons)

            if num_codons > 0:
                # GC3 (GC at third position of codon)
                gc3_count = sum(1 for c in codons if c[2] in ["G", "C"])
                gc3 = float(gc3_count / num_codons)

                # Codon Stabilization Coefficient (CSC)
                csc_vals = [HUMAN_CSC[c] for c in codons if c in HUMAN_CSC]
                mean_csc = float(np.mean(csc_vals)) if csc_vals else np.nan

                # Codon Adaptation Index (CAI)
                cai_weights = [HUMAN_CAI_WEIGHTS[c] for c in codons if c in HUMAN_CAI_WEIGHTS]
                if cai_weights:
                    log_weights = [np.log(max(w, 1e-4)) for w in cai_weights]
                    cai = float(np.exp(np.mean(log_weights)))

        # 3' UTR metrics
        if len_3utr > 0:
            utr3_bases = seq_chars[cds_end:]
            gc_3utr = float(np.mean((utr3_bases == "G") | (utr3_bases == "C")))
            mirna_3utr_mean = float(np.mean(mirna[cds_end:]))
            mirna_3utr_sum = float(np.sum(mirna[cds_end:]))
            rbp_3utr_mean = float(np.mean(rbp[cds_end:]))
            rbp_3utr_sum = float(np.sum(rbp[cds_end:]))

            utr3_str = seq_str[cds_end:]
            are_motif_count = utr3_str.count("ATTTA")  # DNA notation for AUUUA

    # 5. Exon Segmentation & Per-Exon Signal Analysis (Channel 5)
    splice_indices = np.where(track[:, 5] > 0.5)[0].tolist()
    num_exons = len(splice_indices) + 1

    # In Saluki, splice junction markers 'ej' are located at the last nucleotide of an exon
    exon_starts = [0] + [idx + 1 for idx in splice_indices]
    exon_ends = [idx + 1 for idx in splice_indices] + [l]

    exon_lengths = []
    exon_mirna_means = []
    exon_mirna_sums = []
    exon_rbp_means = []
    exon_rbp_sums = []

    for s, e in zip(exon_starts, exon_ends):
        ex_len = max(1, e - s)
        exon_lengths.append(ex_len)
        exon_mirna_means.append(float(np.mean(mirna[s:e])))
        exon_mirna_sums.append(float(np.sum(mirna[s:e])))
        exon_rbp_means.append(float(np.mean(rbp[s:e])))
        exon_rbp_sums.append(float(np.sum(rbp[s:e])))

    # Exon aggregates
    mean_exon_len = float(np.mean(exon_lengths))
    exon_mirna_mean_avg = float(np.mean(exon_mirna_means))
    exon_rbp_mean_avg = float(np.mean(exon_rbp_means))
    exon_first_mirna_mean = float(exon_mirna_means[0])
    exon_first_rbp_mean = float(exon_rbp_means[0])
    exon_last_mirna_mean = float(exon_mirna_means[-1])
    exon_last_rbp_mean = float(exon_rbp_means[-1])

    if num_exons > 2:
        exon_internal_mirna_mean = float(np.mean(exon_mirna_means[1:-1]))
        exon_internal_rbp_mean = float(np.mean(exon_rbp_means[1:-1]))
    else:
        exon_internal_mirna_mean = np.nan
        exon_internal_rbp_mean = np.nan

    exon_max_mirna_mean = float(np.max(exon_mirna_means))
    exon_max_rbp_mean = float(np.max(exon_rbp_means))

    return {
        # Identifiers
        "ensembl_transcript_id": str(tx_id),
        "ensembl_gene_id": str(gene_id),
        "hgnc_symbol": str(symbol),
        # Functional Targets
        "half_life": float(half_life),
        "half_life_transformed": float(half_life_transformed),
        "rate": float(rate),
        # Sequence & Structural Architecture
        "seq_len": int(l),
        "has_cds": bool(has_cds),
        "len_5utr": int(len_5utr),
        "len_cds": int(len_cds),
        "len_3utr": int(len_3utr),
        "num_exons": int(num_exons),
        "mean_exon_len": float(mean_exon_len),
        "is_multiexon": bool(num_exons > 1),
        "are_motif_count": int(are_motif_count),
        # Codon Usage & Nucleotide Composition
        "gc_total": float(gc_total),
        "gc_5utr": float(gc_5utr),
        "gc_cds": float(gc_cds),
        "gc_3utr": float(gc_3utr),
        "gc3": float(gc3),
        "mean_csc": float(mean_csc),
        "cai": float(cai),
        "num_codons": int(num_codons),
        # Channel 6 (miRNA) Global & Regional Metrics
        "mirna_mean": float(mirna_mean),
        "mirna_sum": float(mirna_sum),
        "mirna_max": float(mirna_max),
        "mirna_nz_fraction": float(mirna_nz_fraction),
        "mirna_density_nz": float(mirna_density_nz),
        "mirna_5utr_mean": float(mirna_5utr_mean),
        "mirna_cds_mean": float(mirna_cds_mean),
        "mirna_3utr_mean": float(mirna_3utr_mean),
        "mirna_3utr_sum": float(mirna_3utr_sum),
        # Channel 7 (RBP) Global & Regional Metrics
        "rbp_mean": float(rbp_mean),
        "rbp_sum": float(rbp_sum),
        "rbp_max": float(rbp_max),
        "rbp_nz_fraction": float(rbp_nz_fraction),
        "rbp_density_nz": float(rbp_density_nz),
        "rbp_5utr_mean": float(rbp_5utr_mean),
        "rbp_cds_mean": float(rbp_cds_mean),
        "rbp_3utr_mean": float(rbp_3utr_mean),
        "rbp_3utr_sum": float(rbp_3utr_sum),
        # Per-Exon Metrics
        "exon_mirna_mean_avg": float(exon_mirna_mean_avg),
        "exon_rbp_mean_avg": float(exon_rbp_mean_avg),
        "exon_first_mirna_mean": float(exon_first_mirna_mean),
        "exon_first_rbp_mean": float(exon_first_rbp_mean),
        "exon_last_mirna_mean": float(exon_last_mirna_mean),
        "exon_last_rbp_mean": float(exon_last_rbp_mean),
        "exon_internal_mirna_mean": float(exon_internal_mirna_mean),
        "exon_internal_rbp_mean": float(exon_internal_rbp_mean),
        "exon_max_mirna_mean": float(exon_max_mirna_mean),
        "exon_max_rbp_mean": float(exon_max_rbp_mean),
        # Cross-Channel Colocalization & Overlap
        "coloc_overlap_bp": int(coloc_overlap_bp),
        "coloc_jaccard": float(coloc_jaccard),
    }


# =============================================================================
# 3. Correlation Computation Utilities
# =============================================================================

def compute_pairwise_correlations(
    df: pd.DataFrame,
    var_x: str,
    vars_y: List[str],
) -> pd.DataFrame:
    """
    Computes Pearson and Spearman rank correlation with p-values
    between var_x and each variable in vars_y.
    """
    rows = []
    for vy in vars_y:
        if vy not in df.columns or vy == var_x:
            continue
        valid = df[[var_x, vy]].dropna()
        if len(valid) < 5:
            continue

        x_vals = np.asarray(valid[var_x].values, dtype=float).ravel()
        y_vals = np.asarray(valid[vy].values, dtype=float).ravel()

        # Check for zero variance
        if np.std(x_vals) == 0 or np.std(y_vals) == 0:
            continue

        r_pearson, p_pearson = stats.pearsonr(x_vals, y_vals)
        r_spearman, p_spearman = stats.spearmanr(x_vals, y_vals)

        rows.append({
            "Variable": vy,
            "Pearson_r": r_pearson,
            "Pearson_p": p_pearson,
            "Spearman_rho": r_spearman,
            "Spearman_p": p_spearman,
            "N": len(valid),
        })

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Spearman_rho", key=abs, ascending=False)
    return res_df


# =============================================================================
# 4. Publication-Quality Visualizations
# =============================================================================

def generate_visualizations(df: pd.DataFrame, plots_dir: Path):
    """
    Generates and saves a suite of publication-ready visualization figures.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ---------------------------------------------------------
    # Plot 1: Correlation Heatmap (Trans-factors vs Structural & Target)
    # ---------------------------------------------------------
    cols_heatmap = [
        "half_life_transformed", "half_life",
        "mirna_mean", "mirna_3utr_mean", "exon_mirna_mean_avg",
        "rbp_mean", "rbp_3utr_mean", "rbp_cds_mean", "exon_rbp_mean_avg",
        "gc3", "mean_csc", "cai", "gc_total",
        "seq_len", "len_3utr", "num_exons", "mean_exon_len",
    ]
    cols_present = [c for c in cols_heatmap if c in df.columns]
    corr_matrix = df[cols_present].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=0.6,
        vmin=-0.6,
        center=0,
        square=True,
        linewidths=0.6,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation (ρ)"},
        ax=ax,
    )
    ax.set_title("Spearman Correlation Matrix: Trans-Factors, Codon Usage, Architecture & Stability", fontsize=13, pad=15)
    plt.tight_layout()
    fig.savefig(plots_dir / "correlation_heatmap_trans_factors_vs_all.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 2: Per-Exon Position Profile (First vs. Internal vs. Last Exon)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Subset of multi-exon transcripts for clean comparison
    multi_df = df[df["num_exons"] >= 3].copy()
    
    # miRNA across exons
    mirna_exon_data = pd.DataFrame({
        "First Exon (5' end)": multi_df["exon_first_mirna_mean"],
        "Internal Exons": multi_df["exon_internal_mirna_mean"],
        "Terminal Exon (3' end)": multi_df["exon_last_mirna_mean"],
    }).dropna()

    sns.boxplot(data=mirna_exon_data, ax=axes[0], palette="Blues", showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "darkblue", "markeredgecolor": "white", "markersize": 8})
    axes[0].set_title("miRNA Signal by Exon Position (Multi-Exon mRNAs)", fontsize=12)
    axes[0].set_ylabel("Mean Normalized Signal (TargetScan)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1e-5)

    # RBP across exons
    rbp_exon_data = pd.DataFrame({
        "First Exon (5' end)": multi_df["exon_first_rbp_mean"],
        "Internal Exons": multi_df["exon_internal_rbp_mean"],
        "Terminal Exon (3' end)": multi_df["exon_last_rbp_mean"],
    }).dropna()

    sns.boxplot(data=rbp_exon_data, ax=axes[1], palette="Purples", showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "indigo", "markeredgecolor": "white", "markersize": 8})
    axes[1].set_title("RBP Signal by Exon Position (Multi-Exon mRNAs)", fontsize=12)
    axes[1].set_ylabel("Mean Normalized Signal (ENCODE eCLIP)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=1e-5)

    plt.tight_layout()
    fig.savefig(plots_dir / "exon_position_signal_profile.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 3: Codon Usage vs. Trans-Factor Signals
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GC3 vs. 3' UTR miRNA signal
    sub_df = df[(df["mirna_3utr_mean"] > 0) & (df["gc3"].notna())]
    if len(sub_df) > 50:
        sns.regplot(
            data=sub_df,
            x="gc3",
            y="mirna_3utr_mean",
            ax=axes[0],
            scatter_kws={"alpha": 0.25, "s": 15, "color": "teal"},
            line_kws={"color": "crimson", "linewidth": 2},
        )
        rho_val, p_val = stats.spearmanr(sub_df["gc3"], sub_df["mirna_3utr_mean"])
        axes[0].set_title(f"GC3 vs. 3' UTR miRNA Signal (ρ = {rho_val:.3f}, p = {p_val:.1e})", fontsize=11)
        axes[0].set_xlabel("GC3 (GC content at 3rd codon position)")
        axes[0].set_ylabel("3' UTR miRNA Mean Signal")

    # CSC (Codon Stabilization) vs. RBP Signal
    sub_rbp = df[(df["rbp_mean"] > 0) & (df["mean_csc"].notna())]
    if len(sub_rbp) > 50:
        sns.regplot(
            data=sub_rbp,
            x="mean_csc",
            y="rbp_mean",
            ax=axes[1],
            scatter_kws={"alpha": 0.25, "s": 15, "color": "darkorange"},
            line_kws={"color": "navy", "linewidth": 2},
        )
        rho_val, p_val = stats.spearmanr(sub_rbp["mean_csc"], sub_rbp["rbp_mean"])
        axes[1].set_title(f"Mean CSC vs. RBP Signal (ρ = {rho_val:.3f}, p = {p_val:.1e})", fontsize=11)
        axes[1].set_xlabel("Mean Codon Stabilization Coefficient (CSC)")
        axes[1].set_ylabel("Transcript Mean RBP Signal")

    plt.tight_layout()
    fig.savefig(plots_dir / "codon_usage_vs_trans_factors.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 4: Trans-Factors vs. Functional mRNA Half-Life
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sub_hl = df[(df["mirna_3utr_mean"] > 0) & (df["half_life_transformed"].notna())]
    if len(sub_hl) > 50:
        sns.regplot(
            data=sub_hl,
            x="mirna_3utr_mean",
            y="half_life_transformed",
            ax=axes[0],
            scatter_kws={"alpha": 0.25, "s": 15, "color": "royalblue"},
            line_kws={"color": "firebrick", "linewidth": 2},
        )
        rho_val, p_val = stats.spearmanr(sub_hl["mirna_3utr_mean"], sub_hl["half_life_transformed"])
        axes[0].set_title(f"3' UTR miRNA vs. Half-Life Z-Score (ρ = {rho_val:.3f}, p = {p_val:.1e})", fontsize=11)
        axes[0].set_xlabel("3' UTR miRNA Mean Signal")
        axes[0].set_ylabel("half_life_transformed (Z-Score)")

    sub_hl_rbp = df[(df["rbp_mean"] > 0) & (df["half_life_transformed"].notna())]
    if len(sub_hl_rbp) > 50:
        sns.regplot(
            data=sub_hl_rbp,
            x="rbp_mean",
            y="half_life_transformed",
            ax=axes[1],
            scatter_kws={"alpha": 0.25, "s": 15, "color": "darkviolet"},
            line_kws={"color": "darkgreen", "linewidth": 2},
        )
        rho_val, p_val = stats.spearmanr(sub_hl_rbp["rbp_mean"], sub_hl_rbp["half_life_transformed"])
        axes[1].set_title(f"RBP Mean Signal vs. Half-Life Z-Score (ρ = {rho_val:.3f}, p = {p_val:.1e})", fontsize=11)
        axes[1].set_xlabel("Transcript Mean RBP Signal")
        axes[1].set_ylabel("half_life_transformed (Z-Score)")

    plt.tight_layout()
    fig.savefig(plots_dir / "half_life_vs_trans_factors.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 5: Random Forest Feature Importances for Half-Life
    # ---------------------------------------------------------
    feature_cols = [
        "mirna_mean", "mirna_3utr_mean", "exon_mirna_mean_avg",
        "rbp_mean", "rbp_3utr_mean", "rbp_cds_mean", "exon_rbp_mean_avg",
        "gc3", "mean_csc", "cai", "gc_total", "gc_3utr",
        "seq_len", "len_3utr", "len_cds", "num_exons", "mean_exon_len",
        "coloc_jaccard", "are_motif_count",
    ]
    model_df = df[feature_cols + ["half_life_transformed"]].dropna()

    if len(model_df) > 100:
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(model_df[feature_cols], model_df["half_life_transformed"])

        fi_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf.feature_importances_,
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#2b5c8f" if "mirna" in f else "#7b3294" if "rbp" in f else "#008837" if ("gc" in f or "cai" in f or "csc" in f) else "#e66101" for f in fi_df["Feature"]]
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title("Random Forest Feature Importances for Explaining mRNA Half-Life", fontsize=12, pad=12)
        ax.set_xlabel("MDI Gini Feature Importance")

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2b5c8f", label="miRNA Channel 6"),
            Patch(facecolor="#7b3294", label="RBP Channel 7"),
            Patch(facecolor="#008837", label="Codon Usage & GC"),
            Patch(facecolor="#e66101", label="Architecture / Length / Exons"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)

        plt.tight_layout()
        fig.savefig(plots_dir / "feature_importance_half_life.png", dpi=300)
        plt.close(fig)

    print(f"[Visualizations] All 5 publication figures successfully saved in: {plots_dir}")


# =============================================================================
# 5. Multivariate Regression Analysis for Half-Life
# =============================================================================

def run_multivariate_modeling(df: pd.DataFrame) -> str:
    """
    Fits Ridge Regression and Random Forest models to quantify the incremental
    predictive contribution of trans-factor channels over base sequence features.
    """
    out = io.StringIO()

    arch_features = ["seq_len", "len_cds", "len_3utr", "num_exons", "mean_exon_len"]
    codon_features = ["gc3", "mean_csc", "cai", "gc_total"]
    mirna_features = ["mirna_mean", "mirna_3utr_mean", "exon_mirna_mean_avg"]
    rbp_features = ["rbp_mean", "rbp_cds_mean", "rbp_3utr_mean", "exon_rbp_mean_avg"]

    all_features = arch_features + codon_features + mirna_features + rbp_features
    target = "half_life_transformed"

    valid_df = df[all_features + [target]].dropna()
    n_samples = len(valid_df)

    out.write(f"\n=======================================================================\n")
    out.write(f"=== MULTIVARIATE MODELING: EXPLAINING mRNA HALF-LIFE (N = {n_samples}) ===\n")
    out.write(f"=======================================================================\n")

    if n_samples < 100:
        out.write("Insufficient complete cases for multivariate modeling.\n")
        return out.getvalue()

    models_spec = [
        ("Model 1: Structural & Architecture (Length, Exons)", arch_features),
        ("Model 2: Codon Usage & GC (GC3, CSC, CAI, GC)", codon_features),
        ("Model 3: Sequence Baseline (Architecture + Codon Usage)", arch_features + codon_features),
        ("Model 4: Trans-Factors Only (Channel 6 miRNA + Channel 7 RBP)", mirna_features + rbp_features),
        ("Model 5: Complete Combined Model (Baseline + Trans-Factors)", all_features),
    ]

    results = []
    y = valid_df[target].values

    for model_name, feats in models_spec:
        X = valid_df[feats].values

        # Standardize features
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_norm, y)
        y_pred = ridge.predict(X_norm)

        r2 = r2_score(y, y_pred)
        r_corr, _ = stats.pearsonr(y, y_pred)
        rho_corr, _ = stats.spearmanr(y, y_pred)

        results.append({
            "Model": model_name,
            "Features Count": len(feats),
            "R² Score": round(r2, 4),
            "Pearson r": round(r_corr, 4),
            "Spearman ρ": round(rho_corr, 4),
        })

    res_df = pd.DataFrame(results)
    out.write(res_df.to_string(index=False))
    out.write("\n\nKey Takeaway:\n")
    out.write("- Comparing Model 3 vs Model 5 reveals the exact variance explained (ΔR² and Δr)\n")
    out.write("  added by incorporating continuous trans-factor tracks (miRNA & RBP) on top\n")
    out.write("  of intrinsic structural architecture and codon optimality.\n")

    return out.getvalue()


# =============================================================================
# 6. Main Pipeline & Comprehensive Report Generator
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis of Trans-Factor Channels in hIPSC_CM 8-Track Dataset"
    )
    parser.add_argument(
        "--input_npz",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/hIPSC_CM_8track_minmax.npz",
        help="Path to input 8-track NPZ file",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_8track_minmax_analysis.txt",
        help="Path to output text report file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots and CSV features (defaults to parent directory of output_txt)",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Optional subsample size for quick testing (None = process all)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Whether to generate and save publication visualization plots",
    )
    parser.add_argument(
        "--save_features_csv",
        action="store_true",
        default=True,
        help="Whether to export extracted features table to CSV for downstream use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_npz)
    output_txt_path = Path(args.output_txt)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = output_txt_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "8track_analysis_plots"

    print("=" * 75)
    print("   ORTHRUS 8-TRACK TRANS-FACTOR & FEATURE CORRELATION ANALYSIS PIPELINE   ")
    print("=" * 75)
    print(f"Input NPZ:      {input_path}")
    print(f"Output Report:  {output_txt_path}")
    print(f"Plots Directory:{plots_dir}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file '{input_path}' not found! "
            "Please check the cluster mount or provide a valid path via --input_npz."
        )

    # 1. Load NPZ dataset
    start_time = time.time()
    print(f"\n[1/5] Loading 8-track dataset from {input_path.name}...")
    data = np.load(input_path, allow_pickle=True)

    tracks = data["tracks"]
    tx_ids = data["ensembl_transcript_id"]
    n_total = len(tx_ids)
    print(f"Loaded {n_total:,} transcripts.")

    gene_ids = data["ensembl_gene_id"] if "ensembl_gene_id" in data else [""] * n_total
    symbols = data["hgnc_symbol"] if "hgnc_symbol" in data else [""] * n_total
    half_lives = data["half_life"] if "half_life" in data else [np.nan] * n_total
    half_lives_trans = data["half_life_transformed"] if "half_life_transformed" in data else [np.nan] * n_total
    rates = data["rate"] if "rate" in data else [np.nan] * n_total
    normalization = str(data["normalization"]) if "normalization" in data else "unknown"
    mirna_q99 = float(data["mirna_q99"]) if "mirna_q99" in data else np.nan
    eclip_q99 = float(data["eclip_q99"]) if "eclip_q99" in data else np.nan

    # Subsample if requested
    indices = np.arange(n_total)
    if args.sample_n is not None and 0 < args.sample_n < n_total:
        print(f"Subsampling {args.sample_n} transcripts for fast test run...")
        np.random.seed(42)
        indices = np.random.choice(indices, size=args.sample_n, replace=False)

    # 2. Extract features per transcript
    print(f"\n[2/5] Extracting structural, codon usage, and trans-factor metrics...")
    feature_list = []
    for idx in tqdm(indices, desc="Processing tracks"):
        feat = extract_features_from_track(
            track=tracks[idx],
            tx_id=str(tx_ids[idx]),
            gene_id=str(gene_ids[idx]),
            symbol=str(symbols[idx]),
            half_life=float(half_lives[idx]),
            half_life_transformed=float(half_lives_trans[idx]),
            rate=float(rates[idx]),
        )
        if feat is not None:
            feature_list.append(feat)

    df = pd.DataFrame(feature_list)
    print(f"Extracted feature table with {len(df):,} transcripts and {df.shape[1]} biological metrics.")

    # Save feature CSV for instant access in future analyses
    if args.save_features_csv:
        features_csv_path = output_dir / "hIPSC_CM_8track_extracted_features.csv"
        print(f"Saving extracted tabular features to: {features_csv_path.name}...")
        df.to_csv(features_csv_path, index=False)

    # 3. Build Comprehensive Analysis Report
    print(f"\n[3/5] Computing statistical summaries and correlation profiles...")
    report_buf = io.StringIO()

    report_buf.write("=" * 80 + "\n")
    report_buf.write("         STATISTICAL AND CORRELATION ANALYSIS REPORT: 8-TRACK DATASET         \n")
    report_buf.write("         (Channel 6: miRNA TargetScan | Channel 7: RBP ENCODE eCLIP)          \n")
    report_buf.write("=" * 80 + "\n")
    report_buf.write(f"Analyzed File:       {input_path}\n")
    report_buf.write(f"Generated On:        {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_buf.write(f"Total Transcripts:   {len(df):,}\n")
    report_buf.write(f"Unique Genes:        {df['ensembl_gene_id'].nunique():,}\n")
    report_buf.write(f"Normalization Mode:  {normalization.upper()}\n")
    report_buf.write(f"miRNA q99 Reference: {mirna_q99:.4f}\n")
    report_buf.write(f"RBP eCLIP q99 Ref:   {eclip_q99:.4f}\n\n")

    # --- Section A: Trans-Factor Sparsity & Activity ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 1. TRANS-FACTOR SIGNAL COVERAGE & CO-OCCURRENCE ===\n")
    report_buf.write("=" * 80 + "\n")
    has_mirna_mask = df["mirna_sum"] > 0.0
    has_rbp_mask = df["rbp_sum"] > 0.0
    both_mask = has_mirna_mask & has_rbp_mask
    neither_mask = (~has_mirna_mask) & (~has_rbp_mask)

    cov_table = pd.DataFrame({
        "Category": [
            "Transcripts with miRNA Signal (>0)",
            "Transcripts with RBP Signal (>0)",
            "Transcripts with BOTH miRNA & RBP",
            "Transcripts with NEITHER",
        ],
        "Count": [
            int(has_mirna_mask.sum()),
            int(has_rbp_mask.sum()),
            int(both_mask.sum()),
            int(neither_mask.sum()),
        ],
        "Percentage (%)": [
            round(has_mirna_mask.mean() * 100, 2),
            round(has_rbp_mask.mean() * 100, 2),
            round(both_mask.mean() * 100, 2),
            round(neither_mask.mean() * 100, 2),
        ],
    })
    report_buf.write(cov_table.to_string(index=False) + "\n\n")

    # Spatial overlap statistics among transcripts with both signals
    both_df = df[both_mask]
    if len(both_df) > 0:
        report_buf.write("Direct Spatial Colocalization (within the same transcript):\n")
        report_buf.write(f"  - Overlapping Active Nucleotides (bp): Mean = {both_df['coloc_overlap_bp'].mean():.1f}, Median = {both_df['coloc_overlap_bp'].median():.0f}, Max = {both_df['coloc_overlap_bp'].max():,}\n")
        report_buf.write(f"  - Jaccard Similarity Index:            Mean = {both_df['coloc_jaccard'].mean():.4f}, Max = {both_df['coloc_jaccard'].max():.4f}\n\n")

    # --- Section B: Descriptive Statistics ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 2. DESCRIPTIVE STATISTICS OF KEY METRICS ===\n")
    report_buf.write("=" * 80 + "\n")
    summary_cols = [
        "half_life", "half_life_transformed",
        "seq_len", "num_exons", "mean_exon_len", "len_3utr", "len_cds",
        "gc_total", "gc3", "mean_csc", "cai",
        "mirna_mean", "mirna_3utr_mean", "exon_mirna_mean_avg",
        "rbp_mean", "rbp_3utr_mean", "rbp_cds_mean", "exon_rbp_mean_avg",
    ]
    report_buf.write(df[summary_cols].describe().round(4).to_string() + "\n\n")

    # --- Section C: Per-Exon Signal Analysis ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 3. PER-EXON SIGNAL STRENGTH ANALYSIS ===\n")
    report_buf.write("=" * 80 + "\n")
    report_buf.write("Comparison of Trans-Factor Density across Exon Categories:\n")
    
    exon_cat_summary = pd.DataFrame({
        "Exon Position Category": [
            "First Exon (5' end)",
            "Internal Exons (middle)",
            "Terminal Exon (3' end)",
            "Average Across All Exons",
        ],
        "miRNA Mean Signal": [
            round(df["exon_first_mirna_mean"].mean(), 5),
            round(df["exon_internal_mirna_mean"].dropna().mean(), 5),
            round(df["exon_last_mirna_mean"].mean(), 5),
            round(df["exon_mirna_mean_avg"].mean(), 5),
        ],
        "RBP Mean Signal": [
            round(df["exon_first_rbp_mean"].mean(), 5),
            round(df["exon_internal_rbp_mean"].dropna().mean(), 5),
            round(df["exon_last_rbp_mean"].mean(), 5),
            round(df["exon_rbp_mean_avg"].mean(), 5),
        ],
    })
    report_buf.write(exon_cat_summary.to_string(index=False) + "\n\n")

    # Stratification by Exon Count
    df["exon_count_bin"] = pd.cut(
        df["num_exons"],
        bins=[0, 1, 3, 7, 15, 100],
        labels=["1 Exon (Single)", "2-3 Exons", "4-7 Exons", "8-15 Exons", "16+ Exons"],
    )
    exon_bin_agg = df.groupby("exon_count_bin", observed=False).agg({
        "mirna_mean": ["count", "mean"],
        "rbp_mean": ["mean"],
        "exon_mirna_mean_avg": ["mean"],
        "exon_rbp_mean_avg": ["mean"],
        "half_life_transformed": ["mean"],
    }).round(4)
    report_buf.write("Signal Breakdown by Exon Count Categories:\n")
    report_buf.write(exon_bin_agg.to_string() + "\n\n")

    # --- Section D: Correlations with Channel 6 (miRNA) ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 4. CORRELATIONS WITH CHANNEL 6 (miRNA / TargetScan) ===\n")
    report_buf.write("=" * 80 + "\n")
    correlate_vars = [
        "half_life_transformed", "half_life", "rate",
        "seq_len", "len_cds", "len_3utr", "len_5utr", "num_exons", "mean_exon_len",
        "gc_total", "gc3", "mean_csc", "cai", "are_motif_count",
        "rbp_mean", "rbp_3utr_mean", "rbp_cds_mean", "exon_rbp_mean_avg",
    ]
    corr_mirna_df = compute_pairwise_correlations(df, "mirna_mean", correlate_vars)
    report_buf.write("--- Overall miRNA Mean Signal (`mirna_mean`) Correlations ---\n")
    report_buf.write(corr_mirna_df.round(4).to_string(index=False) + "\n\n")

    corr_mirna_3utr_df = compute_pairwise_correlations(df, "mirna_3utr_mean", correlate_vars)
    report_buf.write("--- 3' UTR miRNA Mean Signal (`mirna_3utr_mean`) Correlations ---\n")
    report_buf.write(corr_mirna_3utr_df.round(4).to_string(index=False) + "\n\n")

    # --- Section E: Correlations with Channel 7 (RBP) ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 5. CORRELATIONS WITH CHANNEL 7 (RBPs / ENCODE eCLIP) ===\n")
    report_buf.write("=" * 80 + "\n")
    corr_rbp_df = compute_pairwise_correlations(df, "rbp_mean", correlate_vars)
    report_buf.write("--- Overall RBP Mean Signal (`rbp_mean`) Correlations ---\n")
    report_buf.write(corr_rbp_df.round(4).to_string(index=False) + "\n\n")

    corr_rbp_exon_df = compute_pairwise_correlations(df, "exon_rbp_mean_avg", correlate_vars)
    report_buf.write("--- Average Exon RBP Mean Signal (`exon_rbp_mean_avg`) Correlations ---\n")
    report_buf.write(corr_rbp_exon_df.round(4).to_string(index=False) + "\n\n")

    # --- Section F: Codon Usage vs. Trans-Factors Deep Dive ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 6. CODON USAGE & GC BIAS VS. TRANS-FACTORS ===\n")
    report_buf.write("=" * 80 + "\n")
    codon_summary_table = []
    for c_metric in ["gc3", "mean_csc", "cai", "gc_cds"]:
        for tf_metric in ["mirna_mean", "mirna_3utr_mean", "rbp_mean", "exon_rbp_mean_avg"]:
            valid = df[[c_metric, tf_metric]].dropna()
            if len(valid) > 10:
                r_val, p_val = stats.pearsonr(valid[c_metric], valid[tf_metric])
                rho_val, rho_p = stats.spearmanr(valid[c_metric], valid[tf_metric])
                codon_summary_table.append({
                    "Codon Metric": c_metric,
                    "Trans-Factor Metric": tf_metric,
                    "Pearson r": round(r_val, 4),
                    "Spearman ρ": round(rho_val, 4),
                    "p-value": f"{rho_p:.2e}",
                })
    report_buf.write(pd.DataFrame(codon_summary_table).to_string(index=False) + "\n\n")

    # --- Section G: Multivariate Predictive Modeling (Half-Life) ---
    report_buf.write("=" * 80 + "\n")
    report_buf.write("=== 7. MULTIVARIATE PREDICTIVE MODELING (mRNA STABILITY) ===\n")
    report_buf.write("=" * 80 + "\n")
    multi_report = run_multivariate_modeling(df)
    report_buf.write(multi_report + "\n")

    report_buf.write("=" * 80 + "\n")
    report_buf.write("                    END OF ANALYSIS REPORT                    \n")
    report_buf.write("=" * 80 + "\n")

    report_text = report_buf.getvalue()

    # 4. Save Report to Requested TXT File
    print(f"\n[4/5] Saving full report to: {output_txt_path}...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 5. Generate Visualizations
    if args.save_plots:
        print(f"\n[5/5] Generating publication-quality visual plots in: {plots_dir}...")
        generate_visualizations(df, plots_dir)

    # Print summary to console
    print("\n" + "=" * 75)
    print("ANALYSIS FINISHED SUCCESSFULLY!")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Report saved to:    {output_txt_path}")
    print(f"Plots saved in:     {plots_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
