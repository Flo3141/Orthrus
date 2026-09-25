#!/usr/bin/env python3
"""
Check Expression and Transcript Presence of Curated RBP Candidates in hIPSC-CM Dataset.

Identifies which candidate RBPs involved in mRNA stability / decay regulation
are actively transcribed and have measured half-life data in the hIPSC-CM experiment.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# Curated list of mRNA stability / decay regulators with functional annotation
RBP_CANDIDATES = [
    {
        "symbol": "ELAVL1",
        "alias": ["HUR", "HUA"],
        "ensembl_id": "ENSG00000066044",
        "category": "ARE Regulation",
        "role": "Stabilizer",
        "mechanism": "Binds AREs; protects from deadenylation & decay",
    },
    {
        "symbol": "KHSRP",
        "alias": ["KSRP", "FUBP2"],
        "ensembl_id": "ENSG00000088247",
        "category": "ARE Regulation",
        "role": "Destabilizer",
        "mechanism": "Recruits exosome and PARN deadenylase",
    },
    {
        "symbol": "HNRNPD",
        "alias": ["AUF1"],
        "ensembl_id": "ENSG00000139667",
        "category": "ARE Regulation",
        "role": "Destabilizer",
        "mechanism": "Promotes ARE-mediated mRNA decay",
    },
    {
        "symbol": "ZFP36L1",
        "alias": ["TIS11B", "BRF1"],
        "ensembl_id": "ENSG00000185650",
        "category": "ARE Regulation (TTP family)",
        "role": "Destabilizer",
        "mechanism": "Recruits CNOT deadenylase complex",
    },
    {
        "symbol": "ZFP36L2",
        "alias": ["TIS11D", "BRF2"],
        "ensembl_id": "ENSG00000152518",
        "category": "ARE Regulation (TTP family)",
        "role": "Destabilizer",
        "mechanism": "Accelerates mRNA decay",
    },
    {
        "symbol": "ZFP36",
        "alias": ["TTP", "GOS24"],
        "ensembl_id": "ENSG00000128016",
        "category": "ARE Regulation (TTP family)",
        "role": "Destabilizer",
        "mechanism": "Prototypical Tristetraprolin decay factor",
    },
    {
        "symbol": "PUM1",
        "alias": ["PUMH1"],
        "ensembl_id": "ENSG00000134640",
        "category": "Pumilio Repression",
        "role": "Destabilizer",
        "mechanism": "Binds PRE (UGUAHAUA); recruits CNOT deadenylases",
    },
    {
        "symbol": "PUM2",
        "alias": ["PUMH2"],
        "ensembl_id": "ENSG00000055917",
        "category": "Pumilio Repression",
        "role": "Destabilizer",
        "mechanism": "Binds PRE; represses translation & accelerates decay",
    },
    {
        "symbol": "YTHDF2",
        "alias": ["HGRG8"],
        "ensembl_id": "ENSG00000198492",
        "category": "m6A-Mediated Decay",
        "role": "Destabilizer",
        "mechanism": "m6A reader; directs transcripts to CCR4-NOT / decay",
    },
    {
        "symbol": "IGF2BP1",
        "alias": ["IMP1", "ZBP1", "CRD-BP"],
        "ensembl_id": "ENSG00000159489",
        "category": "m6A Reader / Shield",
        "role": "Stabilizer",
        "mechanism": "Protects m6A-modified mRNAs from degradation",
    },
    {
        "symbol": "IGF2BP2",
        "alias": ["IMP2", "VICKZ2"],
        "ensembl_id": "ENSG00000073792",
        "category": "m6A Reader / Shield",
        "role": "Stabilizer",
        "mechanism": "Post-transcriptional stabilizer of target mRNAs",
    },
    {
        "symbol": "IGF2BP3",
        "alias": ["IMP3", "KOC1"],
        "ensembl_id": "ENSG00000136231",
        "category": "m6A Reader / Shield",
        "role": "Stabilizer",
        "mechanism": "Promotes mRNA stability and translation",
    },
    {
        "symbol": "UPF1",
        "alias": ["RENT1", "NORF1"],
        "ensembl_id": "ENSG00000005001",
        "category": "NMD Surveillance",
        "role": "Destabilizer",
        "mechanism": "Key helicase initiating Nonsense-Mediated Decay",
    },
    {
        "symbol": "QKI",
        "alias": ["QK", "QK1", "QK3"],
        "ensembl_id": "ENSG00000112531",
        "category": "Cardiomyocyte / Muscle Regulation",
        "role": "Dual / Stabilizer",
        "mechanism": "Controls cardiac myofibrillogenesis and mRNA stability",
    },
    {
        "symbol": "TARDBP",
        "alias": ["TDP-43", "TDP43"],
        "ensembl_id": "ENSG00000120948",
        "category": "Cardiomyocyte / Muscle Regulation",
        "role": "Dual / Stabilizer",
        "mechanism": "Regulates mRNA stability & neuromuscular transcripts",
    },
    {
        "symbol": "RBFOX2",
        "alias": ["RBM9", "HRNBP2"],
        "ensembl_id": "ENSG00000100320",
        "category": "Cardiomyocyte Splicing & Stability",
        "role": "Regulatory",
        "mechanism": "Crucial for heart development; regulates 3' UTR events",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check expression of curated RBP candidates in the hIPSC-CM dataset."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt",
        help="Path to source hIPSC-CM tab-separated dataset file",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Optional path to save text report",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"[Error] Dataset file not found at: {input_path}")
        print("Please check the path or run this on the cluster.")
        sys.exit(1)

    print(f"Loading hIPSC-CM dataset from: {input_path} ...")
    df = pd.read_csv(input_path, sep="\t")

    # Clean columns
    df["hgnc_symbol_upper"] = df["hgnc_symbol"].fillna("").astype(str).str.strip().str.upper()
    if "ensembl_gene_id" in df.columns:
        df["ensembl_gene_id_clean"] = df["ensembl_gene_id"].fillna("").astype(str).str.strip()
    else:
        df["ensembl_gene_id_clean"] = ""

    results = []

    print("\n" + "=" * 95)
    print("                RBP CANDIDATE EXPRESSION CHECK IN hIPSC-CM DATASET")
    print("=" * 95)

    for cand in RBP_CANDIDATES:
        sym = cand["symbol"].upper()
        aliases = [a.upper() for a in cand["alias"]]
        ens_id = cand["ensembl_id"]

        # Search by symbol or aliases or ensembl gene id
        mask = (
            (df["hgnc_symbol_upper"] == sym)
            | (df["hgnc_symbol_upper"].isin(aliases))
            | (df["ensembl_gene_id_clean"] == ens_id)
        )
        matches = df[mask]

        num_tx = len(matches)
        is_expressed = num_tx > 0

        if is_expressed:
            hwz_mean = matches["half_life"].mean() if "half_life" in matches.columns else np.nan
            hwz_median = matches["half_life"].median() if "half_life" in matches.columns else np.nan
            tx_ids = ", ".join(matches["ensembl_transcript_id"].astype(str).tolist()[:3])
            if num_tx > 3:
                tx_ids += f" (+{num_tx - 3} more)"
        else:
            hwz_mean = np.nan
            hwz_median = np.nan
            tx_ids = "-"

        results.append({
            "Symbol": cand["symbol"],
            "Category": cand["category"],
            "Role": cand["role"],
            "Status": "EXPRESSED" if is_expressed else "NOT FOUND",
            "Num_Transcripts": num_tx,
            "Mean_Half_Life_h": round(hwz_mean, 2) if not np.isnan(hwz_mean) else "-",
            "Median_Half_Life_h": round(hwz_median, 2) if not np.isnan(hwz_median) else "-",
            "Sample_Transcripts": tx_ids,
            "Mechanism": cand["mechanism"],
        })

    res_df = pd.DataFrame(results)

    # Format table for output
    summary_cols = ["Symbol", "Role", "Status", "Num_Transcripts", "Mean_Half_Life_h", "Category"]
    print(res_df[summary_cols].to_string(index=False))

    expressed_count = (res_df["Status"] == "EXPRESSED").sum()
    total_count = len(res_df)

    print("\n" + "=" * 95)
    print(f"Summary: {expressed_count} of {total_count} candidate RBPs are CONFIRMED EXPRESSED in hIPSC-CMs.")
    print("=" * 95)

    print("\nRecommended Next Steps for ENCORI:")
    expressed_symbols = res_df[res_df["Status"] == "EXPRESSED"]["Symbol"].tolist()
    stabilizers = res_df[(res_df["Status"] == "EXPRESSED") & (res_df["Role"].str.contains("Stabilizer"))]["Symbol"].tolist()
    destabilizers = res_df[(res_df["Status"] == "EXPRESSED") & (res_df["Role"].str.contains("Destabilizer"))]["Symbol"].tolist()

    print(f"1. Confirmed Stabilizers ({len(stabilizers)}):   {', '.join(stabilizers)}")
    print(f"2. Confirmed Destabilizers ({len(destabilizers)}): {', '.join(destabilizers)}")

    if args.output_txt:
        out_p = Path(args.output_txt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_p, sep="\t", index=False)
        print(f"\nDetailed report saved to: {out_p}")


if __name__ == "__main__":
    main()
