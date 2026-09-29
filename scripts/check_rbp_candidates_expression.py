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
RBP_STABILIZERS = ["BCLAF1", "FUS", "HNRNPC", "HNRNPU", "IGF2BP1", "IGF2BP2", "IGF2BP3", "SRSF1", "TAF15", "YBX3", "QKI"]
RBP_DESTABILIZERS = ["DDX6", "EXOSC5", "FTO", "FXR2", "KHSRP", "NCBP2", "PABPN1", "PUM1", "PUM2", "SND1", "XRN2", "UPF1"]


# Known gene aliases to ensure matches even if alternate nomenclature is used
KNOWN_ALIASES = {
    "UPF1": ["RENT1", "NORF1", "HUPF1"],
    "QKI": ["QK", "QK1", "QK3"],
    "KHSRP": ["KSRP", "FUBP2"],
    "NCBP2": ["CBP20", "NIP1"],
    "IGF2BP1": ["IMP1", "CRD-BP", "ZBP1"],
    "IGF2BP2": ["IMP2", "VICKZ2"],
    "IGF2BP3": ["IMP3", "KOC1"],
    "PUM1": ["PUMH1"],
    "PUM2": ["PUMH2"],
    "DDX6": ["RCK", "HLR2"],
    "EXOSC5": ["RRP46", "RRP41L"],
    "FTO": ["ALKBH9"],
    "FXR2": ["FMR1L2"],
    "PABPN1": ["PAB2", "PABP2", "OPMD"],
    "SND1": ["TDRD11", "P100"],
    "XRN2": ["DHP1"],
    "BCLAF1": ["BTF"],
    "FUS": ["TLS", "FUS1", "HNRPP2"],
    "HNRNPC": ["HNRPC", "HNPC"],
    "HNRNPU": ["HNRPU", "HNPU", "SAF-A"],
    "SRSF1": ["ASF", "SF2", "SFRS1"],
    "TAF15": ["RBP56", "TAF2N", "TSR"],
    "YBX3": ["CSDA", "DBPA", "ZONAB"],
}


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
        help="Optional path to save text/tsv report",
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

    candidates = []
    for sym in RBP_STABILIZERS:
        candidates.append({"symbol": sym, "role": "Stabilizer"})
    for sym in RBP_DESTABILIZERS:
        candidates.append({"symbol": sym, "role": "Destabilizer"})

    results = []

    print("\n" + "=" * 105)
    print("                RBP CANDIDATE EXPRESSION CHECK IN hIPSC-CM DATASET")
    print("=" * 105)

    for cand in candidates:
        sym = cand["symbol"].upper()
        aliases = [a.upper() for a in KNOWN_ALIASES.get(sym, [])]

        # Search by symbol or known aliases
        mask = (
            (df["hgnc_symbol_upper"] == sym)
            | (df["hgnc_symbol_upper"].isin(aliases))
        )
        matches = df[mask]

        num_tx = len(matches)
        is_expressed = num_tx > 0

        if is_expressed:
            hwz_mean = matches["half_life"].mean() if "half_life" in matches.columns else np.nan
            hwz_median = matches["half_life"].median() if "half_life" in matches.columns else np.nan
            matched_sym = matches["hgnc_symbol_upper"].iloc[0]
            tx_ids = ", ".join(matches["ensembl_transcript_id"].astype(str).tolist()[:2])
            if num_tx > 2:
                tx_ids += f" (+{num_tx - 2} more)"
        else:
            hwz_mean = np.nan
            hwz_median = np.nan
            matched_sym = "-"
            tx_ids = "-"

        results.append({
            "Symbol": cand["symbol"],
            "Role": cand["role"],
            "Status": "EXPRESSED" if is_expressed else "NOT FOUND",
            "Matched_Gene": matched_sym,
            "Num_Transcripts": num_tx,
            "Mean_Half_Life_h": round(hwz_mean, 2) if not np.isnan(hwz_mean) else "-",
            "Median_Half_Life_h": round(hwz_median, 2) if not np.isnan(hwz_median) else "-",
            "Sample_Transcripts": tx_ids,
        })

    res_df = pd.DataFrame(results)

    # Format table for output
    summary_cols = ["Symbol", "Role", "Status", "Num_Transcripts", "Mean_Half_Life_h", "Sample_Transcripts"]
    print(res_df[summary_cols].to_string(index=False))

    expressed_count = (res_df["Status"] == "EXPRESSED").sum()
    total_count = len(res_df)

    stab_df = res_df[res_df["Role"] == "Stabilizer"]
    destab_df = res_df[res_df["Role"] == "Destabilizer"]

    stab_exp = (stab_df["Status"] == "EXPRESSED").sum()
    destab_exp = (destab_df["Status"] == "EXPRESSED").sum()

    print("\n" + "=" * 105)
    print(f"Summary: {expressed_count} of {total_count} candidate RBPs are CONFIRMED EXPRESSED in hIPSC-CMs.")
    print(f"  - Stabilizers:   {stab_exp} of {len(stab_df)} expressed")
    print(f"  - Destabilizers: {destab_exp} of {len(destab_df)} expressed")
    print("=" * 105)

    stab_list = stab_df[stab_df["Status"] == "EXPRESSED"]["Symbol"].tolist()
    destab_list = destab_df[destab_df["Status"] == "EXPRESSED"]["Symbol"].tolist()

    print(f"\n1. Confirmed Stabilizers ({len(stab_list)}):   {', '.join(stab_list)}")
    print(f"2. Confirmed Destabilizers ({len(destab_list)}): {', '.join(destab_list)}")

    missing = res_df[res_df["Status"] != "EXPRESSED"]["Symbol"].tolist()
    if missing:
        print(f"\n[Warning] Not found in hIPSC-CM ({len(missing)}): {', '.join(missing)}")
    else:
        print("\n[Success] All tested candidates are actively expressed in hIPSC-CM!")

    if args.output_txt:
        out_p = Path(args.output_txt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_p, sep="\t", index=False)
        print(f"\nDetailed report saved to: {out_p}")


if __name__ == "__main__":
    main()
