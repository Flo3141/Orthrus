#!/usr/bin/env python3
"""
Filter Merged ENCODE eCLIP Peaks for Curated mRNA Stability / Decay Candidates.

Reads the comprehensive ENCODE eCLIP BED file (e.g. all_rbp_peaks_merged.bed),
identifies and extracts peaks belonging to the 16 confirmed hIPSC-CM candidate RBPs,
and produces:
1. curated_16_rbp_peaks.bed (all curated candidates)
2. stabilizers_peaks.bed (ELAVL1, IGF2BP1-3, QKI, TARDBP)
3. destabilizers_peaks.bed (KHSRP, HNRNPD, ZFP36/L1/L2, PUM1/2, YTHDF2, UPF1)
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import re
import sys
import time


# Candidate definitions with classification and regex pattern for matching
CANDIDATE_MAP = {
    # Stabilizers
    "ELAVL1":  {"role": "Stabilizer",   "patterns": [r"^ELAVL1\b", r"^HUR\b", r"^HUA\b"]},
    "IGF2BP1": {"role": "Stabilizer",   "patterns": [r"^IGF2BP1\b", r"^IMP1\b", r"^ZBP1\b"]},
    "IGF2BP2": {"role": "Stabilizer",   "patterns": [r"^IGF2BP2\b", r"^IMP2\b"]},
    "IGF2BP3": {"role": "Stabilizer",   "patterns": [r"^IGF2BP3\b", r"^IMP3\b"]},
    "QKI":     {"role": "Stabilizer",   "patterns": [r"^QKI\b", r"^QK\b"]},
    "TARDBP":  {"role": "Stabilizer",   "patterns": [r"^TARDBP\b", r"^TDP43\b", r"^TDP-43\b"]},
    # Destabilizers
    "KHSRP":   {"role": "Destabilizer", "patterns": [r"^KHSRP\b", r"^KSRP\b"]},
    "HNRNPD":  {"role": "Destabilizer", "patterns": [r"^HNRNPD\b", r"^AUF1\b"]},
    "ZFP36L1": {"role": "Destabilizer", "patterns": [r"^ZFP36L1\b", r"^TIS11B\b", r"^BRF1\b"]},
    "ZFP36L2": {"role": "Destabilizer", "patterns": [r"^ZFP36L2\b", r"^TIS11D\b", r"^BRF2\b"]},
    "ZFP36":   {"role": "Destabilizer", "patterns": [r"^ZFP36\b", r"^TTP\b"]},
    "PUM1":    {"role": "Destabilizer", "patterns": [r"^PUM1\b", r"^PUMH1\b"]},
    "PUM2":    {"role": "Destabilizer", "patterns": [r"^PUM2\b", r"^PUMH2\b"]},
    "YTHDF2":  {"role": "Destabilizer", "patterns": [r"^YTHDF2\b"]},
    "UPF1":    {"role": "Destabilizer", "patterns": [r"^UPF1\b", r"^RENT1\b"]},
    # Regulatory / Cardiac
    "RBFOX2":  {"role": "Regulatory",   "patterns": [r"^RBFOX2\b", r"^RBM9\b"]},
}

# Precompile regexes for ultra-fast matching
COMPILED_MATCHERS = []
for sym, info in CANDIDATE_MAP.items():
    combined_pat = re.compile("|".join(info["patterns"]), re.IGNORECASE)
    COMPILED_MATCHERS.append((sym, info["role"], combined_pat))


def identify_rbp(peak_name: str):
    """Identifies which candidate RBP a peak belongs to based on its name."""
    if not peak_name or peak_name == ".":
        return None, None
    clean_name = peak_name.strip()
    for sym, role, pat in COMPILED_MATCHERS:
        if pat.search(clean_name):
            return sym, role
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter merged eCLIP BED file for 16 curated stability/decay RBPs."
    )
    parser.add_argument(
        "--input_bed",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/all_rbp_peaks_merged.bed",
        help="Path to source all_rbp_peaks_merged.bed file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/curated_candidates",
        help="Directory where filtered BED files will be stored",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_bed)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"[Error] Source BED file not found: {input_path}")
        print("Please check the path or verify that it exists on the cluster.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_out_path = output_dir / "curated_16_rbp_peaks.bed"
    stab_out_path = output_dir / "stabilizers_peaks.bed"
    destab_out_path = output_dir / "destabilizers_peaks.bed"

    print(f"Input BED file:  {input_path}")
    print(f"Output directory: {output_dir}")
    print("Beginning stream filtering line-by-line (low memory footprint)...")

    start_time = time.time()
    total_lines = 0
    matched_peaks = 0
    counts_by_rbp = defaultdict(int)
    scores_by_rbp = defaultdict(list)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(all_out_path, "w", encoding="utf-8") as f_all, \
         open(stab_out_path, "w", encoding="utf-8") as f_stab, \
         open(destab_out_path, "w", encoding="utf-8") as f_destab:

        for line in f_in:
            if line.startswith("#") or line.startswith("track") or not line.strip():
                continue

            total_lines += 1
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            peak_name = parts[3]
            rbp_symbol, role = identify_rbp(peak_name)

            if rbp_symbol is not None:
                matched_peaks += 1
                counts_by_rbp[rbp_symbol] += 1

                # Parse signalValue if available (column 6 in standard narrowPeak)
                if len(parts) >= 7:
                    try:
                        scores_by_rbp[rbp_symbol].append(float(parts[6]))
                    except ValueError:
                        pass

                # Write to the all-curated file
                f_all.write(line)

                # Route to role-specific BED files
                if role == "Stabilizer":
                    f_stab.write(line)
                elif role == "Destabilizer":
                    f_destab.write(line)
                elif role == "Regulatory":
                    # RBFOX2 is written to all_out, and can optionally be examined
                    pass

            if total_lines % 500000 == 0:
                print(f"  Processed {total_lines:,} lines | Matched {matched_peaks:,} candidate peaks...")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("                   eCLIP CANDIDATE FILTERING FINISHED")
    print("=" * 80)
    print(f"Total lines scanned:   {total_lines:,}")
    print(f"Total candidate peaks: {matched_peaks:,} ({matched_peaks/total_lines*100:.2f}% of all peaks)")
    print(f"Elapsed time:          {elapsed:.2f} seconds\n")

    print(f"{'Symbol':<12} {'Role':<15} {'Found Peaks':<15} {'Mean Signal':<12} {'Max Signal':<12}")
    print("-" * 70)

    for sym, info in CANDIDATE_MAP.items():
        role = info["role"]
        cnt = counts_by_rbp[sym]
        scores = scores_by_rbp[sym]
        mean_s = f"{sum(scores)/len(scores):.2f}" if scores else "-"
        max_s = f"{max(scores):.2f}" if scores else "-"
        print(f"{sym:<12} {role:<15} {cnt:<15,} {mean_s:<12} {max_s:<12}")

    print("-" * 70)
    print(f"\nGenerated Filtered BED Files:")
    print(f"1. All 16 Candidates:      {all_out_path}")
    print(f"2. Stabilizers Only:       {stab_out_path}")
    print(f"3. Destabilizers Only:     {destab_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
