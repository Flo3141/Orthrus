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
import sys
import time


# Candidate definitions with functional classification and all known aliases
CANDIDATE_DEFINITIONS = [
    # Stabilizers
    {"canonical": "ELAVL1",  "role": "Stabilizer",   "aliases": ["ELAVL1", "HUR", "HUA"]},
    {"canonical": "IGF2BP1", "role": "Stabilizer",   "aliases": ["IGF2BP1", "IMP1", "ZBP1"]},
    {"canonical": "IGF2BP2", "role": "Stabilizer",   "aliases": ["IGF2BP2", "IMP2"]},
    {"canonical": "IGF2BP3", "role": "Stabilizer",   "aliases": ["IGF2BP3", "IMP3"]},
    {"canonical": "QKI",     "role": "Stabilizer",   "aliases": ["QKI", "QK"]},
    {"canonical": "TARDBP",  "role": "Stabilizer",   "aliases": ["TARDBP", "TDP43", "TDP-43"]},
    # Destabilizers
    {"canonical": "KHSRP",   "role": "Destabilizer", "aliases": ["KHSRP", "KSRP"]},
    {"canonical": "HNRNPD",  "role": "Destabilizer", "aliases": ["HNRNPD", "AUF1"]},
    {"canonical": "ZFP36L1", "role": "Destabilizer", "aliases": ["ZFP36L1", "TIS11B", "BRF1"]},
    {"canonical": "ZFP36L2", "role": "Destabilizer", "aliases": ["ZFP36L2", "TIS11D", "BRF2"]},
    {"canonical": "ZFP36",   "role": "Destabilizer", "aliases": ["ZFP36", "TTP"]},
    {"canonical": "PUM1",    "role": "Destabilizer", "aliases": ["PUM1", "PUMH1"]},
    {"canonical": "PUM2",    "role": "Destabilizer", "aliases": ["PUM2", "PUMH2"]},
    {"canonical": "YTHDF2",  "role": "Destabilizer", "aliases": ["YTHDF2"]},
    {"canonical": "UPF1",    "role": "Destabilizer", "aliases": ["UPF1", "RENT1"]},
    # Regulatory / Cardiac
    {"canonical": "RBFOX2",  "role": "Regulatory",   "aliases": ["RBFOX2", "RBM9"]},
]

# Build fast O(1) lookup dictionary: maps uppercase token -> (canonical_symbol, role)
LOOKUP_DICT = {}
for item in CANDIDATE_DEFINITIONS:
    canon = item["canonical"]
    role = item["role"]
    for alias in item["aliases"]:
        LOOKUP_DICT[alias.upper()] = (canon, role)


def identify_rbp(peak_name: str):
    """
    Identifies candidate RBP from peak name (e.g. 'PUM1_K562_rep01' -> ('PUM1', 'Destabilizer')).
    Splits by underscore, hyphen, and period for maximum robustness.
    """
    if not peak_name or peak_name == ".":
        return None, None

    # ENCODE standard format: {RBP}_{CellLine}_{Replicate}
    # 1. Primary check: First token before underscore
    first_token = peak_name.split("_")[0].strip().upper()
    if first_token in LOOKUP_DICT:
        return LOOKUP_DICT[first_token]

    # 2. Secondary check: Any token in name
    # Useful if named e.g. eCLIP_PUM1_rep01 or similar
    tokens = peak_name.replace("-", "_").replace(".", "_").upper().split("_")
    for tok in tokens:
        if tok in LOOKUP_DICT:
            return LOOKUP_DICT[tok]

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

    print(f"Input BED file:   {input_path}")
    print(f"Output directory: {output_dir}")
    print("Beginning stream filtering line-by-line (O(1) dictionary token matching)...")

    start_time = time.time()
    total_lines = 0
    matched_peaks = 0
    counts_by_rbp = defaultdict(int)
    scores_by_rbp = defaultdict(list)
    all_unique_prefixes = set()

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
            
            # Record unique prefix for diagnostics (first 100k lines)
            if total_lines <= 100000:
                p_tok = peak_name.split("_")[0].strip()
                if p_tok and p_tok != ".":
                    all_unique_prefixes.add(p_tok)

            rbp_symbol, role = identify_rbp(peak_name)

            if rbp_symbol is not None:
                matched_peaks += 1
                counts_by_rbp[rbp_symbol] += 1

                # Parse signalValue if available (column 6 in narrowPeak)
                if len(parts) >= 7:
                    try:
                        scores_by_rbp[rbp_symbol].append(float(parts[6]))
                    except ValueError:
                        pass

                # Write to all-curated file
                f_all.write(line)

                # Route to role-specific BED files
                if role == "Stabilizer":
                    f_stab.write(line)
                elif role == "Destabilizer":
                    f_destab.write(line)
                elif role == "Regulatory":
                    pass

            if total_lines % 5000000 == 0:
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

    for item in CANDIDATE_DEFINITIONS:
        sym = item["canonical"]
        role = item["role"]
        cnt = counts_by_rbp[sym]
        scores = scores_by_rbp[sym]
        mean_s = f"{sum(scores)/len(scores):.2f}" if scores else "-"
        max_s = f"{max(scores):.2f}" if scores else "-"
        print(f"{sym:<12} {role:<15} {cnt:<15,} {mean_s:<12} {max_s:<12}")

    print("-" * 70)
    print(f"\nGenerated Filtered BED Files:")
    print(f"1. All Candidates:         {all_out_path}")
    print(f"2. Stabilizers Only:       {stab_out_path}")
    print(f"3. Destabilizers Only:     {destab_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
