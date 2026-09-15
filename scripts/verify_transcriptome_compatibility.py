#!/usr/bin/env python3
"""
Validation script to verify transcriptome and coordinate compatibility
between hIPSC_CM, GTF (Ensembl 108), and TargetScan 8.0.
"""

import gffutils
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths to files
SALUKI_DATA = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt")
GTF_DB = Path("/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db")
TARGETSCAN_FILE = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt")

print("1. Loading GTF DB...")
db = gffutils.FeatureDB(str(GTF_DB))

print("2. Loading hIPSC_CM dataset...")
df = pd.read_csv(SALUKI_DATA, sep="\t")

print("3. Loading TargetScan data (complete & indexed for fast lookup)...")
ts_df = pd.read_csv(
    TARGETSCAN_FILE,
    sep="\t",
    usecols=["Transcript ID", "UTR_start", "UTR end"],
    dtype={"Transcript ID": str},
    low_memory=False,
).dropna(subset=["Transcript ID", "UTR_start", "UTR end"])

ts_df["clean_tx"] = ts_df["Transcript ID"].str.split(".").str[0]
ts_df["UTR_start"] = ts_df["UTR_start"].astype(int)
ts_df["UTR end"] = ts_df["UTR end"].astype(int)

# Dictionary: clean_tx -> list of (UTR_start, UTR end)
ts_data = {}
for row in tqdm(ts_df[["clean_tx", "UTR_start", "UTR end"]].itertuples(index=False), total=len(ts_df), desc="Indexing TargetScan"):
    ts_data.setdefault(row.clean_tx, []).append((row.UTR_start, getattr(row, "_2")))

print(f"TargetScan data indexed for {len(ts_data)} unique transcripts.")

# GTF metrics
total = len(df)
found_in_gtf = 0
exact_length_match = 0
length_diffs = []

# TargetScan 3' UTR metrics
ts_tx_with_predictions = 0
tx_without_cds = 0
total_sites = 0
fitting_sites = 0
overflowing_sites = 0
transcripts_with_overflow = 0
overflow_distances = []

print(f"\nVerifying {total} transcripts...")

for idx, row in tqdm(df.iterrows(), total=total, desc="Checking transcripts"):
    raw_tx = str(row["ensembl_transcript_id"])
    clean_tx = raw_tx.split(".")[0]
    tokens = [tok.strip() for tok in str(row["sequence"]).split(",") if tok.strip()]
    seq_len = len(tokens)
    
    # Determine CDS / 3' UTR start in dataset
    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    utr3_start = (max(upper_indices) + 3) if upper_indices else 0
    utr3_len = seq_len - utr3_start

    # A. GTF exon length comparison
    tx = None
    for cand in [raw_tx, clean_tx]:
        try:
            tx = db[cand]
            break
        except Exception:
            continue
            
    if tx is not None:
        found_in_gtf += 1
        exons = list(db.children(tx, featuretype="exon"))
        gtf_cdna_len = sum(exon.end - exon.start + 1 for exon in exons)
        diff = seq_len - gtf_cdna_len
        length_diffs.append(diff)
        if diff == 0:
            exact_length_match += 1

    # B. TargetScan comparison & 3' UTR validation
    if clean_tx in ts_data:
        ts_tx_with_predictions += 1
        sites = ts_data[clean_tx]
        tx_has_overflow = False

        if not upper_indices:
            tx_without_cds += 1
            continue

        for u_start, u_end in sites:
            total_sites += 1
            if u_end <= utr3_len:
                fitting_sites += 1
            else:
                overflowing_sites += 1
                tx_has_overflow = True
                overflow_distances.append(u_end - utr3_len)

        if tx_has_overflow:
            transcripts_with_overflow += 1


print("\n" + "=" * 65)
print("           TRANSCRIPTOME COMPATIBILITY CHECK RESULTS         ")
print("=" * 65)
print(f"Total transcripts:                   {total}")
print("-" * 65)
print("1. GTF Comparison (Ensembl 108 vs. hIPSC_CM sequences):")
print(f"  Found in GTF:                      {found_in_gtf} ({found_in_gtf/total*100:.2f}%)")
print(f"  Exact length match:                {exact_length_match} ({exact_length_match/found_in_gtf*100:.2f}% of found)")

if length_diffs:
    arr = np.array(length_diffs)
    print(f"  Length difference (seq_len - gtf_len):")
    print(f"    Mean: {np.mean(arr):.2f} bp | Median: {np.median(arr):.1f} bp")
    print(f"    |Diff| == 0 bp: {np.sum(arr == 0) / len(arr) * 100:.2f}%")

print("-" * 65)
print("2. TargetScan 8.0 3' UTR Comparison:")
print(f"  Transcripts with TargetScan data:  {ts_tx_with_predictions} ({ts_tx_with_predictions/total*100:.2f}%)")
if tx_without_cds > 0:
    print(f"  Of which without annotated CDS:    {tx_without_cds}")
print(f"  Total miRNA binding sites:         {total_sites}")
if total_sites > 0:
    fit_pct = fitting_sites / total_sites * 100
    ov_pct = overflowing_sites / total_sites * 100
    print(f"  Fit completely in 3' UTR:          {fitting_sites} ({fit_pct:.2f}%)")
    print(f"  Overflow beyond 3' UTR:            {overflowing_sites} ({ov_pct:.2f}%)")
    print(f"  Transcripts with >= 1 overflow:    {transcripts_with_overflow} ({transcripts_with_overflow/ts_tx_with_predictions*100:.2f}%)")

    if overflow_distances:
        ov_arr = np.array(overflow_distances)
        print(f"  Overflow distance for overflows:")
        print(f"    Mean: {np.mean(ov_arr):.1f} bp | Median: {np.median(ov_arr):.1f} bp | Max: {np.max(ov_arr)} bp")
print("=" * 65)
