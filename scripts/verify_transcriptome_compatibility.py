#!/usr/bin/env python3
"""
Validierungs-Skript zur Pruefung der Transkriptom- und Koordinaten-Kompatibilitaet
zwischen hIPSC_CM, GTF (Ensembl 108) und TargetScan 8.0.
"""

import gffutils
import pandas as pd
import numpy as np
from pathlib import Path

# Pfade zu Ihren Dateien
SALUKI_DATA = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt")
GTF_DB = Path("/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db")
TARGETSCAN_FILE = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt")

print("1. Lade GTF-DB...")
db = gffutils.FeatureDB(str(GTF_DB))

print("2. Lade hIPSC_CM Datensatz...")
df = pd.read_csv(SALUKI_DATA, sep="\t")

print("3. Lade TargetScan Stichprobe...")
ts_df = pd.read_csv(TARGETSCAN_FILE, sep="\t", nrows=100000)
ts_transcripts = set(ts_df["Transcript ID"].astype(str).str.split(".").str[0])

# Metriken
total = len(df)
found_in_gtf = 0
exact_length_match = 0
length_diffs = []
utr3_overflow_count = 0
ts_match_count = 0

print(f"\nUeberpruefe {total} Transkripte...")

for idx, row in df.iterrows():
    raw_tx = str(row["ensembl_transcript_id"])
    clean_tx = raw_tx.split(".")[0]
    tokens = [tok.strip() for tok in str(row["sequence"]).split(",") if tok.strip()]
    seq_len = len(tokens)
    
    # CDS / 3' UTR Start im Datensatz ermitteln
    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    utr3_start = (max(upper_indices) + 3) if upper_indices else 0
    utr3_len = seq_len - utr3_start

    # A. GTF-Exon-Laengenabgleich
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

    # B. TargetScan Abgleich
    if clean_tx in ts_transcripts:
        ts_match_count += 1

print("\n" + "=" * 60)
print("             ERGEBNIS DER KOMPATIBILITAETSPRUEFUNG           ")
print("=" * 60)
print(f"Gesamtanzahl Transkripte:            {total}")
print(f"In GTF (GRCh38.108) gefunden:        {found_in_gtf} ({found_in_gtf/total*100:.2f}%)")
print(f"Exakte Laengenuebereinstimmung:      {exact_length_match} ({exact_length_match/found_in_gtf*100:.2f}% der gefundenen)")

if length_diffs:
    arr = np.array(length_diffs)
    print(f"Laengendifferenz (seq_len - gtf_len):")
    print(f"  Mittelwert: {np.mean(arr):.2f} bp | Median: {np.median(arr):.1f} bp")
    print(f"  |Diff| <= 5 bp: {np.sum(np.abs(arr) <= 5) / len(arr) * 100:.2f}%")

print("=" * 60)
