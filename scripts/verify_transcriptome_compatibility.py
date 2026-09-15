#!/usr/bin/env python3
"""
Validierungs-Skript zur Pruefung der Transkriptom- und Koordinaten-Kompatibilitaet
zwischen hIPSC_CM, GTF (Ensembl 108) und TargetScan 8.0.
"""

import gffutils
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Pfade zu Ihren Dateien
SALUKI_DATA = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt")
GTF_DB = Path("/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db")
TARGETSCAN_FILE = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt")

print("1. Lade GTF-DB...")
db = gffutils.FeatureDB(str(GTF_DB))

print("2. Lade hIPSC_CM Datensatz...")
df = pd.read_csv(SALUKI_DATA, sep="\t")

print("3. Lade TargetScan Daten (vollstaendig & indexiert fuer schnellen Lookup)...")
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

# Dictionary: clean_tx -> Liste von (UTR_start, UTR end)
ts_data = {}
for row in tqdm(ts_df[["clean_tx", "UTR_start", "UTR end"]].itertuples(index=False), total=len(ts_df), desc="TargetScan indizieren"):
    ts_data.setdefault(row.clean_tx, []).append((row.UTR_start, getattr(row, "_2")))

print(f"TargetScan Daten fuer {len(ts_data)} einzigartige Transkripte indiziert.")

# Metriken GTF
total = len(df)
found_in_gtf = 0
exact_length_match = 0
length_diffs = []

# Metriken TargetScan 3' UTR
ts_tx_with_predictions = 0
tx_without_cds = 0
total_sites = 0
fitting_sites = 0
overflowing_sites = 0
transcripts_with_overflow = 0
overflow_distances = []

print(f"\nUeberpruefe {total} Transkripte...")

for idx, row in tqdm(df.iterrows(), total=total, desc="Transkripte pruefen"):
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

    # B. TargetScan Abgleich & 3' UTR Validierung
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
print("             ERGEBNIS DER KOMPATIBILITAETSPRUEFUNG           ")
print("=" * 65)
print(f"Gesamtanzahl Transkripte:            {total}")
print("-" * 65)
print("1. GTF-Abgleich (Ensembl 108 vs. hIPSC_CM Sequenzen):")
print(f"  In GTF gefunden:                   {found_in_gtf} ({found_in_gtf/total*100:.2f}%)")
print(f"  Exakte Laengenuebereinstimmung:    {exact_length_match} ({exact_length_match/found_in_gtf*100:.2f}% der gefundenen)")

if length_diffs:
    arr = np.array(length_diffs)
    print(f"  Laengendifferenz (seq_len - gtf_len):")
    print(f"    Mittelwert: {np.mean(arr):.2f} bp | Median: {np.median(arr):.1f} bp")
    print(f"    |Diff| == 0 bp: {np.sum(arr == 0) / len(arr) * 100:.2f}%")

print("-" * 65)
print("2. TargetScan 8.0 3' UTR Abgleich:")
print(f"  Transkripte mit TargetScan-Daten:  {ts_tx_with_predictions} ({ts_tx_with_predictions/total*100:.2f}%)")
if tx_without_cds > 0:
    print(f"  Davon ohne annotierte CDS:         {tx_without_cds}")
print(f"  Gesamte miRNA-Bindungsstellen:     {total_sites}")
if total_sites > 0:
    fit_pct = fitting_sites / total_sites * 100
    ov_pct = overflowing_sites / total_sites * 100
    print(f"  Passen vollstaendig in 3' UTR:     {fitting_sites} ({fit_pct:.2f}%)")
    print(f"  Ragen ueber 3' UTR hinaus (Over):  {overflowing_sites} ({ov_pct:.2f}%)")
    print(f"  Transkripte mit >= 1 Overflow:     {transcripts_with_overflow} ({transcripts_with_overflow/ts_tx_with_predictions*100:.2f}%)")

    if overflow_distances:
        ov_arr = np.array(overflow_distances)
        print(f"  Ueberhang-Distanz bei Overflows:")
        print(f"    Mittelwert: {np.mean(ov_arr):.1f} bp | Median: {np.median(ov_arr):.1f} bp | Max: {np.max(ov_arr)} bp")
print("=" * 65)
