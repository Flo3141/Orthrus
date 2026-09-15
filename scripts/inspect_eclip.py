import re
from pathlib import Path
import pandas as pd

# Path to downloaded eCLIP BED file
bed_path = Path.home() / "Downloads" / "rer1_eclip_peaks.bed"
if not bed_path.exists():
    bed_path = Path("rer1_eclip_peaks.bed")

output_all_csv = "eclip_rer1_peaks_all.csv"
output_exonic_csv = "eclip_rer1_exonic_peaks.csv"

print(f"Reading eCLIP BED file: {bed_path}...")

# Exon coordinates for RER1 (ENST00000378512 / GRCh38, plus strand)
# Exon 1..7 (1-based closed in genome)
RER1_EXONS = [
    {"exon": 1, "start": 2391763, "end": 2391958, "region": "5' UTR / Exon 1"},
    {"exon": 2, "start": 2395784, "end": 2395871, "region": "CDS / Exon 2"},
    {"exon": 3, "start": 2397116, "end": 2397220, "region": "CDS / Exon 3"},
    {"exon": 4, "start": 2399415, "end": 2399514, "region": "CDS / Exon 4"},
    {"exon": 5, "start": 2400857, "end": 2400935, "region": "CDS / Exon 5"},
    {"exon": 6, "start": 2402097, "end": 2402342, "region": "CDS + Stop + 3' UTR / Exon 6"},
    {"exon": 7, "start": 2403035, "end": 2403751, "region": "3' UTR / Exon 7"},
]

rows = []
with open(bed_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Skip header / metadata
        if line.startswith("track") or line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 6:
            rows.append(parts)

print(f"Found eCLIP entries: {len(rows)}")

# narrowPeak columns
columns = [
    "chrom", "chromStart", "chromEnd", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak"
]
df = pd.DataFrame([r[:10] for r in rows], columns=columns[:len(rows[0])])

# Convert numeric columns
df["chromStart"] = pd.to_numeric(df["chromStart"], errors="coerce")
df["chromEnd"] = pd.to_numeric(df["chromEnd"], errors="coerce")
df["peak_length"] = df["chromEnd"] - df["chromStart"]

if "score" in df.columns:
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
if "signalValue" in df.columns:
    df["signalValue"] = pd.to_numeric(df["signalValue"], errors="coerce")
if "pValue" in df.columns:
    df["pValue"] = pd.to_numeric(df["pValue"], errors="coerce")

# Extract RBP and cell line from 'name' column (e.g. 'NONO_K562_rep01')
def parse_rbp_name(val):
    if not val or val == ".":
        return "Unknown", "", ""
    parts = str(val).split("_")
    rbp = parts[0]
    cell = parts[1] if len(parts) > 1 else ""
    rep = parts[2] if len(parts) > 2 else ""
    return rbp, cell, rep

parsed = df["name"].apply(parse_rbp_name)
df["RBP"] = [p[0] for p in parsed]
df["Cell_Line"] = [p[1] for p in parsed]
df["Replicate"] = [p[2] for p in parsed]

# Sort by genomic coordinates
df = df.sort_values(by=["chromStart", "chromEnd"])

# 1. Save all peaks
df.to_csv(output_all_csv, index=False)
print(f"All peaks saved as: {output_all_csv}")

# 2. Determine exonic overlaps for RER1 (plus strand)
# These are precisely the peaks mapped onto the mRNA by generate_trans_factor_tracks.py!
exonic_records = []

# Calculate cumulative mRNA position (5' -> 3')
curr_tx_offset = 0
for ex in RER1_EXONS:
    ex["tx_start"] = curr_tx_offset
    ex_len = ex["end"] - ex["start"] + 1
    ex["tx_end"] = curr_tx_offset + ex_len
    curr_tx_offset += ex_len

for _, row in df.iterrows():
    # RER1 is on the plus strand
    if row["strand"] != "+":
        continue

    p_start = row["chromStart"] + 1  # 0-based -> 1-based
    p_end = row["chromEnd"]
    sig = row.get("signalValue", 0.0)

    for ex in RER1_EXONS:
        # Check for intersection with exon
        if p_end >= ex["start"] and p_start <= ex["end"]:
            overlap_start = max(p_start, ex["start"])
            overlap_end = min(p_end, ex["end"])
            overlap_len = overlap_end - overlap_start + 1

            rel_tx_start = ex["tx_start"] + (overlap_start - ex["start"])
            rel_tx_end = ex["tx_start"] + (overlap_end - ex["start"]) + 1

            rec = dict(row)
            rec["Exon"] = ex["exon"]
            rec["Exon_Region"] = ex["region"]
            rec["Overlap_Genomic"] = f"{overlap_start}-{overlap_end}"
            rec["Overlap_Length_nt"] = overlap_len
            rec["mRNA_rel_start"] = rel_tx_start
            rec["mRNA_rel_end"] = rel_tx_end
            exonic_records.append(rec)

df_exonic = pd.DataFrame(exonic_records)
if not df_exonic.empty:
    df_exonic = df_exonic.sort_values(by=["mRNA_rel_start", "signalValue"], ascending=[True, False])
    df_exonic.to_csv(output_exonic_csv, index=False)
    print(f"Exonic peaks (mapped to RER1) saved as: {output_exonic_csv}")

# =============================================================================
# Summary & statistics output
# =============================================================================
print("\n" + "=" * 70)
print("             eCLIP PEAK STATISTICS FOR RER1 REGION            ")
print("=" * 70)
print(f"Total peaks in window:              {len(df)}")
print(f"Peaks on plus strand (+):           {sum(df['strand'] == '+')}")
print(f"Peaks on minus strand (-):          {sum(df['strand'] == '-')}")
print(f"Peaks with direct exon overlap:     {len(df_exonic)}")

if not df_exonic.empty:
    print("\nDistribution of eCLIP peaks across exons of RER1:")
    print(df_exonic["Exon_Region"].value_counts().to_string())

    print("\nTop 15 RBPs with highest signalValue on RER1:")
    top_rbps = df_exonic.groupby("RBP")["signalValue"].max().sort_values(ascending=False).head(15)
    for rbp, val in top_rbps.items():
        print(f"  - {rbp:<16}: max SignalValue = {val:.3f}")

    print("\nFirst 15 exonic peaks (sorted by relative mRNA position):")
    preview_cols = ["Exon", "RBP", "Cell_Line", "mRNA_rel_start", "mRNA_rel_end", "signalValue", "Overlap_Length_nt"]
    print(df_exonic[[c for c in preview_cols if c in df_exonic.columns]].head(15).to_string(index=False))
print("=" * 70)
