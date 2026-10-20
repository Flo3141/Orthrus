#!/usr/bin/env bash
# =============================================================================
# Helper script to prepare a compact test dataset (20 transcripts) on the Cluster
#
# Usage (run directly on the Slurm cluster):
#   bash /beegfs/prj/RNA_NLP/FlorianMasterThesis/code/Orthrus/scripts/prepare_cluster_test_sample.sh
# =============================================================================

set -e

CLUSTER_BASE="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"
DATA_SRC="$CLUSTER_BASE/data/hIPSC_CM"
OUT_DIR="$HOME/orthrus_local_test_data"

echo "============================================================================="
echo "   Preparing Compact Test Dataset for Local WSL Testing (20 Transcripts)    "
echo "============================================================================="

mkdir -p "$OUT_DIR"

if [ ! -f "$DATA_SRC/hIPSC_CM_ej_cds_transformed.txt" ]; then
    echo "[Error] File not found: $DATA_SRC/hIPSC_CM_ej_cds_transformed.txt"
    exit 1
fi

if [ ! -f "$DATA_SRC/hipsc_cm_10folds_lookup.csv" ]; then
    echo "[Error] File not found: $DATA_SRC/hipsc_cm_10folds_lookup.csv"
    exit 1
fi

echo "[1/3] Copying 10-folds lookup table..."
cp "$DATA_SRC/hipsc_cm_10folds_lookup.csv" "$OUT_DIR/hipsc_cm_10folds_lookup.csv"

echo "[2/3] Selecting 2 transcripts per fold (splits 0..9) for balanced CV testing..."
python3 - << 'EOF'
import pandas as pd
from pathlib import Path

lookup_path = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")
out_dir = Path.home() / "orthrus_local_test_data"

df = pd.read_csv(lookup_path)
# Pick 2 canonical transcripts per split (0 to 9)
sample_df = df.groupby("split").head(2)
sample_txs = set(sample_df["ensembl_transcript_id"].astype(str).str.strip())

with open(out_dir / "sample_transcript_ids.txt", "w") as f:
    for tx in sample_txs:
        f.write(f"{tx}\n")

print(f"      Selected {len(sample_txs)} transcripts across {df['split'].nunique()} folds.")
EOF

echo "[3/3] Extracting selected transcripts from hIPSC_CM_ej_cds_transformed.txt..."
# Extract header
head -n 1 "$DATA_SRC/hIPSC_CM_ej_cds_transformed.txt" > "$OUT_DIR/hIPSC_CM_ej_cds_transformed.txt"
# Extract matching rows
grep -F -f "$OUT_DIR/sample_transcript_ids.txt" "$DATA_SRC/hIPSC_CM_ej_cds_transformed.txt" >> "$OUT_DIR/hIPSC_CM_ej_cds_transformed.txt"

N_ROWS=$(wc -l < "$OUT_DIR/hIPSC_CM_ej_cds_transformed.txt")
echo "      Extracted $((N_ROWS - 1)) transcripts."

echo "============================================================================="
echo "   DONE! Files are ready in: $OUT_DIR"
echo "============================================================================="
echo "Copy these 2 files to your local Windows workspace at:"
echo "   F:\\Workspace\\MIM\\Masterarbeit\\data\\hIPSC_CM\\"
echo ""
echo "Command to copy to your local machine (run from Windows PowerShell):"
echo "   scp user@cluster:$OUT_DIR/hIPSC_CM_ej_cds_transformed.txt F:\\Workspace\\MIM\\Masterarbeit\\data\\hIPSC_CM\\"
echo "   scp user@cluster:$OUT_DIR/hipsc_cm_10folds_lookup.csv F:\\Workspace\\MIM\\Masterarbeit\\data\\hIPSC_CM\\"
echo "============================================================================="
