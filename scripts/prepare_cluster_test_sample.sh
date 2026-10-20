#!/usr/bin/env bash
# =============================================================================
# Helper script to prepare a compact test dataset (20 transcripts) on the Cluster
# for ALL 4 track generation scripts:
#   - generate_6track.py
#   - generate_m6a_tracks.py (7-track m6A)
#   - generate_2_rbp_tracks.py (8-track RBP Stabilizer/Destabilizer)
#   - generate_miRNA_rbp_tracks.py (8-track miRNA & eCLIP)
#
# Usage (run directly on the Slurm cluster):
#   bash /beegfs/prj/RNA_NLP/FlorianMasterThesis/code/Orthrus/scripts/prepare_cluster_test_sample.sh
# =============================================================================

set -e

CLUSTER_BASE="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"
DATA_SRC="$CLUSTER_BASE/data/hIPSC_CM"
ALPHAGENOME_GTF="/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db"
OUT_ROOT="$HOME/orthrus_local_test_data"
OUT_DATA="$OUT_ROOT/data"

echo "============================================================================="
echo "   Preparing Compact Test Dataset for Local WSL Testing (All 4 Track Types) "
echo "============================================================================="

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_DATA/hIPSC_CM/m6a"
mkdir -p "$OUT_DATA/eclip/curated_candidates"
mkdir -p "$OUT_DATA/targetscan"

# -----------------------------------------------------------------------------
# 1. Select 20 Transcripts (2 per split 0..9)
# -----------------------------------------------------------------------------
echo "[1/7] Selecting 2 transcripts per fold (splits 0..9) from 10-folds lookup..."
if [ ! -f "$DATA_SRC/hipsc_cm_10folds_lookup.csv" ]; then
    echo "[Error] Lookup table not found: $DATA_SRC/hipsc_cm_10folds_lookup.csv"
    exit 1
fi

cp "$DATA_SRC/hipsc_cm_10folds_lookup.csv" "$OUT_DATA/hIPSC_CM/hipsc_cm_10folds_lookup.csv"

python3 - << 'EOF'
import pandas as pd
from pathlib import Path

lookup_path = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")
out_dir = Path.home() / "orthrus_local_test_data"

df = pd.read_csv(lookup_path)
sample_df = df.groupby("split").head(2)
sample_txs = set(sample_df["ensembl_transcript_id"].astype(str).str.strip())

with open(out_dir / "sample_transcript_ids.txt", "w") as f:
    for tx in sorted(sample_txs):
        f.write(f"{tx}\n")

print(f"      Selected {len(sample_txs)} transcripts across {df['split'].nunique()} folds.")
EOF

# -----------------------------------------------------------------------------
# 2. Extract Base 6-Track hIPSC_CM Sequences
# -----------------------------------------------------------------------------
echo "[2/7] Extracting 20 transcripts from hIPSC_CM_ej_cds_transformed.txt..."
head -n 1 "$DATA_SRC/hIPSC_CM_ej_cds_transformed.txt" > "$OUT_DATA/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt"
grep -F -f "$OUT_ROOT/sample_transcript_ids.txt" "$DATA_SRC/hIPSC_CM_ej_cds_transformed.txt" >> "$OUT_DATA/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt" || true
echo "      Extracted $(( $(wc -l < "$OUT_DATA/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt") - 1 )) sequence records."

# -----------------------------------------------------------------------------
# 3. Extract Validated m6A Sites (for generate_m6a_tracks.py)
# -----------------------------------------------------------------------------
echo "[3/7] Extracting m6A records (validated_m6a_sites.tsv)..."
M6A_SRC="$DATA_SRC/m6a/validated_m6a_sites.tsv"
if [ -f "$M6A_SRC" ]; then
    head -n 1 "$M6A_SRC" > "$OUT_DATA/hIPSC_CM/m6a/validated_m6a_sites.tsv"
    grep -F -f "$OUT_ROOT/sample_transcript_ids.txt" "$M6A_SRC" >> "$OUT_DATA/hIPSC_CM/m6a/validated_m6a_sites.tsv" || true
    echo "      Saved m6A slice ($(( $(wc -l < "$OUT_DATA/hIPSC_CM/m6a/validated_m6a_sites.tsv") - 1 )) sites)."
else
    echo "      [Notice] $M6A_SRC not found, creating dummy header."
    echo -e "TxId\tcDNAstart\tcDNAend\tmean_score\tis_drach" > "$OUT_DATA/hIPSC_CM/m6a/validated_m6a_sites.tsv"
fi

# -----------------------------------------------------------------------------
# 4. Extract RBP Peaks (for generate_2_rbp_tracks.py)
# -----------------------------------------------------------------------------
echo "[4/7] Copying RBP Stabilizers & Destabilizers BED files..."
STAB_SRC="$CLUSTER_BASE/data/eclip/curated_candidates/stabilizers_peaks.bed"
DESTAB_SRC="$CLUSTER_BASE/data/eclip/curated_candidates/destabilizers_peaks.bed"

if [ -f "$STAB_SRC" ]; then
    cp "$STAB_SRC" "$OUT_DATA/eclip/curated_candidates/stabilizers_peaks.bed"
else
    touch "$OUT_DATA/eclip/curated_candidates/stabilizers_peaks.bed"
fi

if [ -f "$DESTAB_SRC" ]; then
    cp "$DESTAB_SRC" "$OUT_DATA/eclip/curated_candidates/destabilizers_peaks.bed"
else
    touch "$OUT_DATA/eclip/curated_candidates/destabilizers_peaks.bed"
fi

# -----------------------------------------------------------------------------
# 5. Extract miRNA & Merged eCLIP (for generate_miRNA_rbp_tracks.py)
# -----------------------------------------------------------------------------
echo "[5/7] Extracting TargetScan and merged eCLIP records..."
TS_SRC="$CLUSTER_BASE/data/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt"
if [ -f "$TS_SRC" ]; then
    head -n 1 "$TS_SRC" > "$OUT_DATA/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt"
    grep -F -f "$OUT_ROOT/sample_transcript_ids.txt" "$TS_SRC" >> "$OUT_DATA/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt" || true
    echo "      TargetScan slice extracted."
else
    echo "      [Notice] TargetScan source file not found, creating dummy header."
    echo -e "Transcript ID\tGene Symbol\tmiRNA family\tUTR_start\tUTR end\tweighted context++ score" > "$OUT_DATA/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt"
fi

ECLIP_MERGED="$CLUSTER_BASE/data/eclip/all_rbp_peaks_merged.bed"
if [ -f "$ECLIP_MERGED" ]; then
    head -n 1000 "$ECLIP_MERGED" > "$OUT_DATA/eclip/all_rbp_peaks_merged.bed"
    echo "      eCLIP merged slice extracted (1000 peaks)."
else
    touch "$OUT_DATA/eclip/all_rbp_peaks_merged.bed"
fi

# -----------------------------------------------------------------------------
# 6. Extract Mini GTF SQLite Database (gffutils FeatureDB)
# -----------------------------------------------------------------------------
echo "[6/7] Extracting mini GTF database for the 20 transcripts..."
python3 - << 'EOF'
import sys
from pathlib import Path

out_root = Path.home() / "orthrus_local_test_data"
out_db = out_root / "data" / "Homo_sapiens.GRCh38.108.gtf.db"
src_db_path = Path("/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db")

try:
    import gffutils
    if src_db_path.exists():
        print(f"      Reading source GTF database: {src_db_path}...")
        src_db = gffutils.FeatureDB(str(src_db_path))
        with open(out_root / "sample_transcript_ids.txt") as f:
            tx_ids = [line.strip() for line in f if line.strip()]

        features = []
        found_count = 0
        for tx_id in tx_ids:
            clean_id = tx_id.split(".")[0]
            for cand in [tx_id, clean_id]:
                try:
                    tx = src_db[cand]
                    features.append(tx)
                    for child in src_db.children(tx):
                        features.append(child)
                    found_count += 1
                    break
                except Exception:
                    continue

        if features:
            gffutils.create_db(features, str(out_db), force=True, merge_strategy="create_unique")
            print(f"      Created mini GTF DB with {found_count} transcripts ({len(features)} total features).")
        else:
            print("      [Warning] None of the 20 transcripts found in source GTF DB.")
    else:
        print(f"      [Notice] Source GTF DB not found at: {src_db_path}")
except ImportError:
    print("      [Notice] gffutils not installed in active Python on cluster; skipping mini GTF generation.")
except Exception as e:
    print(f"      [Notice] Failed to extract mini GTF DB: {e}")
EOF

# -----------------------------------------------------------------------------
# 7. Package into a single compact archive
# -----------------------------------------------------------------------------
echo "[7/7] Packaging test dataset into tar.gz archive..."
TAR_FILE="$HOME/orthrus_test_data.tar.gz"
tar -czf "$TAR_FILE" -C "$OUT_ROOT" data

ARCHIVE_SIZE=$(ls -lh "$TAR_FILE" | awk '{print $5}')
echo "============================================================================="
echo "   DONE! Complete test archive created: $TAR_FILE ($ARCHIVE_SIZE)"
echo "============================================================================="
echo "To copy and extract on your local Windows PC:"
echo ""
echo "1. Run from Windows PowerShell:"
echo "   scp user@cluster:~/orthrus_test_data.tar.gz F:\\Workspace\\MIM\\Masterarbeit\\"
echo ""
echo "2. Extract in Windows PowerShell:"
echo "   tar -xzvf F:\\Workspace\\MIM\\Masterarbeit\\orthrus_test_data.tar.gz -C F:\\Workspace\\MIM\\Masterarbeit\\"
echo "============================================================================="
