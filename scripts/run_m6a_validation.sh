#!/usr/bin/env bash
# =============================================================================
# run_m6a_validation.sh
# Execute hiPSC-CM m6A validation and filtering pipeline
# =============================================================================
#SBATCH --job-name=val_m6a
#SBATCH --output=val_m6a_%j.log
#SBATCH --error=val_m6a_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# 1. Project and Environment Configuration
BAMBU_DIR="/prj/Heart_on_m6A/Isabel/bambu"
FASTA_REF="/biodb/genomes/homo_sapiens/GRCh38_102/GRCh38_102.fa"
OUTPUT_DIR="./results/m6a_validated_tracks"

# Input Files
DMR_FILE="${BAMBU_DIR}/DMSO_M3I_replicates_cov10_baseA_moda.dmr"
CTRL_BEDS="${BAMBU_DIR}/hiPSC-CM_CasRx_ctrl_*_RTA_m6A_cDNAfinal.bed"

echo "============================================================================="
echo " Starting m6A Modification Validation for hiPSC-CM Dataset"
echo "============================================================================="
echo " DMR File   : ${DMR_FILE}"
echo " Control BED: ${CTRL_BEDS}"
echo " FASTA Ref  : ${FASTA_REF}"
echo " Output Dir : ${OUTPUT_DIR}"
echo "============================================================================="

# 2. Run Python Validation Script
python3 "$(dirname "$0")/validate_and_filter_m6a.py" \
    --output-dir "${OUTPUT_DIR}" \
    --min-effect-size 0.15 \
    --max-fdr 0.05 \
    --min-pct-samples 75.0 \
    --min-ctrl-mod 0.10 \
    --min-rep-count 2

echo "============================================================================="
echo " m6A Validation Finished! Results are saved in ${OUTPUT_DIR}"
echo "============================================================================="
