#!/bin/bash
# =============================================================================
# Run Comprehensive Trans-Factor & Feature Analysis on 8-Track Dataset
# =============================================================================
set -e

PROJECT_DIR="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"
SCRIPTS_DIR="${PROJECT_DIR}/Orthrus/scripts"
DATA_NPZ="${PROJECT_DIR}/data/hIPSC_CM/orthrus/hIPSC_CM_8track_minmax.npz"
OUTPUT_TXT="${PROJECT_DIR}/data/hIPSC_CM/hIPSC_CM_8track_minmax_analysis.txt"
OUTPUT_DIR="${PROJECT_DIR}/data/hIPSC_CM"

echo "============================================================================="
echo "   ORTHRUS 8-TRACK TRANS-FACTOR STATISTICAL & CORRELATION ANALYSIS           "
echo "============================================================================="
date

echo "Input NPZ:       ${DATA_NPZ}"
echo "Output Report:   ${OUTPUT_TXT}"
echo "Plots Directory: ${OUTPUT_DIR}/8track_analysis_plots"

python "${SCRIPTS_DIR}/analyze_8track_trans_factors.py" \
    --input_npz "${DATA_NPZ}" \
    --output_txt "${OUTPUT_TXT}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_plots \
    --save_features_csv

echo -e "\n============================================================================="
echo "   Analysis Completed Successfully! "
echo "============================================================================="
date
