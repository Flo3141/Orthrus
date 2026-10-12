#!/usr/bin/env bash
# =============================================================================
# run_m6a_7track_pipeline.sh
# End-to-End Pipeline for m6A 7-Track Generation & Orthrus Model Fine-Tuning
# =============================================================================
#SBATCH --job-name=orthrus_m6a_7t
#SBATCH --output=orthrus_m6a_7t_%j.log
#SBATCH --error=orthrus_m6a_7t_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

echo "============================================================================="
echo " Starting Orthrus 7-Track (m6A Augmented) Pipeline"
echo "============================================================================="
date

# 1. Environment & Paths
PROJECT_DIR="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"
SCRIPTS_DIR="${PROJECT_DIR}/Orthrus/scripts"

# Base Data & Validation Outputs
SALUKI_DATA="${PROJECT_DIR}/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt"
M6A_VALIDATED_TSV="./results/m6a_validated_tracks/validated_m6a_sites.tsv"

# 7-Track Outputs
DATA_7T_DIR="${PROJECT_DIR}/data/hIPSC_CM/orthrus"
OUTPUT_7T_NPZ="${DATA_7T_DIR}/hIPSC_CM_7track_m6a.npz"
NORMALIZATION="unit"  # Options: unit (0-100% -> [0, 1]), minmax, binary, log, none

# Checkpoint Destinations
CHECKPOINT_7T_BASE="${PROJECT_DIR}/checkpoints/orthrus/orthrus-large-7-track"
FINETUNED_7T_DIR="${PROJECT_DIR}/checkpoints/orthrus/orthrus_7track_finetuned_hIPSC_CM"

# -----------------------------------------------------------------------------
# Step 1: Generate 7-Track NPZ with m6A Channel (cDNA Mapping)
# -----------------------------------------------------------------------------
echo -e "\n[STEP 1/3] Generating 7-track NPZ dataset with m6A modification channel..."
python3 "${SCRIPTS_DIR}/generate_m6a_tracks.py" \
    --saluki_data "${SALUKI_DATA}" \
    --m6a_file "${M6A_VALIDATED_TSV}" \
    --output_file "${OUTPUT_7T_NPZ}" \
    --score_col "mean_score" \
    --normalization "${NORMALIZATION}" \
    --chunk_size 500

ACTUAL_NPZ="${DATA_7T_DIR}/hIPSC_CM_7track_m6a_${NORMALIZATION}.npz"
if [ ! -f "${ACTUAL_NPZ}" ]; then
    ACTUAL_NPZ="${OUTPUT_7T_NPZ}"
fi

# -----------------------------------------------------------------------------
# Step 2: Weight Surgery (Convert 6-Track to 7-Track Orthrus Checkpoint)
# -----------------------------------------------------------------------------
echo -e "\n[STEP 2/3] Converting Orthrus base model from 6 to 7 tracks..."
python3 "${SCRIPTS_DIR}/convert_6track_to_ntrack.py" \
    --base_model "quietflamingo/orthrus-large-6-track" \
    --n_target_tracks 7 \
    --output_dir "${CHECKPOINT_7T_BASE}" \
    --init_method "normal" \
    --init_std 0.02

# -----------------------------------------------------------------------------
# Step 3: Supervised Fine-Tuning of 7-Track Orthrus on Half-Life
# -----------------------------------------------------------------------------
echo -e "\n[STEP 3/3] Fine-tuning 7-track Orthrus model..."
python3 "${SCRIPTS_DIR}/finetune_8track.py" \
    --data_path "${ACTUAL_NPZ}" \
    --model_checkpoint "${CHECKPOINT_7T_BASE}" \
    --output_dir "${FINETUNED_7T_DIR}" \
    --target_col "half_life_transformed" \
    --epochs 25 \
    --batch_size 16 \
    --lr_backbone 5e-5 \
    --lr_embedding 2e-4 \
    --lr_head 3e-4 \
    --loss_fn "huber"

echo -e "\n============================================================================="
echo " Orthrus 7-Track m6A Pipeline Completed Successfully!"
echo " Results and Checkpoints saved in ${FINETUNED_7T_DIR}"
echo "============================================================================="
date
