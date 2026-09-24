#!/bin/bash
# =============================================================================
# End-to-End Pipeline for Orthrus 8-Track Adaptation, Fine-Tuning & Evaluation
# =============================================================================
# Exit on error
set -e

echo "============================================================================="
echo "   ORTHRUS 8-TRACK TRANS-FACTOR PIPELINE (Strategy A: Supervised Fine-Tuning) "
echo "============================================================================="
date

# Default Paths (Adjust if necessary)
PROJECT_DIR="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"
SCRIPTS_DIR="${PROJECT_DIR}/Orthrus/scripts"

DATA_NPZ="${PROJECT_DIR}/data/hIPSC_CM/orthrus/hIPSC_CM_8track_minmax.npz"
SPLITS_LOOKUP="${PROJECT_DIR}/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv"
CHECKPOINT_8T_DIR="${PROJECT_DIR}/checkpoints/orthrus/orthrus-large-8-track"
FINETUNED_DIR="${PROJECT_DIR}/checkpoints/orthrus/orthrus_8track_finetuned_hIPSC_CM"
EMBEDDING_OUT="${PROJECT_DIR}/data/hIPSC_CM/orthrus/orthrus_8track_embeddings_hIPSC_CM_minmax.npz"

# -----------------------------------------------------------------------------
# Step 0: Standardized 10-Fold Gene-Grouped Split Table Creation
# -----------------------------------------------------------------------------
if [ ! -f "${SPLITS_LOOKUP}" ]; then
    echo -e "\n[STEP 0/4] Generating standardized 10-fold split lookup table..."
    python "${SCRIPTS_DIR}/create_hipsc_cm_splits.py" \
        --input_path "${DATA_NPZ}" \
        --output_path "${SPLITS_LOOKUP}" \
        --n_splits 10 \
        --random_seed 42
else
    echo -e "\n[STEP 0/4] Found existing split lookup table: ${SPLITS_LOOKUP}"
fi

# -----------------------------------------------------------------------------
# Step 1: Weight Surgery (Convert 6-Track to 8-Track Checkpoint)
# -----------------------------------------------------------------------------
echo -e "\n[STEP 1/4] Converting 6-track checkpoint to 8-track model..."
python "${SCRIPTS_DIR}/convert_6track_to_8track.py" \
    --base_model "quietflamingo/orthrus-large-6-track" \
    --output_dir "${CHECKPOINT_8T_DIR}" \
    --init_method "normal" \
    --init_std 0.02

# -----------------------------------------------------------------------------
# Step 2: End-to-End Fine-Tuning on Augmented 8-Track Dataset
# -----------------------------------------------------------------------------
echo -e "\n[STEP 2/4] Fine-tuning 8-track Orthrus on hIPSC_CM half-life..."
python "${SCRIPTS_DIR}/finetune_8track.py" \
    --data_path "${DATA_NPZ}" \
    --model_checkpoint "${CHECKPOINT_8T_DIR}" \
    --output_dir "${FINETUNED_DIR}" \
    --target_col "half_life_transformed" \
    --epochs 25 \
    --batch_size 16 \
    --lr_backbone 5e-5 \
    --lr_embedding 2e-4 \
    --lr_head 3e-4 \
    --weight_decay 0.01 \
    --warmup_epochs 3 \
    --loss_fn "huber"

# -----------------------------------------------------------------------------
# Step 3: Extract 512-D Representations from Fine-Tuned 8-Track Model
# -----------------------------------------------------------------------------
echo -e "\n[STEP 3/4] Extracting fine-tuned 8-track representations (4-Fold CV)..."
python "${SCRIPTS_DIR}/extract_embeddings_hIPSC_CM_8track.py" \
    --data_path "${DATA_NPZ}" \
    --model_checkpoint "${FINETUNED_DIR}" \
    --output_dir "${PROJECT_DIR}/data/hIPSC_CM/orthrus" \
    --output_filename "orthrus_8track_embeddings_hIPSC_CM_minmax.npz" \
    --batch_size 16

# -----------------------------------------------------------------------------
# Step 4: Downstream Ridge Regression & Comparison
# -----------------------------------------------------------------------------
echo -e "\n[STEP 4/4] Evaluating downstream Ridge Regression on 8-track embeddings..."
python "${SCRIPTS_DIR}/train_ridge_regression_hIPSC_CM.py" \
    --embeddings_path "${EMBEDDING_OUT}" \
    --splits_lookup_path "${SPLITS_LOOKUP}" \
    --target_col "half_life_transformed" \
    --plot

echo -e "\n============================================================================="
echo "   8-Track Pipeline Finished Successfully! "
echo "============================================================================="
date
