#!/usr/bin/env python3
"""
Unified Ridge Regression Training & Evaluation Pipeline on Extracted Orthrus Embeddings.

Features:
- Unifies hIPSC_CM and mRNABench (Human & Mouse) into a single, standardized pipeline.
- Automatically infers the correct standardized 10-fold lookup table from NPZ 'dataset' metadata.
- Custom 4-fold Cross-Validation over training splits [0..7] with RidgeCV:
    Fold 0: Train [0, 1, 2, 3, 4, 5], Val [6, 7]
    Fold 1: Train [2, 3, 4, 5, 6, 7], Val [0, 1]
    Fold 2: Train [0, 1, 4, 5, 6, 7], Val [2, 3]
    Fold 3: Train [0, 1, 2, 3, 6, 7], Val [4, 5]
- Strictly held-out Test Set on splits [8, 9] (unseen during training and CV).
- Automatically calculates and logs metrics for raw targets (target_raw / half-life in hours) if available.
- Saves trained model (.joblib), detailed predictions table (.csv), metrics (.json), and optional publication plots (.png).
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Headless backend for cluster servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Standardized Cluster File Paths & Lookup Tables
# =============================================================================

SPLITS_LOOKUP_HIPSC_CM = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv"
)
SPLITS_LOOKUP_RNAHL_HUMAN = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/rnahl-human_10folds_lookup.csv"
)
SPLITS_LOOKUP_RNAHL_MOUSE = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/rnahl-mouse_10folds_lookup.csv"
)

SPLITS_LOOKUP_TABLES = {
    "hipsc_cm": SPLITS_LOOKUP_HIPSC_CM,
    "mrna_bench_human": SPLITS_LOOKUP_RNAHL_HUMAN,
    "mrna_bench_mouse": SPLITS_LOOKUP_RNAHL_MOUSE,
}

DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Calculates comprehensive regression metrics and correlation statistics."""
    p_corr, p_val = pearsonr(y_true, y_pred)
    s_corr, s_val = spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    prefix_str = f"{prefix}_" if prefix else ""
    return {
        f"{prefix_str}pearson_r": float(p_corr),
        f"{prefix_str}pearson_pvalue": float(p_val),
        f"{prefix_str}spearman_rho": float(s_corr),
        f"{prefix_str}spearman_pvalue": float(s_val),
        f"{prefix_str}mse": float(mse),
        f"{prefix_str}rmse": rmse,
        f"{prefix_str}mae": float(mae),
        f"{prefix_str}r2": float(r2),
    }


def print_metrics(metrics: dict, title: str):
    """Formatted console output for metrics."""
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        if "pvalue" in k:
            print(f"  {k:28s}: {v:.3e}")
        else:
            print(f"  {k:28s}: {v:.4f}")




# =============================================================================
# NPZ Embeddings Loader
# =============================================================================

def load_embeddings_npz(file_path: Path) -> dict:
    """Loads an NPZ file containing Orthrus embeddings and all metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"[Error] Embedding file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    available_keys = list(data.keys())

    # 1. Embeddings
    if "embeddings_arr" not in data:
        raise KeyError(f"[Error] 'embeddings_arr' not found in {file_path}! Available keys: {available_keys}")
    embeddings = data["embeddings_arr"].astype(np.float32)

    n_samples = len(embeddings)

    # 2. Target (Transformed / Model target)
    if "targets_arr" not in data:
        raise KeyError(f"[Error] 'targets_arr' not found in {file_path}! Available keys: {available_keys}")
    targets = data["targets_arr"].astype(np.float32)

    # 3. Target Raw (Unnormalized / Raw half-life in hours)
    if "targets_raw_arr" in data:
        targets_raw = data["targets_raw_arr"].astype(np.float32)
    else:
        print("[Note] 'targets_raw_arr' not found in {file_path}!")
        targets_raw = None

    # 4. Transcript IDs
    if "ensembl_transcript_id_arr" not in data:
        raise KeyError(f"[Error] 'ensembl_transcript_id_arr' not found in {file_path}! Available keys: {available_keys}")
    tx_ids = data["ensembl_transcript_id_arr"].astype(str)

    # 5. Gene Symbols
    if "genes_arr" not in data:
        raise KeyError(f"[Error] 'genes_arr' not found in {file_path}! Available keys: {available_keys}")
    genes = data["genes_arr"].astype(str)

    # 6. Dataset Identity Metadata
    dataset_name = "unknown"
    if "dataset" in data:
        raw_val = data["dataset"]
        if isinstance(raw_val, np.ndarray) and raw_val.ndim > 0 and len(raw_val) > 0:
            dataset_name = str(raw_val.flat[0]).strip()
        elif hasattr(raw_val, "item"):
            try:
                dataset_name = str(raw_val.item()).strip()
            except Exception:
                dataset_name = str(raw_val).strip()
        else:
            dataset_name = str(raw_val).strip()

    # 7. Additional Metadata
    normalization = str(data["normalization"]).strip().lower() if "normalization" in data else None
    if normalization in ["none_not_set", "nan", "none"]:
        normalization = None

    is_finetuned = bool(data["is_finetuned"]) if "is_finetuned" in data else None
    is_4fold_cv = bool(data.get("is_4fold_cv", False)) or ("fold_assignment_arr" in data) or ("fold_assignment" in data)

    return {
        "embeddings": embeddings,
        "targets": targets,
        "targets_raw": targets_raw,
        "transcript_ids": tx_ids,
        "genes": genes,
        "dataset": dataset_name,
        "normalization": normalization,
        "is_finetuned": is_finetuned,
        "is_4fold_cv": is_4fold_cv,
        "seq_lens": data.get("seq_lens_arr", data.get("seq_lens", None)),
        "archive_keys": available_keys,
    }


# =============================================================================
# Helper: Resolve Dataset, Split Lookup Table & Output Directory
# =============================================================================

def resolve_dataset_key(dataset_meta: str, file_path: Path) -> str:
    """Normalizes dataset identifier string to canonical key ('hipsc_cm', 'mrna_bench_human', 'mrna_bench_mouse')."""
    d_clean = str(dataset_meta).strip().lower()
    if d_clean in SPLITS_LOOKUP_TABLES:
        return d_clean
    else:
        raise ValueError(f"[Error] Unknown dataset key: {d_clean}! Known datasets: {list(SPLITS_LOOKUP_TABLES.keys())}.")

def resolve_splits_lookup(dataset_key: str) -> Path:
    """Resolves the 10-fold lookup table path automatically from dataset key."""
    if dataset_key in SPLITS_LOOKUP_TABLES:
        table_path = SPLITS_LOOKUP_TABLES[dataset_key]
        if table_path.exists():
            return table_path
        raise FileNotFoundError(
            f"[Error] Standardized lookup table for '{dataset_key}' not found at: {table_path}! "
            f"Please verify the cluster file path."
        )

    raise ValueError(
        f"[Error] Unable to resolve lookup table for dataset '{dataset_key}'! "
        f"Known datasets: {list(SPLITS_LOOKUP_TABLES.keys())}."
    )


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Ridge Regression training & evaluation on Orthrus embeddings."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to NPZ embeddings archive (required)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for models, predictions, metrics, and plots (default: auto-structured by dataset)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate high-resolution scatter plot (y_true vs. y_pred) as PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings_path)

    # 1. Load Embeddings NPZ
    print("=" * 75)
    print("      ORTHRUS UNIFIED RIDGE REGRESSION EVALUATION PIPELINE       ")
    print("=" * 75)
    print(f"Loading embeddings archive: {emb_path}...")
    data = load_embeddings_npz(emb_path)

    # 2. Identify Dataset and Auto-resolve Lookup Table
    dataset_key = resolve_dataset_key(data["dataset"], emb_path)
    lookup_path = resolve_splits_lookup(dataset_key)

    # 3. Determine Output Directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        norm_suffix = f"_{data['normalization']}" if data["normalization"] else ""
        variant = "finetuned" if data["is_finetuned"] else "pretrained"
        out_dir = Path(f"/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/Orthrus/{dataset_key}/{variant}{norm_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset detected:      {dataset_key} (from NPZ metadata: '{data['dataset']}')")
    print(f"Splits lookup table:   {lookup_path}")
    print(f"Output directory:      {out_dir}")
    print(f"Model variant:         {'Fine-Tuned' if data['is_finetuned'] else 'Pretrained Base'}")
    if data["normalization"]:
        print(f"Normalization:         {data['normalization'].upper()}")

    # 4. Data Extraction & Target Verification
    X = data["embeddings"]
    y = data["targets"]
    y_raw = data["targets_raw"]
    genes = data["genes"]
    tx_ids = data["transcript_ids"]

    # Filter NaN / Inf
    valid_mask = ~np.isnan(y) & ~np.isinf(y)
    if not np.all(valid_mask):
        dropped = int(np.sum(~valid_mask))
        print(f"[Data Cleaning] Filtered {dropped} samples with invalid target values.")
        X = X[valid_mask]
        y = y[valid_mask]
        genes = genes[valid_mask]
        tx_ids = tx_ids[valid_mask]
        if y_raw is not None:
            y_raw = y_raw[valid_mask]

    print(f"Total valid samples:   {len(y):,}")
    print(f"Feature dimension:     {X.shape[1]}")
    print(f"Unique genes:          {len(np.unique(genes)):,}")
    print(f"Raw target present:    {'Yes (target_raw)' if y_raw is not None else 'No'}")

    # 5. Load Standardized 10-Fold Split Lookup Table
    print(f"\n[Split Alignment] Reading lookup splits from {lookup_path.name}...")
    lookup_df = pd.read_csv(lookup_path)

    lookup_tx_col = "ensembl_transcript_id" if "ensembl_transcript_id" in lookup_df.columns else "transcript_id"
    if lookup_tx_col not in lookup_df.columns or "split" not in lookup_df.columns:
        raise KeyError(
            f"[Error] Required columns ('{lookup_tx_col}', 'split') not found in {lookup_path}! "
            f"Columns: {list(lookup_df.columns)}"
        )

    tx_to_split = dict(zip(lookup_df[lookup_tx_col].astype(str).str.strip(), lookup_df["split"].astype(int)))
    sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in tx_ids])

    unmatched = int(np.sum(sample_splits == -1))
    if unmatched > 0:
        print(f"[Warning] {unmatched} transcripts not found in lookup table! Filtering them out.")
        matched_mask = (sample_splits != -1)
        X = X[matched_mask]
        y = y[matched_mask]
        genes = genes[matched_mask]
        tx_ids = tx_ids[matched_mask]
        sample_splits = sample_splits[matched_mask]
        if y_raw is not None:
            y_raw = y_raw[matched_mask]

    # Test Set: Folds 8 & 9 (20% strictly held-out)
    test_mask = np.isin(sample_splits, [8, 9])
    X_test, y_test = X[test_mask], y[test_mask]
    test_genes = genes[test_mask]
    test_tx = tx_ids[test_mask]
    y_test_raw_true = y_raw[test_mask] if y_raw is not None else None

    # CV Pool: Folds 0-7 (80% for training & cross-validation)
    cv_mask = np.isin(sample_splits, [0, 1, 2, 3, 4, 5, 6, 7])
    X_train_cv, y_train_cv = X[cv_mask], y[cv_mask]
    cv_splits = sample_splits[cv_mask]
    train_genes = genes[cv_mask]

    print(f"\n[Dataset Partitioning]")
    print(f"  CV Pool (Folds 0-7): {len(y_train_cv):,} samples ({len(np.unique(train_genes)):,} unique genes)")
    print(f"  Test Set (Folds 8-9): {len(y_test):,} samples ({len(np.unique(test_genes)):,} unique genes)")

    # 6. Custom 4-Fold Cross-Validation Setup
    cv_fold_definitions = [
        {"name": "Fold 0", "train_splits": [0, 1, 2, 3, 4, 5], "val_splits": [6, 7]},
        {"name": "Fold 1", "train_splits": [2, 3, 4, 5, 6, 7], "val_splits": [0, 1]},
        {"name": "Fold 2", "train_splits": [0, 1, 4, 5, 6, 7], "val_splits": [2, 3]},
        {"name": "Fold 3", "train_splits": [0, 1, 2, 3, 6, 7], "val_splits": [4, 5]},
    ]

    custom_cv = []
    for fold_def in cv_fold_definitions:
        tr_idx = np.where(np.isin(cv_splits, fold_def["train_splits"]))[0]
        val_idx = np.where(np.isin(cv_splits, fold_def["val_splits"]))[0]
        custom_cv.append((tr_idx, val_idx))

    print(f"\n[Training] Running RidgeCV across 4-fold Cross-Validation (Alphas: {DEFAULT_ALPHAS})...")
    model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=custom_cv)
    model.fit(X_train_cv, y_train_cv)

    best_alpha = float(model.alpha_)
    print(f"\n>>> Selected Optimal Alpha: {best_alpha:.4e} <<<")

    # Evaluate Individual CV Folds
    print("\n" + "-" * 72)
    print("          4-FOLD CROSS-VALIDATION DIAGNOSTICS (VAL FOLDS)         ")
    print("-" * 72)
    print(f"{'Fold':<10} | {'Train Folds':<18} | {'Val Folds':<12} | {'Val Pearson r':<14} | {'Val RMSE':<10}")
    print("-" * 72)

    fold_evaluations = []
    val_pearsons = []
    val_rmses = []
    val_spearmans = []

    for f_idx, fold_def in enumerate(cv_fold_definitions):
        tr_idx, val_idx = custom_cv[f_idx]
        fold_ridge = Ridge(alpha=best_alpha, random_state=args.random_state)
        fold_ridge.fit(X_train_cv[tr_idx], y_train_cv[tr_idx])
        val_pred = fold_ridge.predict(X_train_cv[val_idx])

        f_metrics = calculate_metrics(y_train_cv[val_idx], val_pred, prefix=f"fold_{f_idx}")
        fold_evaluations.append({
            "fold_name": fold_def["name"],
            "train_splits": fold_def["train_splits"],
            "val_splits": fold_def["val_splits"],
            "val_samples": int(len(val_idx)),
            **f_metrics,
        })
        r_val = f_metrics[f"fold_{f_idx}_pearson_r"]
        rmse_val = f_metrics[f"fold_{f_idx}_rmse"]
        val_pearsons.append(r_val)
        val_rmses.append(rmse_val)
        val_spearmans.append(f_metrics[f"fold_{f_idx}_spearman_rho"])

        tr_str = ",".join(map(str, fold_def["train_splits"]))
        val_str = ",".join(map(str, fold_def["val_splits"]))
        print(f"{fold_def['name']:<10} | {tr_str:<18} | {val_str:<12} | {r_val:>12.4f}  | {rmse_val:>8.4f}")

    print("-" * 72)
    mean_cv_r = float(np.mean(val_pearsons))
    std_cv_r = float(np.std(val_pearsons))
    mean_cv_rmse = float(np.mean(val_rmses))
    std_cv_rmse = float(np.std(val_rmses))
    print(f"{'Mean ± Std':<10} | {'-':<18} | {'-':<12} | {mean_cv_r:>7.4f} ± {std_cv_r:.4f} | {mean_cv_rmse:>6.4f} ± {std_cv_rmse:.4f}")
    print("-" * 72)

    # Train predictions on entire CV pool (0-7)
    y_train_pred = model.predict(X_train_cv)
    train_metrics = calculate_metrics(y_train_cv, y_train_pred, prefix="train_cv_pool")

    # 7. Test Set Evaluation (Unseen Folds 8 & 9)
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

    # 8. Optional Raw Target Evaluation (target_raw)
    test_raw_metrics = {}
    if y_test_raw_true is not None:
        print("\n[Target Raw] Computing direct correlation metrics against target_raw...")
        test_raw_metrics = calculate_metrics(y_test_raw_true, y_test_pred, prefix="test_raw")

    # Print Summary Diagnostics
    print_metrics(train_metrics, "Training / CV Pool (Splits 0-7) Metrics")
    print_metrics(test_metrics, "Test Set (Splits 8 & 9) Metrics - Transformed")
    if test_raw_metrics:
        print_metrics(test_raw_metrics, "Test Set (Splits 8 & 9) Metrics - Raw Target")

    # 9. Save Trained Model
    model_file = out_dir / f"ridge_model_{dataset_key}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved to: {model_file}")

    # 10. Save Predictions CSV
    pred_dict = {
        "transcript_id": test_tx,
        "gene": test_genes,
        "true_target": y_test,
        "predicted_target": y_test_pred,
        "residual": y_test - y_test_pred,
    }
    if y_test_raw_true is not None:
        pred_dict["true_target_raw"] = y_test_raw_true

    pred_df = pd.DataFrame(pred_dict)
    pred_file = out_dir / f"predictions_{dataset_key}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    # 11. Save Comprehensive Metrics JSON
    all_metrics = {
        "dataset": dataset_key,
        "raw_dataset_meta": data["dataset"],
        "is_finetuned": data["is_finetuned"],
        "is_4fold_cv": data["is_4fold_cv"],
        "normalization": data["normalization"],
        "embeddings_path": str(emb_path),
        "splits_lookup_path": str(lookup_path),
        "best_alpha": best_alpha,
        "cv_pool_size": int(len(y_train_cv)),
        "test_size": int(len(y_test)),
        "cv_pool_unique_genes": int(len(np.unique(train_genes))),
        "test_unique_genes": int(len(np.unique(test_genes))),
        "cv_folds_metrics": fold_evaluations,
        "cv_mean_pearson_r": mean_cv_r,
        "cv_std_pearson_r": std_cv_r,
        "cv_mean_rmse": mean_cv_rmse,
        "cv_std_rmse": std_cv_rmse,
        **train_metrics,
        **test_metrics,
        **test_raw_metrics,
    }
    metrics_file = out_dir / f"metrics_{dataset_key}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

    # 12. Optional Scatter Plot
    if args.plot:
        try:
            if y_test_raw_true is not None:
                # Dual subplot: Left = Transformed Target, Right = Raw Target
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Subplot 1: Transformed Target
                ax1.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
                p_r = test_metrics["test_pearson_r"]
                s_rho = test_metrics["test_spearman_rho"]
                rmse = test_metrics["test_rmse"]
                r2 = test_metrics["test_r2"]
                ax1.set_title(
                    f"Transformed Target\n"
                    f"Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | RMSE = {rmse:.3f} | R² = {r2:.3f}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax1.set_xlabel("True Value (Transformed)", fontsize=10)
                ax1.set_ylabel("Predicted Value (Transformed)", fontsize=10)
                min_v1 = min(float(np.min(y_test)), float(np.min(y_test_pred)))
                max_v1 = max(float(np.max(y_test)), float(np.max(y_test_pred)))
                margin1 = (max_v1 - min_v1) * 0.05
                ax1.plot([min_v1 - margin1, max_v1 + margin1], [min_v1 - margin1, max_v1 + margin1], "r--", linewidth=1.5, label="Ideal (y=x)")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.5)

                # Subplot 2: Raw Target vs. Prediction
                ax2.scatter(y_test_raw_true, y_test_pred, alpha=0.35, s=18, color="#2ca02c", edgecolors="none")
                p_r_raw = test_raw_metrics.get("test_raw_pearson_r", np.nan)
                s_rho_raw = test_raw_metrics.get("test_raw_spearman_rho", np.nan)
                ax2.set_title(
                    f"Raw Target vs. Predicted Value\n"
                    f"Pearson r = {p_r_raw:.3f} | Spearman rho = {s_rho_raw:.3f}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax2.set_xlabel("True Raw Value (target_raw)", fontsize=10)
                ax2.set_ylabel("Predicted Value", fontsize=10)
                ax2.grid(True, linestyle="--", alpha=0.5)

                fig.suptitle(f"Orthrus Ridge Regression -> {dataset_key} Test Set Evaluation", fontsize=12, fontweight="bold")
                plt.tight_layout()
            else:
                # Single plot
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
                p_r = test_metrics["test_pearson_r"]
                s_rho = test_metrics["test_spearman_rho"]
                r2 = test_metrics["test_r2"]
                rmse = test_metrics["test_rmse"]
                ax.set_title(
                    f"Orthrus Ridge Regression -> {dataset_key}\n"
                    f"Test Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | RMSE = {rmse:.3f} | R² = {r2:.3f}",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel("True Value", fontsize=10)
                ax.set_ylabel("Predicted Value", fontsize=10)
                min_val = min(float(np.min(y_test)), float(np.min(y_test_pred)))
                max_val = max(float(np.max(y_test)), float(np.max(y_test_pred)))
                margin = (max_val - min_val) * 0.05
                ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], "r--", linewidth=1.5, label="Ideal (y=x)")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()

            plot_file = out_dir / f"scatter_{dataset_key}.png"
            fig.savefig(plot_file, dpi=300)
            plt.close(fig)
            print(f"Scatter plot saved to: {plot_file}")
        except Exception as e:
            print(f"[Notice] Plot generation skipped: {e}")

    print("\nEvaluation successfully completed!")


if __name__ == "__main__":
    main()
