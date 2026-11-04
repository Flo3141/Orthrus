#!/usr/bin/env python3
"""
Unified Ridge Regression Training & Evaluation Pipeline on Extracted Orthrus Embeddings.

Features:
- Unifies hIPSC_CM and mRNABench (Human & Mouse) into a single, standardized pipeline.
- Automatically infers the correct standardized 10-fold lookup table from the 'dataset' field of the
  track metadata (.meta.json) referenced by the embeddings archive ('source_tracks_meta').
- Custom 4-fold Cross-Validation over training splits [0..7] with GridSearchCV (fold scheme: util.CV_FOLDS).
- Strictly held-out Test Set on splits [8, 9] (util.TEST_SPLITS, unseen during training and CV).
- Calculates raw-scale metrics (target_raw / half-life in hours) on back-transformed predictions
  if target_raw and the target transformation parameters (log + z-score) are available.
- Saves trained model (.joblib), detailed predictions table (.csv), metrics (.json), and optional publication plots (.png).
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_predict

from util import (
    load_metadata, SPLITS_LOOKUP_TABLES, resolve_dataset_key,
    CV_FOLDS, CV_POOL_SPLITS, TEST_SPLITS, DATASET_DIRS, RESULTS_DIR,
    assign_splits, require_meta, regression_metrics, print_metrics,
)

# Headless backend for cluster servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Embeddings are deliberately not standardized (same protocol as the mRNABench/Orthrus linear probing).
ALPHA_GRID = np.logspace(-2, 6, 17)

# Target transformation parameters (written by transform_hIPSC_CM_dataset.py):
#   target = (ln(target_raw + pseudocount) - mu_log) / sigma_log
# mRNABench targets (PC1 of half-lives) have no invertible raw-scale counterpart.
TARGET_TRANSFORM_PARAMS = {
    "hipsc_cm": DATASET_DIRS["hipsc_cm"] / "transformation_params.json",
}


def resolve_target_transform(meta: dict, dataset_key: str) -> Optional[dict]:
    """
    Returns the log + z-score parameters of the target transformation, or None if unavailable.
    Priority: 'target_transform' in track metadata -> dataset-specific transformation_params.json.
    """
    params = meta.get("target_transform")
    if params is None:
        params_path = TARGET_TRANSFORM_PARAMS.get(dataset_key)
        if params_path is None or not params_path.is_file():
            return None
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        print(f"[Target Transform] Loaded parameters from: {params_path}")

    return {
        "pseudocount": float(params["pseudocount"]),
        "mu_log": float(params["mu_log"]),
        "sigma_log": float(params["sigma_log"]),
    }


def inverse_transform_target(y: np.ndarray, transform: dict) -> np.ndarray:
    """Maps transformed targets / predictions back to raw half-lives (hours)."""
    return np.exp(y * transform["sigma_log"] + transform["mu_log"]) - transform["pseudocount"]


def verify_target_transform(y: np.ndarray, y_raw: np.ndarray, transform: dict, atol: float = 1e-3):
    """Fails if the transformation parameters do not reproduce the stored targets from target_raw."""
    mask = np.isfinite(y) & np.isfinite(y_raw) & (y_raw + transform["pseudocount"] > 0)
    y_expected = (np.log(y_raw[mask] + transform["pseudocount"]) - transform["mu_log"]) / transform["sigma_log"]
    max_dev = float(np.max(np.abs(y_expected - y[mask]))) if np.any(mask) else 0.0
    if max_dev > atol:
        raise ValueError(
            f"[Error] Target transformation parameters {transform} do not reproduce the stored targets "
            f"(max. deviation {max_dev:.4e} > {atol})! Raw-scale metrics would be invalid."
        )
    print(f"[Target Transform] Verified against target_raw (max. deviation: {max_dev:.2e}).")


# =============================================================================
# NPZ Embeddings Loader
# =============================================================================

def load_embeddings_npz(file_path: Path) -> dict:
    """
    Loads an NPZ file containing Orthrus embeddings.
    Besides the data arrays, the archive only carries 'source_tracks_meta', 'variant' and 'model_checkpoint';
    dataset, track name and normalization are read from the referenced track .meta.json (single source).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"[Error] Embedding file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    available_keys = list(data.keys())
    required_keys = [
        "embeddings_arr", "targets_arr", "ensembl_transcript_id_arr", "genes_arr",
        "source_tracks_meta", "variant", "model_checkpoint",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"[Error] Required fields {missing} not found in {file_path}! Available keys: {available_keys}\n"
            f"Please re-extract the embeddings with extract_embeddings.py."
        )

    # Track metadata from Track Generation (lies next to the embeddings archive)
    track_meta_path = file_path.parent / str(data["source_tracks_meta"])
    if not track_meta_path.is_file():
        raise FileNotFoundError(
            f"[Error] Track metadata '{track_meta_path}' referenced by {file_path} not found!"
        )
    print(f"[Metadata] Using Track Generation metadata from: {track_meta_path}")
    meta = load_metadata(track_meta_path, required=True)
    require_meta(meta, track_meta_path, "dataset", "track_name")

    normalization = str(meta.get("normalization", "none")).strip().lower() or "none"
    variant = str(data["variant"]).strip().lower()
    if variant not in ("pretrained", "finetuned"):
        raise ValueError(f"[Error] Unknown variant {variant!r} in {file_path}! Expected 'pretrained' or 'finetuned'.")

    # Target Raw (unnormalized half-life in hours); all-NaN means not available
    targets_raw = None
    if "targets_raw_arr" in data:
        cand_raw = data["targets_raw_arr"].astype(np.float32)
        if not np.all(np.isnan(cand_raw)):
            targets_raw = cand_raw

    return {
        "embeddings": data["embeddings_arr"].astype(np.float32),
        "targets": data["targets_arr"].astype(np.float32),
        "targets_raw": targets_raw,
        "transcript_ids": data["ensembl_transcript_id_arr"].astype(str),
        "genes": data["genes_arr"].astype(str),
        "dataset": str(meta["dataset"]).strip(),
        "track_name": str(meta["track_name"]).strip(),
        "normalization": normalization,
        "variant": variant,
        "model_checkpoint": str(data["model_checkpoint"]),
        "track_meta_path": track_meta_path,
        "metadata": meta,
    }


# =============================================================================
# Helper: Output Directory
# =============================================================================

def resolve_output_dir(
    dataset_key: str,
    track_type: str,
    normalization: str,
    variant: str,
) -> Path:
    """
    Constructs the canonical output folder based strictly on metadata:
    - 6-track:                  <base_dir>/<dataset_key>/6track/<variant> (variant: finetuned/pretrained)
    - All other tracks (N-track): <base_dir>/<dataset_key>/<track_type>/<normalization> (e.g. minmax, log, none)
    """
    sub_dir = variant if track_type == "6track" else (normalization if normalization and normalization != "none" else "none")
    return RESULTS_DIR / dataset_key / track_type / sub_dir


# =============================================================================
# Helpers: Array Subsets & Scatter Plot
# =============================================================================

def subset(arrays: dict, mask: np.ndarray) -> dict:
    """Applies a boolean mask to all sample-aligned arrays; optional arrays stay None."""
    return {k: None if v is None else v[mask] for k, v in arrays.items()}


def scatter_panel(ax, y_true, y_pred, title, xlabel, ylabel, color="#1f77b4", log_scale=False):
    """Draws y_true vs. y_pred with the y=x reference line, on log-log axes if log_scale is set."""
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=color, edgecolors="none")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    values = np.concatenate([y_true, y_pred])
    if log_scale:
        values = values[values > 0]
        if len(values) == 0:
            return
        ax.set_xscale("log")
        ax.set_yscale("log")
        low, high = float(values.min()), float(values.max())
    else:
        margin = (float(values.max()) - float(values.min())) * 0.05
        low, high = float(values.min()) - margin, float(values.max()) + margin
    ax.plot([low, high], [low, high], "r--", linewidth=1.5, label="Ideal (y=x)")
    ax.legend()


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
        "--plot",
        action="store_true",
        help="Generate high-resolution scatter plot (y_true vs. y_pred) as PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings_path)

    # 1. Load Embeddings NPZ and strictly companion metadata
    print("=" * 75)
    print("      ORTHRUS UNIFIED RIDGE REGRESSION EVALUATION PIPELINE       ")
    print("=" * 75)
    print(f"Loading embeddings archive: {emb_path}...")
    data = load_embeddings_npz(emb_path)

    # 2. Identify Dataset and Auto-resolve Lookup Table
    dataset_key = resolve_dataset_key(data["dataset"])
    lookup_path = SPLITS_LOOKUP_TABLES[dataset_key]

    # 3. Determine Track Type, Normalization, Variant & Output Directory strictly from metadata
    track_type = data["track_name"]
    normalization = data["normalization"]
    variant = data["variant"]
    out_dir = resolve_output_dir(dataset_key, track_type, normalization, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset detected:      {dataset_key} (from metadata: '{data['dataset']}')")
    print(f"Track identity:        {track_type}")
    print(f"Normalization:         {normalization.upper()}")
    print(f"Model variant:         {variant.upper()}")
    print(f"Splits lookup table:   {lookup_path}")
    print(f"Output directory:      {out_dir}")

    # 4. Target Verification
    y_all = data["targets"]
    y_raw_all = data["targets_raw"]
    print(f"Raw target present:    {'Yes (target_raw)' if y_raw_all is not None else 'No'}")

    # Target transformation is required to map predictions back to raw half-lives
    target_transform = None
    if y_raw_all is not None:
        target_transform = resolve_target_transform(data["metadata"], dataset_key)
        if target_transform is None:
            print("[Target Transform] No transformation parameters found -> raw-scale metrics are skipped.")
        else:
            verify_target_transform(y_all, y_raw_all, target_transform)

    # 5. Assign Standardized 10-Fold Splits (Lookup Table) to samples with a finite target; all others stay -1
    valid_mask = np.isfinite(y_all)
    sample_splits = np.full(len(y_all), -1, dtype=int)
    sample_splits[valid_mask] = assign_splits(dataset_key, data["transcript_ids"][valid_mask])

    keep_mask = sample_splits != -1
    n_invalid = int(np.sum(~valid_mask))
    n_unmatched = int(np.sum(valid_mask & ~keep_mask))
    if n_invalid or n_unmatched:
        print(f"[Data Cleaning] Filtered {n_invalid} samples with invalid targets, {n_unmatched} without split assignment.")

    samples = subset({
        "X": data["embeddings"],
        "y": y_all,
        "y_raw": y_raw_all,
        "genes": data["genes"],
        "tx_ids": data["transcript_ids"],
        "splits": sample_splits,
    }, keep_mask)

    print(f"Total valid samples:   {len(samples['y']):,}")
    print(f"Feature dimension:     {samples['X'].shape[1]}")
    print(f"Unique genes:          {len(np.unique(samples['genes'])):,}")

    # Test Set: Splits 8 & 9 (20% strictly held-out)
    test_set = subset(samples, np.isin(samples["splits"], TEST_SPLITS))
    X_test, y_test = test_set["X"], test_set["y"]
    test_genes = test_set["genes"]
    y_test_raw_true = test_set["y_raw"]

    # CV Pool: Splits 0-7 (80% for training & cross-validation)
    cv_pool = subset(samples, np.isin(samples["splits"], CV_POOL_SPLITS))
    X_train_cv, y_train_cv = cv_pool["X"], cv_pool["y"]
    cv_splits = cv_pool["splits"]
    train_genes = cv_pool["genes"]

    print(f"\n[Dataset Partitioning]")
    print(f"  CV Pool (Folds 0-7): {len(y_train_cv):,} samples ({len(np.unique(train_genes)):,} unique genes)")
    print(f"  Test Set (Folds 8-9): {len(y_test):,} samples ({len(np.unique(test_genes)):,} unique genes)")

    # 6. Custom 4-Fold Cross-Validation Setup
    custom_cv = []
    for fold_def in CV_FOLDS.values():
        tr_idx = np.where(np.isin(cv_splits, fold_def["train"]))[0]
        val_idx = np.where(np.isin(cv_splits, fold_def["val"]))[0]
        custom_cv.append((tr_idx, val_idx))

    print(f"\n[Training] Running GridSearchCV across 4-fold Cross-Validation (Alphas: {ALPHA_GRID.tolist()})...")
    grid_search = GridSearchCV(
        Ridge(), {"alpha": ALPHA_GRID}, cv=custom_cv, scoring="neg_mean_squared_error", refit=True
    )
    grid_search.fit(X_train_cv, y_train_cv)
    model = grid_search.best_estimator_  # refitted on the entire CV pool (0-7)

    best_alpha = float(grid_search.best_params_["alpha"])
    print(f"\n>>> Selected Optimal Alpha: {best_alpha:.4e} <<<")
    if best_alpha in (ALPHA_GRID[0], ALPHA_GRID[-1]):
        print(f"[Warning] Selected alpha lies on the edge of the grid ({ALPHA_GRID[0]:.0e} to {ALPHA_GRID[-1]:.0e}); the optimum may lie outside.")

    # Evaluate Individual CV Folds on out-of-fold predictions
    oof_pred = cross_val_predict(model, X_train_cv, y_train_cv, cv=custom_cv)

    print("\n" + "-" * 72)
    print("          4-FOLD CROSS-VALIDATION DIAGNOSTICS (VAL FOLDS)         ")
    print("-" * 72)
    print(f"{'Fold':<10} | {'Train Folds':<18} | {'Val Folds':<12} | {'Val Pearson r':<14} | {'Val RMSE':<10}")
    print("-" * 72)

    fold_evaluations = []
    val_pearsons = []
    val_rmses = []

    for f_idx, fold_def in CV_FOLDS.items():
        _, val_idx = custom_cv[f_idx]
        f_metrics = regression_metrics(y_train_cv[val_idx], oof_pred[val_idx], prefix=f"fold_{f_idx}")
        fold_evaluations.append({
            "fold_name": f"Fold {f_idx}",
            "train_splits": fold_def["train"],
            "val_splits": fold_def["val"],
            "val_samples": int(len(val_idx)),
            **f_metrics,
        })
        r_val = f_metrics[f"fold_{f_idx}_pearson_r"]
        rmse_val = f_metrics[f"fold_{f_idx}_rmse"]
        val_pearsons.append(r_val)
        val_rmses.append(rmse_val)

        tr_str = ",".join(map(str, fold_def["train"]))
        val_str = ",".join(map(str, fold_def["val"]))
        print(f"{'Fold ' + str(f_idx):<10} | {tr_str:<18} | {val_str:<12} | {r_val:>12.4f}  | {rmse_val:>8.4f}")

    print("-" * 72)
    mean_cv_r = float(np.mean(val_pearsons))
    std_cv_r = float(np.std(val_pearsons))
    mean_cv_rmse = float(np.mean(val_rmses))
    std_cv_rmse = float(np.std(val_rmses))
    print(f"{'Mean ± Std':<10} | {'-':<18} | {'-':<12} | {mean_cv_r:>7.4f} ± {std_cv_r:.4f} | {mean_cv_rmse:>6.4f} ± {std_cv_rmse:.4f}")
    print("-" * 72)

    # Train predictions on entire CV pool (0-7)
    y_train_pred = model.predict(X_train_cv)
    train_metrics = regression_metrics(y_train_cv, y_train_pred, prefix="train_cv_pool")

    # 7. Test Set Evaluation (Unseen Folds 8 & 9)
    y_test_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, y_test_pred, prefix="test")

    # 8. Optional Raw Target Evaluation (target_raw, half-life in hours)
    # Predictions are back-transformed to the raw scale before comparison with target_raw.
    test_raw_metrics = {}
    y_test_pred_raw = None
    has_valid_raw = (
        target_transform is not None
        and y_test_raw_true is not None
        and not np.all(np.isnan(y_test_raw_true))
    )
    if has_valid_raw:
        y_test_pred_raw = inverse_transform_target(y_test_pred, target_transform)
        valid_raw_mask = np.isfinite(y_test_raw_true)
        if np.any(valid_raw_mask):
            print("\n[Target Raw] Computing metrics on back-transformed predictions vs. target_raw (hours)...")
            test_raw_metrics = regression_metrics(
                y_test_raw_true[valid_raw_mask],
                y_test_pred_raw[valid_raw_mask],
                prefix="test_raw"
            )
            # Robust error measure (raw half-lives are extremely right-skewed)
            test_raw_metrics["test_raw_median_ae"] = float(
                np.median(np.abs(y_test_raw_true[valid_raw_mask] - y_test_pred_raw[valid_raw_mask]))
            )

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
        "transcript_id": test_set["tx_ids"],
        "gene": test_genes,
        "true_target": y_test,
        "predicted_target": y_test_pred,
        "residual": y_test - y_test_pred,
    }
    if has_valid_raw:
        pred_dict["true_target_raw"] = y_test_raw_true
        pred_dict["predicted_target_raw"] = y_test_pred_raw

    pred_df = pd.DataFrame(pred_dict)
    pred_file = out_dir / f"predictions_{dataset_key}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    # 11. Save Comprehensive Metrics JSON
    all_metrics = {
        "dataset": dataset_key,
        "track_type": track_type,
        "track_name": track_type,
        "normalization": normalization,
        "variant": variant,
        "raw_dataset_meta": data["dataset"],
        "model_checkpoint": data["model_checkpoint"],
        "embeddings_path": str(emb_path),
        "track_meta_path": str(data["track_meta_path"]),
        "splits_lookup_path": str(lookup_path),
        "target_transform": target_transform,
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
            test_stats = (
                f"Pearson r = {test_metrics['test_pearson_r']:.3f} | Spearman rho = {test_metrics['test_spearman_rho']:.3f} | "
                f"RMSE = {test_metrics['test_rmse']:.3f} | R² = {test_metrics['test_r2']:.3f}"
            )
            if test_raw_metrics:
                # Left = Transformed Target, Right = Raw half-life vs. back-transformed prediction (right-skewed -> log-log)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                scatter_panel(
                    ax1, y_test, y_test_pred, f"Transformed Target\n{test_stats}",
                    "True Value (Transformed)", "Predicted Value (Transformed)",
                )
                scatter_panel(
                    ax2, y_test_raw_true[valid_raw_mask], y_test_pred_raw[valid_raw_mask],
                    f"Raw Half-Life (back-transformed)\n"
                    f"Pearson r = {test_raw_metrics['test_raw_pearson_r']:.3f} | "
                    f"Spearman rho = {test_raw_metrics['test_raw_spearman_rho']:.3f} | "
                    f"Median AE = {test_raw_metrics['test_raw_median_ae']:.2f} h",
                    "True Half-Life (h, target_raw)", "Predicted Half-Life (h)",
                    color="#2ca02c", log_scale=True,
                )
                fig.suptitle(f"Orthrus Ridge Regression -> {dataset_key} Test Set Evaluation", fontsize=12, fontweight="bold")
            else:
                fig, ax = plt.subplots(figsize=(7, 6))
                scatter_panel(
                    ax, y_test, y_test_pred, f"Orthrus Ridge Regression -> {dataset_key}\nTest {test_stats}",
                    "True Value", "Predicted Value",
                )
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
