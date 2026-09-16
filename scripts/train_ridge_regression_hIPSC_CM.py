#!/usr/bin/env python3
"""
Ridge regression training & evaluation on extracted Orthrus embeddings for the hIPSC_CM dataset.
Supports both 6-track and 8-track representations.
Integrates standardized 10-fold gene-grouped split lookup table:
- Evaluates on test splits [8, 9]
- Conducts custom 4-fold Cross-Validation over splits [0..7]:
    Fold 1: Train [0, 1, 2, 3, 4, 5], Val [6, 7]
    Fold 2: Train [2, 3, 4, 5, 6, 7], Val [0, 1]
    Fold 3: Train [0, 1, 4, 5, 6, 7], Val [2, 3]
    Fold 4: Train [0, 1, 2, 3, 6, 7], Val [4, 5]
"""

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Headless backend for cluster servers without X11/display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Alpha grid according to linear_probe_eval.py from the Orthrus paper
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Calculates regression metrics including correlations."""
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
            print(f"  {k:26s}: {v:.3e}")
        else:
            print(f"  {k:26s}: {v:.4f}")


def load_hIPSC_CM_npz(file_path: Path, target_col: str) -> dict:
    """Loads an NPZ file containing hIPSC_CM embeddings and metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    available_keys = list(data.keys())

    if "embeddings" not in data:
        raise KeyError(f"'embeddings' not found in NPZ. Available keys: {available_keys}")

    if target_col not in data:
        raise KeyError(
            f"Target variable '{target_col}' not found in NPZ archive. "
            f"Available keys: {available_keys}"
        )

    targets = data[target_col].astype(np.float32)

    # Gene column for group split (prefer hgnc_symbol, else ensembl_gene_id, else gene)
    if "hgnc_symbol" in data:
        genes = data["hgnc_symbol"].astype(str)
    elif "ensembl_gene_id" in data:
        genes = data["ensembl_gene_id"].astype(str)
    elif "genes" in data:
        genes = data["genes"].astype(str)
    else:
        genes = np.array([f"gene_{i}" for i in range(len(targets))])

    # Transcript IDs
    if "ensembl_transcript_id" in data:
        transcript_ids = data["ensembl_transcript_id"].astype(str)
    elif "transcript_id" in data:
        transcript_ids = data["transcript_id"].astype(str)
    elif "tx_ids" in data:
        transcript_ids = data["tx_ids"].astype(str)
    else:
        transcript_ids = np.array([f"tx_{i}" for i in range(len(targets))])

    return {
        "embeddings": data["embeddings"],
        "targets": targets.astype(np.float32),
        "genes": genes,
        "transcript_ids": transcript_ids,
        "seq_lens": data.get("seq_lens", None),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Ridge regression head on Orthrus hIPSC_CM embeddings"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus_6track_embeddings_hIPSC_CM.npz",
        help="Path to NPZ file containing hIPSC_CM embeddings",
    )
    parser.add_argument(
        "--splits_lookup_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv",
        help="Path to standardized 10-fold split lookup table CSV (from create_hipsc_cm_splits.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/Orthrus/hIPSC_CM",
        help="Output directory for models, metrics, plots, and predictions",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="half_life_transformed",
        choices=["half_life_transformed", "half_life", "rate"],
        help="Target variable to train on (default: half_life_transformed)",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["lookup", "gene", "random"],
        default="lookup",
        help="'lookup' (standardized 10-fold table with 4-fold CV and test 8,9), 'gene' (ad-hoc GroupShuffleSplit), or 'random'",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of test split if fallback to 'gene' or 'random' (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optional: create scatter plot (y_true vs. y_pred) as PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings_path)
    lookup_path = Path(args.splits_lookup_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("        Orthrus Ridge Regression: hIPSC_CM Evaluation Pipeline        ")
    print("=" * 75)
    print(f"Embeddings file:    {emb_path}")
    print(f"Target variable:    {args.target_col}")
    print(f"Split mechanism:    {args.split_type}")
    print(f"Random state:       {args.random_state}")
    print(f"Output directory:   {out_dir}")

    print("\nLoading embeddings and target variables...")
    data = load_hIPSC_CM_npz(emb_path, target_col=args.target_col)

    X = data["embeddings"]
    y = data["targets"]
    genes = data["genes"]
    transcript_ids = data["transcript_ids"]

    # Filter invalid target values (NaN / Inf)
    valid_mask = ~np.isnan(y) & ~np.isinf(y)
    if not np.all(valid_mask):
        num_invalid = int(np.sum(~valid_mask))
        print(f"[Data Cleaning] Removed {num_invalid} samples with NaN/Inf target values.")
        X = X[valid_mask]
        y = y[valid_mask]
        genes = genes[valid_mask]
        transcript_ids = transcript_ids[valid_mask]

    print(f"Valid samples:       {len(y)}")
    print(f"Feature dimension:   {X.shape[1]}")
    print(f"Unique genes:        {len(np.unique(genes))}")

    if len(y) == 0:
        raise ValueError(f"No valid data points found for target '{args.target_col}'.")

    # -------------------------------------------------------------------------
    # Split Strategy
    # -------------------------------------------------------------------------
    use_lookup = (args.split_type == "lookup" and lookup_path.exists())

    if args.split_type == "lookup" and not lookup_path.exists():
        print(f"\n[Warning] Splits lookup table not found at: {lookup_path}")
        print("Falling back to standard gene-grouped GroupShuffleSplit (test_size=0.2).")
        use_lookup = False

    if use_lookup:
        print(f"\nUsing Standardized 10-Fold Lookup Table: {lookup_path}")
        lookup_df = pd.read_csv(lookup_path)

        # Standardize matching IDs
        lookup_tx_col = "ensembl_transcript_id" if "ensembl_transcript_id" in lookup_df.columns else "transcript_id"
        tx_to_split = dict(zip(lookup_df[lookup_tx_col].astype(str).str.strip(), lookup_df["split"].astype(int)))

        # Assign split to each sample
        sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in transcript_ids])

        # Verify all transcripts matched
        unmatched_count = int(np.sum(sample_splits == -1))
        if unmatched_count > 0:
            print(f"[Warning] {unmatched_count} transcripts were not found in lookup table! Filtering them out.")
            matched_mask = sample_splits != -1
            X = X[matched_mask]
            y = y[matched_mask]
            genes = genes[matched_mask]
            transcript_ids = transcript_ids[matched_mask]
            sample_splits = sample_splits[matched_mask]

        # Folds 8 & 9 are reserved for Test Set (20%)
        test_mask = np.isin(sample_splits, [8, 9])
        cv_mask = np.isin(sample_splits, [0, 1, 2, 3, 4, 5, 6, 7])

        X_train_cv, y_train_cv = X[cv_mask], y[cv_mask]
        cv_splits = sample_splits[cv_mask]
        train_genes = genes[cv_mask]

        X_test, y_test = X[test_mask], y[test_mask]
        test_genes = genes[test_mask]
        test_tx = transcript_ids[test_mask]

        print(f"\n[Split Breakdown]")
        print(f"  CV Pool (Folds 0-7): {len(y_train_cv)} samples ({len(np.unique(train_genes))} unique genes)")
        print(f"  Test Set (Folds 8-9): {len(y_test)} samples ({len(np.unique(test_genes))} unique genes)")

        cv_fold_definitions = [
            {"name": "Fold 1", "train_splits": [0, 1, 2, 3, 4, 5], "val_splits": [6, 7]},
            {"name": "Fold 2", "train_splits": [2, 3, 4, 5, 6, 7], "val_splits": [0, 1]},
            {"name": "Fold 3", "train_splits": [0, 1, 4, 5, 6, 7], "val_splits": [2, 3]},
            {"name": "Fold 4", "train_splits": [0, 1, 2, 3, 6, 7], "val_splits": [4, 5]},
        ]

        custom_cv = []
        for fold_def in cv_fold_definitions:
            tr_idx = np.where(np.isin(cv_splits, fold_def["train_splits"]))[0]
            val_idx = np.where(np.isin(cv_splits, fold_def["val_splits"]))[0]
            custom_cv.append((tr_idx, val_idx))

        print(f"\nTraining RidgeCV across custom 4-Fold Cross-Validation (Alphas: {DEFAULT_ALPHAS})...")
        model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=custom_cv)
        model.fit(X_train_cv, y_train_cv)

        best_alpha = float(model.alpha_)
        print(f"\n>>> Selected Optimal Alpha: {best_alpha:.4e} <<<")

        # Evaluate individual CV folds with optimal alpha
        print("\n" + "-" * 65)
        print("          4-FOLD CROSS-VALIDATION DIAGNOSTICS (VAL FOLDS)         ")
        print("-" * 65)
        print(f"{'Fold':<10} | {'Train Folds':<18} | {'Val Folds':<12} | {'Val Pearson r':<14} | {'Val RMSE':<10}")
        print("-" * 65)

        fold_evaluations = []
        val_pearsons = []
        val_rmses = []
        val_spearmans = []

        for f_idx, fold_def in enumerate(cv_fold_definitions):
            tr_idx, val_idx = custom_cv[f_idx]
            fold_ridge = Ridge(alpha=best_alpha)
            fold_ridge.fit(X_train_cv[tr_idx], y_train_cv[tr_idx])
            val_pred = fold_ridge.predict(X_train_cv[val_idx])

            f_metrics = calculate_metrics(y_train_cv[val_idx], val_pred, prefix=f"fold_{f_idx+1}")
            fold_evaluations.append({
                "fold_name": fold_def["name"],
                "train_splits": fold_def["train_splits"],
                "val_splits": fold_def["val_splits"],
                "val_samples": int(len(val_idx)),
                **f_metrics,
            })
            r_val = f_metrics[f"fold_{f_idx+1}_pearson_r"]
            rmse_val = f_metrics[f"fold_{f_idx+1}_rmse"]
            s_val = f_metrics[f"fold_{f_idx+1}_spearman_rho"]
            val_pearsons.append(r_val)
            val_rmses.append(rmse_val)
            val_spearmans.append(s_val)

            tr_str = ",".join(map(str, fold_def["train_splits"]))
            val_str = ",".join(map(str, fold_def["val_splits"]))
            print(f"{fold_def['name']:<10} | {tr_str:<18} | {val_str:<12} | {r_val:>12.4f}  | {rmse_val:>8.4f}")

        print("-" * 65)
        mean_cv_r = float(np.mean(val_pearsons))
        std_cv_r = float(np.std(val_pearsons))
        mean_cv_rmse = float(np.mean(val_rmses))
        std_cv_rmse = float(np.std(val_rmses))
        print(f"{'Mean ± Std':<10} | {'-':<18} | {'-':<12} | {mean_cv_r:>7.4f} ± {std_cv_r:.4f} | {mean_cv_rmse:>6.4f} ± {std_cv_rmse:.4f}")
        print("-" * 65)

        # Train predictions on entire CV pool (0-7)
        y_train_pred = model.predict(X_train_cv)
        train_metrics = calculate_metrics(y_train_cv, y_train_pred, prefix="train_cv_pool")

    else:
        # Fallback to standard split
        if args.split_type == "gene":
            print("Performing ad-hoc gene-based split (GroupShuffleSplit)...")
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=genes))
        else:
            print("Performing random split (train_test_split)...")
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=args.test_size, random_state=args.random_state
            )

        X_train_cv, y_train_cv = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_genes, test_genes = genes[train_idx], genes[test_idx]
        test_tx = transcript_ids[test_idx]

        print(f"Training set: {len(y_train_cv)} samples ({len(np.unique(train_genes))} unique genes)")
        print(f"Test set:     {len(y_test)} samples ({len(np.unique(test_genes))} unique genes)")

        print(f"\nTraining RidgeCV with 5-fold CV over alphas {DEFAULT_ALPHAS}...")
        model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=5)
        model.fit(X_train_cv, y_train_cv)
        best_alpha = float(model.alpha_)
        print(f"Optimal alpha: {best_alpha}")

        fold_evaluations = []
        mean_cv_r, std_cv_r = 0.0, 0.0
        mean_cv_rmse, std_cv_rmse = 0.0, 0.0

        y_train_pred = model.predict(X_train_cv)
        train_metrics = calculate_metrics(y_train_cv, y_train_pred, prefix="train")

    # -------------------------------------------------------------------------
    # Test Set Evaluation (Unseen Splits 8 & 9)
    # -------------------------------------------------------------------------
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

    print_metrics(train_metrics, f"Training/CV Pool Metrics ({args.target_col})")
    print_metrics(test_metrics, f"Test Set Metrics (Folds 8 & 9) ({args.target_col})")

    # 1. Save trained model
    model_file = out_dir / f"ridge_model_hIPSC_CM_{args.target_col}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved to: {model_file}")

    # 2. Save predictions as CSV
    pred_df = pd.DataFrame({
        "transcript_id": test_tx,
        "gene": test_genes,
        "true_target": y_test,
        "predicted_target": y_test_pred,
        "residual": y_test - y_test_pred,
    })
    pred_file = out_dir / f"predictions_hIPSC_CM_{args.target_col}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    # 3. Save comprehensive metrics as JSON
    all_metrics = {
        "dataset": "hIPSC_CM",
        "target_col": args.target_col,
        "split_mechanism": "lookup_10folds" if use_lookup else args.split_type,
        "splits_lookup_path": str(lookup_path) if use_lookup else None,
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
    }
    metrics_file = out_dir / f"metrics_hIPSC_CM_{args.target_col}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

    # 4. Save scatter plot (y_true vs. y_pred)
    if args.plot:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
            p_r = test_metrics["test_pearson_r"]
            s_rho = test_metrics["test_spearman_rho"]
            r2 = test_metrics["test_r2"]

            emb_type = "8-Track" if X.shape[1] == 512 else "6-Track"
            ax.set_title(
                f"Orthrus {emb_type} -> hIPSC_CM ({args.target_col})\n"
                f"Test Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | R² = {r2:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel(f"True Value ({args.target_col})", fontsize=10)
            ax.set_ylabel(f"Predicted Value ({args.target_col})", fontsize=10)

            # Diagonal reference line (Ideal)
            min_val = min(float(np.min(y_test)), float(np.min(y_test_pred)))
            max_val = max(float(np.max(y_test)), float(np.max(y_test_pred)))
            margin = (max_val - min_val) * 0.05
            ax.plot(
                [min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                "r--",
                linewidth=1.5,
                label="Ideal (y=x)",
            )
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plot_file = out_dir / f"scatter_hIPSC_CM_{args.target_col}.png"
            fig.savefig(plot_file, dpi=300)
            plt.close(fig)
            print(f"Scatter plot saved to: {plot_file}")
        except Exception as e:
            print(f"Notice: Plot could not be created: {e}")

    print("\nEvaluation successfully completed!")


if __name__ == "__main__":
    main()
