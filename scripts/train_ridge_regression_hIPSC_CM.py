#!/usr/bin/env python3
"""
Ridge regression training & evaluation on extracted Orthrus 6-track embeddings for the hIPSC_CM dataset.
Supports 'half_life_transformed' as well as 'half_life' and 'rate'.
Runs on cluster (CPU or GPU node).
"""

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
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
            print(f"  {k:24s}: {v:.3e}")
        else:
            print(f"  {k:24s}: {v:.4f}")


def load_hIPSC_CM_npz(file_path: Path, target_col: str) -> dict:
    """Loads an NPZ file containing hIPSC_CM embeddings and metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    available_keys = list(data.keys())

    if "embeddings" not in data:
        raise KeyError(f"'embeddings' not found in NPZ. Available keys: {available_keys}")

    # Select target variable
    if target_col not in data:
        raise KeyError(
            f"Target variable '{target_col}' not found in NPZ archive. "
            f"Available keys: {available_keys}"
        )

    targets = data[target_col].astype(np.float32)

    # Gene column for group split (prefer hgnc_symbol, else ensembl_gene_id)
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
        choices=["gene", "random"],
        default="gene",
        help="'gene' (GroupShuffleSplit against leakage of isoforms from the same gene) or 'random'",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of test split (default: 0.2)",
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      Orthrus 6-Track Ridge Regression: hIPSC_CM Evaluation      ")
    print("=" * 70)
    print(f"Embeddings file:  {emb_path}")
    print(f"Target variable:  {args.target_col}")
    print(f"Split type:       {args.split_type}")
    print(f"Test size:        {args.test_size}")
    print(f"Random state:     {args.random_state}")
    print(f"Output directory: {out_dir}")

    print("\nLoading embeddings and target variables...")
    data = load_hIPSC_CM_npz(emb_path, target_col=args.target_col)

    X = data["embeddings"]
    y = data["targets"]
    genes = data["genes"]
    transcript_ids = data["transcript_ids"]

    # Check for NaN values in target
    valid_mask = ~np.isnan(y)
    if not np.all(valid_mask):
        num_invalid = np.sum(~valid_mask)
        print(f"Warning: {num_invalid} samples with NaN in target were removed.")
        X = X[valid_mask]
        y = y[valid_mask]
        genes = genes[valid_mask]
        transcript_ids = transcript_ids[valid_mask]

    print(f"Valid samples: {len(y)}, Feature dimension: {X.shape[1]}")
    print(f"Number of unique genes: {len(np.unique(genes))}")

    if len(y) == 0:
        raise ValueError(
            f"No valid data points found for target variable '{args.target_col}'. "
            f"All values are NaN! Please check the embedding file or select --target_col half_life."
        )

    # Perform split
    if args.split_type == "gene":
        print("Performing gene-based split (GroupShuffleSplit)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=genes))
    else:
        print("Performing random split (train_test_split)...")
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size, random_state=args.random_state
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_genes, test_genes = genes[train_idx], genes[test_idx]
    test_tx = transcript_ids[test_idx]

    print(f"Training set: {len(y_train)} samples ({len(np.unique(train_genes))} unique genes)")
    print(f"Test set:     {len(y_test)} samples ({len(np.unique(test_genes))} unique genes)")

    # Train RidgeCV
    print(f"\nTraining RidgeCV with 5-fold CV over alphas {DEFAULT_ALPHAS}...")
    model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=5)
    model.fit(X_train, y_train)

    print(f"Optimal alpha: {model.alpha_}")

    # Compute predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="train")
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

    print_metrics(train_metrics, f"Training Metrics ({args.target_col})")
    print_metrics(test_metrics, f"Test Metrics ({args.target_col})")

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

    # 3. Save metrics as JSON
    all_metrics = {
        "dataset": "hIPSC_CM",
        "target_col": args.target_col,
        "split_type": args.split_type,
        "best_alpha": float(model.alpha_),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_unique_genes": int(len(np.unique(train_genes))),
        "test_unique_genes": int(len(np.unique(test_genes))),
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
            ax.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="royalblue", edgecolors="none")
            p_r = test_metrics["test_pearson_r"]
            s_rho = test_metrics["test_spearman_rho"]
            r2 = test_metrics["test_r2"]

            ax.set_title(
                f"Orthrus 6-Track -> hIPSC_CM ({args.target_col})\n"
                f"Pearson R = {p_r:.3f} | Spearman Rho = {s_rho:.3f} | R² = {r2:.3f}",
                fontsize=11,
            )
            ax.set_xlabel(f"True Value ({args.target_col})")
            ax.set_ylabel(f"Predicted Value ({args.target_col})")

            # Diagonal reference line (Ideal)
            min_val = min(float(np.min(y_test)), float(np.min(y_test_pred)))
            max_val = max(float(np.max(y_test)), float(np.max(y_test_pred)))
            margin = (max_val - min_val) * 0.05
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], "r--", linewidth=1.5, label="Ideal (y=x)")
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
