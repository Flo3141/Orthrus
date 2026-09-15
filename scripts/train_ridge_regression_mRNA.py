#!/usr/bin/env python3
"""
Ridge regression training & evaluation on extracted Orthrus 6-track embeddings.
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


# Alpha grid according to linear_probe_eval.py from the Orthrus paper
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Calculates regression metrics including correlations."""
    p_corr, p_val = pearsonr(y_true, y_pred)
    s_corr, s_val = spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    prefix_str = f"{prefix}_" if prefix else ""
    return {
        f"{prefix_str}pearson_r": float(p_corr),
        f"{prefix_str}pearson_pvalue": float(p_val),
        f"{prefix_str}spearman_rho": float(s_corr),
        f"{prefix_str}spearman_pvalue": float(s_val),
        f"{prefix_str}mse": float(mse),
        f"{prefix_str}rmse": float(rmse),
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


def load_npz(file_path: Path) -> dict:
    """Loads an NPZ file containing embeddings and metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "targets": data["targets"],
        "genes": data["genes"],
        "chromosomes": data["chromosomes"],
        "seq_lens": data["seq_lens"]
    }


def resolve_embedding_file(species: str, embeddings_dir: Path | None, data_dir: Path | None) -> Path:
    """Finds the path to the embedding file based on data_dir or embeddings_dir."""
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir / f"rnahl-{species}" / "embeddings" / "orthrus_6track_embeddings.npz")
        candidates.append(data_dir / f"orthrus_6track_embeddings_{species}.npz")
    if embeddings_dir is not None:
        candidates.append(embeddings_dir / f"rnahl-{species}" / "embeddings" / "orthrus_6track_embeddings.npz")
        candidates.append(embeddings_dir / f"orthrus_6track_embeddings_{species}.npz")

    for c in candidates:
        if c.exists():
            return c
            
    # Default if not yet existing (for error message)
    return candidates[0] if candidates else Path(f"orthrus_6track_embeddings_{species}.npz")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Ridge regression on Orthrus embeddings")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench",
        help="Path to mrna_bench data directory (looks in <data_dir>/rnahl-<species>/embeddings/)"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Optional alternative path to a central embeddings directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/Orthrus/mRNABench",
        help="Output directory for models, metrics, and predictions"
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["human", "mouse", "cross_species"],
        default="human",
        help="Training mode: 'human', 'mouse', or 'cross_species' (Train: Human, Test: Mouse)"
    )
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["gene", "random"],
        default="gene",
        help="'gene' (GroupShuffleSplit by gene against leakage of isoforms) or 'random' (random split)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of test split (default: 0.2)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optional: create scatter plot (y_true vs. y_pred) as PNG"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("        Orthrus 6-Track Ridge Regression Evaluation              ")
    print("=================================================================")
    print(f"Mode:         {args.species}")
    print(f"Split type:   {args.split_type}")
    print(f"Test size:    {args.test_size}")
    print(f"Random state: {args.random_state}")

    if args.species in ["human", "mouse"]:
        file_path = resolve_embedding_file(args.species, emb_dir, data_dir)
        print(f"\nLoading data: {file_path}")
        data = load_npz(file_path)

        X = data["embeddings"]
        y = data["targets"]
        genes = data["genes"]

        print(f"Entries: {len(y)}, Feature dimension: {X.shape[1]}")

        # Perform split
        if args.split_type == "gene":
            print("Performing gene-based split (GroupShuffleSplit)...")
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=genes))
        else:
            print("Performing random split...")
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=args.test_size, random_state=args.random_state
            )

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        test_genes = genes[test_idx]

        print(f"Training set: {len(y_train)} samples")
        print(f"Test set:     {len(y_test)} samples")

        # Fit RidgeCV
        print(f"\nTraining RidgeCV with 5-fold cross-validation over alphas {DEFAULT_ALPHAS}...")
        model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=5)
        model.fit(X_train, y_train)

        print(f"Optimal alpha: {model.alpha_}")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_train_pred, prefix="train")
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

        print_metrics(train_metrics, f"Training Metrics ({args.species})")
        print_metrics(test_metrics, f"Test Metrics ({args.species})")

        # Save model
        model_file = out_dir / f"ridge_model_{args.species}.joblib"
        joblib.dump(model, model_file)
        print(f"\nModel saved to: {model_file}")

        # Save predictions
        pred_df = pd.DataFrame({
            "gene": test_genes,
            "true_target": y_test,
            "predicted_target": y_test_pred
        })
        pred_file = out_dir / f"predictions_{args.species}.csv"
        pred_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")

        # Save metrics as JSON
        all_metrics = {
            "species": args.species,
            "split_type": args.split_type,
            "best_alpha": float(model.alpha_),
            "train_size": len(y_train),
            "test_size": len(y_test),
            **train_metrics,
            **test_metrics
        }
        metrics_file = out_dir / f"metrics_{args.species}.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")

    elif args.species == "cross_species":
        # Train on Human, test on Mouse
        human_file = resolve_embedding_file("human", emb_dir, data_dir)
        mouse_file = resolve_embedding_file("mouse", emb_dir, data_dir)
        
        print(f"\nLoading human data: {human_file}")
        human_data = load_npz(human_file)
        print(f"Loading mouse data: {mouse_file}")
        mouse_data = load_npz(mouse_file)

        X_train, y_train = human_data["embeddings"], human_data["targets"]
        X_test, y_test = mouse_data["embeddings"], mouse_data["targets"]
        test_genes = mouse_data["genes"]

        print(f"Training set (Human): {len(y_train)} samples")
        print(f"Test set (Mouse):      {len(y_test)} samples")

        print(f"\nTraining RidgeCV on human...")
        model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=5)
        model.fit(X_train, y_train)

        print(f"Optimal alpha: {model.alpha_}")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_train_pred, prefix="human_train")
        test_metrics = calculate_metrics(y_test, y_test_pred, prefix="mouse_test")

        print_metrics(train_metrics, "Human (In-Domain Train)")
        print_metrics(test_metrics, "Mouse (Cross-Species Test)")

        model_file = out_dir / "ridge_model_cross_species_h2m.joblib"
        joblib.dump(model, model_file)
        print(f"\nModel saved to: {model_file}")

        pred_df = pd.DataFrame({
            "gene": test_genes,
            "true_target": y_test,
            "predicted_target": y_test_pred
        })
        pred_file = out_dir / "predictions_cross_species_h2m.csv"
        pred_df.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")

        all_metrics = {
            "mode": "cross_species_human_to_mouse",
            "best_alpha": float(model.alpha_),
            "train_size_human": len(y_train),
            "test_size_mouse": len(y_test),
            **train_metrics,
            **test_metrics
        }
        metrics_file = out_dir / "metrics_cross_species_h2m.json"
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.3, s=15, color="teal")
            p_val = test_metrics.get("test_pearson_r") or test_metrics.get("mouse_test_pearson_r")
            ax.set_title(f"Ridge Regression Test: {args.species} (Pearson R = {p_val:.3f})")
            ax.set_xlabel("True Target Value (PC1 Half-Life)")
            ax.set_ylabel("Predicted Value")
            
            # Diagonal reference line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, "r--", alpha=0.7, label="Ideal")
            ax.legend()
            plt.tight_layout()
            plot_file = out_dir / f"scatter_{args.species}.png"
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Scatter plot saved to: {plot_file}")
        except Exception as e:
            print(f"Plot could not be created: {e}")

    print("\nEvaluation successfully completed!")


if __name__ == "__main__":
    main()
