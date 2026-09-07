#!/usr/bin/env python3
"""
Ridge-Regression Training & Evaluierung auf extrahierten Orthrus 6-Track Embeddings fuer den Saluki-Datensatz.
Unterstuetzt sowohl 'half_life_transformed' als auch 'half_life' und 'rate'.
Laeuft auf dem Cluster (CPU oder GPU-Node).
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

# Headless Backend fuer Cluster-Server ohne X11/Display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Alpha-Raster gemaess linear_probe_eval.py aus dem Orthrus-Paper
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Berechnet Regressions-Metriken inklusive Korrelationen."""
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
    """Formatierte Konsolenausgabe fuer Metriken."""
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        if "pvalue" in k:
            print(f"  {k:24s}: {v:.3e}")
        else:
            print(f"  {k:24s}: {v:.4f}")


def load_saluki_npz(file_path: Path, target_col: str) -> dict:
    """Laedt eine NPZ-Datei mit Saluki-Embeddings und Metadaten."""
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding-Datei nicht gefunden: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    available_keys = list(data.keys())

    if "embeddings" not in data:
        raise KeyError(f"'embeddings' nicht in NPZ gefunden. Vorhandene Keys: {available_keys}")

    # Zielvariable auswaehlen
    if target_col in data:
        targets = data[target_col]
    elif target_col == "half_life_transformed" and "targets" in data:
        targets = data["targets"]
    elif target_col == "half_life" and "targets" in data:
        targets = data["targets"]
    else:
        raise KeyError(
            f"Zielvariable '{target_col}' nicht im NPZ-Archiv vorhanden. "
            f"Verfuegbare Schluessel: {available_keys}"
        )

    # Gene-Spalte fuer Group-Split (bevorzugt hgnc_symbol, sonst ensembl_gene_id)
    if "hgnc_symbol" in data:
        genes = data["hgnc_symbol"].astype(str)
    elif "ensembl_gene_id" in data:
        genes = data["ensembl_gene_id"].astype(str)
    elif "genes" in data:
        genes = data["genes"].astype(str)
    else:
        genes = np.array([f"gene_{i}" for i in range(len(targets))])

    # Transkript-IDs
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
        description="Ridge Regression Head auf Orthrus Saluki-Embeddings trainieren und evaluieren"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/saluki/orthrus_6track_embeddings_saluki.npz",
        help="Pfad zur NPZ-Datei mit Saluki-Embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/Orthrus/saluki",
        help="Ausgabeverzeichnis fuer Modelle, Metriken, Plots und Vorhersagen",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="half_life_transformed",
        choices=["half_life_transformed", "half_life", "rate"],
        help="Zu lernende Zielvariable (Standard: half_life_transformed)",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["gene", "random"],
        default="gene",
        help="'gene' (GroupShuffleSplit gegen Leakage von Isoformen desselben Gens) oder 'random'",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Anteil des Test-Splits (Standard: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Zufallssamen fuer Reproduzierbarkeit",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optional: Erstelle Streudiagramm (y_true vs. y_pred) als PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    emb_path = Path(args.embeddings_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      Orthrus 6-Track Ridge Regression: Saluki Evaluierung      ")
    print("=" * 70)
    print(f"Embeddings-Datei: {emb_path}")
    print(f"Zielvariable:     {args.target_col}")
    print(f"Split-Typ:        {args.split_type}")
    print(f"Test-Groesse:     {args.test_size}")
    print(f"Random State:     {args.random_state}")
    print(f"Ausgabeordner:    {out_dir}")

    print(f"\nLade Embeddings und Zielvariablen...")
    data = load_saluki_npz(emb_path, target_col=args.target_col)

    X = data["embeddings"]
    y = data["targets"]
    genes = data["genes"]
    transcript_ids = data["transcript_ids"]

    # Pruefen auf NaN-Werte im Target
    valid_mask = ~np.isnan(y)
    if not np.all(valid_mask):
        num_invalid = np.sum(~valid_mask)
        print(f"Achtung: {num_invalid} Proben mit NaN im Target wurden entfernt.")
        X = X[valid_mask]
        y = y[valid_mask]
        genes = genes[valid_mask]
        transcript_ids = transcript_ids[valid_mask]

    print(f"Gueltige Proben: {len(y)}, Feature-Dimension: {X.shape[1]}")
    print(f"Anzahl eindeutiger Gene: {len(np.unique(genes))}")

    # Split durchfuehren
    if args.split_type == "gene":
        print("Fuehre gen-basierten Split (GroupShuffleSplit) durch...")
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=genes))
    else:
        print("Fuehre zufaelligen Split (train_test_split) durch...")
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size, random_state=args.random_state
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_genes, test_genes = genes[train_idx], genes[test_idx]
    test_tx = transcript_ids[test_idx]

    print(f"Trainings-Set: {len(y_train)} Proben ({len(np.unique(train_genes))} unique Gene)")
    print(f"Test-Set:      {len(y_test)} Proben ({len(np.unique(test_genes))} unique Gene)")

    # RidgeCV trainieren
    print(f"\nTrainiere RidgeCV mit 5-Fold CV ueber Alphas {DEFAULT_ALPHAS}...")
    model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=5)
    model.fit(X_train, y_train)

    print(f"Optimales Alpha: {model.alpha_}")

    # Vorhersagen berechnen
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, y_train_pred, prefix="train")
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

    print_metrics(train_metrics, f"Trainings-Metriken ({args.target_col})")
    print_metrics(test_metrics, f"Test-Metriken ({args.target_col})")

    # 1. Speichern des trainierten Modells
    model_file = out_dir / f"ridge_model_saluki_{args.target_col}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModell gespeichert unter: {model_file}")

    # 2. Speichern der Vorhersagen als CSV
    pred_df = pd.DataFrame({
        "transcript_id": test_tx,
        "gene": test_genes,
        "true_target": y_test,
        "predicted_target": y_test_pred,
        "residual": y_test - y_test_pred,
    })
    pred_file = out_dir / f"predictions_saluki_{args.target_col}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Vorhersagen gespeichert unter: {pred_file}")

    # 3. Speichern der Metriken als JSON
    all_metrics = {
        "dataset": "saluki",
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
    metrics_file = out_dir / f"metrics_saluki_{args.target_col}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metriken gespeichert unter: {metrics_file}")

    # 4. Streudiagramm (y_true vs. y_pred) speichern
    if args.plot:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="royalblue", edgecolors="none")
            p_r = test_metrics["test_pearson_r"]
            s_rho = test_metrics["test_spearman_rho"]
            r2 = test_metrics["test_r2"]

            ax.set_title(
                f"Orthrus 6-Track -> Saluki ({args.target_col})\n"
                f"Pearson R = {p_r:.3f} | Spearman Rho = {s_rho:.3f} | R² = {r2:.3f}",
                fontsize=11,
            )
            ax.set_xlabel(f"Wahrer Wert ({args.target_col})")
            ax.set_ylabel(f"Vorhergesagter Wert ({args.target_col})")

            # Diagonale Referenzlinie (Ideal)
            min_val = min(float(np.min(y_test)), float(np.min(y_test_pred)))
            max_val = max(float(np.max(y_test)), float(np.max(y_test_pred)))
            margin = (max_val - min_val) * 0.05
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], "r--", linewidth=1.5, label="Ideal (y=x)")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plot_file = out_dir / f"scatter_saluki_{args.target_col}.png"
            fig.savefig(plot_file, dpi=300)
            plt.close(fig)
            print(f"Streudiagramm gespeichert unter: {plot_file}")
        except Exception as e:
            print(f"Hinweis: Plot konnte nicht erstellt werden: {e}")

    print("\nEvaluierung erfolgreich abgeschlossen!")


if __name__ == "__main__":
    main()
