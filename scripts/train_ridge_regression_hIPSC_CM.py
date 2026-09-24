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


def load_transformation_params(params_path: Path) -> tuple:
    """
    Loads mu_log, sigma_log, and pseudocount from transformation_params.json.
    Raises FileNotFoundError if the file does not exist, and KeyError if mu_log or sigma_log are missing.
    """
    if not params_path.exists():
        raise FileNotFoundError(f"Transformation parameter file not found: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    if "mu_log" not in params:
        raise KeyError(f"Key 'mu_log' is missing in transformation parameter file: {params_path}")
    if "sigma_log" not in params:
        raise KeyError(f"Key 'sigma_log' is missing in transformation parameter file: {params_path}")

    mu_log = float(params["mu_log"])
    sigma_log = float(params["sigma_log"])
    pseudocount = float(params.get("pseudocount", 0.1))

    return mu_log, sigma_log, pseudocount, params_path


def inverse_transform_half_life(
    y_transformed: np.ndarray,
    mu: float,
    sigma: float,
    pseudocount: float = 0.1,
) -> np.ndarray:
    """
    Inverts the log + z-score transformation:
      y_log = (y_transformed * sigma) + mu
      y_raw = exp(y_log) - pseudocount
    Clips output to >= 0 to avoid negative physical half-lives.
    """
    y_log = (y_transformed * sigma) + mu
    y_raw = np.exp(y_log) - pseudocount
    return np.clip(y_raw, a_min=0.0, a_max=None)


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

    normalization = None
    if "normalization" in data:
        try:
            norm_val = str(data["normalization"]).strip().lower()
            if norm_val and norm_val != "none_not_set":
                normalization = norm_val
        except Exception:
            pass

    is_finetuned = None
    if "is_finetuned" in data:
        try:
            is_finetuned = bool(data["is_finetuned"])
        except Exception:
            pass

    raw_half_life = None
    if "half_life" in data:
        raw_half_life = data["half_life"].astype(np.float32)

    is_4fold_cv = bool(data.get("is_4fold_cv", False)) or ("fold_assignment" in data)

    return {
        "embeddings": data["embeddings"],
        "targets": targets.astype(np.float32),
        "genes": genes,
        "transcript_ids": transcript_ids,
        "seq_lens": data.get("seq_lens", None),
        "normalization": normalization,
        "is_finetuned": is_finetuned,
        "is_4fold_cv": is_4fold_cv,
        "archive_keys": available_keys,
        "raw_half_life": raw_half_life,
    }


def resolve_track_and_normalization(
    data: dict,
    file_path: Path,
    user_track_type: str = "auto",
    user_normalization: str = "auto",
) -> tuple:
    """
    Infers the track type ('6track' vs '8track') and normalization scheme.
    Returns: (track_type, normalization)
    """
    path_str = str(file_path).lower()
    archive_keys = data.get("archive_keys", [])

    # 1. Resolve track type
    if user_track_type != "auto":
        track_type = user_track_type.lower()
    else:
        if "8track" in path_str or "8_track" in path_str or "8-track" in path_str:
            track_type = "8track"
        elif "6track" in path_str or "6_track" in path_str or "6-track" in path_str:
            track_type = "6track"
        elif "has_mirna" in archive_keys or "has_eclip" in archive_keys:
            track_type = "8track"
        elif data.get("normalization") is not None:
            track_type = "8track"
        else:
            track_type = "6track"

    # 2. Resolve normalization (for 8track)
    if user_normalization != "auto":
        normalization = user_normalization.lower()
    else:
        norm_from_data = data.get("normalization")
        if norm_from_data and norm_from_data not in ["none", "nan", "none_not_set"]:
            normalization = norm_from_data
        elif "minmax" in path_str or "min_max" in path_str:
            normalization = "minmax"
        elif "log" in path_str:
            normalization = "log"
        elif norm_from_data == "none" or "none" in path_str:
            normalization = "none"
        else:
            normalization = "none"

    return track_type, normalization


def resolve_output_dir(
    base_output_dir: Path,
    track_type: str,
    normalization: str,
    is_finetuned: bool = False,
) -> Path:
    """
    Constructs the structured output folder:
    - 6-track:  <base_output_dir>/6track/pretrained or <base_output_dir>/6track/finetuned
    - 8-track:  <base_output_dir>/8track/<normalization>
    Avoids redundant nesting if the user already passed the full subfolder path.
    """
    norm_clean = normalization.lower().strip() if normalization else "none"

    if track_type == "6track":
        variant = "finetuned" if is_finetuned else "pretrained"
        if (
            base_output_dir.name.lower() == variant
            and base_output_dir.parent.name.lower() == "6track"
        ):
            target_dir = base_output_dir
        elif base_output_dir.name.lower() == "6track":
            target_dir = base_output_dir / variant
        else:
            target_dir = base_output_dir / "6track" / variant
    else:  # 8track
        if (
            base_output_dir.name.lower() == norm_clean
            and base_output_dir.parent.name.lower() == "8track"
        ):
            target_dir = base_output_dir
        elif base_output_dir.name.lower() == "8track":
            target_dir = base_output_dir / norm_clean
        else:
            target_dir = base_output_dir / "8track" / norm_clean

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Ridge regression head on Orthrus hIPSC_CM embeddings"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/orthrus_6track_embeddings_hIPSC_CM.npz",
        help="Path to NPZ file containing hIPSC_CM embeddings",
    )
    parser.add_argument(
        "--track_type",
        type=str,
        choices=["auto", "6track", "8track"],
        default="auto",
        help="Track representation type ('6track', '8track', or 'auto' to infer from embeddings file)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="auto",
        help="Normalization method for 8-track ('none', 'minmax', 'log', or 'auto' to infer from file)",
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
        help="Base output directory (subfolders '6track' or '8track/<normalization>' will be created automatically)",
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
        choices=["lookup"],
        default="lookup",
        help=argparse.SUPPRESS,  # Deprecated: standardized 10-fold lookup table is always used
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--is_finetuned",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether embeddings stem from a fine-tuned model ('auto', 'true', or 'false')",
    )
    parser.add_argument(
        "--transformation_params_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/transformation_params.json",
        help="Path to transformation_params.json containing mu_log and sigma_log (from transform_hIPSC_CM_dataset.py)",
    )
    parser.add_argument(
        "--evaluation_scheme",
        type=str,
        choices=["auto", "4fold_cv", "single_split"],
        default="auto",
        help="Evaluation scheme for standardized lookup table: '4fold_cv' (4-fold CV on splits 0-7, test 8,9), 'single_split' (legacy single split train 0-5, val 6-7, test 8,9), or 'auto' (4fold_cv for 4-fold CV embeddings and base model). Default: auto",
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
    base_out_dir = Path(args.output_dir)

    print("\nLoading embeddings and target variables...")
    data = load_hIPSC_CM_npz(emb_path, target_col=args.target_col)

    track_type, normalization = resolve_track_and_normalization(
        data=data,
        file_path=emb_path,
        user_track_type=args.track_type,
        user_normalization=args.normalization,
    )

    if args.is_finetuned == "true":
        is_finetuned = True
    elif args.is_finetuned == "false":
        is_finetuned = False
    else:
        is_finetuned = (
            bool(data.get("is_finetuned"))
            if data.get("is_finetuned") is not None
            else ("finetun" in str(emb_path).lower())
        )

    out_dir = resolve_output_dir(
        base_out_dir,
        track_type=track_type,
        normalization=normalization,
        is_finetuned=is_finetuned,
    )

    print("=" * 75)
    print("        Orthrus Ridge Regression: hIPSC_CM Evaluation Pipeline        ")
    print("=" * 75)
    print(f"Embeddings file:    {emb_path}")
    print(f"Track type:         {track_type.upper()}")
    if track_type == "8track":
        print(f"Normalization:      {normalization.upper()}")
    elif track_type == "6track":
        print(f"Model variant:      {'Fine-Tuned' if is_finetuned else 'Pretrained Base'}")
    print(f"Target variable:    {args.target_col}")
    print("Split mechanism:    Standardized 10-Fold Lookup Table (Strict)")
    print(f"Random state:       {args.random_state}")
    print(f"Output directory:   {out_dir}")

    X = data["embeddings"]
    y = data["targets"]
    genes = data["genes"]
    transcript_ids = data["transcript_ids"]
    raw_half_life = data.get("raw_half_life")

    # Filter invalid target values (NaN / Inf)
    valid_mask = ~np.isnan(y) & ~np.isinf(y)
    if not np.all(valid_mask):
        num_invalid = int(np.sum(~valid_mask))
        print(f"[Data Cleaning] Removed {num_invalid} samples with NaN/Inf target values.")
        X = X[valid_mask]
        y = y[valid_mask]
        genes = genes[valid_mask]
        transcript_ids = transcript_ids[valid_mask]
        if raw_half_life is not None:
            raw_half_life = raw_half_life[valid_mask]

    print(f"Valid samples:       {len(y)}")
    print(f"Feature dimension:   {X.shape[1]}")
    print(f"Unique genes:        {len(np.unique(genes))}")

    if len(y) == 0:
        raise ValueError(f"No valid data points found for target '{args.target_col}'.")

    # -------------------------------------------------------------------------
    # Split Strategy: Standardized 10-Fold Lookup Table (Strict, No Fallback)
    # -------------------------------------------------------------------------
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"[Error] Standardized splits lookup table not found at: {lookup_path}! "
            f"A valid lookup table is strictly required."
        )

    print(f"\nUsing Standardized 10-Fold Lookup Table: {lookup_path}")
    lookup_df = pd.read_csv(lookup_path)

    lookup_tx_col = "ensembl_transcript_id" if "ensembl_transcript_id" in lookup_df.columns else "transcript_id"
    if lookup_tx_col not in lookup_df.columns:
        raise KeyError(
            f"[Error] Transcript ID column ('ensembl_transcript_id' or 'transcript_id') not found in lookup table {lookup_path}. "
            f"Columns: {list(lookup_df.columns)}"
        )
    if "split" not in lookup_df.columns:
        raise KeyError(
            f"[Error] 'split' column not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}"
        )

    available_splits = set(lookup_df["split"].dropna().astype(int).unique())
    expected_splits = set(range(10))
    missing_splits = expected_splits - available_splits
    if missing_splits:
        raise ValueError(
            f"[Error] Missing folds in lookup table {lookup_path}! Expected all 10 splits (0-9), but missing: {sorted(missing_splits)}."
        )

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
        if raw_half_life is not None:
            raw_half_life = raw_half_life[matched_mask]

    # Folds 8 & 9 are reserved for Test Set (20%)
    test_mask = np.isin(sample_splits, [8, 9])
    X_test, y_test = X[test_mask], y[test_mask]
    test_genes = genes[test_mask]
    test_tx = transcript_ids[test_mask]
    y_test_raw_true = raw_half_life[test_mask] if raw_half_life is not None else None

    is_4fold_cv = bool(data.get("is_4fold_cv", False)) or ("fold_assignment" in data.get("archive_keys", []))
    if args.evaluation_scheme == "4fold_cv":
        run_4fold_cv = True
    elif args.evaluation_scheme == "single_split":
        run_4fold_cv = False
    else:
        # 'auto': 4-fold CV for 4-fold out-of-fold embeddings and base models
        run_4fold_cv = is_4fold_cv or (not is_finetuned)

    if not run_4fold_cv:
        # -----------------------------------------------------------------
        # Legacy Single Fine-Tuned Model Pipeline: Train [0..5], Val [6..7], Test [8..9]
        # -----------------------------------------------------------------
        train_mask = np.isin(sample_splits, [0, 1, 2, 3, 4, 5])
        val_mask = np.isin(sample_splits, [6, 7])

        X_train, y_train = X[train_mask], y[train_mask]
        train_genes = genes[train_mask]

        X_val, y_val = X[val_mask], y[val_mask]
        val_genes = genes[val_mask]

        print(f"\n[Split Breakdown - Legacy Single-Model Protocol]")
        print(f"  Training Set   (Splits 0-5): {len(y_train)} samples ({len(np.unique(train_genes))} unique genes)")
        print(f"  Validation Set (Splits 6-7): {len(y_val)} samples ({len(np.unique(val_genes))} unique genes)")
        print(f"  Test Set       (Splits 8-9): {len(y_test)} samples ({len(np.unique(test_genes))} unique genes)")

        print(f"\nTuning Ridge alpha on Validation Set [6, 7] (Candidates: {DEFAULT_ALPHAS})...")
        print("-" * 70)
        print(f"{'Alpha':<12} | {'Val Pearson r':<16} | {'Val Spearman rho':<18} | {'Val RMSE':<10}")
        print("-" * 70)

        alpha_evaluations = []
        best_alpha = None
        best_val_r = -float("inf")
        best_model = None

        for alpha_cand in DEFAULT_ALPHAS:
            cand_ridge = Ridge(alpha=alpha_cand)
            cand_ridge.fit(X_train, y_train)
            val_pred_cand = cand_ridge.predict(X_val)

            cand_metrics = calculate_metrics(y_val, val_pred_cand, prefix=f"alpha_{alpha_cand}")
            cand_r = cand_metrics[f"alpha_{alpha_cand}_pearson_r"]
            cand_rho = cand_metrics[f"alpha_{alpha_cand}_spearman_rho"]
            cand_rmse = cand_metrics[f"alpha_{alpha_cand}_rmse"]

            alpha_evaluations.append({
                "alpha": float(alpha_cand),
                "val_pearson_r": cand_r,
                "val_spearman_rho": cand_rho,
                "val_rmse": cand_rmse,
            })
            print(f"{alpha_cand:<12.4e} | {cand_r:>14.4f}   | {cand_rho:>16.4f}   | {cand_rmse:>8.4f}")

            if cand_r > best_val_r:
                best_val_r = cand_r
                best_alpha = float(alpha_cand)
                best_model = cand_ridge

        print("-" * 70)
        print(f"\n>>> Selected Optimal Alpha: {best_alpha:.4e} (Validation Pearson r = {best_val_r:.4f}) <<<")

        model = best_model

        # Train and Val predictions
        y_train_pred = model.predict(X_train)
        train_metrics = calculate_metrics(y_train, y_train_pred, prefix="train")

        y_val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val, y_val_pred, prefix="val")

        X_train_cv = X_train
        y_train_cv = y_train
        fold_evaluations = alpha_evaluations
        mean_cv_r, std_cv_r = float(best_val_r), 0.0
        mean_cv_rmse, std_cv_rmse = float(val_metrics["val_rmse"]), 0.0

    else:
        # -----------------------------------------------------------------
        # 4-Fold Cross-Validation Pipeline across [0..7], Holdout Test [8..9]
        # -----------------------------------------------------------------
        cv_mask = np.isin(sample_splits, [0, 1, 2, 3, 4, 5, 6, 7])
        X_train_cv, y_train_cv = X[cv_mask], y[cv_mask]
        cv_splits = sample_splits[cv_mask]
        train_genes = genes[cv_mask]

        print(f"\n[Split Breakdown - 4-Fold Cross-Validation Protocol]")
        print(f"  CV Pool (Folds 0-7): {len(y_train_cv)} samples ({len(np.unique(train_genes))} unique genes)")
        print(f"  Test Set (Folds 8-9): {len(y_test)} samples ({len(np.unique(test_genes))} unique genes)")

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
            s_val = f_metrics[f"fold_{f_idx}_spearman_rho"]
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
        val_metrics = {}

    # -------------------------------------------------------------------------
    # Test Set Evaluation (Unseen Splits 8 & 9)
    # -------------------------------------------------------------------------
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

    # Inverse transformation for half_life_transformed
    test_raw_metrics = {}
    y_test_pred_raw = None
    trans_info = None

    if args.target_col == "half_life_transformed":
        params_path = Path(args.transformation_params_path)
        mu_log, sigma_log, pseudocount, resolved_params_path = load_transformation_params(params_path)
        print(f"\n[Transformation Params Loaded from: {resolved_params_path}]")
        print(f"  mu_log:      {mu_log:.6f}")
        print(f"  sigma_log:   {sigma_log:.6f}")
        print(f"  pseudocount: {pseudocount}")

        y_test_pred_raw = inverse_transform_half_life(
            y_test_pred, mu=mu_log, sigma=sigma_log, pseudocount=pseudocount
        )
        if y_test_raw_true is None:
            y_test_raw_true = inverse_transform_half_life(
                y_test, mu=mu_log, sigma=sigma_log, pseudocount=pseudocount
            )

        test_raw_metrics = calculate_metrics(y_test_raw_true, y_test_pred_raw, prefix="test_raw_hwz")
        trans_info = {
            "params_path": str(resolved_params_path),
            "mu_log": mu_log,
            "sigma_log": sigma_log,
            "pseudocount": pseudocount,
        }

    if run_4fold_cv:
        print_metrics(train_metrics, f"Training/CV Pool Metrics (Splits 0-7) ({args.target_col})")
    elif is_finetuned:
        print_metrics(train_metrics, f"Training Set Metrics (Splits 0-5) ({args.target_col})")
        print_metrics(val_metrics, f"Validation Set Metrics (Splits 6-7) ({args.target_col})")
    else:
        print_metrics(train_metrics, f"Training/CV Pool Metrics (Splits 0-7) ({args.target_col})")

    print_metrics(test_metrics, f"Test Set Metrics (Folds 8 & 9) - Transformed ({args.target_col})")
    if test_raw_metrics:
        print_metrics(test_raw_metrics, "Test Set Metrics (Folds 8 & 9) - Back-transformed (Actual Half-Life in Hours)")

    # 1. Save trained model
    model_file = out_dir / f"ridge_model_hIPSC_CM_{args.target_col}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved to: {model_file}")

    # 2. Save predictions as CSV
    pred_dict = {
        "transcript_id": test_tx,
        "gene": test_genes,
        "true_target": y_test,
        "predicted_target": y_test_pred,
        "residual": y_test - y_test_pred,
    }
    if y_test_pred_raw is not None and y_test_raw_true is not None:
        pred_dict["true_half_life_hours"] = y_test_raw_true
        pred_dict["predicted_half_life_hours"] = y_test_pred_raw
        pred_dict["residual_hours"] = y_test_raw_true - y_test_pred_raw

    pred_df = pd.DataFrame(pred_dict)
    pred_file = out_dir / f"predictions_hIPSC_CM_{args.target_col}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    # 3. Save comprehensive metrics as JSON
    all_metrics = {
        "dataset": "hIPSC_CM",
        "track_type": track_type,
        "is_finetuned": is_finetuned,
        "is_4fold_cv": is_4fold_cv,
        "normalization": normalization if track_type == "8track" else None,
        "target_col": args.target_col,
        "embeddings_path": str(emb_path),
        "split_mechanism": "lookup_10folds",
        "evaluation_scheme": "4fold_cv_pool_0_7" if run_4fold_cv else "train_0_5_val_6_7_test_8_9",
        "splits_lookup_path": str(lookup_path),
        "best_alpha": best_alpha,
        "cv_pool_size": int(len(y_train_cv)),
        "test_size": int(len(y_test)),
        "cv_pool_unique_genes": int(len(np.unique(train_genes))),
        "test_unique_genes": int(len(np.unique(test_genes))),
        **({"val_size": int(len(y_val)), "val_unique_genes": int(len(np.unique(val_genes)))} if not run_4fold_cv else {}),
        "transformation_params": trans_info,
        "cv_folds_metrics": fold_evaluations,
        "cv_mean_pearson_r": mean_cv_r,
        "cv_std_pearson_r": std_cv_r,
        "cv_mean_rmse": mean_cv_rmse,
        "cv_std_rmse": std_cv_rmse,
        **train_metrics,
        **val_metrics,
        **test_metrics,
        **test_raw_metrics,
    }
    metrics_file = out_dir / f"metrics_hIPSC_CM_{args.target_col}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

    # 4. Save scatter plot (y_true vs. y_pred)
    if args.plot:
        try:
            if track_type == "8track":
                emb_label = f"8-Track ({normalization.upper()})"
            elif is_finetuned:
                emb_label = "6-Track (Fine-Tuned)"
            else:
                emb_label = "6-Track (Pretrained)"

            if y_test_pred_raw is not None and y_test_raw_true is not None:
                # Dual subplot: Left = Transformed Z-Score, Right = Back-transformed (Hours)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Subplot 1: Transformed Z-score
                ax1.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
                p_r = test_metrics["test_pearson_r"]
                s_rho = test_metrics["test_spearman_rho"]
                r2 = test_metrics["test_r2"]
                rmse = test_metrics["test_rmse"]
                ax1.set_title(
                    f"Transformed Target (Z-Score)\n"
                    f"Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | RMSE = {rmse:.3f} | R² = {r2:.3f}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax1.set_xlabel(f"True Value ({args.target_col})", fontsize=10)
                ax1.set_ylabel(f"Predicted Value ({args.target_col})", fontsize=10)
                min_v1 = min(float(np.min(y_test)), float(np.min(y_test_pred)))
                max_v1 = max(float(np.max(y_test)), float(np.max(y_test_pred)))
                margin1 = (max_v1 - min_v1) * 0.05
                ax1.plot([min_v1 - margin1, max_v1 + margin1], [min_v1 - margin1, max_v1 + margin1], "r--", linewidth=1.5, label="Ideal (y=x)")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.5)

                # Subplot 2: Back-transformed (Actual Half-Life in Hours)
                ax2.scatter(y_test_raw_true, y_test_pred_raw, alpha=0.35, s=18, color="#2ca02c", edgecolors="none")
                p_r_raw = test_raw_metrics.get("test_raw_hwz_pearson_r", np.nan)
                s_rho_raw = test_raw_metrics.get("test_raw_hwz_spearman_rho", np.nan)
                rmse_raw = test_raw_metrics.get("test_raw_hwz_rmse", np.nan)
                mae_raw = test_raw_metrics.get("test_raw_hwz_mae", np.nan)
                r2_raw = test_raw_metrics.get("test_raw_hwz_r2", np.nan)
                ax2.set_title(
                    f"Back-transformed: Actual Half-Life (Hours)\n"
                    f"Pearson r = {p_r_raw:.3f} | Spearman rho = {s_rho_raw:.3f} | RMSE = {rmse_raw:.2f}h | MAE = {mae_raw:.2f}h",
                    fontsize=10,
                    fontweight="bold",
                )
                ax2.set_xlabel("True Half-Life (Hours)", fontsize=10)
                ax2.set_ylabel("Predicted Half-Life (Hours)", fontsize=10)
                min_v2 = min(float(np.min(y_test_raw_true)), float(np.min(y_test_pred_raw)))
                max_v2 = max(float(np.max(y_test_raw_true)), float(np.max(y_test_pred_raw)))
                margin2 = (max_v2 - min_v2) * 0.05
                ax2.plot([min_v2 - margin2, max_v2 + margin2], [min_v2 - margin2, max_v2 + margin2], "r--", linewidth=1.5, label="Ideal (y=x)")
                ax2.legend()
                ax2.grid(True, linestyle="--", alpha=0.5)

                fig.suptitle(f"Orthrus {emb_label} -> hIPSC_CM Test Set Evaluation", fontsize=12, fontweight="bold")
                plt.tight_layout()
            else:
                # Single plot (standard behavior)
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
                p_r = test_metrics["test_pearson_r"]
                s_rho = test_metrics["test_spearman_rho"]
                r2 = test_metrics["test_r2"]

                ax.set_title(
                    f"Orthrus {emb_label} -> hIPSC_CM ({args.target_col})\n"
                    f"Test Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | R² = {r2:.3f}",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel(f"True Value ({args.target_col})", fontsize=10)
                ax.set_ylabel(f"Predicted Value ({args.target_col})", fontsize=10)

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
