#!/usr/bin/env python3
"""
Recursive Metrics Aggregator & Summary Table Generator for Orthrus Ridge Regression.

Recursively scans a results directory (default: /beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/Orthrus)
for files matching 'metrics_*.json' and compiles them into a structured comparison table where:
  - Rows: Regression metrics (CV Val, Test Holdout, Train CV-Pool, Overfitting Gap, Hyperparameters)
  - Columns: Datasets and Tracks (e.g. mrna_bench_human | 7track_m6a)

Outputs:
  - Rich Console Table
  - CSV export (formatted & raw)
  - GitHub-flavored Markdown table
  - LaTeX table (publication-ready booktabs format)
"""

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Safe UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


DEFAULT_CLUSTER_RESULTS_DIR = Path(
    "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/Orthrus"
)


def get_default_results_dir() -> Path:
    """Returns cluster default dir if it exists, otherwise falls back to local downloaded results."""
    if DEFAULT_CLUSTER_RESULTS_DIR.exists():
        return DEFAULT_CLUSTER_RESULTS_DIR
    
    # Try workspace relative path: downloaded_results/results/Orthrus
    repo_root = Path(__file__).resolve().parent.parent
    local_candidates = [
        repo_root.parent / "downloaded_results" / "results" / "Orthrus",
        repo_root / "results" / "Orthrus",
        repo_root / "downloaded_results" / "results",
        Path.cwd() / "downloaded_results" / "results" / "Orthrus",
    ]
    for cand in local_candidates:
        if cand.exists():
            return cand
    return DEFAULT_CLUSTER_RESULTS_DIR


def extract_metadata_from_path_and_json(file_path: Path, data: dict) -> Tuple[str, str, str, str]:
    """
    Robustly extracts (dataset, track_type, variant, normalization) from JSON content
    and directory hierarchy.
    """
    parts = file_path.parts

    # 1. Dataset identification
    ds = data.get("dataset") or data.get("raw_dataset_meta")
    if not ds:
        if "mode" in data and "cross" in data["mode"]:
            ds = "mrna_cross_h2m"
        elif "species" in data:
            ds = f"mrna_bench_{data['species']}"
        elif any("hipsc" in str(p).lower() for p in parts):
            ds = "hipsc_cm"
        elif any("mrna" in str(p).lower() for p in parts):
            ds = "mrna_bench"
        else:
            ds = file_path.parent.name

    # 2. Track identification
    track = data.get("track_name") or data.get("track_type")
    if not track:
        for p in parts:
            if "track" in p.lower():
                track = p
                break
        if not track:
            track = "6track" if "mrna" in str(ds).lower() else "unknown_track"

    # 3. Model Variant (finetuned vs pretrained)
    variant = data.get("variant")
    if not variant:
        if data.get("is_finetuned") is True:
            variant = "finetuned"
        elif data.get("is_finetuned") is False:
            variant = "pretrained"
        elif any("finetuned" in str(p).lower() for p in parts):
            variant = "finetuned"
        elif any("pretrained" in str(p).lower() for p in parts):
            variant = "pretrained"
        else:
            variant = ""

    # 4. Normalization (minmax, log, none)
    norm = data.get("normalization")
    if not norm or str(norm).lower() == "none":
        for n in ["minmax", "log", "standard"]:
            if any(n in str(p).lower() for p in parts):
                norm = n
                break
        if not norm or str(norm).lower() == "none":
            norm = ""

    return str(ds), str(track), str(variant), str(norm)


def format_column_name(dataset: str, track: str, variant: str, norm: str) -> str:
    """Creates a human-readable column identifier."""
    tags = []
    if variant and variant.lower() != "none":
        tags.append(variant)
    if norm and norm.lower() != "none":
        tags.append(norm)
    tag_str = f" ({', '.join(tags)})" if tags else ""
    return f"{dataset} | {track}{tag_str}"


def _get_val(d: dict, *keys: str, default: Any = None) -> Any:
    """Helper to get the first matching key from a dictionary."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def collect_metrics_files(base_dir: Path, pattern: str = "metrics_*.json") -> List[Tuple[Path, dict]]:
    """Recursively finds and loads all JSON metrics files matching the pattern."""
    if not base_dir.exists():
        raise FileNotFoundError(f"[Error] Target directory does not exist: {base_dir}")

    results = []
    for json_file in sorted(base_dir.rglob(pattern)):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                results.append((json_file, data))
        except Exception as e:
            print(f"[Warning] Could not parse JSON file {json_file}: {e}", file=sys.stderr)
    return results


def build_summary_dataframe(
    metrics_entries: List[Tuple[Path, dict]],
    sort_by: str = "dataset"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs:
      1. A formatted, publication-ready summary DataFrame (with Mean ± Std).
      2. A full numeric raw DataFrame with exact floating point metrics.
    """
    cols_data = {}
    cols_meta = {}

    for path, data in metrics_entries:
        ds, track, variant, norm = extract_metadata_from_path_and_json(path, data)
        col_name = format_column_name(ds, track, variant, norm)
        
        # Handle duplicates if same experiment was run into multiple subfolders
        if col_name in cols_data:
            col_name = f"{col_name} ({path.parent.name})"
            
        test_r_val = _get_val(data, "test_pearson_r", "mouse_test_pearson_r")
        cols_data[col_name] = data
        cols_meta[col_name] = {
            "dataset": ds,
            "track": track,
            "variant": variant,
            "norm": norm,
            "path": str(path),
            "test_r": float(test_r_val) if test_r_val is not None else -1.0,
        }

    # Sorting columns
    sorted_cols = list(cols_data.keys())
    if sort_by == "test_r":
        sorted_cols.sort(key=lambda c: cols_meta[c]["test_r"], reverse=True)
    elif sort_by == "dataset":
        sorted_cols.sort(key=lambda c: (cols_meta[c]["dataset"], cols_meta[c]["track"], cols_meta[c]["variant"]))

    # -------------------------------------------------------------------------
    # 1. Publication-Ready Formatted Summary Table
    # -------------------------------------------------------------------------
    summary_metric_definitions = [
        # CV Out-of-fold Metrics
        ("CV Pearson r (Mean ± Std)", lambda d: (
            f"{float(d['cv_mean_pearson_r']):.4f} ± {float(d.get('cv_std_pearson_r', 0.0)):.4f}"
            if "cv_mean_pearson_r" in d else "-"
        )),
        ("CV RMSE (Mean ± Std)", lambda d: (
            f"{float(d['cv_mean_rmse']):.4f} ± {float(d.get('cv_std_rmse', 0.0)):.4f}"
            if "cv_mean_rmse" in d else "-"
        )),
        
        # Held-out Test Set (Splits 8 & 9 or species test)
        ("Test Pearson r", lambda d: (
            f"{float(_get_val(d, 'test_pearson_r', 'mouse_test_pearson_r')):.4f}"
            if _get_val(d, 'test_pearson_r', 'mouse_test_pearson_r') is not None else "-"
        )),
        ("Test Spearman ρ", lambda d: (
            f"{float(_get_val(d, 'test_spearman_rho', 'mouse_test_spearman_rho')):.4f}"
            if _get_val(d, 'test_spearman_rho', 'mouse_test_spearman_rho') is not None else "-"
        )),
        ("Test R²", lambda d: (
            f"{float(_get_val(d, 'test_r2', 'mouse_test_r2')):.4f}"
            if _get_val(d, 'test_r2', 'mouse_test_r2') is not None else "-"
        )),
        ("Test RMSE", lambda d: (
            f"{float(_get_val(d, 'test_rmse', 'mouse_test_rmse')):.4f}"
            if _get_val(d, 'test_rmse', 'mouse_test_rmse') is not None else "-"
        )),
        ("Test MAE", lambda d: (
            f"{float(_get_val(d, 'test_mae', 'mouse_test_mae')):.4f}"
            if _get_val(d, 'test_mae', 'mouse_test_mae') is not None else "-"
        )),
        
        # Training / CV-Pool Set (Splits 0 - 7)
        ("Train Pearson r", lambda d: (
            f"{float(_get_val(d, 'train_cv_pool_pearson_r', 'train_pearson_r', 'human_train_pearson_r')):.4f}"
            if _get_val(d, 'train_cv_pool_pearson_r', 'train_pearson_r', 'human_train_pearson_r') is not None else "-"
        )),
        ("Train Spearman ρ", lambda d: (
            f"{float(_get_val(d, 'train_cv_pool_spearman_rho', 'train_spearman_rho', 'human_train_spearman_rho')):.4f}"
            if _get_val(d, 'train_cv_pool_spearman_rho', 'train_spearman_rho', 'human_train_spearman_rho') is not None else "-"
        )),
        ("Train R²", lambda d: (
            f"{float(_get_val(d, 'train_cv_pool_r2', 'train_r2', 'human_train_r2')):.4f}"
            if _get_val(d, 'train_cv_pool_r2', 'train_r2', 'human_train_r2') is not None else "-"
        )),
        ("Train RMSE", lambda d: (
            f"{float(_get_val(d, 'train_cv_pool_rmse', 'train_rmse', 'human_train_rmse')):.4f}"
            if _get_val(d, 'train_cv_pool_rmse', 'train_rmse', 'human_train_rmse') is not None else "-"
        )),
        ("Train MAE", lambda d: (
            f"{float(_get_val(d, 'train_cv_pool_mae', 'train_mae', 'human_train_mae')):.4f}"
            if _get_val(d, 'train_cv_pool_mae', 'train_mae', 'human_train_mae') is not None else "-"
        )),
        
        # Overfitting Diagnostic
        ("Overfitting Gap Δr (Train - Test)", lambda d: (
            f"{(float(_get_val(d, 'train_cv_pool_pearson_r', 'train_pearson_r', 'human_train_pearson_r')) - float(_get_val(d, 'test_pearson_r', 'mouse_test_pearson_r'))):.4f}"
            if (_get_val(d, 'test_pearson_r', 'mouse_test_pearson_r') is not None and 
                _get_val(d, 'train_cv_pool_pearson_r', 'train_pearson_r', 'human_train_pearson_r') is not None) else "-"
        )),
        
        # Hyperparameters & Sample sizes
        ("Best Alpha", lambda d: f"{d['best_alpha']}" if "best_alpha" in d else "-"),
        ("Samples (Train / Test)", lambda d: (
            f"{_get_val(d, 'cv_pool_size', 'train_size', 'train_size_human', default='-')} / {_get_val(d, 'test_size', 'test_size_mouse', default='-')}"
        )),
    ]


    summary_rows = {}
    for metric_label, extractor in summary_metric_definitions:
        summary_rows[metric_label] = {col: extractor(cols_data[col]) for col in sorted_cols}

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.index.name = "Metric"

    # -------------------------------------------------------------------------
    # 2. Raw Numerical Metrics Table
    # -------------------------------------------------------------------------
    raw_metric_keys = [
        "cv_mean_pearson_r",
        "cv_std_pearson_r",
        "cv_mean_rmse",
        "cv_std_rmse",
        "train_cv_pool_pearson_r",
        "train_cv_pool_pearson_pvalue",
        "train_cv_pool_spearman_rho",
        "train_cv_pool_spearman_pvalue",
        "train_cv_pool_mse",
        "train_cv_pool_rmse",
        "train_cv_pool_mae",
        "train_cv_pool_r2",
        "test_pearson_r",
        "test_pearson_pvalue",
        "test_spearman_rho",
        "test_spearman_pvalue",
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "best_alpha",
        "cv_pool_size",
        "test_size",
    ]

    raw_rows = {}
    for k in raw_metric_keys:
        raw_rows[k] = {col: cols_data[col].get(k, np.nan) for col in sorted_cols}

    raw_df = pd.DataFrame(raw_rows).T
    raw_df.index.name = "Metric_Key"

    return summary_df, raw_df


# =============================================================================
# Exporters: Markdown & LaTeX (Dependency-free)
# =============================================================================

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Formats DataFrame as GitHub Flavored Markdown table without external dependencies."""
    idx_col_name = str(df.index.name or "Metric")
    headers = [idx_col_name] + [str(col).replace("\n", " ") for col in df.columns]
    
    col_widths = [len(h) for h in headers]
    for idx_val, row in df.iterrows():
        col_widths[0] = max(col_widths[0], len(str(idx_val)))
        for c_idx, val in enumerate(row):
            col_widths[c_idx + 1] = max(col_widths[c_idx + 1], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join(
        (":" + "-" * (col_widths[i] - 1) if i == 0 else ":" + "-" * (col_widths[i] - 2) + ":")
        for i in range(len(headers))
    ) + " |"

    data_lines = []
    for idx_val, row in df.iterrows():
        vals = [str(idx_val).ljust(col_widths[0])]
        for c_idx, val in enumerate(row):
            vals.append(str(val).center(col_widths[c_idx + 1]))
        data_lines.append("| " + " | ".join(vals) + " |")

    return "\n".join([header_line, separator_line] + data_lines)


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str = "Comparison of Orthrus Regression Metrics across Datasets and Feature Tracks",
    label: str = "tab:orthrus_metrics_comparison",
) -> str:
    """Generates LaTeX table code with booktabs styling."""
    num_cols = len(df.columns)
    col_align = "l" + "c" * num_cols

    clean_headers = [
        str(c).replace("\n", " ").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
        for c in df.columns
    ]
    header_str = " & ".join(["\\textbf{Metric}"] + [f"\\textbf{{{c}}}" for c in clean_headers]) + " \\\\"

    lines = [
        "% Auto-generated by summarize_results.py",
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\toprule",
        header_str,
        "\\midrule",
    ]

    for idx_val, row in df.iterrows():
        clean_idx = (
            str(idx_val)
            .replace("_", "\\_")
            .replace("±", "$\\pm$")
            .replace("ρ", "$\\rho$")
            .replace("Δ", "$\\Delta$")
            .replace("²", "$^2$")
        )
        row_cells = [clean_idx]
        for val in row:
            clean_val = str(val).replace("±", "$\\pm$")
            row_cells.append(clean_val)
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


# =============================================================================
# CLI Main
# =============================================================================

def parse_args():
    default_dir = get_default_results_dir()
    parser = argparse.ArgumentParser(
        description="Recursively scans Orthrus results directories for metrics_*.json and creates comparison tables."
    )
    parser.add_argument(
        "-d", "--base_dir",
        type=Path,
        default=default_dir,
        help=f"Directory to search recursively for metrics_*.json (default: {default_dir})",
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="metrics_*.json",
        help="Glob pattern to identify metric JSON files (default: 'metrics_*.json')",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default=None,
        help="Optional directory to save generated summary files (CSV, Markdown, LaTeX)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        choices=["dataset", "test_r"],
        default="dataset",
        help="Column ordering strategy: 'dataset' (alphabetical by dataset/track) or 'test_r' (by Test Pearson r descending)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Display the raw numerical metrics table instead of the formatted summary table",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Automatically saves 'metrics_summary.csv', 'metrics_summary.md', 'metrics_summary.tex', and 'metrics_raw.csv'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir.resolve()

    print("=" * 80)
    print("      ORTHRUS METRICS AGGREGATOR & TABLE GENERATOR      ")
    print("=" * 80)
    print(f"Scanning directory: {base_dir}")
    print(f"Pattern:            {args.pattern}")
    print(f"Column sorting:     {args.sort_by}")

    metrics_entries = collect_metrics_files(base_dir, pattern=args.pattern)
    if not metrics_entries:
        print(f"\n[Warning] No files matching '{args.pattern}' were found in '{base_dir}'.")
        print("Please check your results directory path.")
        sys.exit(0)

    print(f"Found {len(metrics_entries)} metrics files.\n")

    summary_df, raw_df = build_summary_dataframe(metrics_entries, sort_by=args.sort_by)

    # 1. Print formatted markdown table to console
    print("\n" + "=" * 80)
    print("                  SUMMARY METRICS COMPARISON TABLE                  ")
    print("=" * 80 + "\n")
    
    active_df = raw_df if args.raw else summary_df
    md_table = dataframe_to_markdown(active_df)
    print(md_table)
    print("\n")

    # 2. File Export if requested
    out_dir = args.output_dir or (base_dir if base_dir.is_dir() else Path.cwd())
    if args.save_all or args.output_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSVs
        summary_csv_path = out_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_csv_path, encoding="utf-8")
        print(f"[Export] Saved formatted CSV to:  {summary_csv_path}")

        raw_csv_path = out_dir / "metrics_raw.csv"
        raw_df.to_csv(raw_csv_path, encoding="utf-8")
        print(f"[Export] Saved raw numeric CSV to: {raw_csv_path}")

        # Save Markdown
        summary_md_path = out_dir / "metrics_summary.md"
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("# Orthrus Model Evaluation: Metrics Summary\n\n")
            f.write(dataframe_to_markdown(summary_df))
            f.write("\n\n## Raw Metrics Table\n\n")
            f.write(dataframe_to_markdown(raw_df))
            f.write("\n")
        print(f"[Export] Saved Markdown report to: {summary_md_path}")

        # Save LaTeX
        summary_tex_path = out_dir / "metrics_summary.tex"
        with open(summary_tex_path, "w", encoding="utf-8") as f:
            f.write(dataframe_to_latex(summary_df))
            f.write("\n")
        print(f"[Export] Saved LaTeX table to:     {summary_tex_path}")

    print("\nDone! All metrics aggregated successfully.")


if __name__ == "__main__":
    main()
