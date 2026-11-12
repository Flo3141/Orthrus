#!/usr/bin/env python3
"""
Analyses of fine-tuned N-track Orthrus models on the held-out test set.

Runs after finetune_ntrack.py on the checkpoints it wrote (fold_k/best_model.pt + best_finetuned_backbone).
No training is involved, so the analyses can be repeated on any existing checkpoint.

--analyses ablation (default): every input channel is set to zero in turn and the test set is predicted again.
The drop in Pearson r shows how much the model relies on that channel.

--analyses drach: motif perturbation (DRACH, AUUUA, UGUANAUA) in float32, see drach_analysis.py. Compares the
predicted half-lives with mutated motifs and with mutated control positions to the original predictions.

--analyses drach_shift: the central A of every DRACH site in the 3' UTR is moved one position to the left and to the
right (swap with the neighbouring base) in float32, see drach_shift_analysis.py. Compares WT, both shifts and swapped
control dinucleotides.

Outputs:
- <run_dir>/fold_k/xai_channel_summary.{json,png,pdf}                (ablation)
- <run_dir>/xai_aggregated.{json,png,pdf}: mean and standard deviation across folds (ablation, more than one fold)
- <run_dir>/fold_k/drach_summary.json, drach_predictions.csv, drach_effects.{png,pdf}, drach_scatter.{png,pdf}  (drach)
- <run_dir>/drach_aggregated.{json,png,pdf}: mean and standard deviation across folds (drach, more than one fold)
- <run_dir>/fold_k/drach_shift/drach_shift_summary.json, drach_shift_predictions.csv, drach_shift_effects.{png,pdf},
  drach_shift_scatter.{png,pdf}  (drach_shift)
- <run_dir>/drach_shift/drach_shift_aggregated.{json,png,pdf}: mean and standard deviation across folds (drach_shift, more than one fold)

Usage:
  python xai_analysis.py --data_path <tracks.npz>
  python xai_analysis.py --data_path <tracks.npz> --analyses ablation drach
  python xai_analysis.py --data_path <tracks.npz> --analyses drach_shift
  python xai_analysis.py --data_path <tracks.npz> --fold 0
  python xai_analysis.py --data_path <tracks.npz> --run_name lr_test
  python xai_analysis.py --data_path <tracks.npz> --checkpoint_dir /path/to/run_dir

Local run on checkpoints downloaded with sync_results.sh --checkpoints (inside WSL):
  export ORTHRUS_BASE=/mnt/f/Workspace/MIM/Masterarbeit/local_xai
  python xai_analysis.py --data_path $ORTHRUS_BASE/data/hIPSC_CM/orthrus/6track/hipsc_cm_6tracks.npz --batch_size 4
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finetune_ntrack import TrackData, build_regression_model, get_amp_dtype, load_track_data, make_loader, predict, resolve_run_dir, run_name_arg
from util import CV_FOLDS, TEST_SPLITS, regression_metrics


BAR_COLOR = "#2a78d6"

# Largest accepted difference to the test r of the training run. Larger differences point to another
# test set (lookup table) or another model implementation (mamba_ssm version), not to rounding.
MAX_REPRODUCTION_DIFF_R = 0.005


def compute_channel_ablation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    baseline_metrics: dict,
    channel_names: list,
    amp_dtype: torch.dtype,
) -> dict:
    """
    Evaluates model performance drop when each channel is systematically ablated (zeroed out).
    Delta r = baseline_r - ablated_r. A higher drop indicates greater reliance on that channel.
    """
    ablation_results = {}

    for c_idx, c_name in enumerate(tqdm(channel_names, desc="[xAI] Computing Channel Ablation Impact")):
        index = torch.tensor([c_idx], device=device)
        y_true, y_pred = predict(model, dataloader, device, amp_dtype, x_transform=lambda x: x.index_fill(2, index, 0.0))
        ablated = regression_metrics(y_true, y_pred)

        ablation_results[c_name] = {
            "channel_index": c_idx,
            "ablated_pearson_r": float(ablated["pearson_r"]),
            "drop_pearson_r": float(baseline_metrics["pearson_r"] - ablated["pearson_r"]),
            "ablated_spearman_rho": float(ablated["spearman_rho"]),
            "drop_spearman_rho": float(baseline_metrics["spearman_rho"] - ablated["spearman_rho"]),
            "ablated_mse": ablated["mse"],
            "delta_mse": float(ablated["mse"] - baseline_metrics["mse"]),
        }

    return ablation_results


def plot_channel_ablation(drops: list, channel_names: list, title: str, output_png: Path, errors: list = None):
    """Bar chart of the drop in Pearson r per channel, in channel order; errors are optional error bars."""
    positions = range(len(channel_names))
    fig, ax = plt.subplots(figsize=(max(6, len(channel_names) * 1.1), 4.5))

    ax.bar(positions, drops, yerr=errors, width=0.7, capsize=3, color=BAR_COLOR, error_kw={"linewidth": 1, "ecolor": "#333333"}, zorder=2)
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_ylabel("Drop in Pearson r when the channel is zeroed", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(channel_names, rotation=35, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0, color="#666666", linewidth=0.8, zorder=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.12)  # Room for the value labels, which do not extend the axis limits

    for i, drop in enumerate(drops):
        err = errors[i] if errors is not None else 0.0
        above = drop >= 0
        ax.annotate(
            f"{drop:.3f}",
            xy=(i, drop + err if above else drop - err),
            xytext=(0, 3 if above else -3), textcoords="offset points",
            ha="center", va="bottom" if above else "top", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    pdf_path = output_png.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[xAI] Saved channel ablation plot to: {output_png} and {pdf_path}")


def load_fold_model(fold_dir: Path, channel_names: list, device: torch.device) -> tuple:
    """Loads backbone and regression head of a fine-tuned fold. Returns (model, best_epoch, checkpoint_path)."""
    ckpt_path = fold_dir / "best_model.pt"
    backbone_dir = fold_dir / "best_finetuned_backbone"
    if not ckpt_path.is_file() or not backbone_dir.is_dir():
        raise FileNotFoundError(f"[Error] Fold checkpoint incomplete, expected {ckpt_path} and {backbone_dir}. Run finetune_ntrack.py first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt["channel_names"] != channel_names:
        raise ValueError(f"[Error] Checkpoint {ckpt_path} was trained on channels {ckpt['channel_names']}, the track archive has {channel_names}!")

    model = build_regression_model(str(backbone_dir), len(channel_names), device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("best_epoch"), ckpt_path


def check_reproduces_training(fold_dir: Path, test_metrics: dict) -> None:
    """Compares the baseline with the test r of the training run, if its summary is available."""
    summary_path = fold_dir / "training_summary.json"
    if not summary_path.is_file():
        print(f"[Check] No {summary_path.name} found, the baseline cannot be compared with the training run.")
        return

    with open(summary_path, "r") as f:
        training_r = json.load(f)["test_metrics"]["pearson_r"]
    diff = abs(test_metrics["pearson_r"] - training_r)
    print(f"[Check] Test Pearson r: {test_metrics['pearson_r']:.4f} here, {training_r:.4f} in the training run (difference {diff:.5f})")
    if diff > MAX_REPRODUCTION_DIFF_R:
        print(f"[Warning] Difference above {MAX_REPRODUCTION_DIFF_R}: check the lookup table, the track archive and the mamba_ssm version before using these results.")


def analyze_fold(
    fold_dir: Path,
    data: TrackData,
    test_idx: np.ndarray,
    test_loader: Optional[DataLoader],
    device: torch.device,
    amp_dtype: torch.dtype,
    analyses: list,
    args: argparse.Namespace,
) -> dict:
    """
    Runs the requested analyses for one fold with a single loaded model. Returns {"ablation": ..., "motif": ...,
    "shift": ...} with the summaries of the analyses that ran. The ablation writes xai_channel_summary.{json,png,pdf}
    into fold_dir, the motif analysis drach_summary.json and its plots, the shift analysis everything into fold_dir/drach_shift/.
    """
    channel_names = data.channel_names
    model, best_epoch, ckpt_path = load_fold_model(fold_dir, channel_names, device)
    results = {}

    if "ablation" in analyses:
        y_true, y_pred = predict(model, test_loader, device, amp_dtype)
        test_metrics = regression_metrics(y_true, y_pred)
        check_reproduces_training(fold_dir, test_metrics)

        ablation_results = compute_channel_ablation(model, test_loader, device, test_metrics, channel_names, amp_dtype)

        print("\n" + "=" * 45)
        print(f"{'Channel':<18} {'Ablation Drop (Delta r)':<25}")
        print("=" * 45)
        for name in channel_names:
            print(f"{name:<18} {ablation_results[name]['drop_pearson_r']:>23.4f}")
        print("=" * 45)

        plot_channel_ablation(
            drops=[ablation_results[n]["drop_pearson_r"] for n in channel_names],
            channel_names=channel_names,
            title=f"{fold_dir.name}: baseline Pearson r = {test_metrics['pearson_r']:.3f}",
            output_png=fold_dir / "xai_channel_summary.png",
        )

        results["ablation"] = {
            "checkpoint": str(ckpt_path),
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "channel_names": channel_names,
            "channel_ablation": ablation_results,
        }
        summary_file = fold_dir / "xai_channel_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results["ablation"], f, indent=2)
        print(f"[xAI] Saved xAI summary report to: {summary_file}")

    if "drach" in analyses:
        # Imported here: an ablation-only run does not need scipy.stats/pandas, and an error in the new module cannot break it
        from drach_analysis import run_motif_analysis
        results["motif"] = run_motif_analysis(
            model, data, test_idx, fold_dir, device,
            batch_size=args.drach_batch_size or args.batch_size, max_length=args.max_length,
            substitutions=tuple(args.drach_substitutions),
        )

    if "drach_shift" in analyses:
        from drach_shift_analysis import SHIFTS, run_shift_analysis
        results["shift"] = run_shift_analysis(
            model, data, test_idx, fold_dir, device,
            batch_size=args.drach_batch_size or args.batch_size, max_length=args.max_length,
            shifts=tuple(s for s in SHIFTS if s.name in args.shift_directions),
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def aggregate_folds(fold_summaries: dict, channel_names: list) -> dict:
    """
    Mean and standard deviation (np.std, as in the fine-tuning summary) of the ablation drop across folds.
    All folds share one test set and overlapping training data: the spread shows the sensitivity to the fold,
    it is not a confidence interval.
    """
    def stats(values: list) -> dict:
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_fold": values}

    return {
        "folds": sorted(fold_summaries),
        "channel_names": channel_names,
        "baseline_pearson_r": stats([float(s["test_metrics"]["pearson_r"]) for s in fold_summaries.values()]),
        "drop_pearson_r": {
            name: stats([float(s["channel_ablation"][name]["drop_pearson_r"]) for s in fold_summaries.values()])
            for name in channel_names
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Channel ablation analysis of fine-tuned N-track Orthrus models", allow_abbrev=False)
    parser.add_argument("--data_path", type=str, required=True, help="Path to the N-track .npz archive the model was trained on")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Run directory containing fold_0..fold_3 (default: derived from the track metadata and --run_name, as in finetune_ntrack.py)",
    )
    parser.add_argument("--run_name", type=run_name_arg, default=None, help="--run_name of the fine-tuning run (ignored with --checkpoint_dir)")
    parser.add_argument("--fold", type=str, default="all", choices=["all", "0", "1", "2", "3"], help="Fold to analyze (default: all)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--max_length", type=int, default=12288, help="Maximum sequence length; longer transcripts keep their 3' end (default: 12288)")
    parser.add_argument(
        "--analyses", nargs="+", default=["ablation"], choices=["ablation", "drach", "drach_shift"],
        help="Analyses to run (default: ablation). 'drach' is the motif perturbation, 'drach_shift' moves the central A of DRACH by one position.",
    )
    parser.add_argument(
        "--drach_batch_size", type=int, default=None,
        help="Batch size of the float32 motif and shift analyses (default: --batch_size). Needs about 6.6 GB for 16 x 12288 nt, so use 4 to 8 on 8 GB of VRAM",
    )
    parser.add_argument(
        "--shift_directions", nargs="+", default=["left", "right"], choices=["left", "right"],
        help="Directions of the DRACH shift analysis (default: left right)",
    )
    parser.add_argument(
        "--drach_substitutions", nargs="+", default=["G", "C", "U"], choices=["G", "C", "U"],
        help="Bases that replace the A in the motif analysis (default: G C U). Each one adds a full set of passes, so the run time scales with their number",
    )
    args = parser.parse_args()
    analyses = list(dict.fromkeys(args.analyses))
    args.drach_substitutions = list(dict.fromkeys(args.drach_substitutions))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    print(f"Using device: {device} (precision: {amp_dtype}; the motif analysis always uses float32)")
    print(f"Analyses: {analyses}")

    data = load_track_data(Path(args.data_path))
    run_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else resolve_run_dir(data, args.run_name)
    print(f"[Checkpoints] Run directory: {run_dir}")

    test_idx = np.where(np.isin(data.sample_splits, TEST_SPLITS))[0]
    if len(test_idx) == 0:
        raise ValueError(f"[Error] No test samples found! Test splits: {TEST_SPLITS}")
    print(f"Test samples: {len(test_idx)} (Splits {TEST_SPLITS})")
    test_loader = make_loader(data, test_idx, args.batch_size, args.max_length, shuffle=False, device=device) if "ablation" in analyses else None

    channel_names = data.channel_names
    folds = list(CV_FOLDS) if args.fold == "all" else [int(args.fold)]

    fold_results = {}
    for fold_id in folds:
        print("\n" + "=" * 60)
        print(f"                 xAI ANALYSIS: FOLD {fold_id}                 ")
        print("=" * 60)
        fold_results[fold_id] = analyze_fold(run_dir / f"fold_{fold_id}", data, test_idx, test_loader, device, amp_dtype, analyses, args)

    if len(folds) > 1 and "drach" in analyses:
        from drach_analysis import aggregate_motif_folds
        aggregate_motif_folds({k: r["motif"] for k, r in fold_results.items()}, run_dir, f"{data.track_name}")

    if len(folds) > 1 and "drach_shift" in analyses:
        from drach_shift_analysis import aggregate_shift_folds
        aggregate_shift_folds({k: r["shift"] for k, r in fold_results.items()}, run_dir, f"{data.track_name}")

    if len(folds) > 1 and "ablation" in analyses:
        fold_summaries = {k: r["ablation"] for k, r in fold_results.items()}
        aggregated = aggregate_folds(fold_summaries, channel_names)
        baseline = aggregated["baseline_pearson_r"]
        plot_channel_ablation(
            drops=[aggregated["drop_pearson_r"][n]["mean"] for n in channel_names],
            channel_names=channel_names,
            title=f"{data.track_name}, {len(folds)} folds: baseline Pearson r = {baseline['mean']:.3f} ± {baseline['std']:.3f}",
            output_png=run_dir / "xai_aggregated.png",
            errors=[aggregated["drop_pearson_r"][n]["std"] for n in channel_names],
        )
        aggregated_file = run_dir / "xai_aggregated.json"
        with open(aggregated_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"[xAI] Saved aggregated report across folds to: {aggregated_file}")


if __name__ == "__main__":
    main()
