#!/usr/bin/env python3
"""
Explainable AI (xAI) analysis of fine-tuned N-track Orthrus models on the held-out test set.

Runs after finetune_ntrack.py on the checkpoints it wrote (fold_k/best_model.pt + best_finetuned_backbone).
No training is involved, so the analysis can be repeated on any existing checkpoint.

For each fold:
- Input * Gradient saliency attributions (%)
- Systematic single-channel ablation (drop in Pearson r)
- Input embedding weight L2 norms
- 2D sequence saliency heatmaps (N channels x position) for selected transcripts

Outputs:
- <run_dir>/fold_k/xai_channel_summary.{json,png,pdf} and <run_dir>/fold_k/xai_heatmaps/
- <run_dir>/xai_aggregated.{json,png,pdf}: mean and standard deviation across folds (if more than one fold ran)

Usage:
  python xai_analysis.py --data_path <tracks.npz>
  python xai_analysis.py --data_path <tracks.npz> --fold 0 --tx_ids ENST00000000001,ENST00000000002
  python xai_analysis.py --data_path <tracks.npz> --run_name lr_test
  python xai_analysis.py --data_path <tracks.npz> --checkpoint_dir /path/to/run_dir
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finetune_ntrack import (
    MultiTrackDataset, amp_autocast, build_regression_model, get_amp_dtype, load_track_data,
    make_loader, predict, resolve_run_dir, run_name_arg,
)
from util import CV_FOLDS, TEST_SPLITS, regression_metrics


# Standard color palette for input channels
STANDARD_PALETTE = [
    "#4e79a7",  # A (Blue)
    "#f28e2c",  # C (Orange)
    "#59a14f",  # G (Green)
    "#e15759",  # U (Red)
    "#b07aa1",  # CDS (Purple)
    "#76b7b2",  # Splice (Cyan)
    "#2ca02c",  # Channel 6 (Green / m6A / RBP Stabilizer)
    "#d62728",  # Channel 7 (Crimson / RBP Destabilizer)
    "#edc949",  # Channel 8 (Gold)
    "#af7aa1",  # Channel 9 (Plum)
    "#ff9da7",  # Channel 10 (Pink)
    "#9c755f",  # Channel 11 (Brown)
    "#bab0ac",  # Channel 12 (Gray)
]


def get_channel_colors(channel_names: list) -> list:
    """
    Returns a distinct list of hex colors for the given channel names.
    Uses STANDARD_PALETTE for up to 13 channels, falling back to tab20 colormap for larger N.
    """
    n = len(channel_names)
    if n <= len(STANDARD_PALETTE):
        return STANDARD_PALETTE[:n]
    cmap = plt.get_cmap("tab20")
    return [matplotlib.colors.to_hex(cmap(i % 20)) for i in range(n)]


def input_gradients(model: nn.Module, x: torch.Tensor, lengths: torch.Tensor, device: torch.device, amp_dtype: torch.dtype) -> tuple:
    """Returns (predictions, d sum(predictions) / dx). Only the input gradient is computed, no parameter gradients."""
    x = x.detach().requires_grad_(True)
    with amp_autocast(device, amp_dtype):
        preds = model(x, lengths)
    (grad_x,) = torch.autograd.grad(preds.sum(), x)
    return preds.detach(), grad_x


def compute_embedding_norms(model: nn.Module, num_channels: int) -> list:
    """
    Computes L2 norm of the learned input embedding weights for each channel 0..num_channels-1.
    Shows the direct parametric weight assigned to each track.
    """
    for name, param in model.backbone.named_parameters():
        if "embedding" in name and "weight" in name:
            w = param.detach()
            norms = []
            for c in range(num_channels):
                if w.ndim == 2:
                    col = w[:, c]
                elif w.ndim == 3:
                    col = w[:, c, :]
                else:
                    col = w[c]
                norms.append(float(torch.norm(col.float(), p=2).item()))
            return norms
    return [0.0] * num_channels


def compute_channel_importance_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    channel_names: list,
    max_samples: int,
    amp_dtype: torch.dtype,
) -> dict:
    """
    Computes global channel attribution across test samples using Input * Gradient (Feature Saliency).
    Returns mean absolute attribution per channel and percentage contribution.

    The test loader is sorted by length, so max_samples selects the shortest transcripts of the test set.
    """
    model.eval()
    num_channels = len(channel_names)
    total_attribution = torch.zeros(num_channels, device=device, dtype=torch.float64)
    total_valid_nt = 0
    samples_processed = 0

    for batch in tqdm(dataloader, desc="[xAI] Computing Input*Gradient Channel Saliency"):
        if samples_processed >= max_samples:
            break

        x = batch["x"].to(device)
        lengths = batch["lengths"].to(device)
        _, grad_x = input_gradients(model, x, lengths, device, amp_dtype)

        sal = torch.abs(x * grad_x)
        for i in range(x.shape[0]):
            l_i = lengths[i].item()
            total_attribution += sal[i, :l_i, :num_channels].sum(dim=0).to(torch.float64)
            total_valid_nt += l_i
            samples_processed += 1
            if samples_processed >= max_samples:
                break

    mean_attribution = (total_attribution / max(1, total_valid_nt)).cpu().numpy()
    sum_attr = float(mean_attribution.sum())
    pct_attribution = (mean_attribution / sum_attr * 100.0) if sum_attr > 0 else np.zeros(num_channels)

    results = {}
    for idx, name in enumerate(channel_names):
        results[name] = {
            "channel_index": idx,
            "mean_attribution": float(mean_attribution[idx]),
            "pct_attribution": float(pct_attribution[idx]),
        }
    return results


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


def plot_channel_importance_summary(
    grad_pcts: list,
    ablation_drops: list,
    embedding_norms: list,
    channel_names: list,
    channel_colors: list,
    output_png: Path,
    errors: dict = None,
):
    """
    Plots a 3-panel publication-grade summary of channel importance:
    1. Input * Gradient Attribution (% of total)
    2. Channel Ablation Drop in Pearson r (Delta r)
    3. Embedding Weight L2 Norms

    errors: optional error bars per panel ({"grad": [...], "ablation": [...], "norms": [...]}).
    """
    errors = errors or {}
    panels = [
        (grad_pcts, errors.get("grad"), "Feature Saliency (Input * Grad)", "Attribution Share (%)", "{:.1f}%"),
        (ablation_drops, errors.get("ablation"), "Channel Ablation Impact", "Drop in Pearson r (Delta r)", "{:.3f}"),
        (embedding_norms, errors.get("norms"), "Embedding Weight Norms", "L2 Norm (||W[:, c]||)", "{:.2f}"),
    ]
    positions = range(len(channel_names))
    fig, axes = plt.subplots(1, 3, figsize=(max(16, len(channel_names) * 2), 5.5))

    for ax, (values, errs, title, ylabel, label_fmt) in zip(axes, panels):
        ax.bar(positions, values, yerr=errs, capsize=3, color=channel_colors, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(channel_names, rotation=35, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.12)  # Room for the value labels, which do not extend the axis limits
        if min(values) < 0:
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        for i, h in enumerate(values):
            err = errs[i] if errs is not None else 0.0
            above = h >= 0
            ax.annotate(
                label_fmt.format(h),
                xy=(i, h + err if above else h - err),
                xytext=(0, 3 if above else -10), textcoords="offset points",
                ha="center", va="bottom" if above else "top", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    pdf_path = output_png.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[xAI] Saved channel importance summary to: {output_png} and {pdf_path}")


def select_heatmap_samples(samples: list, num_channels: int, num_heatmaps: int, specific_tx_ids: str = None) -> list:
    """Requested transcripts if given and found; otherwise prefers transcripts with signal in the augmented channels."""
    if specific_tx_ids:
        target_ids = {t.strip() for t in specific_tx_ids.split(",") if t.strip()}
        selected = [idx for idx, s in enumerate(samples) if s["tx_id"] in target_ids or s["tx_id"].split(".")[0] in target_ids]
        if selected:
            return selected

    # Channel index >= 6 if present, else >= 4
    first_augmented = 6 if num_channels > 6 else 4
    active, other = [], []
    for idx, s in enumerate(samples):
        has_augmented = num_channels > 4 and bool(torch.any(s["track"][:, first_augmented:num_channels] > 0).item())
        (active if has_augmented else other).append(idx)
    return (active + other)[:num_heatmaps]


def generate_saliency_heatmaps(
    model: nn.Module,
    dataset: MultiTrackDataset,
    device: torch.device,
    output_dir: Path,
    channel_names: list,
    channel_colors: list,
    amp_dtype: torch.dtype,
    num_heatmaps: int = 5,
    specific_tx_ids: str = None,
) -> list:
    """Generates 2D Sequence Saliency Heatmaps (N Channels x Position) for selected transcripts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = dataset.samples
    num_channels = len(channel_names)

    selected_indices = select_heatmap_samples(samples, num_channels, num_heatmaps, specific_tx_ids)
    print(f"\n[xAI] Generating 2D Saliency Heatmaps for {len(selected_indices)} transcripts...")
    heatmap_records = []

    model.eval()
    for idx in selected_indices:
        sample = samples[idx]
        tx_id = sample["tx_id"]
        gene = sample["gene"]
        seq_len = sample["length"]
        target = sample["target"]

        x = sample["track"].unsqueeze(0).to(device)
        length = torch.tensor([seq_len], dtype=torch.long, device=device)
        pred, grad_x = input_gradients(model, x, length, device, amp_dtype)
        pred_val = float(pred.item())

        # Attribution: signed (x * grad)
        signed_attr = (x * grad_x)[0, :seq_len, :num_channels].float().cpu().numpy()  # (L, N)
        x_raw = x[0, :seq_len, :num_channels].cpu().numpy()  # (L, N)

        attr_t = signed_attr.T  # (N, L)

        # Calculate symmetric color limits based on 99th percentile of attribution
        vlim = float(np.percentile(np.abs(attr_t), 99.0))
        if vlim <= 1e-6:
            vlim = float(np.max(np.abs(attr_t)))
        if vlim <= 1e-6:
            vlim = 1.0

        # Plot 3-panel figure: 2D Heatmap, Input Annotations, 1D Importance Profile
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3.2, 1.2, 1.4]}, sharex=True)

        # 1. 2D Saliency Heatmap
        im = axes[0].imshow(
            attr_t,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        axes[0].set_yticks(range(num_channels))
        axes[0].set_yticklabels(channel_names, fontsize=10, fontweight="bold")
        axes[0].set_title(
            f"Transcript: {tx_id} ({gene}) | Length: {seq_len} nt | True HL: {target:.2f} | Pred HL: {pred_val:.2f}",
            fontsize=13, fontweight="bold", pad=10
        )
        cbar = fig.colorbar(im, ax=axes[0], orientation="vertical", pad=0.015, fraction=0.02)
        cbar.set_label("Attribution (x * Grad)\nBlue: Destabilizing | Red: Stabilizing", fontsize=9, fontweight="bold")

        # 2. Input Track Annotations (Channels 4 to N-1)
        positions = np.arange(seq_len)
        has_annotation_plot = False
        for c in range(4, num_channels):
            if np.any(x_raw[:, c] > 0):
                c_name = channel_names[c]
                c_col = channel_colors[c]
                axes[1].fill_between(positions, 0, x_raw[:, c], color=c_col, alpha=0.5, label=f"{c_name} (Ch {c})")
                has_annotation_plot = True

        if not has_annotation_plot:
            axes[1].text(0.5, 0.5, "No non-sequence tracks (CDS/Splice/etc.) present for this sequence",
                         horizontalalignment="center", verticalalignment="center", transform=axes[1].transAxes, fontsize=10)

        axes[1].set_ylabel("Input Signals", fontsize=10, fontweight="bold")
        axes[1].set_ylim(-0.05, 1.1)
        if has_annotation_plot:
            axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.85)
        axes[1].grid(True, alpha=0.3)

        # 3. 1D Integrated Position Importance Profile
        # Sum of absolute attributions across all N channels
        pos_importance = np.sum(np.abs(signed_attr), axis=1)
        axes[2].fill_between(positions, 0, pos_importance, color="#4a148c", alpha=0.45)
        axes[2].plot(positions, pos_importance, color="#4a148c", lw=1.2, label="Positional Impact (Sum |Attr|)")
        # Also net attribution (dashed line)
        net_attr = np.sum(signed_attr, axis=1)
        axes[2].plot(positions, net_attr, color="#ff7f0e", lw=1.0, linestyle="--", label="Net Direction (Sum Attr)")
        axes[2].axhline(0, color="gray", linestyle=":", lw=0.8)
        axes[2].set_ylabel("1D Saliency", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Nucleotide Position along mRNA (5' -> 3')", fontsize=11, fontweight="bold")
        axes[2].legend(loc="upper right", fontsize=8.5, framealpha=0.85)
        axes[2].grid(True, alpha=0.3)

        clean_tx = tx_id.replace(".", "_")
        clean_gene = gene.replace(" ", "_") if gene else "unknown"
        png_path = output_dir / f"heatmap_{clean_tx}_{clean_gene}.png"
        pdf_path = output_dir / f"heatmap_{clean_tx}_{clean_gene}.pdf"

        plt.tight_layout()
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)

        heatmap_records.append({
            "transcript_id": tx_id,
            "gene": gene,
            "length": seq_len,
            "target": target,
            "prediction": pred_val,
            "png_file": str(png_path.name),
            "pdf_file": str(pdf_path.name),
        })

    print(f"[xAI] Saved {len(heatmap_records)} 2D Saliency Heatmaps to: {output_dir}")
    return heatmap_records


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
    model.requires_grad_(False)  # Input gradients only: the graph does not need to track the parameters
    return model, ckpt.get("best_epoch"), ckpt_path


def analyze_fold(
    fold_dir: Path,
    test_loader: DataLoader,
    channel_names: list,
    channel_colors: list,
    device: torch.device,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict:
    """Runs all analyses for one fold and writes xai_channel_summary.{json,png,pdf} and xai_heatmaps/ into fold_dir."""
    model, best_epoch, ckpt_path = load_fold_model(fold_dir, channel_names, device)

    y_true, y_pred = predict(model, test_loader, device, amp_dtype)
    test_metrics = regression_metrics(y_true, y_pred)

    norms = compute_embedding_norms(model, num_channels=len(channel_names))
    grad_results = compute_channel_importance_gradients(model, test_loader, device, channel_names, args.max_samples, amp_dtype)
    ablation_results = compute_channel_ablation(model, test_loader, device, test_metrics, channel_names, amp_dtype)

    print("\n" + "=" * 80)
    print(f"{'Channel':<18} {'Input*Grad (%)':<16} {'Ablation Drop (Delta r)':<25} {'Embedding Norm':<15}")
    print("=" * 80)
    for idx, name in enumerate(channel_names):
        print(f"{name:<18} {grad_results[name]['pct_attribution']:>10.2f}% {ablation_results[name]['drop_pearson_r']:>20.4f} {norms[idx]:>16.4f}")
    print("=" * 80)

    plot_channel_importance_summary(
        grad_pcts=[grad_results[n]["pct_attribution"] for n in channel_names],
        ablation_drops=[ablation_results[n]["drop_pearson_r"] for n in channel_names],
        embedding_norms=norms,
        channel_names=channel_names,
        channel_colors=channel_colors,
        output_png=fold_dir / "xai_channel_summary.png",
    )

    heatmap_records = generate_saliency_heatmaps(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        output_dir=fold_dir / "xai_heatmaps",
        channel_names=channel_names,
        channel_colors=channel_colors,
        amp_dtype=amp_dtype,
        num_heatmaps=args.num_heatmaps,
        specific_tx_ids=args.tx_ids,
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "channel_names": channel_names,
        "embedding_norms": {name: norms[i] for i, name in enumerate(channel_names)},
        "gradient_attribution": grad_results,
        "channel_ablation": ablation_results,
        "heatmaps": heatmap_records,
    }
    summary_file = fold_dir / "xai_channel_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[xAI] Saved comprehensive xAI summary report to: {summary_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_folds(fold_summaries: dict, channel_names: list) -> dict:
    """
    Mean and standard deviation (np.std, as in the fine-tuning summary) of the channel statistics across folds.
    All folds share one test set and overlapping training data: the spread shows the sensitivity to the fold,
    it is not a confidence interval.
    """
    def channel_stats(values_of) -> dict:
        stats = {}
        for name in channel_names:
            values = [float(values_of(s, name)) for s in fold_summaries.values()]
            stats[name] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_fold": values}
        return stats

    return {
        "folds": sorted(fold_summaries),
        "channel_names": channel_names,
        "pct_attribution": channel_stats(lambda s, n: s["gradient_attribution"][n]["pct_attribution"]),
        "drop_pearson_r": channel_stats(lambda s, n: s["channel_ablation"][n]["drop_pearson_r"]),
        "embedding_norm": channel_stats(lambda s, n: s["embedding_norms"][n]),
    }


def main():
    parser = argparse.ArgumentParser(description="xAI analysis of fine-tuned N-track Orthrus models", allow_abbrev=False)
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
        "--max_samples",
        type=int,
        default=500,
        help="Test transcripts used for Input*Gradient. The loader is sorted by length, so a smaller value keeps only the shortest transcripts (default: 500)",
    )
    parser.add_argument("--num_heatmaps", type=int, default=5, help="Number of transcripts to generate 2D saliency heatmaps for (default: 5)")
    parser.add_argument("--tx_ids", type=str, default=None, help="Optional comma-separated transcript IDs to generate 2D saliency heatmaps for")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    print(f"Using device: {device} (precision: {amp_dtype})")

    data = load_track_data(Path(args.data_path))
    run_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else resolve_run_dir(data, args.run_name)
    print(f"[Checkpoints] Run directory: {run_dir}")

    test_idx = np.where(np.isin(data.sample_splits, TEST_SPLITS))[0]
    if len(test_idx) == 0:
        raise ValueError(f"[Error] No test samples found! Test splits: {TEST_SPLITS}")
    print(f"Test samples: {len(test_idx)} (Splits {TEST_SPLITS})")
    test_loader = make_loader(data, test_idx, args.batch_size, args.max_length, shuffle=False, device=device)

    channel_names = data.channel_names
    channel_colors = get_channel_colors(channel_names)
    folds = list(CV_FOLDS) if args.fold == "all" else [int(args.fold)]

    fold_summaries = {}
    for fold_id in folds:
        print("\n" + "=" * 60)
        print(f"                 xAI ANALYSIS: FOLD {fold_id}                 ")
        print("=" * 60)
        fold_summaries[fold_id] = analyze_fold(run_dir / f"fold_{fold_id}", test_loader, channel_names, channel_colors, device, amp_dtype, args)

    if len(folds) > 1:
        aggregated = aggregate_folds(fold_summaries, channel_names)
        plot_channel_importance_summary(
            grad_pcts=[aggregated["pct_attribution"][n]["mean"] for n in channel_names],
            ablation_drops=[aggregated["drop_pearson_r"][n]["mean"] for n in channel_names],
            embedding_norms=[aggregated["embedding_norm"][n]["mean"] for n in channel_names],
            channel_names=channel_names,
            channel_colors=channel_colors,
            output_png=run_dir / "xai_aggregated.png",
            errors={
                "grad": [aggregated["pct_attribution"][n]["std"] for n in channel_names],
                "ablation": [aggregated["drop_pearson_r"][n]["std"] for n in channel_names],
                "norms": [aggregated["embedding_norm"][n]["std"] for n in channel_names],
            },
        )
        aggregated_file = run_dir / "xai_aggregated.json"
        with open(aggregated_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"[xAI] Saved aggregated report across folds to: {aggregated_file}")


if __name__ == "__main__":
    main()
