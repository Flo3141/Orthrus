#!/usr/bin/env python3
"""
End-to-End Supervised Fine-Tuning of the 6-Track Orthrus Model on hIPSC_CM Dataset.

Features:
- Loads the pretrained 6-track Orthrus model (quietflamingo/orthrus-large-6-track)
- Attaches the exact same regression projection head as 8-track fine-tuning
- Slices/loads 6-channel multi-track RNA representations (channels 0-5: A, C, G, U, CDS, Splice)
- Standardized 10-fold gene-grouped split integration (Train: 0-5, Val: 6-7, Test: 8-9)
- Dynamic length padding with bucketing to minimize padding overhead
- Cosine Annealing with Linear Warmup + Mixed Precision (bf16) + Huber loss
- Saves best checkpoint based on validation Pearson r and exports fine-tuned backbone in Hugging Face format
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoModel

# Headless backend for cluster servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. Dataset & Length-Grouped Dynamic Batching
# =============================================================================

class MultiTrackDataset(Dataset):
    """
    Dataset wrapping variable-length 6-track RNA representations and target values.
    """
    def __init__(self, tracks: list, targets: np.ndarray, genes: np.ndarray, transcript_ids: np.ndarray, max_length: int = 12288):
        self.samples = []
        for i in range(len(tracks)):
            tr = tracks[i]
            # Ensure strictly 6 channels (slice if input contains 8 channels)
            if tr.shape[1] > 6:
                tr = tr[:, :6]
            elif tr.shape[1] < 6:
                raise ValueError(f"Sample {i} has only {tr.shape[1]} channels, but 6-track requires 6 channels.")

            if tr.shape[0] > max_length:
                tr = tr[:max_length, :]
            self.samples.append({
                "track": torch.from_numpy(tr).float(),
                "length": tr.shape[0],
                "target": float(targets[i]),
                "gene": str(genes[i]),
                "tx_id": str(transcript_ids[i]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BucketBatchSampler(Sampler):
    """
    Batches sequences of similar lengths together to minimize zero-padding overhead.
    """
    def __init__(self, dataset: MultiTrackDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        indices_and_lens = [(i, dataset.samples[i]["length"]) for i in range(len(dataset))]
        indices_and_lens.sort(key=lambda x: x[1])
        self.sorted_indices = [x[0] for x in indices_and_lens]

    def __iter__(self):
        batches = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pad_collate_fn(batch: list) -> dict:
    """
    Dynamically pads sequences in the batch to the maximum sequence length of that specific batch.
    """
    b_lens = [s["length"] for s in batch]
    max_b_len = max(b_lens)
    n_channels = 6

    padded_tracks = torch.zeros(len(batch), max_b_len, n_channels, dtype=torch.float32)
    targets = torch.tensor([s["target"] for s in batch], dtype=torch.float32)
    lengths = torch.tensor(b_lens, dtype=torch.long)

    for i, s in enumerate(batch):
        l = s["length"]
        padded_tracks[i, :l, :] = s["track"]

    return {
        "x": padded_tracks,
        "lengths": lengths,
        "targets": targets,
        "genes": [s["gene"] for s in batch],
        "tx_ids": [s["tx_id"] for s in batch],
    }


# =============================================================================
# 2. Model Architecture: 6-Track Backbone + Projection Head
# =============================================================================

class OrthrusRegressionModel(nn.Module):
    """
    Wraps the 6-track Orthrus backbone with the identical regression head used for 8-track.
    """
    def __init__(self, backbone: nn.Module, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        # Projection head: Linear -> LayerNorm -> GELU -> Dropout -> Linear(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        rep = self.backbone.representation(x, lengths, channel_last=True)
        out = self.head(rep).squeeze(-1)
        return out


# =============================================================================
# 3. Evaluation Loops & Plotting
# =============================================================================

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, loss_fn: nn.Module, amp_dtype: torch.dtype = torch.bfloat16) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
                preds = model(x, lengths)
                loss = loss_fn(preds, targets)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * len(targets)
            all_preds.extend(preds.detach().cpu().float().numpy())
            all_targets.extend(targets.detach().cpu().float().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    if np.isnan(y_pred).any() or np.isnan(y_true).any():
        nan_count = int(np.isnan(y_pred).sum())
        print(f"\n[Warning] {nan_count} NaN values detected in predictions during evaluation! Replacing with fallback 0.0.")
        valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        if valid_mask.sum() > 2:
            p_corr, p_val = pearsonr(y_true[valid_mask], y_pred[valid_mask])
            s_corr, s_val = spearmanr(y_true[valid_mask], y_pred[valid_mask])
        else:
            p_corr, p_val = 0.0, 1.0
            s_corr, s_val = 0.0, 1.0
        y_pred = np.nan_to_num(y_pred, nan=0.0)
    else:
        p_corr, p_val = pearsonr(y_true, y_pred)
        s_corr, s_val = spearmanr(y_true, y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    avg_loss = total_loss / max(1, len(y_true))

    return {
        "loss": avg_loss,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(p_corr),
        "pearson_pval": float(p_val),
        "spearman_rho": float(s_corr),
        "spearman_pval": float(s_val),
    }


def plot_and_save_training_curves(history: list, output_file: Path):
    """
    Plots training & validation loss, Pearson r, Spearman rho, and RMSE across epochs.
    """
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_metrics"]["loss"] for h in history]
    val_pearson = [h["val_metrics"]["pearson_r"] for h in history]
    val_spearman = [h["val_metrics"]["spearman_rho"] for h in history]
    val_rmse = [h["val_metrics"]["rmse"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train vs Val Loss
    axes[0, 0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=2, marker="o", markersize=4)
    axes[0, 0].plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", lw=2, linestyle="--", marker="s", markersize=4)
    axes[0, 0].set_title("Loss Curves (Train vs. Validation)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=11)

    # 2. Validation Pearson r
    best_r_idx = int(np.argmax(val_pearson))
    axes[0, 1].plot(epochs, val_pearson, label="Validation Pearson r", color="#2ca02c", lw=2, marker="o", markersize=4)
    axes[0, 1].scatter([epochs[best_r_idx]], [val_pearson[best_r_idx]], color="red", s=90, zorder=5,
                       label=f"Best r: {val_pearson[best_r_idx]:.4f} (Ep {epochs[best_r_idx]})")
    axes[0, 1].set_title("Validation Pearson Correlation (r)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("Pearson r", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=11)

    # 3. Validation Spearman rho
    axes[1, 0].plot(epochs, val_spearman, label="Validation Spearman rho", color="#9467bd", lw=2, marker="o", markersize=4)
    axes[1, 0].set_title("Validation Spearman Rank Correlation (rho)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("Spearman rho", fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=11)

    # 4. Validation RMSE
    axes[1, 1].plot(epochs, val_rmse, label="Validation RMSE", color="#d62728", lw=2, marker="o", markersize=4)
    axes[1, 1].set_title("Validation Root Mean Squared Error (RMSE)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("RMSE", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=11)

    plt.suptitle("Orthrus 6-Track Fine-Tuning Performance Across Epochs", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Training curves saved to: {output_file}")


# =============================================================================
# 4. Training Loop
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    warmup_epochs: int,
    output_dir: Path,
    resume: bool = True,
    precision: str = "bf16",
    loss_fn_name: str = "huber",
    max_grad_norm: float = 1.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select loss function
    if loss_fn_name == "huber":
        loss_fn = nn.HuberLoss(delta=1.0)
    elif loss_fn_name == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()

    # Mixed precision setup
    if precision == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False
        print("[Precision] Using bfloat16 Mixed Precision (bf16). GradScaler disabled.")
    elif precision == "fp16":
        amp_dtype = torch.float16
        use_scaler = True
        print("[Precision] Using float16 Mixed Precision (fp16) with GradScaler.")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        print("[Precision] Using Full Precision (fp32).")

    # Optimizer with differential learning rate for backbone vs head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ])

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    # Cosine scheduler with linear warmup
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_scaler))

    best_val_r = -1.0
    best_epoch = -1
    start_epoch = 1
    history = []

    best_ckpt_path = output_dir / "best_model.pt"
    latest_ckpt_path = output_dir / "latest_checkpoint.pt"
    best_backbone_dir = output_dir / "best_finetuned_backbone"
    curves_path = output_dir / "loss_curves.png"
    csv_log_path = output_dir / "training_log.csv"

    # Check for resume
    if resume and latest_ckpt_path.exists():
        print(f"\n[Resume] Found existing checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_r = ckpt.get("best_val_r", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resumed from Epoch {ckpt['epoch']}. Next training epoch: {start_epoch}/{epochs}")

    print("\nStarting 6-Track Fine-Tuning Training...")
    print(f"Total Epochs:      {epochs}")
    print(f"Batches per Epoch: {len(train_loader)}")
    print(f"Warmup Epochs:     {warmup_epochs} ({warmup_steps} steps)")
    print(f"Loss Function:     {loss_fn_name.upper()}")
    print(f"LR Backbone:       {lr_backbone:.2e}")
    print(f"LR Head:           {lr_head:.2e}\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch_idx, batch in enumerate(pbar):
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and precision != "fp32")):
                preds = model(x, lengths)
                loss = loss_fn(preds, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Warning] NaN/Inf loss encountered at Epoch {epoch}, Batch {batch_idx}! Skipping step.")
                optimizer.zero_grad()
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            scheduler.step()
            running_loss += loss.item() * len(targets)
            n_samples += len(targets)

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{cur_lr:.2e}"})

        epoch_train_loss = running_loss / max(1, n_samples)

        # Validation
        val_metrics = evaluate(model, val_loader, device, loss_fn, amp_dtype=amp_dtype)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Pearson r: {val_metrics['pearson_r']:.4f} | "
            f"Val Spearman rho: {val_metrics['spearman_rho']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_metrics": val_metrics,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        })

        is_best = val_metrics["pearson_r"] > best_val_r
        if is_best:
            best_val_r = val_metrics["pearson_r"]
            best_epoch = epoch
            print(f"  >>> New Best Validation Pearson r: {best_val_r:.4f} at Epoch {epoch}! Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_r": best_val_r,
                "best_epoch": best_epoch,
                "history": history,
            }, best_ckpt_path)

            best_backbone_dir.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_backbone_dir)

        # Save latest checkpoint
        latest_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_r": best_val_r,
            "best_epoch": best_epoch,
            "history": history,
        }
        if use_scaler:
            latest_dict["scaler_state_dict"] = scaler.state_dict()
        torch.save(latest_dict, latest_ckpt_path)

        # Periodic curves & CSV update
        if epoch % 2 == 0 or epoch == epochs:
            plot_and_save_training_curves(history, curves_path)

    # Save final CSV log
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_pearson_r", "val_spearman_rho", "val_mse", "val_rmse", "val_r2"])
        for h in history:
            vm = h["val_metrics"]
            writer.writerow([h["epoch"], h["train_loss"], vm["loss"], vm["pearson_r"], vm["spearman_rho"], vm["mse"], vm["rmse"], vm["r2"]])

    # Evaluate best model on test set
    if test_loader is not None and best_ckpt_path.exists():
        print("\nLoading best model checkpoint for Test Set Evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, loss_fn, amp_dtype=amp_dtype)

        print("=" * 60)
        print("           6-TRACK TEST SET METRICS (Folds 8 & 9)         ")
        print("=" * 60)
        for k, v in test_metrics.items():
            if "pval" in k:
                print(f"  {k:20s}: {v:.3e}")
            else:
                print(f"  {k:20s}: {v:.4f}")
        print("=" * 60)

        summary_file = output_dir / "training_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "model_type": "6-track",
                "best_epoch": best_epoch,
                "best_val_pearson_r": best_val_r,
                "test_metrics": test_metrics,
                "history": history,
            }, f, indent=2)
        print(f"Summary report saved to: {summary_file}")


# =============================================================================
# 5. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning of 6-Track Orthrus Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_multitrack_with_trans_factors_minmax.npz",
        help="Path to dataset NPZ file (first 6 tracks will be used)",
    )
    parser.add_argument(
        "--splits_lookup_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv",
        help="Path to standardized 10-fold split lookup table CSV (from create_hipsc_cm_splits.py)",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["lookup", "gene"],
        default="lookup",
        help="'lookup' (standardized 10-fold table: Train 0-5, Val 6-7, Test 8-9) or 'gene' (ad-hoc GroupShuffleSplit)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Path or HF ID for pretrained 6-track Orthrus model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus_6track_finetuned_hIPSC_CM",
        help="Directory to save fine-tuned checkpoints and logs",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="half_life_transformed",
        help="Target column in NPZ archive (default: half_life_transformed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of fine-tuning epochs (default: 25)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=5e-5,
        help="Learning rate for Mamba backbone layers (default: 5e-5)",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=3e-4,
        help="Learning rate for regression head (default: 3e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW (default: 0.01)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=3,
        help="Linear warmup epochs (default: 3)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.10,
        help="Fraction of genes reserved for test set if fallback to 'gene' (default: 0.10)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.10,
        help="Fraction of genes reserved for validation set if fallback to 'gene' (default: 0.10)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximum sequence length (default: 12288)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume training from latest_checkpoint.pt if it exists in output_dir",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Mixed precision mode (default: bf16)",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        choices=["huber", "mse", "smooth_l1"],
        default="huber",
        help="Loss function: 'huber', 'mse', or 'smooth_l1'",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping (default: 1.0)",
    )
    args = parser.parse_args()

    # Set seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load multi-track NPZ dataset
    data_file = Path(args.data_path)
    print(f"Loading dataset for 6-track fine-tuning: {data_file}...")
    npz_data = np.load(data_file, allow_pickle=True)

    tracks = npz_data["tracks"]
    targets = npz_data[args.target_col].astype(np.float32)

    # Use gene symbol if present, else gene id
    if "hgnc_symbol" in npz_data:
        genes = npz_data["hgnc_symbol"].astype(str)
    elif "ensembl_gene_id" in npz_data:
        genes = npz_data["ensembl_gene_id"].astype(str)
    else:
        genes = np.array([f"gene_{i}" for i in range(len(targets))])

    tx_ids = npz_data["ensembl_transcript_id"].astype(str) if "ensembl_transcript_id" in npz_data else np.array([f"tx_{i}" for i in range(len(targets))])

    # Clean invalid samples (NaN targets, Inf targets, empty sequences)
    valid_mask = ~np.isnan(targets) & ~np.isinf(targets)
    if "seq_lens" in npz_data:
        valid_mask &= (npz_data["seq_lens"] > 0)
    else:
        valid_mask &= np.array([len(t) > 0 for t in tracks])

    n_invalid = int(np.sum(~valid_mask))
    if n_invalid > 0:
        print(f"[Data Cleaning] Filtered out {n_invalid} samples with NaN/Inf targets or empty sequence length.")
        tracks = [t for i, t in enumerate(tracks) if valid_mask[i]]
        targets = targets[valid_mask]
        genes = genes[valid_mask]
        tx_ids = tx_ids[valid_mask]

    # Strictly enforce/slice 6 tracks
    raw_channels = tracks[0].shape[1]
    if raw_channels > 6:
        print(f"[6-Track Slicing] Input has {raw_channels} channels. Slicing first 6 channels (A, C, G, U, CDS, Splice)...")
        tracks = [t[:, :6] for t in tracks]
    elif raw_channels < 6:
        raise ValueError(f"Expected at least 6 channels, but input only has {raw_channels} channels.")

    print(f"Total valid samples loaded: {len(targets)}")
    print(f"Unique genes:              {len(np.unique(genes))}")
    print(f"Input channels:            6 (A, C, G, U, CDS, Splice)")
    print(f"Target column:             {args.target_col}")

    # 2. Train / Val / Test splitting
    lookup_path = Path(args.splits_lookup_path)
    use_lookup = (args.split_type == "lookup" and lookup_path.exists())

    if args.split_type == "lookup" and not lookup_path.exists():
        print(f"\n[Warning] Splits lookup table not found at: {lookup_path}")
        print("Falling back to ad-hoc Gene-Grouped GroupShuffleSplit.")
        use_lookup = False

    if use_lookup:
        print(f"\nUsing Standardized 10-Fold Lookup Table: {lookup_path}")
        lookup_df = pd.read_csv(lookup_path)

        lookup_tx_col = "ensembl_transcript_id" if "ensembl_transcript_id" in lookup_df.columns else "transcript_id"
        tx_to_split = dict(zip(lookup_df[lookup_tx_col].astype(str).str.strip(), lookup_df["split"].astype(int)))

        sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in tx_ids])
        unmatched_count = int(np.sum(sample_splits == -1))
        if unmatched_count > 0:
            print(f"[Warning] {unmatched_count} transcripts not found in lookup table! Filtering them out.")
            matched_mask = (sample_splits != -1)
            tracks = [t for i, t in enumerate(tracks) if matched_mask[i]]
            targets = targets[matched_mask]
            genes = genes[matched_mask]
            tx_ids = tx_ids[matched_mask]
            sample_splits = sample_splits[matched_mask]

        train_idx = np.where(np.isin(sample_splits, [0, 1, 2, 3, 4, 5]))[0]
        val_idx = np.where(np.isin(sample_splits, [6, 7]))[0]
        test_idx = np.where(np.isin(sample_splits, [8, 9]))[0]

        print(f"\n[Split Breakdown from Lookup Table]")
        print(f"  Train (Splits 0-5): {len(train_idx)} samples ({len(np.unique(genes[train_idx]))} genes, {len(train_idx)/len(targets)*100:.1f}%)")
        print(f"  Val   (Splits 6-7): {len(val_idx)} samples ({len(np.unique(genes[val_idx]))} genes, {len(val_idx)/len(targets)*100:.1f}%)")
        print(f"  Test  (Splits 8-9): {len(test_idx)} samples ({len(np.unique(genes[test_idx]))} genes, {len(test_idx)/len(targets)*100:.1f}%)")
    else:
        print("\nCreating Ad-hoc Gene-Grouped splits (GroupShuffleSplit)...")
        gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_seed)
        train_val_idx, test_idx = next(gss_test.split(tracks, targets, groups=genes))

        val_rel_size = args.val_size / (1.0 - args.test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_size, random_state=args.random_seed)
        train_sub_idx, val_sub_idx = next(gss_val.split(tracks[train_val_idx], targets[train_val_idx], groups=genes[train_val_idx]))

        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

        print(f"  Train samples: {len(train_idx)} ({len(np.unique(genes[train_idx]))} genes)")
        print(f"  Val samples:   {len(val_idx)} ({len(np.unique(genes[val_idx]))} genes)")
        print(f"  Test samples:  {len(test_idx)} ({len(np.unique(genes[test_idx]))} genes)")

    # 3. Create Datasets & DataLoaders
    train_ds = MultiTrackDataset([tracks[i] for i in train_idx], targets[train_idx], genes[train_idx], tx_ids[train_idx], args.max_length)
    val_ds = MultiTrackDataset([tracks[i] for i in val_idx], targets[val_idx], genes[val_idx], tx_ids[val_idx], args.max_length)
    test_ds = MultiTrackDataset([tracks[i] for i in test_idx], targets[test_idx], genes[test_idx], tx_ids[test_idx], args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(train_ds, batch_size=args.batch_size, shuffle=True),
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(val_ds, batch_size=args.batch_size, shuffle=False),
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=BucketBatchSampler(test_ds, batch_size=args.batch_size, shuffle=False),
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Load 6-track model and build regression model
    print(f"\nLoading 6-track Orthrus checkpoint from: {args.model_checkpoint}...")
    backbone = AutoModel.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    d_model = getattr(backbone.config, "ssm_model_dim", 512)

    model = OrthrusRegressionModel(backbone=backbone, d_model=d_model, hidden_dim=256, dropout=0.1)
    model.to(device)

    # 5. Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        output_dir=output_dir,
        resume=args.resume,
        precision=args.precision,
        loss_fn_name=args.loss_fn,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()
