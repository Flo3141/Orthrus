#!/usr/bin/env python3
"""
End-to-End Supervised Fine-Tuning of the 8-Track Orthrus Model on Trans-Factor Augmented Datasets.

Features:
- Loads the 8-track converted Orthrus model (AutoModel with trust_remote_code=True)
- Attaches an Orthrus-style regression projection head
- Loads multi-track data (channels: A, C, G, U, CDS, Splice, miRNA, eCLIP) from generate_trans_factor_tracks.py
- Gene-grouped Train / Validation / Test splitting (avoids data leakage across isoforms)
- Dynamic length padding with bucketing to minimize padding overhead
- Differential learning rates:
    - Backbone Mamba layers: low LR (e.g. 5e-5)
    - Input embedding (channels 6 & 7): higher LR (e.g. 2e-4) to rapidly adapt new signals
    - Regression projection head: higher LR (e.g. 3e-4)
- Mixed-precision training (torch.cuda.amp) with Cosine Annealing + Linear Warmup
- Saves the best checkpoint based on validation Pearson r and exports fine-tuned backbone in Hugging Face format
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
    Dataset wrapping variable-length multi-track RNA representations and target values.
    """
    def __init__(self, tracks: list, targets: np.ndarray, genes: np.ndarray, transcript_ids: np.ndarray, max_length: int = 12288):
        self.samples = []
        for i in range(len(tracks)):
            tr = tracks[i]
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
    Batches sequences of similar lengths together to dramatically minimize zero-padding overhead.
    """
    def __init__(self, dataset: MultiTrackDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Sort indices by sequence length
        indices_and_lens = [(i, dataset.samples[i]["length"]) for i in range(len(dataset))]
        indices_and_lens.sort(key=lambda x: x[1])
        self.sorted_indices = [x[0] for x in indices_and_lens]

    def __iter__(self):
        # Create batches of similar length
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
    n_channels = batch[0]["track"].shape[1]

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
# 2. Model Architecture: 8-Track Backbone + Projection Head
# =============================================================================

class OrthrusRegressionModel(nn.Module):
    """
    Wraps the Orthrus backbone with a regression head (matching the Orthrus downstream setup).
    """
    def __init__(self, backbone: nn.Module, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        # Projection head: Linear -> LayerNorm -> ReLU -> Dropout -> Linear(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Representation: (B, d_model) via mean pooling over unpadded sequence
        rep = self.backbone.representation(x, lengths, channel_last=True)
        out = self.head(rep).squeeze(-1)  # (B,)
        return out


# =============================================================================
# 3. Training & Evaluation Loops
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

    # Protect against any NaNs in predictions
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
    Plots training & validation loss, Pearson r, Spearman rho, and MSE/RMSE across epochs.
    """
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_metrics"]["loss"] for h in history]
    val_pearson = [h["val_metrics"]["pearson_r"] for h in history]
    val_spearman = [h["val_metrics"]["spearman_rho"] for h in history]
    val_mse = [h["val_metrics"]["mse"] for h in history]
    val_rmse = [h["val_metrics"]["rmse"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train vs Val Loss
    axes[0, 0].plot(epochs, train_loss, label="Train Loss (MSE)", color="#1f77b4", lw=2, marker="o", markersize=4)
    axes[0, 0].plot(epochs, val_loss, label="Val Loss (MSE)", color="#ff7f0e", lw=2, linestyle="--", marker="s", markersize=4)
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

    # 4. Validation Error (MSE & RMSE)
    axes[1, 1].plot(epochs, val_mse, label="Val MSE", color="#d62728", lw=2, marker="o", markersize=4)
    axes[1, 1].plot(epochs, val_rmse, label="Val RMSE", color="#8c564b", lw=2, linestyle=":", marker="^", markersize=4)
    axes[1, 1].set_title("Validation Error (MSE & RMSE)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Error", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def save_csv_log(history: list, csv_path: Path):
    """
    Saves or appends training history to a CSV file.
    """
    fieldnames = ["epoch", "train_loss", "val_loss", "val_pearson_r", "val_spearman_rho", "val_mse", "val_rmse", "val_r2"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in history:
            vm = h["val_metrics"]
            writer.writerow({
                "epoch": h["epoch"],
                "train_loss": f"{h['train_loss']:.6f}",
                "val_loss": f"{vm['loss']:.6f}",
                "val_pearson_r": f"{vm['pearson_r']:.6f}",
                "val_spearman_rho": f"{vm['spearman_rho']:.6f}",
                "val_mse": f"{vm['mse']:.6f}",
                "val_rmse": f"{vm['rmse']:.6f}",
                "val_r2": f"{vm['r2']:.6f}",
            })


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_backbone: float,
    lr_embedding: float,
    lr_head: float,
    weight_decay: float,
    warmup_epochs: int,
    output_dir: Path,
    resume: bool = True,
    loss_fn_name: str = "huber",
    max_grad_norm: float = 1.0,
):
    # Loss Function: Huber loss prevents gradient explosions from outliers
    if loss_fn_name == "huber":
        loss_fn = nn.HuberLoss(delta=1.0)
    elif loss_fn_name == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        loss_fn = nn.MSELoss()

    # Precision: BFloat16 is strictly required for Mamba SSM models (falls back to fp32 on CPU/unsupported hardware)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
        print("[Precision] Using Native BFloat16 (bf16) mixed precision (strictly required for Mamba SSM).")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        print("[Precision] Using Full Precision (fp32) (CUDA bf16 not available).")

    # Differential learning rate parameter groups
    backbone_params = []
    embedding_params = []

    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if "embedding" in name:
            embedding_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": embedding_params, "lr": lr_embedding, "weight_decay": weight_decay},
        {"params": model.head.parameters(), "lr": lr_head, "weight_decay": weight_decay},
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
        print("Loading model, optimizer, scheduler, and scaler state...")
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

        if "random_states" in ckpt:
            try:
                rs = ckpt["random_states"]
                if rs.get("torch") is not None:
                    # torch.set_rng_state strictly expects a CPU ByteTensor
                    torch_state = rs["torch"].cpu() if isinstance(rs["torch"], torch.Tensor) else rs["torch"]
                    torch.set_rng_state(torch_state)
                if rs.get("cuda") is not None and torch.cuda.is_available():
                    cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in rs["cuda"]]
                    torch.cuda.set_rng_state_all(cuda_states)
                if rs.get("numpy") is not None:
                    np.random.set_state(rs["numpy"])
                if rs.get("python") is not None:
                    random.setstate(rs["python"])
            except Exception as e:
                print(f"[Warning] Could not fully restore random states ({e}), proceeding with current RNG state.")

        print(f"[Resume] Successfully resumed from Epoch {ckpt['epoch']}. Next training epoch: {start_epoch}/{epochs}")
        if start_epoch > epochs:
            print(f"[Notice] Training already completed ({ckpt['epoch']} >= {epochs} epochs). Skipping to evaluation.")

    print("\nStarting Fine-Tuning Training...")
    print(f"Total Epochs:      {epochs}")
    print(f"Batches per Epoch: {len(train_loader)}")
    print(f"Warmup Epochs:     {warmup_epochs} ({warmup_steps} steps)")
    print(f"Loss Function:     {loss_fn_name.upper()}")
    print(f"LR Backbone:       {lr_backbone:.2e}")
    print(f"LR Embedding:      {lr_embedding:.2e}")
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
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
                preds = model(x, lengths)
                loss = loss_fn(preds, targets)

            # Safeguard 1: Skip batch if loss is NaN/Inf to prevent corrupting weights
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Warning] NaN/Inf loss encountered at Epoch {epoch}, Batch {batch_idx}! Skipping step.")
                optimizer.zero_grad()
                continue

            if use_scaler:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                # Safeguard 2: Check gradient norm for NaN
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[Warning] NaN/Inf gradient norm encountered at Epoch {epoch}, Batch {batch_idx}! Skipping step.")
                    optimizer.zero_grad()
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after >= scale_before:
                    scheduler.step()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                # Safeguard 2: Check gradient norm for NaN
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n[Warning] NaN/Inf gradient norm encountered at Epoch {epoch}, Batch {batch_idx}! Skipping step.")
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                scheduler.step()

            running_loss += loss.item() * len(targets)
            n_samples += len(targets)
            pbar.set_postfix({"train_loss": f"{running_loss / max(1, n_samples):.4f}"})

        train_loss = running_loss / max(1, n_samples)
        val_metrics = evaluate(model, val_loader, device, loss_fn, amp_dtype=amp_dtype)

        val_r = val_metrics["pearson_r"]
        val_rho = val_metrics["spearman_rho"]
        val_mse = val_metrics["mse"]

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Pearson r: {val_r:.4f} | Val Spearman rho: {val_rho:.4f} | Val MSE: {val_mse:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        })

        # 1. Update Best Model Checkpoint
        if val_r > best_val_r:
            best_val_r = val_r
            best_epoch = epoch
            print(f"  --> [*] New best validation Pearson r: {val_r:.4f} (saving best model)")

            # Save best full state
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, best_ckpt_path)

            # Export best fine-tuned backbone in Hugging Face format
            best_backbone_dir.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_backbone_dir)

            # Copy orthrus_hf.py if available anywhere in related directories
            import shutil
            for candidate_dir in [output_dir, output_dir.parent, Path(output_dir.parent).parent]:
                hf_script = candidate_dir / "orthrus_hf.py"
                if hf_script.exists():
                    shutil.copy2(hf_script, best_backbone_dir / "orthrus_hf.py")
                    break

        # 2. Save Automatic Resume Checkpoint (latest_checkpoint.pt)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_r": best_val_r,
            "best_epoch": best_epoch,
            "history": history,
            "random_states": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
        }, latest_ckpt_path)

        # 3. Update Plots and CSV logs incrementally
        try:
            plot_and_save_training_curves(history, curves_path)
            save_csv_log(history, csv_log_path)
        except Exception as e:
            print(f"[Warning] Could not update loss curves/CSV: {e}")

    print(f"\nTraining completed. Best validation Pearson r: {best_val_r:.4f} at epoch {best_epoch}.")
    print(f"Loss curves saved to: {curves_path}")
    print(f"CSV training log saved to: {csv_log_path}")

    # Evaluate best model on test set
    if test_loader is not None and best_ckpt_path.exists():
        print("\nLoading best model checkpoint for Test Set Evaluation...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, loss_fn, amp_dtype=amp_dtype)

        print("=" * 60)
        print("                 TEST SET METRICS                 ")
        print("=" * 60)
        for k, v in test_metrics.items():
            if "pval" in k:
                print(f"  {k:20s}: {v:.3e}")
            else:
                print(f"  {k:20s}: {v:.4f}")
        print("=" * 60)

        # Save training summary
        summary_file = output_dir / "training_summary.json"
        summary_data = {
            "model_type": "8-track",
            "best_epoch": best_epoch,
            "best_val_pearson_r": best_val_r,
            "test_metrics": test_metrics,
            "history": history,
        }
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary report saved to: {summary_file}")

    return {
        "best_epoch": best_epoch,
        "best_val_pearson_r": best_val_r,
        "test_metrics": test_metrics if (test_loader is not None and best_ckpt_path.exists()) else None,
    }


# =============================================================================
# 4. Main Pipeline
# =============================================================================

SPLITS_LOOKUP_PATH = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning of 8-Track Orthrus Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/orthrus/hIPSC_CM_8track_minmax.npz",
        help="Path to augmented 8-track NPZ file (from generate_trans_factor_tracks.py)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus/orthrus_8track_from_finetuned_6track",
        help="Path to 8-track converted Orthrus model directory (from convert_6track_to_8track.py)",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        choices=["all", "0", "1", "2", "3"],
        help="Which fold to train: '0', '1', '2', '3', or 'all' to train all 4 folds. (Default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus/orthrus_8track_finetuned_hIPSC_CM",
        help="Directory to save fine-tuned checkpoints and logs (subfolder fold_X will be created inside)",
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
        "--lr_embedding",
        type=float,
        default=2e-4,
        help="Learning rate for input embedding layer, especially new channels 6 & 7 (default: 2e-4)",
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
        help="Automatically resume training from latest_checkpoint.pt if it exists in output_dir (default: True, use --no-resume to start fresh)",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        choices=["huber", "mse", "smooth_l1"],
        default="huber",
        help="Loss function: 'huber' (default, robust against outlier gradients), 'mse', or 'smooth_l1'",
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
    print(f"Loading dataset: {data_file}...")
    npz_data = np.load(data_file, allow_pickle=True)

    tracks = npz_data["tracks"]
    targets = npz_data[args.target_col].astype(np.float32)

    # Ensure required metadata fields exist in NPZ archive
    if "hgnc_symbol" not in npz_data:
        raise KeyError(
            f"[Error] Required field 'hgnc_symbol' not found in dataset {data_file}! "
            f"Available fields: {list(npz_data.files)}"
        )
    genes = npz_data["hgnc_symbol"].astype(str)

    if "ensembl_transcript_id" not in npz_data:
        raise KeyError(
            f"[Error] Required field 'ensembl_transcript_id' not found in dataset {data_file}! "
            f"Available fields: {list(npz_data.files)}"
        )
    tx_ids = npz_data["ensembl_transcript_id"].astype(str)

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

    n_channels = tracks[0].shape[1]
    if n_channels != 8:
        raise ValueError(
            f"[Error] finetune_8track.py is strictly configured for 8 tracks, but input data has {n_channels} channels. "
            f"Please use finetune_6track.py for 6-track data."
        )
    print(f"Total valid samples loaded: {len(targets)}")
    print(f"Unique genes:              {len(np.unique(genes))}")
    print(f"Input channels:            8 (A, C, G, U, CDS, Splice, miRNA, eCLIP)")
    print(f"Target column:             {args.target_col}")

    # 2. Standardized 10-Fold Lookup Table Verification & Splitting
    lookup_path = SPLITS_LOOKUP_PATH
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"[Error] Standardized splits lookup table not found at: {lookup_path}! "
            f"A valid lookup table is strictly required."
        )

    print(f"\nUsing Standardized 10-Fold Lookup Table: {lookup_path}")
    lookup_df = pd.read_csv(lookup_path)

    if "ensembl_transcript_id" not in lookup_df.columns:
        raise KeyError(f"[Error] 'ensembl_transcript_id' column not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}")
    if "split" not in lookup_df.columns:
        raise KeyError(f"[Error] 'split' column not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}")

    available_splits = set(lookup_df["split"].dropna().astype(int).unique())
    expected_splits = set(range(10))
    missing_splits = expected_splits - available_splits
    if missing_splits:
        raise ValueError(
            f"[Error] Missing folds in lookup table {lookup_path}! Expected all 10 splits (0-9), but missing: {sorted(missing_splits)}."
        )

    tx_to_split = dict(zip(lookup_df["ensembl_transcript_id"].astype(str).str.strip(), lookup_df["split"].astype(int)))
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

    cv_fold_definitions = {
        0: {"name": "Fold 0", "train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]},
        1: {"name": "Fold 1", "train": [2, 3, 4, 5, 6, 7], "val": [0, 1], "test": [8, 9]},
        2: {"name": "Fold 2", "train": [0, 1, 4, 5, 6, 7], "val": [2, 3], "test": [8, 9]},
        3: {"name": "Fold 3", "train": [0, 1, 2, 3, 6, 7], "val": [4, 5], "test": [8, 9]},
    }

    folds_to_run = [0, 1, 2, 3] if args.fold == "all" else [int(args.fold)]
    all_fold_summaries = {}

    for fold_id in folds_to_run:
        fold_def = cv_fold_definitions[fold_id]
        fold_output_dir = output_dir / f"fold_{fold_id}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        train_idx = np.where(np.isin(sample_splits, fold_def["train"]))[0]
        val_idx = np.where(np.isin(sample_splits, fold_def["val"]))[0]
        test_idx = np.where(np.isin(sample_splits, fold_def["test"]))[0]

        if len(train_idx) == 0:
            raise ValueError(f"[Error] Fold {fold_id} has 0 training samples! Train splits: {fold_def['train']}")
        if len(val_idx) == 0:
            raise ValueError(f"[Error] Fold {fold_id} has 0 validation samples! Val splits: {fold_def['val']}")
        if len(test_idx) == 0:
            raise ValueError(f"[Error] Fold {fold_id} has 0 test samples! Test splits: {fold_def['test']}")

        print("\n" + "=" * 70)
        print(f"       STARTING 8-TRACK FINE-TUNING: {fold_def['name'].upper()}")
        print("=" * 70)
        print(f"  Train (Splits {fold_def['train']}): {len(train_idx)} samples ({len(np.unique(genes[train_idx]))} genes, {len(train_idx)/len(targets)*100:.1f}%)")
        print(f"  Val   (Splits {fold_def['val']}):   {len(val_idx)} samples ({len(np.unique(genes[val_idx]))} genes, {len(val_idx)/len(targets)*100:.1f}%)")
        print(f"  Test  (Splits {fold_def['test']}):  {len(test_idx)} samples ({len(np.unique(genes[test_idx]))} genes, {len(test_idx)/len(targets)*100:.1f}%)")
        print(f"  Output Directory: {fold_output_dir}")

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

        print(f"\nLoading 8-track Orthrus checkpoint from: {args.model_checkpoint}...")
        backbone = AutoModel.from_pretrained(args.model_checkpoint, trust_remote_code=True)
        d_model = getattr(backbone.config, "ssm_model_dim", 512)

        model = OrthrusRegressionModel(backbone=backbone, d_model=d_model, hidden_dim=256, dropout=0.1)
        model.to(device)

        fold_summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_embedding=args.lr_embedding,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            output_dir=fold_output_dir,
            resume=args.resume,
            loss_fn_name=args.loss_fn,
            max_grad_norm=args.max_grad_norm,
        )
        all_fold_summaries[fold_id] = fold_summary

        del model, backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(folds_to_run) > 1:
        print("\n" + "=" * 70)
        print("         8-TRACK 4-FOLD CROSS-VALIDATION SUMMARY REPORT         ")
        print("=" * 70)
        val_rs = [all_fold_summaries[f]["best_val_pearson_r"] for f in folds_to_run if all_fold_summaries[f]]
        for f in folds_to_run:
            r_f = all_fold_summaries[f]["best_val_pearson_r"]
            ep_f = all_fold_summaries[f]["best_epoch"]
            print(f"  Fold {f}: Best Val Pearson r = {r_f:.4f} (Epoch {ep_f})")
        if val_rs:
            print(f"  Mean Val Pearson r: {np.mean(val_rs):.4f} ± {np.std(val_rs):.4f}")
        print("=" * 70)

        overall_summary_file = output_dir / "all_folds_summary.json"
        with open(overall_summary_file, "w") as f:
            json.dump({
                "model_type": "8-track",
                "folds": all_fold_summaries,
                "mean_val_pearson_r": float(np.mean(val_rs)) if val_rs else None,
                "std_val_pearson_r": float(np.std(val_rs)) if val_rs else None,
            }, f, indent=2)
        print(f"All folds summary saved to: {overall_summary_file}")


if __name__ == "__main__":
    main()
