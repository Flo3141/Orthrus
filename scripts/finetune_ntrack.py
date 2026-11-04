#!/usr/bin/env python3
"""
End-to-End Supervised Fine-Tuning of Arbitrary N-Track Orthrus Models (6-track, 7-track, 8-track, etc.)
on Experimental & Trans-Factor Augmented Datasets.

Features:
- Operates exclusively on precomputed .npz track archives; always trains on the 'target' field.
- Channel names (and thus N), dataset and track identity come from the companion .meta.json.
- Slices the channel dimension if the archive holds more channels than the metadata lists.
- Homology-grouped Train / Validation / Test splitting (standardized 10-fold lookup table from create_deterministic_splits.py).
- Dynamic length padding with megabatch length bucketing (low padding overhead, new batch composition every epoch).
- Sequences longer than --max_length are truncated keeping their 3' end (3' UTR preserved).
- Differential learning rates:
    - Backbone Mamba layers: lr_backbone (e.g. 5e-5)
    - Input embedding layer: lr_embedding (e.g. 2e-4) to rapidly adapt new/expanded channels
    - Regression projection head: lr_head (e.g. 3e-4)
- Mixed-precision training (bf16/fp32) with Cosine Annealing + Linear Warmup + Huber/SmoothL1/MSE loss.
- AdamW without weight decay (as in the original Orthrus fine-tuning configs).
- Skips batches with NaN/Inf loss and aborts on NaN/Inf gradient norms or predictions.
- Incremental CSV training logs and publication-quality loss curve plots.
- Saves best checkpoint and exports fine-tuned backbone in Hugging Face format.
- Records the resolved arguments of each fold in fold_k/run_config.json.

The channel attribution analysis of the trained checkpoints lives in xai_analysis.py.

Arguments can come from a YAML file whose keys are the argument names; the command line overrides it:
  python finetune_ntrack.py --config run.yaml --epochs 5

  # run.yaml
  data_path: /path/to/hipsc_cm_8track_2rbp_minmax.npz
  model_checkpoint: /path/to/orthrus-large-8-track
  run_name: lr_test
  lr_backbone: 1.0e-5
  resume: false

--run_name stores the checkpoints in a subdirectory of the default output directory, so that runs with different
hyperparameters do not overwrite each other. Pass that subdirectory to extract_embeddings.py as --model_checkpoint.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoModel
import yaml

from util import (
    reset_peak_memory_stats, print_memory_profile, load_metadata, normalize_run_config,
    resolve_dataset_key, truncate_to_max_length, require_truncation_mode,
    CV_FOLDS, TEST_SPLITS, CHECKPOINTS_DIR, RUN_CONFIG_FILENAME, assign_splits, require_meta, regression_metrics,
)

# Headless backend for cluster servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MultiTrackDataset(Dataset):
    """
    Dataset wrapping variable-length multi-track RNA representations and target values.
    """
    def __init__(self, tracks: list, targets: np.ndarray, genes: np.ndarray, transcript_ids: np.ndarray, max_length: int = 12288):
        self.samples = []
        for i in range(len(tracks)):
            tr = tracks[i]
            tr = truncate_to_max_length(tr, max_length)  # keeps the 3' end
            self.samples.append({
                "track": torch.from_numpy(tr).float(),
                "target": float(targets[i]),
                "gene": str(genes[i]),
                "tx_id": str(transcript_ids[i]),
                "length": int(tr.shape[0]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class BucketBatchSampler(Sampler):
    """
    Batches sequences of similar lengths together to minimize zero-padding overhead.

    - shuffle=True (training): megabatch bucketing. Each epoch the indices are shuffled and split into
      megabatches of megabatch_size * batch_size samples; within each megabatch they are sorted by length
      and cut into batches, then all batches are shuffled. Batch composition changes every epoch at ~3 %
      padding overhead (simulated on hIPSC_CM lengths with megabatch_size=50), whereas global sorting
      yields identical batches in every epoch.
    - shuffle=False (val/test): globally sorted by length (deterministic, minimal padding).
    """
    def __init__(self, dataset: MultiTrackDataset, batch_size: int, shuffle: bool = True, megabatch_size: int = 50):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.megabatch_size = megabatch_size
        self.lengths = [s["length"] for s in dataset.samples]

    def _sorted_batches(self, indices: list) -> list:
        indices = sorted(indices, key=lambda i: self.lengths[i])
        return [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    def __iter__(self):
        if not self.shuffle:
            batches = self._sorted_batches(list(range(len(self.dataset))))
        else:
            # Python RNG: seeded via --random_seed and restored on --resume
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            megabatch_len = self.batch_size * self.megabatch_size
            batches = []
            for start in range(0, len(indices), megabatch_len):
                batches.extend(self._sorted_batches(indices[start : start + megabatch_len]))
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pad_collate_fn(batch: list) -> dict:
    """
    Dynamically pads batch sequences to the maximum length in the current batch.
    Detects channel count dynamically from batch elements.
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
# 2. Model Architecture: N-Track Backbone + Projection Head
# =============================================================================

class OrthrusRegressionModel(nn.Module):
    """
    Wraps the Orthrus backbone with a regression head (matching the Orthrus downstream setup).
    Linear(d_model -> hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim -> 1)
    """
    def __init__(self, backbone: nn.Module, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: (B, L, C)
          lengths: (B,)
        Returns:
          out: (B,) scalar predictions
        """
        # Representation: (B, d_model) via mean pooling over unpadded sequence
        rep = self.backbone.representation(x, lengths, channel_last=True)
        out = self.head(rep).squeeze(-1)  # (B,)
        return out


def build_regression_model(checkpoint: str, num_channels: int, device: torch.device) -> OrthrusRegressionModel:
    """Loads an Orthrus backbone (directory or Hugging Face repo), verifies its channel count and adds the regression head."""
    backbone = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    d_model = getattr(backbone.config, "ssm_model_dim", 512)

    # Check input embedding dimension (must match dataset channels)
    emb_layer = getattr(backbone, "embedding", None)
    if emb_layer is not None and hasattr(emb_layer, "in_features") and emb_layer.in_features != num_channels:
        raise ValueError(
            f"[Error] Checkpoint input embedding dimension ({emb_layer.in_features}) does not match dataset tracks ({num_channels})!\n"
            f"Please convert your base model using 'convert_6track_to_ntrack.py --n_target_tracks {num_channels}' first."
        )
    if hasattr(backbone, "config") and getattr(backbone.config, "n_tracks", None) != num_channels:
        backbone.config.n_tracks = num_channels

    model = OrthrusRegressionModel(backbone=backbone, d_model=d_model, hidden_dim=256, dropout=0.1)
    return model.to(device)


# =============================================================================
# 3. Training & Evaluation Loops, Plots & CSV Logging
# =============================================================================

def get_amp_dtype(device: torch.device) -> torch.dtype:
    """bf16 on CUDA where supported (strictly required for Mamba SSM), otherwise full precision."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def amp_autocast(device: torch.device, amp_dtype: torch.dtype) -> torch.amp.autocast:
    """Autocast context for the given precision; disabled on CPU and for fp32."""
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32))


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    x_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple:
    """Returns (y_true, y_pred) in loader order. x_transform can modify each input batch first (e.g. channel ablation)."""
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            if x_transform is not None:
                x = x_transform(x)
            with amp_autocast(device, amp_dtype):
                preds = model(x, batch["lengths"].to(device))
            all_preds.append(preds.cpu().float().numpy())
            all_targets.append(batch["targets"].numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, loss_fn: nn.Module, amp_dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Evaluates the model over the dataset and computes Pearson r, Spearman rho, MSE, RMSE, R2, and Loss.
    """
    y_true, y_pred = predict(model, dataloader, device, amp_dtype)

    # Fail-Fast: Detect numerical instability immediately instead of silently masking NaNs
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        nan_count = int(np.isnan(y_pred).sum())
        inf_count = int(np.isinf(y_pred).sum())
        raise FloatingPointError(
            f"[Fatal Error] Numerical instability detected in evaluation predictions! "
            f"Found {nan_count} NaNs and {inf_count} Infs out of {len(y_pred)} predictions. "
            f"The model weights have likely collapsed or diverged. "
            f"Possible causes: learning rate too high, gradient clipping threshold too large, "
            f"or numerical instability in Mamba SSM state updates."
        )

    if np.isnan(y_true).any() or np.isinf(y_true).any():
        nan_t_count = int(np.isnan(y_true).sum())
        inf_t_count = int(np.isinf(y_true).sum())
        raise ValueError(
            f"[Fatal Error] Found {nan_t_count} NaNs and {inf_t_count} Infs in ground truth targets (y_true)! "
            f"Please clean or filter the dataset."
        )

    loss = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_true)).item()
    return {"loss": float(loss), **regression_metrics(y_true, y_pred)}


LOSS_DISPLAY_NAMES = {"huber": "Huber", "mse": "MSE", "smooth_l1": "Smooth L1"}


def plot_and_save_training_curves(history: list, output_file: Path, num_tracks: int = None, loss_fn_name: str = "huber"):
    """
    Plots training & validation loss, Pearson r, Spearman rho, and MSE/RMSE across epochs.
    The loss curves are labelled with the loss function actually used for training (loss_fn_name).
    """
    loss_label = LOSS_DISPLAY_NAMES.get(loss_fn_name, loss_fn_name)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_metrics"]["loss"] for h in history]
    val_pearson = [h["val_metrics"]["pearson_r"] for h in history]
    val_spearman = [h["val_metrics"]["spearman_rho"] for h in history]
    val_mse = [h["val_metrics"]["mse"] for h in history]
    val_rmse = [h["val_metrics"]["rmse"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss curves
    axes[0, 0].plot(epochs, train_loss, label=f"Train Loss ({loss_label})", color="#1f77b4", lw=2, marker="o", markersize=4)
    axes[0, 0].plot(epochs, val_loss, label=f"Val Loss ({loss_label})", color="#ff7f0e", lw=2, linestyle="--", marker="s", markersize=4)
    axes[0, 0].set_title(f"Training & Validation Loss ({loss_label})", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel(f"{loss_label} Loss", fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=11)

    # 2. Validation Pearson r
    axes[0, 1].plot(epochs, val_pearson, label="Validation Pearson r", color="#2ca02c", lw=2, marker="o", markersize=4)
    best_p_idx = int(np.argmax(val_pearson))
    axes[0, 1].axvline(epochs[best_p_idx], color="gray", linestyle=":", label=f"Best r ({val_pearson[best_p_idx]:.4f} @ Ep {epochs[best_p_idx]})")
    axes[0, 1].set_title("Validation Pearson Correlation (r)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("Pearson r", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=11)

    # 3. Validation Spearman rho
    axes[1, 0].plot(epochs, val_spearman, label="Validation Spearman rho", color="#9467bd", lw=2, marker="o", markersize=4)
    axes[1, 0].set_title("Validation Spearman Correlation (rho)", fontsize=13, fontweight="bold")
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

    title_suffix = f" ({num_tracks}-Track)" if num_tracks else ""
    plt.suptitle(f"Orthrus{title_suffix} Fine-Tuning Performance Across Epochs", fontsize=15, fontweight="bold")
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


# =============================================================================
# 4. Optimizer Setup & Checkpoints
# =============================================================================

def build_param_groups(model: nn.Module, lr_backbone: float, lr_embedding: float, lr_head: float) -> list:
    """
    Differential learning rates for backbone, input embedding and regression head.

    No weight decay, as in the original Orthrus fine-tuning configs ('no_wd' in orthrus/finetune_config/optimizer.yaml).
    Must be set explicitly: AdamW defaults to weight_decay=0.01.
    The group order is part of the optimizer state in existing checkpoints and must stay fixed.
    """
    backbone_params = []
    embedding_params = []
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            (embedding_params if "embedding" in name else backbone_params).append(param)
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    return [
        {"params": params, "lr": lr, "weight_decay": 0.0}
        for params, lr in ((backbone_params, lr_backbone), (embedding_params, lr_embedding), (head_params, lr_head))
    ]


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, **extra) -> None:
    """Saves model, optimizer and scheduler state together with any additional fields."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        **extra,
    }, path)


def get_random_states() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_random_states(states: dict) -> None:
    try:
        if states.get("torch") is not None:
            torch_state = states["torch"].cpu() if isinstance(states["torch"], torch.Tensor) else states["torch"]
            torch.set_rng_state(torch_state)
        if states.get("cuda") is not None and torch.cuda.is_available():
            cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in states["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        if states.get("numpy") is not None:
            np.random.set_state(states["numpy"])
        if states.get("python") is not None:
            random.setstate(states["python"])
    except Exception as e:
        print(f"[Warning] Could not fully restore random states ({e}), proceeding with current RNG state.")


def verify_remote_code_export(backbone_dir: Path):
    """Fails if a module referenced in config.json 'auto_map' is missing as .py file in the exported backbone."""
    cfg_file = backbone_dir / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"[Error] Exported backbone has no config.json: {backbone_dir}")
    with open(cfg_file, "r") as f_cfg:
        auto_map = json.load(f_cfg).get("auto_map", {})

    refs = []
    for ref in auto_map.values():
        refs.extend(ref if isinstance(ref, (list, tuple)) else [ref])
    # References look like 'module.Class' or 'repo_id--module.Class'
    modules = {str(r).split("--")[-1].rsplit(".", 1)[0] for r in refs if r}
    missing = sorted(m for m in modules if not (backbone_dir / f"{m}.py").exists())
    if missing:
        raise FileNotFoundError(
            f"[Error] Exported backbone {backbone_dir} is missing remote code module(s) {missing} referenced in auto_map! "
            f"The checkpoint could not be loaded with trust_remote_code."
        )


def export_backbone(model: nn.Module, backbone_dir: Path, num_channels: int, source_checkpoint: Optional[str]) -> None:
    """Exports the fine-tuned backbone in Hugging Face format and verifies that it stays loadable."""
    backbone_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(backbone_dir)

    # Ensure config.json has updated n_tracks
    cfg_file = backbone_dir / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r") as f_cfg:
                c_dict = json.load(f_cfg)
            c_dict["n_tracks"] = num_channels
            with open(cfg_file, "w") as f_cfg:
                json.dump(c_dict, f_cfg, indent=2)
        except Exception:
            pass

    # save_pretrained() already exports the remote code (auto_map) of trust_remote_code models.
    # Fallback: copy missing *.py files from the source checkpoint only (never from output directories).
    if source_checkpoint and Path(source_checkpoint).is_dir():
        for py_file in Path(source_checkpoint).glob("*.py"):
            dst_file = backbone_dir / py_file.name
            if not dst_file.exists():
                shutil.copy2(py_file, dst_file)
    verify_remote_code_export(backbone_dir)


# =============================================================================
# 5. Core Training Loop
# =============================================================================

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
    warmup_epochs: int,
    output_dir: Path,
    channel_names: list,
    resume: bool = False,
    loss_fn_name: str = "huber",
    max_grad_norm: float = 1.0,
    source_checkpoint: Optional[str] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_channels = len(channel_names)

    # Loss Function
    if loss_fn_name == "huber":
        loss_fn = nn.HuberLoss(delta=1.0)
    elif loss_fn_name == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        loss_fn = nn.MSELoss()

    # Precision
    amp_dtype = get_amp_dtype(device)
    if amp_dtype == torch.bfloat16:
        print("[Precision] Using Native BFloat16 (bf16) mixed precision (strictly required for Mamba SSM).")
    else:
        print("[Precision] Using Full Precision (fp32) (CUDA bf16 not available).")

    optimizer = torch.optim.AdamW(build_param_groups(model, lr_backbone, lr_embedding, lr_head))

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    # Cosine scheduler with linear warmup
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        print("Loading model, optimizer, and scheduler state...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # load_state_dict restores the stored hyperparameters; older checkpoints still carry weight_decay=0.01
        for group in optimizer.param_groups:
            group["weight_decay"] = 0.0
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_r = ckpt.get("best_val_r", -1.0)
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        start_epoch = ckpt["epoch"] + 1

        if "random_states" in ckpt:
            restore_random_states(ckpt["random_states"])

        print(f"[Resume] Successfully resumed from Epoch {ckpt['epoch']}. Next training epoch: {start_epoch}/{epochs}")
        if start_epoch > epochs:
            print(f"[Notice] Training already completed ({ckpt['epoch']} >= {epochs} epochs). Skipping to evaluation.")

    print(f"\nStarting {num_channels}-Track Fine-Tuning Training...")
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
            with amp_autocast(device, amp_dtype):
                preds = model(x, lengths)
                loss = loss_fn(preds, targets)

            # Safeguard 1: Skip batch if loss is NaN/Inf to prevent corrupting weights
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Warning] NaN/Inf loss encountered at Epoch {epoch}, Batch {batch_idx}! Skipping batch update.")
                optimizer.zero_grad()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            # Fail-Fast: Abort training if gradient norm explodes or collapses to NaN/Inf
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                raise FloatingPointError(
                    f"[Fatal Error] NaN/Inf gradient norm encountered at Epoch {epoch}, Batch {batch_idx}! "
                    f"Gradients have exploded or collapsed. Training aborted."
                )

            optimizer.step()
            scheduler.step()

            b_len = len(targets)
            running_loss += loss.item() * b_len
            n_samples += b_len

            pbar.set_postfix({"loss": f"{running_loss / max(1, n_samples):.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        train_epoch_loss = running_loss / max(1, n_samples)

        # Validation
        val_metrics = evaluate(model, val_loader, device, loss_fn, amp_dtype=amp_dtype)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val r: {val_metrics['pearson_r']:.4f} | "
            f"Val rho: {val_metrics['spearman_rho']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "val_metrics": val_metrics,
        })

        # Check for best model
        is_best = val_metrics["pearson_r"] > best_val_r
        if is_best:
            best_val_r = val_metrics["pearson_r"]
            best_epoch = epoch
            print(f"  >>> New Best Validation Pearson r: {best_val_r:.4f} (Epoch {best_epoch})! Saving checkpoint...")

        ckpt_fields = {"epoch": epoch, "best_val_r": best_val_r, "best_epoch": best_epoch, "channel_names": channel_names}

        # 1. Save best model as full PyTorch checkpoint and as Hugging Face backbone
        if is_best:
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, val_metrics=val_metrics, **ckpt_fields)
            export_backbone(model, best_backbone_dir, num_channels, source_checkpoint)

        # 2. Save Automatic Resume Checkpoint (latest_checkpoint.pt)
        save_checkpoint(latest_ckpt_path, model, optimizer, scheduler, history=history, random_states=get_random_states(), **ckpt_fields)

        # 3. Update Plots and CSV logs incrementally
        try:
            plot_and_save_training_curves(history, curves_path, num_tracks=num_channels, loss_fn_name=loss_fn_name)
            save_csv_log(history, csv_log_path)
        except Exception as e:
            print(f"[Warning] Could not update loss curves/CSV: {e}")

    print(f"\nTraining completed. Best validation Pearson r: {best_val_r:.4f} at epoch {best_epoch}.")
    print(f"Loss curves saved to: {curves_path}")
    print(f"CSV training log saved to: {csv_log_path}")

    # Evaluate best model on test set
    test_metrics = None
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
            "model_type": f"{num_channels}-track",
            "channel_names": channel_names,
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
        "test_metrics": test_metrics,
    }


# =============================================================================
# 6. Data Loading & Run Directories (shared with xai_analysis.py)
# =============================================================================

@dataclass
class TrackData:
    """Cleaned track archive with its metadata; all arrays hold the same transcripts in the same order."""
    dataset_name: str
    track_name: str
    normalization: str
    channel_names: list
    tracks: list
    targets: np.ndarray
    genes: np.ndarray
    tx_ids: np.ndarray
    sample_splits: np.ndarray


def load_track_data(data_file: Path) -> TrackData:
    """Loads an .npz track archive and its required metadata, removes invalid samples and assigns the splits."""
    if not data_file.exists():
        raise FileNotFoundError(f"[Error] Dataset file not found: {data_file}")
    if data_file.suffix.lower() != ".npz":
        raise ValueError(f"[Error] Unsupported file extension for dataset: {data_file.suffix}. Expected .npz.")

    print(f"\nLoading dataset: {data_file}...")
    meta = load_metadata(data_file, required=True)

    # Strictly extract fields from meta.json - no fallbacks!
    require_meta(meta, data_file, "dataset", "track_name", "channel_names")
    dataset_name = resolve_dataset_key(meta["dataset"])
    track_name = str(meta["track_name"]).strip()
    channel_names = [str(x) for x in meta["channel_names"]]
    num_channels = len(channel_names)
    if num_channels == 0:
        raise ValueError(f"[Error] 'channel_names' in metadata for {data_file} is empty!")

    normalization = str(meta.get("normalization", "none")).strip().lower()
    require_truncation_mode(meta, data_file)

    npz_data = np.load(data_file, allow_pickle=True)
    if "tracks" not in npz_data:
        raise KeyError(f"[Error] Key 'tracks' not found in NPZ archive {data_file}! Available: {list(npz_data.files)}")
    tracks = list(npz_data["tracks"])

    # Always train on 'target' (identical to extract_embeddings.py / train_ridge_regression.py)
    if "target" not in npz_data:
        raise KeyError(f"[Error] Required field 'target' not found in {data_file}! Available: {list(npz_data.files)}")
    targets = npz_data["target"].astype(np.float32)

    if "gene" not in npz_data:
        raise KeyError(f"[Error] 'gene' not found in dataset {data_file}! Available: {list(npz_data.files)}")
    genes = npz_data["gene"].astype(str)

    if "ensembl_transcript_id" not in npz_data:
        raise KeyError(f"[Error] Required field 'ensembl_transcript_id' not found in dataset {data_file}!")
    tx_ids = npz_data["ensembl_transcript_id"].astype(str)

    # Cleaning mask
    valid_mask = ~np.isnan(targets) & ~np.isinf(targets)
    if "seq_lens" in npz_data:
        valid_mask &= (npz_data["seq_lens"] > 0)
    else:
        valid_mask &= np.array([len(t) > 0 for t in tracks])

    # Clean invalid samples
    n_invalid = int(np.sum(~valid_mask))
    if n_invalid > 0:
        print(f"[Data Cleaning] Filtered out {n_invalid} samples with NaN/Inf targets or empty sequence length.")
        tracks = [t for i, t in enumerate(tracks) if valid_mask[i]]
        targets = targets[valid_mask]
        genes = genes[valid_mask]
        tx_ids = tx_ids[valid_mask]

    # Check and align channel dimensions
    data_channels = tracks[0].shape[1]
    if data_channels < num_channels:
        raise ValueError(
            f"[Error] Dataset has only {data_channels} channels, but {num_channels} channels were specified in metadata ({channel_names})!"
        )
    elif data_channels > num_channels:
        print(
            f"[Info] Dataset has {data_channels} channels. Slicing the first {num_channels} channels matching {channel_names}."
        )
        tracks = [t[:, :num_channels] for t in tracks]

    print(f"Total valid samples loaded: {len(targets)}")
    print(f"Dataset:                   {dataset_name}")
    print(f"Track identity:            {track_name} (Normalization: {normalization.upper()})")
    print(f"Unique genes:              {len(np.unique(genes))}")
    print(f"Configured tracks (N):     {num_channels}")
    print(f"Channel names:             {channel_names}")

    # Standardized 10-Fold Lookup Table Verification & Splitting
    sample_splits = assign_splits(dataset_name, tx_ids)
    if np.any(sample_splits == -1):
        print("[Warning] Filtering out transcripts without split assignment.")
        matched_mask = (sample_splits != -1)
        tracks = [t for i, t in enumerate(tracks) if matched_mask[i]]
        targets = targets[matched_mask]
        genes = genes[matched_mask]
        tx_ids = tx_ids[matched_mask]
        sample_splits = sample_splits[matched_mask]

    return TrackData(dataset_name, track_name, normalization, channel_names, tracks, targets, genes, tx_ids, sample_splits)


def make_loader(data: TrackData, indices: np.ndarray, batch_size: int, max_length: int, shuffle: bool, device: torch.device) -> DataLoader:
    """Length-bucketed DataLoader over the given samples (see BucketBatchSampler)."""
    dataset = MultiTrackDataset([data.tracks[i] for i in indices], data.targets[indices], data.genes[indices], data.tx_ids[indices], max_length)
    return DataLoader(
        dataset,
        batch_sampler=BucketBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle),
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )


def run_name_arg(value: str) -> str:
    """argparse type for --run_name: a single directory name that cannot be mistaken for a fold directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) or value.startswith("fold_"):
        raise argparse.ArgumentTypeError(
            f"invalid run name '{value}': allowed are letters, digits, '.', '_' and '-', and it must not start with 'fold_'"
        )
    return value


def resolve_run_dir(data: TrackData, run_name: Optional[str] = None) -> Path:
    """Checkpoint directory of a run, derived from the track metadata: <dataset>/<track>[/<normalization>][/<run_name>]."""
    run_dir = CHECKPOINTS_DIR / data.dataset_name / data.track_name
    if data.normalization and data.normalization != "none":
        run_dir = run_dir / data.normalization
    return run_dir / run_name if run_name else run_dir


# =============================================================================
# 7. Main Pipeline
# =============================================================================

def config_to_argv(config_path: str) -> list:
    """Converts a YAML config (keys = argument names) to command line arguments, so that it passes the same parser checks."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"[Error] Config file {config_path} must contain a mapping of argument names to values.")

    argv = []
    for key, value in config.items():
        if isinstance(value, bool):
            argv.append(f"--{key}" if value else f"--no-{key}")
        elif isinstance(value, (str, int, float)):
            argv.extend([f"--{key}", str(value)])
        else:
            raise ValueError(f"[Error] Config file {config_path}: '{key}' must be a single value, got {value!r}.")
    return argv


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv

    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file with argument names as keys; arguments on the command line take precedence",
    )
    config_path = config_parser.parse_known_args(argv)[0].config

    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning of Arbitrary N-Track Orthrus Model",
        parents=[config_parser],
        allow_abbrev=False,
    )

    # Required parameters (No defaults)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to precomputed N-track .npz archive (companion .meta.json required)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to Orthrus model checkpoint directory or HuggingFace repo (e.g. from convert_6track_to_ntrack.py)",
    )
    # Optional configuration parameters
    parser.add_argument(
        "--run_name",
        type=run_name_arg,
        default=None,
        help="Store the checkpoints in a subdirectory of this name, so that runs do not overwrite each other (default: none)",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        choices=["all", "0", "1", "2", "3"],
        help="Which fold to train: '0', '1', '2', '3', or 'all' to train all 4 folds. (Default: all)",
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
        help="Learning rate for input embedding layer, especially newly added channels (default: 2e-4)",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=3e-4,
        help="Learning rate for regression head (default: 3e-4)",
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
        help="Maximum sequence length; longer transcripts keep their 3' end (default: 12288)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume training from latest_checkpoint.pt if it exists in output_dir (default: False; "
             "without --resume an existing run in output_dir is overwritten)",
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

    # The command line comes last, so that its values override the config file
    config_argv = config_to_argv(config_path) if config_path else []
    return parser.parse_args(config_argv + argv)


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    reset_peak_memory_stats(device)

    # 1. Load dataset (.npz) with its required metadata and assign the splits
    data = load_track_data(Path(args.data_path))
    channel_names = data.channel_names
    num_channels = len(channel_names)
    genes = data.genes
    sample_splits = data.sample_splits

    # Automatically determine output directory based on metadata
    output_dir = resolve_run_dir(data, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output Directory] Auto-resolved checkpoint output directory: {output_dir}")

    folds_to_run = list(CV_FOLDS) if args.fold == "all" else [int(args.fold)]
    all_fold_summaries = {}

    for fold_id in folds_to_run:
        fold_def = CV_FOLDS[fold_id]
        fold_output_dir = output_dir / f"fold_{fold_id}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        train_idx = np.where(np.isin(sample_splits, fold_def["train"]))[0]
        val_idx = np.where(np.isin(sample_splits, fold_def["val"]))[0]
        test_idx = np.where(np.isin(sample_splits, TEST_SPLITS))[0]

        if len(train_idx) == 0:
            raise ValueError(f"[Error] Fold {fold_id} has 0 training samples! Train splits: {fold_def['train']}")
        if len(val_idx) == 0:
            raise ValueError(f"[Error] Fold {fold_id} has 0 validation samples! Val splits: {fold_def['val']}")
        if len(test_idx) == 0:
            raise ValueError(f"[Error] Fold {fold_id} has 0 test samples! Test splits: {TEST_SPLITS}")

        print("\n" + "=" * 70)
        print(f"       STARTING {num_channels}-TRACK FINE-TUNING: FOLD {fold_id}")
        print("=" * 70)
        print(f"  Train (Splits {fold_def['train']}): {len(train_idx)} samples ({len(np.unique(genes[train_idx]))} genes, {len(train_idx)/len(genes)*100:.1f}%)")
        print(f"  Val   (Splits {fold_def['val']}):   {len(val_idx)} samples ({len(np.unique(genes[val_idx]))} genes, {len(val_idx)/len(genes)*100:.1f}%)")
        print(f"  Test  (Splits {TEST_SPLITS}):  {len(test_idx)} samples ({len(np.unique(genes[test_idx]))} genes, {len(test_idx)/len(genes)*100:.1f}%)")
        print(f"  Output Directory: {fold_output_dir}")

        train_loader = make_loader(data, train_idx, args.batch_size, args.max_length, shuffle=True, device=device)
        val_loader = make_loader(data, val_idx, args.batch_size, args.max_length, shuffle=False, device=device)
        test_loader = make_loader(data, test_idx, args.batch_size, args.max_length, shuffle=False, device=device)

        print(f"\nLoading {num_channels}-track Orthrus checkpoint from: {args.model_checkpoint}...")
        model = build_regression_model(args.model_checkpoint, num_channels, device)

        # Reproducibility: resolved arguments of this run (a resumed run records its latest invocation)
        with open(fold_output_dir / RUN_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(normalize_run_config({**vars(args), "fold": fold_id, "started_at": datetime.now().isoformat(timespec="seconds")}), f, indent=2)

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
            warmup_epochs=args.warmup_epochs,
            output_dir=fold_output_dir,
            channel_names=channel_names,
            resume=args.resume,
            loss_fn_name=args.loss_fn,
            max_grad_norm=args.max_grad_norm,
            source_checkpoint=args.model_checkpoint,
        )
        all_fold_summaries[fold_id] = fold_summary

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(folds_to_run) > 1:
        print("\n" + "=" * 70)
        print(f"         {num_channels}-TRACK 4-FOLD CROSS-VALIDATION SUMMARY REPORT         ")
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
                "model_type": f"{num_channels}-track",
                "channel_names": channel_names,
                "folds": all_fold_summaries,
                "mean_val_pearson_r": float(np.mean(val_rs)) if val_rs else None,
                "std_val_pearson_r": float(np.std(val_rs)) if val_rs else None,
            }, f, indent=2)
        print(f"All folds summary saved to: {overall_summary_file}")

    # Memory profiling summary
    mem_prof = print_memory_profile(device=device, title=f"ORTHRUS {num_channels}-TRACK FINETUNING SPEICHER-PROFILING")
    mem_profile_file = output_dir / "memory_profile.json"
    try:
        with open(mem_profile_file, "w") as f:
            json.dump(mem_prof, f, indent=2)
        print(f"Memory profile saved to: {mem_profile_file}")
    except Exception as e:
        print(f"Notice: Could not write {mem_profile_file}: {e}")


if __name__ == "__main__":
    main()
