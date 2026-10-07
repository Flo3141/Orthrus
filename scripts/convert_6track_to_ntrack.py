#!/usr/bin/env python3
"""
Converts the pre-trained Orthrus 6-track model to an N-track model (Weight Surgery).
Expands the input embedding layer from 6 to N channels (default: N=8, or N=7 for m6A):
  - Channels 0-3: A, C, G, T/U (One-Hot)
  - Channel 4:    CDS marker
  - Channel 5:    Splice-site marker
  - Channels 6+:  Augmented experimental or genomic tracks (e.g., m6A, miRNA, eCLIP)

Transfers existing weights for channels 0-5 and all Mamba backbone layers 1:1,
initializes the new channels (6 to N-1) with small random values (or zeros/kaiming),
and exports a complete Hugging Face checkpoint compatible with AutoModel.from_pretrained(..., trust_remote_code=True).
"""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from util import reset_peak_memory_stats, print_memory_profile


def convert_6track_to_ntrack(
    base_model_name: str,
    output_dir: Path,
    target_tracks: int = 8,
    init_method: str = "normal",
    init_std: float = 0.02,
    device: str = "cpu"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"   Orthrus 6-Track -> {target_tracks}-Track Checkpoint Conversion (Weight Surgery)   ")
    print("=" * 70)
    print(f"Base model:          {base_model_name}")
    print(f"Target tracks (N):   {target_tracks}")
    print(f"Output directory:    {output_dir}")
    print(f"Init method (6-{target_tracks-1}):  {init_method} (std={init_std if init_method == 'normal' else 'N/A'})")
    print(f"Device:              {device}")

    # 1. Load base 6-track model
    print(f"\n[1/5] Loading base model '{base_model_name}'...")
    base_path = Path(base_model_name)
    if base_path.is_dir() and (base_path / "best_finetuned_backbone").is_dir():
        print(f"      Found 'best_finetuned_backbone' inside {base_path} - using it as model source.")
        base_path = base_path / "best_finetuned_backbone"
        base_model_name = str(base_path)

    if base_path.is_file() and base_path.suffix in [".pt", ".pth", ".bin"]:
        print(f"      Restoring weights from PyTorch checkpoint file: {base_path}...")
        model_base = AutoModel.from_pretrained("quietflamingo/orthrus-large-6-track", trust_remote_code=True)
        ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_dict[k[len("backbone."):]] = v
            elif not k.startswith("head."):
                backbone_dict[k] = v
        model_base.load_state_dict(backbone_dict, strict=False)
    else:
        model_base = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

    model_base.eval()
    config_base = model_base.config

    old_n_tracks = getattr(config_base, "n_tracks", 6)
    d_model = getattr(config_base, "ssm_model_dim", 512)
    print(f"      Loaded model config: n_tracks={old_n_tracks}, d_model={d_model}, layers={getattr(config_base, 'ssm_n_layers', 4)}")
    assert target_tracks > old_n_tracks, (
        f"Target tracks ({target_tracks}) must be greater than base model tracks ({old_n_tracks})."
    )

    # 2. Create N-track config and instantiate N-track model
    print(f"\n[2/5] Creating {target_tracks}-track model architecture...")
    config_dict = config_base.to_dict()
    config_dict["n_tracks"] = target_tracks

    # Recreate config & model instance
    ConfigClass = config_base.__class__
    ModelClass = model_base.__class__

    config_target = ConfigClass(**config_dict)
    model_target = ModelClass(config_target)
    model_target.eval()

    # 3. Perform weight surgery
    print(f"\n[3/5] Performing weight surgery on embedding layer (6 -> {target_tracks} channels)...")
    state_base = model_base.state_dict()
    state_target = model_target.state_dict()

    for k, v in state_base.items():
        if k == "embedding.weight":
            print(f"      Transferring '{k}': {v.shape} -> {state_target[k].shape}")
            with torch.no_grad():
                # Copy channels 0 to old_n_tracks-1 exactly
                state_target[k][:, 0:old_n_tracks] = v[:, 0:old_n_tracks]

                # Initialize new channels (e.g. channel 6 for 7-track, channels 6 & 7 for 8-track)
                new_slice = slice(old_n_tracks, target_tracks)
                if init_method == "zeros":
                    nn.init.zeros_(state_target[k][:, new_slice])
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(state_target[k][:, new_slice])
                else:  # 'normal'
                    nn.init.normal_(state_target[k][:, new_slice], mean=0.0, std=init_std)

            print(f"      Channel weights initialized:")
            print(f"        Tracks 0-{old_n_tracks-1} mean norm: {state_target[k][:, 0:old_n_tracks].norm().item():.4f}")
            print(f"        Tracks {old_n_tracks}-{target_tracks-1} mean norm: {state_target[k][:, new_slice].norm().item():.4f}")
        else:
            # Copy all other backbone parameters (Mamba blocks, LayerNorms, biases) 1:1
            state_target[k] = v

    model_target.load_state_dict(state_target)
    print("      Weight transfer successfully completed.")

    # 4. Save model to output_dir
    print(f"\n[4/5] Saving converted model to: {output_dir}")
    model_target.save_pretrained(output_dir)

    # Copy orthrus_hf.py code for standalone trust_remote_code loading
    src_code_file = Path(inspect.getfile(ModelClass))
    dst_code_file = output_dir / "orthrus_hf.py"
    if src_code_file.exists():
        print(f"      Copying remote code file '{src_code_file.name}' to '{dst_code_file.name}'...")
        shutil.copy2(src_code_file, dst_code_file)

    # Ensure config.json has appropriate auto_map
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg = json.load(f)
        cfg["n_tracks"] = target_tracks
        cfg["auto_map"] = {
            "AutoConfig": "orthrus_hf.OrthrusConfig",
            "AutoModel": "orthrus_hf.OrthrusPretrainedModel"
        }
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2)

    # 5. Verification forward pass
    print(f"\n[5/5] Running verification forward pass with dummy {target_tracks}-track input...")
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    reset_peak_memory_stats(dev)
    model_target.to(dev)

    dummy_seq_len = 128
    dummy_x = torch.randn(2, dummy_seq_len, target_tracks, device=dev)
    dummy_lens = torch.tensor([dummy_seq_len, dummy_seq_len // 2], device=dev)

    with torch.no_grad():
        rep = model_target.representation(dummy_x, dummy_lens, channel_last=True)
        unpooled = model_target(dummy_x, channel_last=True)

    print(f"      Input shape:               {tuple(dummy_x.shape)} (B, L, C={target_tracks})")
    print(f"      Representation shape:      {tuple(rep.shape)} (expected: (2, {d_model}))")
    print(f"      Unpooled output shape:     {tuple(unpooled.shape)} (expected: (2, {dummy_seq_len}, {d_model}))")

    assert rep.shape == (2, d_model), f"Shape mismatch: {rep.shape} vs (2, {d_model})"
    print_memory_profile(device=dev, title=f"ORTHRUS 6-TO-{target_tracks} TRACK CONVERSION SPEICHER-PROFILING")
    print("\n" + "=" * 70)
    print(f"SUCCESS: {target_tracks}-Track Orthrus model created and verified!")
    print(f"You can now load it via:")
    print(f"  model = AutoModel.from_pretrained('{output_dir}', trust_remote_code=True)")
    print("=" * 70 + "\n")

    return output_dir


# Backwards compatibility alias
convert_6track_to_8track = convert_6track_to_ntrack


def main():
    parser = argparse.ArgumentParser(
        description="Convert Orthrus 6-track model to N-track checkpoint (default: 8 tracks, or 7 tracks for m6A)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Source Hugging Face model or local directory (default: quietflamingo/orthrus-large-6-track)",
    )
    parser.add_argument(
        "--n_target_tracks",
        type=int,
        default=8,
        help="Target number of tracks (default: 8; use 7 for 6-track + m6A)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Destination path to save the converted checkpoint (default: /beegfs/.../orthrus-large-{n_target_tracks}-track)",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["normal", "zeros", "kaiming"],
        default="normal",
        help="Initialization method for the new channels (default: normal)",
    )
    parser.add_argument(
        "--init_std",
        type=float,
        default=0.02,
        help="Standard deviation for normal initialization of new channels (default: 0.02)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for verification forward pass (default: cuda)",
    )
    args = parser.parse_args()

    # Determine default output_dir based on n_target_tracks if not explicitly set
    if args.output_dir is None:
        args.output_dir = f"/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus/orthrus-large-{args.n_target_tracks}-track"

    convert_6track_to_ntrack(
        base_model_name=args.base_model,
        output_dir=Path(args.output_dir),
        target_tracks=args.n_target_tracks,
        init_method=args.init_method,
        init_std=args.init_std,
        device=args.device
    )


if __name__ == "__main__":
    main()
