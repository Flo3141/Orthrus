#!/usr/bin/env python3
"""
Converts the pre-trained Orthrus 6-track model to an 8-track model (Weight Surgery).
Expands the input embedding layer from 6 to 8 channels:
  - Channels 0-3: A, C, G, T/U (One-Hot)
  - Channel 4:    CDS marker
  - Channel 5:    Splice-site marker
  - Channel 6:    miRNA binding affinity (TargetScan context++ score)
  - Channel 7:    RBP binding intensity (ENCODE eCLIP signal)

Transfers existing weights for channels 0-5 and all Mamba backbone layers 1:1,
initializes the new channels 6 & 7 with small random values, and exports a complete
Hugging Face checkpoint compatible with AutoModel.from_pretrained(..., trust_remote_code=True).
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


def convert_6track_to_8track(
    base_model_name: str,
    output_dir: Path,
    init_method: str = "normal",
    init_std: float = 0.02,
    device: str = "cpu"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("   Orthrus 6-Track -> 8-Track Checkpoint Conversion (Weight Surgery)   ")
    print("=" * 70)
    print(f"Base model:       {base_model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Init method (6-7):{init_method} (std={init_std if init_method == 'normal' else 'N/A'})")
    print(f"Device:           {device}")

    # 1. Load base 6-track model
    print(f"\n[1/5] Loading base model '{base_model_name}'...")
    base_path = Path(base_model_name)
    if base_path.is_dir() and (base_path / "best_finetuned_backbone").is_dir():
        print(f"      Found 'best_finetuned_backbone' inside {base_path} - using it as model source.")
        base_path = base_path / "best_finetuned_backbone"
        base_model_name = str(base_path)

    if base_path.is_file() and base_path.suffix in [".pt", ".pth", ".bin"]:
        print(f"      Restoring weights from PyTorch checkpoint file: {base_path}...")
        model_6t = AutoModel.from_pretrained("quietflamingo/orthrus-large-6-track", trust_remote_code=True)
        ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_dict[k[len("backbone."):]] = v
            elif not k.startswith("head."):
                backbone_dict[k] = v
        model_6t.load_state_dict(backbone_dict, strict=False)
    else:
        model_6t = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

    model_6t.eval()
    config_6t = model_6t.config

    old_n_tracks = getattr(config_6t, "n_tracks", 6)
    d_model = getattr(config_6t, "ssm_model_dim", 512)
    print(f"      Loaded model config: n_tracks={old_n_tracks}, d_model={d_model}, layers={getattr(config_6t, 'ssm_n_layers', 4)}")
    assert old_n_tracks == 6, f"Expected base model with 6 tracks, but got {old_n_tracks}."

    # 2. Create 8-track config and instantiate 8-track model
    print("\n[2/5] Creating 8-track model architecture...")
    config_dict = config_6t.to_dict()
    config_dict["n_tracks"] = 8

    # Recreate config & model instance
    ConfigClass = config_6t.__class__
    ModelClass = model_6t.__class__

    config_8t = ConfigClass(**config_dict)
    model_8t = ModelClass(config_8t)
    model_8t.eval()

    # 3. Perform weight surgery
    print("\n[3/5] Performing weight surgery on embedding layer...")
    state_6t = model_6t.state_dict()
    state_8t = model_8t.state_dict()

    for k, v in state_6t.items():
        if k == "embedding.weight":
            print(f"      Transferring '{k}': {v.shape} -> {state_8t[k].shape}")
            with torch.no_grad():
                # Copy channels 0 to 5 exactly
                state_8t[k][:, 0:6] = v[:, 0:6]

                # Initialize channels 6 & 7 (miRNA and eCLIP)
                if init_method == "zeros":
                    nn.init.zeros_(state_8t[k][:, 6:8])
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(state_8t[k][:, 6:8])
                else:  # 'normal'
                    nn.init.normal_(state_8t[k][:, 6:8], mean=0.0, std=init_std)

            print(f"      Channel weights initialized:")
            print(f"        Tracks 0-5 mean norm: {state_8t[k][:, 0:6].norm().item():.4f}")
            print(f"        Tracks 6-7 mean norm: {state_8t[k][:, 6:8].norm().item():.4f}")
        else:
            # Copy all other backbone parameters (Mamba blocks, LayerNorms, biases) 1:1
            state_8t[k] = v

    model_8t.load_state_dict(state_8t)
    print("      Weight transfer successfully completed.")

    # 4. Save model to output_dir
    print(f"\n[4/5] Saving converted model to: {output_dir}")
    model_8t.save_pretrained(output_dir)

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
        cfg["n_tracks"] = 8
        cfg["auto_map"] = {
            "AutoConfig": "orthrus_hf.OrthrusConfig",
            "AutoModel": "orthrus_hf.OrthrusPretrainedModel"
        }
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2)

    # 5. Verification forward pass
    print("\n[5/5] Running verification forward pass with dummy 8-track input...")
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model_8t.to(dev)

    dummy_seq_len = 128
    dummy_x = torch.randn(2, dummy_seq_len, 8, device=dev)
    dummy_lens = torch.tensor([dummy_seq_len, dummy_seq_len // 2], device=dev)

    with torch.no_grad():
        rep = model_8t.representation(dummy_x, dummy_lens, channel_last=True)
        unpooled = model_8t(dummy_x, channel_last=True)

    print(f"      Input shape:               {tuple(dummy_x.shape)} (B, L, C=8)")
    print(f"      Representation shape:      {tuple(rep.shape)} (expected: (2, {d_model}))")
    print(f"      Unpooled output shape:     {tuple(unpooled.shape)} (expected: (2, {dummy_seq_len}, {d_model}))")

    assert rep.shape == (2, d_model), f"Shape mismatch: {rep.shape} vs (2, {d_model})"
    print("\n" + "=" * 70)
    print("SUCCESS: 8-Track Orthrus model created and verified!")
    print(f"You can now load it via:")
    print(f"  model = AutoModel.from_pretrained('{output_dir}', trust_remote_code=True)")
    print("=" * 70 + "\n")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert Orthrus 6-track model to 8-track checkpoint")
    parser.add_argument(
        "--base_model",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Source Hugging Face model or local directory (default: quietflamingo/orthrus-large-6-track)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/checkpoints/orthrus-large-8-track",
        help="Destination path to save the 8-track checkpoint",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["normal", "zeros", "kaiming"],
        default="normal",
        help="Initialization method for the new channels 6 and 7 (default: normal)",
    )
    parser.add_argument(
        "--init_std",
        type=float,
        default=0.02,
        help="Standard deviation for normal initialization of channels 6 & 7 (default: 0.02)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for verification forward pass (default: cuda)",
    )
    args = parser.parse_args()

    convert_6track_to_8track(
        base_model_name=args.base_model,
        output_dir=Path(args.output_dir),
        init_method=args.init_method,
        init_std=args.init_std,
        device=args.device
    )


if __name__ == "__main__":
    main()
