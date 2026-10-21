"""
Utility functions for profiling GPU VRAM and CPU RAM usage during PyTorch execution.
Helps choose optimal cluster GPU nodes (e.g. 16GB vs 24GB vs 40GB vs 80GB).
"""

import sys
import os
from typing import Optional, Dict, Any, Union

try:
    import torch
except ImportError:
    torch = None

try:
    import resource  # Available on Linux / Cluster environments
except ImportError:
    resource = None

try:
    import psutil  # Optional cross-platform memory tracking
except ImportError:
    psutil = None


def reset_peak_memory_stats(device: Optional[Any] = None) -> None:
    """
    Resets the peak memory statistics for CUDA devices.
    Call this right before the heavy workload (e.g. training loop or inference).
    """
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
            if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                torch.cuda.reset_accumulated_memory_stats(device)
        except Exception:
            pass


def get_memory_profile(device: Optional[Any] = None) -> Dict[str, Any]:
    """
    Collects peak and current VRAM (GPU) and RAM (CPU) usage metrics.
    """
    profile: Dict[str, Any] = {
        "cuda_available": False,
        "device_name": "N/A",
        "device_index": 0,
        "total_vram_gb": 0.0,
        "peak_vram_allocated_gb": 0.0,
        "peak_vram_reserved_gb": 0.0,
        "current_vram_allocated_gb": 0.0,
        "current_vram_reserved_gb": 0.0,
        "peak_ram_gb": 0.0,
        "current_ram_gb": 0.0,
        "recommended_gpu": "N/A",
    }

    # --- CPU RAM (Peak & Current) ---
    if resource is not None:
        try:
            # ru_maxrss is in KiB on Linux, in Bytes on macOS
            raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                profile["peak_ram_gb"] = raw_rss / (1024 ** 3)
            else:
                profile["peak_ram_gb"] = raw_rss / (1024 ** 2)
        except Exception:
            pass

    if psutil is not None:
        try:
            proc = psutil.Process()
            profile["current_ram_gb"] = proc.memory_info().rss / (1024 ** 3)
            if profile["peak_ram_gb"] == 0.0:
                profile["peak_ram_gb"] = profile["current_ram_gb"]
        except Exception:
            pass

    # --- GPU VRAM ---
    if torch is not None and torch.cuda.is_available():
        try:
            if device is None:
                dev_idx = torch.cuda.current_device()
            elif isinstance(device, str):
                dev_idx = torch.device(device).index or 0
            elif isinstance(device, torch.device):
                dev_idx = device.index or 0
            elif isinstance(device, int):
                dev_idx = device
            else:
                dev_idx = 0

            profile["cuda_available"] = True
            profile["device_index"] = int(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            profile["device_name"] = props.name
            profile["total_vram_gb"] = props.total_memory / (1024 ** 3)

            profile["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
            profile["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 3)
            profile["current_vram_allocated_gb"] = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
            profile["current_vram_reserved_gb"] = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)

            # Node recommendation based on peak reserved memory (with ~15% safety margin)
            peak_res = profile["peak_vram_reserved_gb"]
            if peak_res <= 7.0:
                profile["recommended_gpu"] = "8 GB GPU (z. B. RTX 3070 / RTX 2080 / T4)"
            elif peak_res <= 13.5:
                profile["recommended_gpu"] = "16 GB GPU (z. B. V100 16GB / T4 16GB / RTX 4080)"
            elif peak_res <= 21.0:
                profile["recommended_gpu"] = "24 GB GPU (z. B. RTX 3090 / RTX 4090 / A10 / A30)"
            elif peak_res <= 36.0:
                profile["recommended_gpu"] = "40 GB GPU (z. B. A100 40GB / A40)"
            elif peak_res <= 72.0:
                profile["recommended_gpu"] = "80 GB GPU (z. B. A100 80GB / H100 80GB)"
            else:
                profile["recommended_gpu"] = ">80 GB (Multi-GPU / DeepSpeed / Gradient Checkpointing empfohlen)"

        except Exception as e:
            profile["cuda_error"] = str(e)

    return profile


def print_memory_profile(device: Optional[Any] = None, title: str = "SPEICHER-PROFILING (PEAK)") -> Dict[str, Any]:
    """
    Prints a formatted profiling overview of peak VRAM and CPU RAM usage,
    along with cluster GPU recommendation. Returns the profile dict.
    """
    prof = get_memory_profile(device=device)

    print("\n" + "=" * 65)
    print(f"[{title.strip()}]")
    print("=" * 65)

    if prof["cuda_available"]:
        pct_used = (
            (prof["peak_vram_reserved_gb"] / prof["total_vram_gb"] * 100)
            if prof["total_vram_gb"] > 0
            else 0.0
        )
        print(f"  GPU-Geraet:               [{prof['device_index']}] {prof['device_name']}")
        print(f"  GPU-Gesamtspeicher:       {prof['total_vram_gb']:.2f} GB")
        print("-" * 65)
        print(f"  Peak VRAM (Reserviert)*:  {prof['peak_vram_reserved_gb']:.2f} GB  ({pct_used:.1f}% der GPU)")
        print(f"  Peak VRAM (Tensoren):     {prof['peak_vram_allocated_gb']:.2f} GB")
        print(f"  Ende VRAM (Reserviert):   {prof['current_vram_reserved_gb']:.2f} GB")
        print(f"  Ende VRAM (Tensoren):     {prof['current_vram_allocated_gb']:.2f} GB")
    else:
        print("  CUDA/GPU:                 Nicht aktiv / Nicht verfuegbar")

    print("-" * 65)
    if prof["peak_ram_gb"] > 0:
        print(f"  Peak CPU-RAM (MaxRSS):    {prof['peak_ram_gb']:.2f} GB")
    if prof["current_ram_gb"] > 0:
        print(f"  Ende CPU-RAM:             {prof['current_ram_gb']:.2f} GB")

    if prof["cuda_available"]:
        print("=" * 65)
        print("Hinweis / Cluster-GPU-Empfehlung:")
        print(f"   -> {prof['recommended_gpu']}")
        print("   (*Hinweis: 'Reserviert' ist die PyTorch Caching-Grenze, die OOMs bestimmt)")
    print("=" * 65 + "\n")

    return prof


# =============================================================================
# Track & Dataset Metadata Helpers (.meta.json & NPZ synchronization)
# =============================================================================

import json
from pathlib import Path


class _NumpyMetadataEncoder(json.JSONEncoder):
    """Encodes NumPy primitives and arrays to standard Python types for JSON."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def get_meta_filepath(file_path: Union[str, Path]) -> Path:
    """
    Returns the canonical .meta.json file path for a given dataset or model file.
    For example: 'data/hipsc_cm_6tracks.npz' -> 'data/hipsc_cm_6tracks.meta.json'
    """
    p = Path(file_path)
    if p.name.endswith(".meta.json"):
        return p
    if p.suffix.lower() == ".json":
        return p
    return p.parent / f"{p.stem}.meta.json"


def save_metadata(output_path: Union[str, Path], meta_dict: Dict[str, Any]) -> Path:
    """
    Saves a metadata dictionary as formatted JSON alongside the given file.
    Automatically converts NumPy scalars and arrays to JSON-serializable types.
    """
    meta_file = get_meta_filepath(output_path)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, cls=_NumpyMetadataEncoder, ensure_ascii=False)

    print(f"[Metadata] Saved metadata file: {meta_file}")
    return meta_file


def load_metadata(file_path: Union[str, Path], required: bool = True) -> Dict[str, Any]:
    """
    Strictly loads companion metadata (.meta.json / model_meta.json / results.meta.json).
    If required=True and no metadata file is found, raises FileNotFoundError.
    """
    p = Path(file_path)

    # Check for companion JSON file
    candidates = [
        get_meta_filepath(p),
        p.parent / f"{p.stem}.json",
        p.with_suffix(".json"),
        p / "model_meta.json" if p.is_dir() else None,
        p / "results.meta.json" if p.is_dir() else None,
        p / "meta.json" if p.is_dir() else None,
    ]

    for cand in candidates:
        if cand is not None and cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return meta
            except Exception as e:
                raise ValueError(f"[Metadata Error] Failed to parse JSON from '{cand}': {e}") from e

    if required:
        raise FileNotFoundError(
            f"[Metadata Error] Required companion metadata file (.meta.json) not found for: '{p}'! "
            f"Please ensure the companion metadata file exists."
        )

    return {}

