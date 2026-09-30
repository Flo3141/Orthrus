"""
Utility functions for profiling GPU VRAM and CPU RAM usage during PyTorch execution.
Helps choose optimal cluster GPU nodes (e.g. 16GB vs 24GB vs 40GB vs 80GB).
"""

import sys
import os
from typing import Optional, Dict, Any

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
                device = torch.cuda.current_device()
            elif isinstance(device, str):
                device = torch.device(device).index or 0

            profile["cuda_available"] = True
            profile["device_index"] = device
            props = torch.cuda.get_device_properties(device)
            profile["device_name"] = props.name
            profile["total_vram_gb"] = props.total_memory / (1024 ** 3)

            profile["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            profile["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            profile["current_vram_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
            profile["current_vram_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024 ** 3)

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
