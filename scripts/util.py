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


def to_object_array(tracks) -> "np.ndarray":
    """
    Packs variable-length (L, C) tracks into a 1D object array of float32 arrays.
    np.array(list_of_arrays, dtype=object) would instead create an (n, L, C) object array whenever all
    tracks share the same length (e.g. a chunk with a single transcript), breaking torch.from_numpy later.
    """
    import numpy as np
    tracks = list(tracks)
    arr = np.empty(len(tracks), dtype=object)
    for i, t in enumerate(tracks):
        arr[i] = np.asarray(t, dtype=np.float32)
    return arr


# Transcripts longer than max_length keep their 3' end, i.e. the 3' UTR with most miRNA / RBP signal is preserved
TRUNCATION_MODE = "keep_3prime"


def truncate_to_max_length(track, max_length: int):
    """Truncates an (L, C) track to its last max_length positions (see TRUNCATION_MODE)."""
    return track[-max_length:] if track.shape[0] > max_length else track


def require_truncation_mode(meta: Dict[str, Any], source: Union[str, Path]) -> None:
    """Rejects track archives that were generated with a different truncation convention (e.g. 5' end kept)."""
    found = meta.get("truncation")
    if found != TRUNCATION_MODE:
        raise ValueError(
            f"[Error] Track archive {source} was generated with truncation={found!r}, expected {TRUNCATION_MODE!r}! "
            f"Older archives kept the 5' end of long transcripts. Please regenerate the tracks with --recreate."
        )


RUN_CONFIG_FILENAME = "run_config.json"


def normalize_run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a run configuration to plain JSON types (Path -> str, NumPy -> Python) for comparison."""
    return json.loads(json.dumps(config, cls=_NumpyMetadataEncoder, sort_keys=True))


def diff_run_config(stored: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {key: (stored_value, current_value)} for all keys whose values differ."""
    stored, current = normalize_run_config(stored), normalize_run_config(current)
    return {
        k: (stored.get(k), current.get(k))
        for k in sorted(set(stored) | set(current))
        if stored.get(k) != current.get(k)
    }


def ensure_run_config(state_dir: Union[str, Path], config: Dict[str, Any], has_existing_state: bool, reset_hint: str) -> Path:
    """
    Guards resumable intermediate state (e.g. chunk directories) against mixing runs with different parameters.
    - run_config.json present: must match `config`, otherwise RuntimeError.
    - No run_config.json but existing state: state cannot be verified -> RuntimeError.
    - Otherwise: writes run_config.json.
    """
    cfg_file = Path(state_dir) / RUN_CONFIG_FILENAME

    if cfg_file.is_file():
        with open(cfg_file, "r", encoding="utf-8") as f:
            stored = json.load(f)
        diffs = diff_run_config(stored, config)
        if diffs:
            diff_str = "\n".join(f"  - {k}: existing={v[0]!r} | current={v[1]!r}" for k, v in diffs.items())
            raise RuntimeError(
                f"[Error] Existing state in '{state_dir}' was created with a different configuration:\n{diff_str}\n"
                f"Resuming would mix both runs. {reset_hint}"
            )
        print(f"[Resume] Run configuration matches: {cfg_file}")
        return cfg_file

    if has_existing_state:
        raise RuntimeError(
            f"[Error] Existing state in '{state_dir}' has no {RUN_CONFIG_FILENAME} and cannot be verified. {reset_hint}"
        )

    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(normalize_run_config(config), f, indent=2, ensure_ascii=False)
    print(f"[Run Config] Saved run configuration: {cfg_file}")
    return cfg_file


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



# =============================================================================
# Base Paths (override with the environment variables ORTHRUS_BASE / ORTHRUS_GTF_DB)
# =============================================================================

BASE_DIR = Path(os.environ.get("ORTHRUS_BASE", "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"))
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints" / "orthrus"
RESULTS_DIR = BASE_DIR / "results" / "Orthrus"
GTF_DB_PATH = Path(os.environ.get("ORTHRUS_GTF_DB", "/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db"))


# =============================================================================
# Dataset Registry (single source of truth for dataset keys & split lookup tables)
# =============================================================================

# Data directory of each dataset; generated tracks are stored in <dir>/orthrus/<track_name>/
DATASET_DIRS = {
    "hipsc_cm": DATA_DIR / "hIPSC_CM",
    "mrna_bench_human": DATA_DIR / "mrna_bench",
    "mrna_bench_mouse": DATA_DIR / "mrna_bench",
}

SPLITS_LOOKUP_TABLES = {
    "hipsc_cm": DATASET_DIRS["hipsc_cm"] / "hipsc_cm_10folds_lookup.csv",
    "mrna_bench_human": DATASET_DIRS["mrna_bench_human"] / "rnahl-human_10folds_lookup.csv",
    "mrna_bench_mouse": DATASET_DIRS["mrna_bench_mouse"] / "rnahl-mouse_10folds_lookup.csv",
}


def resolve_dataset_key(dataset_name: str) -> str:
    """
    Maps the 'dataset' field of the track metadata to a canonical dataset key (see SPLITS_LOOKUP_TABLES).
    Only case and '-' vs. '_' are normalized; any other name raises a KeyError instead of being guessed.
    """
    key = str(dataset_name).strip().lower().replace("-", "_")
    if key not in SPLITS_LOOKUP_TABLES:
        raise KeyError(
            f"[Error] Unknown dataset '{dataset_name}' in metadata! Known datasets: {list(SPLITS_LOOKUP_TABLES.keys())}."
        )
    return key


# =============================================================================
# Cross-Validation Scheme (single source of truth for fine-tuning, extraction & ridge regression)
# =============================================================================

# Splits 8-9 are held out as test set in every fold; splits 0-7 form the CV pool.
TEST_SPLITS = [8, 9]
CV_FOLDS = {
    0: {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7]},
    1: {"train": [2, 3, 4, 5, 6, 7], "val": [0, 1]},
    2: {"train": [0, 1, 4, 5, 6, 7], "val": [2, 3]},
    3: {"train": [0, 1, 2, 3, 6, 7], "val": [4, 5]},
}
CV_POOL_SPLITS = sorted({s for fold in CV_FOLDS.values() for s in fold["train"] + fold["val"]})

# Abort if more transcripts than this are missing from the lookup table (e.g. wrong or outdated table)
MAX_UNMATCHED_FRACTION = 0.01


def assign_splits(dataset_key: str, tx_ids, max_unmatched_frac: float = MAX_UNMATCHED_FRACTION) -> "np.ndarray":
    """
    Maps each transcript ID to its split (0-9) from the dataset's 10-fold lookup table; unmatched transcripts get -1.
    Raises if the lookup table is missing/incomplete or more than max_unmatched_frac of the transcripts are unmatched.
    """
    import numpy as np
    import pandas as pd

    lookup_path = SPLITS_LOOKUP_TABLES[dataset_key]
    if not lookup_path.exists():
        raise FileNotFoundError(f"[Error] Standardized 10-fold lookup table for '{dataset_key}' not found at: {lookup_path}!")

    print(f"\n[Splits Lookup] Loading lookup table ({dataset_key}): {lookup_path}")
    lookup_df = pd.read_csv(lookup_path)
    missing_cols = {"ensembl_transcript_id", "split"} - set(lookup_df.columns)
    if missing_cols:
        raise KeyError(f"[Error] Columns {sorted(missing_cols)} not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}")

    missing_splits = set(range(10)) - set(lookup_df["split"].dropna().astype(int).unique())
    if missing_splits:
        raise ValueError(f"[Error] Missing splits in lookup table {lookup_path}! Expected all 10 splits (0-9), but missing: {sorted(missing_splits)}.")

    tx_to_split = dict(zip(lookup_df["ensembl_transcript_id"].astype(str).str.strip(), lookup_df["split"].astype(int)))
    sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in tx_ids], dtype=int)

    n_unmatched = int(np.sum(sample_splits == -1))
    if n_unmatched > max_unmatched_frac * len(sample_splits):
        raise ValueError(
            f"[Error] {n_unmatched:,} of {len(sample_splits):,} transcripts ({n_unmatched / len(sample_splits):.1%}) are missing "
            f"from the lookup table {lookup_path} (allowed: {max_unmatched_frac:.0%}). Wrong or outdated lookup table?"
        )
    if n_unmatched > 0:
        print(f"[Warning] {n_unmatched} transcripts not found in lookup table (split = -1).")
    return sample_splits


def require_meta(meta: Dict[str, Any], source: Union[str, Path], *keys: str) -> None:
    """Raises a KeyError if any of the required keys is missing in the companion metadata of 'source'."""
    missing = [k for k in keys if k not in meta]
    if missing:
        raise KeyError(f"[Error] Required metadata key(s) {missing} not found in companion metadata for {source}!")


# =============================================================================
# Regression Metrics (shared by fine-tuning, channel ablation & ridge regression)
# =============================================================================

def regression_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    """
    Pearson r, Spearman rho (with p-values), MSE, RMSE, MAE and R2.
    Correlations are reported as 0 (p = 1) if fewer than 3 samples or a constant prediction/target make them undefined.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) > 2 and np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        p_corr, p_val = pearsonr(y_true, y_pred)
        s_corr, s_val = spearmanr(y_true, y_pred)
    else:
        p_corr, p_val, s_corr, s_val = 0.0, 1.0, 0.0, 1.0
    mse = float(mean_squared_error(y_true, y_pred))

    prefix_str = f"{prefix}_" if prefix else ""
    return {
        f"{prefix_str}pearson_r": float(p_corr),
        f"{prefix_str}pearson_pvalue": float(p_val),
        f"{prefix_str}spearman_rho": float(s_corr),
        f"{prefix_str}spearman_pvalue": float(s_val),
        f"{prefix_str}mse": mse,
        f"{prefix_str}rmse": float(np.sqrt(mse)),
        f"{prefix_str}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix_str}r2": float(r2_score(y_true, y_pred)) if len(y_true) > 2 else 0.0,
    }


def print_metrics(metrics: Dict[str, float], title: str) -> None:
    """Formatted console output for metrics."""
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        if "pvalue" in k:
            print(f"  {k:28s}: {v:.3e}")
        else:
            print(f"  {k:28s}: {v:.4f}")
