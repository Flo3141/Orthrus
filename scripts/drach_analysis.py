#!/usr/bin/env python3
"""
Motif perturbation analysis (in-silico mutagenesis with matched controls) of fine-tuned N-track Orthrus models.

Tests whether the model uses a sequence motif (DRACH, the m6A consensus, plus the positive controls AUUUA and
UGUANAUA) for its half-life prediction. For every test transcript with an annotated 3' UTR and for every
substitution (A>G, A>C, A>U), three predictions are computed:

  base     unchanged input
  motif    the central A of every motif site in the 3' UTR is replaced
  control  the same number of A replacements at A's outside the motif sites, as close as possible to one site each

Effect = pred_motif - pred_control. An effect > 0 means that destroying the motif raises the predicted half-life,
i.e. the model treats the motif as destabilizing. The control removes the shift that any replacement causes.

Several substitutions test whether an effect belongs to the loss of the motif or to one particular base change.
A>C can create a new DRACH site (AAC becomes ACC), so only sites whose replacement creates no new site are changed;
the skipped sites are counted.

Besides the effect, the correlation of the predicted half-lives of the mutated variants with the original
predictions (and with the measured half-lives) is reported.

All predictions run in float32: in bf16 the rounding noise from the batch composition (max 7.8e-3) is larger than the
effect to be measured (about 3.5e-3). Only the four base channels are changed, all other channels keep their values.

Not meant to be run directly: xai_analysis.py calls run_motif_analysis() and aggregate_motif_folds().
Usage: python xai_analysis.py --data_path <tracks.npz> --analyses drach
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, wilcoxon
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Motif definitions live in drach_motifs.py (no model dependencies), shared with the shift analysis and the Saluki files
from drach_motifs import MOTIFS, Motif
from finetune_ntrack import BucketBatchSampler, MultiTrackDataset, TrackData, pad_collate_fn, predict
from util import SPLITS_LOOKUP_TABLES, truncate_to_max_length


BAR_COLORS = ["#2a78d6", "#e07b39", "#3a9d5d"]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# The check against a batch size of 1 uses at most this many transcripts (evenly spread over the set)
INVARIANCE_CHECK_SIZE = 48
MAX_INVARIANCE_DIFF = 1e-5

# Channel order of the base channels in every track archive (BASE_CHANNEL_NAMES in generate_tracks.py)
BASE_BYTES = np.frombuffer(b"ACGUN", dtype=np.uint8)
BASE_CHANNEL = {"A": 0, "C": 1, "G": 2, "U": 3}
SUBSTITUTIONS = ("G", "C", "U")


def substitution_label(base: str) -> str:
    return f"A>{base}"


# =============================================================================
# 1. Motifs, sequence and perturbations
# =============================================================================

def decode_sequence(track: np.ndarray) -> str:
    """One-hot base channels to a string; positions without any base become N (argmax would silently give A)."""
    bases = track[:, :4]
    indices = bases.argmax(axis=1)
    indices[bases.sum(axis=1) == 0] = 4
    return BASE_BYTES[indices].tobytes().decode()


def creates_site(seq: str, position: int, utr3_start: int, motif: Motif, base: str) -> bool:
    """True if replacing the base at `position` creates a motif site that did not exist before (checked in a local window)."""
    lo = max(0, position - motif.length + 1)
    window = seq[lo:position + motif.length]
    mutated = window[:position - lo] + base + window[position - lo + 1:]

    def starts(w: str) -> set:
        return {lo + m.start() for m in re.finditer(f"(?={motif.pattern})", w) if lo + m.start() >= utr3_start}

    return bool(starts(mutated) - starts(window))


def match_controls(seq: str, utr3_start: int, motif: Motif, sites: list, anchors: list) -> Optional[list]:
    """
    One control A per anchor position, greedily by distance and without reuse. Candidates are A's in the 3' UTR
    outside all motif sites with the required following base. Returns None if there are fewer candidates than anchors.
    """
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    mask = arr == ord("A")
    mask[:utr3_start] = False
    if motif.control_next_base is not None:
        mask[-1] = False
        mask[:-1] &= arr[1:] == ord(motif.control_next_base)
    for site in sites:
        mask[site:site + motif.length] = False

    candidates = np.flatnonzero(mask)
    if len(candidates) < len(anchors):
        return None

    free = np.ones(len(candidates), dtype=bool)
    controls = []
    for anchor in anchors:
        distance = np.abs(candidates - anchor).astype(np.float64)
        distance[~free] = np.inf
        best = int(distance.argmin())
        free[best] = False
        controls.append(int(candidates[best]))
    return controls


def substitute(seq: str, positions: list, base: str) -> str:
    chars = list(seq)
    for p in positions:
        if chars[p] != "A":
            raise ValueError(f"[Error] Position {p} is {chars[p]!r}, not A!")
        chars[p] = base
    return "".join(chars)


def mutate_track(track: np.ndarray, positions: list, base: str) -> np.ndarray:
    """Copy of the track with A replaced by `base` at the given positions; only the base channels change."""
    mutated = track.copy()
    mutated[positions, BASE_CHANNEL["A"]] = 0.0
    mutated[positions, BASE_CHANNEL[base]] = 1.0
    return mutated


@dataclass
class Perturbation:
    motif_positions: list
    control_positions: list
    n_sites: int                  # mutated sites (= mutations in each variant)
    n_skipped: int                # sites left unchanged because the replacement would create a new site
    original_sites_left: int      # sites of the original that still exist after the motif mutation (expected: n_skipped)
    created_sites: int            # sites that did not exist before and exist after the motif mutation (expected: 0)
    control_sites_changed: bool   # the control mutation changed the number of motif sites (expected False)


def build_perturbation(seq: str, utr3_start: int, motif: Motif, base: str):
    """Returns (Perturbation, None) or (None, reason) with reason 'no_site', 'no_safe_site' or 'not_enough_controls'."""
    sites = motif.sites(seq, utr3_start)
    if not sites:
        return None, "no_site"
    safe = [s for s in sites if not creates_site(seq, s + motif.mutated_offset, utr3_start, motif, base)]
    if not safe:
        return None, "no_safe_site"

    motif_positions = [s + motif.mutated_offset for s in safe]
    controls = match_controls(seq, utr3_start, motif, sites, motif_positions)
    if controls is None:
        return None, "not_enough_controls"

    after = set(motif.sites(substitute(seq, motif_positions, base), utr3_start))
    before = set(sites)
    control_changed = len(motif.sites(substitute(seq, controls, base), utr3_start)) != len(sites)
    return Perturbation(motif_positions, controls, len(safe), len(sites) - len(safe), len(after & before), len(after - before), control_changed), None


# =============================================================================
# 2. Predictions (float32, input order)
# =============================================================================

def predict_tracks(model: nn.Module, tracks: list, tx_ids: np.ndarray, device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    """
    Predictions in float32 in the order of `tracks`. The sampler sorts by length and predict() returns loader order,
    so the result is sorted back explicitly.
    """
    n = len(tracks)
    if n == 0:
        return np.zeros(0)
    dataset = MultiTrackDataset(tracks, np.zeros(n, dtype=np.float32), np.array(["-"] * n), np.asarray(tx_ids), max_length)
    sampler = BucketBatchSampler(dataset, batch_size=batch_size, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pad_collate_fn, num_workers=2, pin_memory=(device.type == "cuda"))
    _, preds = predict(model, loader, device, torch.float32)

    order = np.array([k for batch in sampler for k in batch])
    result = np.zeros(n)
    result[order] = preds
    return result


def check_batch_invariance(model: nn.Module, tracks: list, tx_ids: np.ndarray, base: np.ndarray, device: torch.device, batch_size: int, max_length: int) -> float:
    """Largest difference between `base` (predicted with batch_size) and batch size 1 on up to INVARIANCE_CHECK_SIZE transcripts."""
    check_idx = np.unique(np.linspace(0, len(tracks) - 1, min(INVARIANCE_CHECK_SIZE, len(tracks))).astype(int))
    single = predict_tracks(model, [tracks[k] for k in check_idx], tx_ids[check_idx], device, 1, max_length)
    diff = float(np.max(np.abs(single - base[check_idx])))
    print(f"[Check] Largest difference between batch size {batch_size} and 1: {diff:.2e} ({len(check_idx)} transcripts)")
    if diff > MAX_INVARIANCE_DIFF:
        print(f"[Warning] Difference above {MAX_INVARIANCE_DIFF:.0e}: the predictions depend on the batch composition, effects of this size are noise.")
    return diff


def load_homology_groups(dataset_key: str, tx_ids: np.ndarray) -> np.ndarray:
    """
    Homology group of each transcript from the split lookup table. Isoforms of one gene (and homologous genes)
    are no independent observations, so the statistics run on groups. A transcript without a group counts as its own.
    """
    lookup_path = SPLITS_LOOKUP_TABLES[dataset_key]
    lookup = pd.read_csv(lookup_path)
    if "homology_group" not in lookup.columns:
        raise KeyError(
            f"[Error] Column 'homology_group' not found in lookup table {lookup_path} (columns: {list(lookup.columns)}). "
            f"Is this the old gene-based table? The analysis needs the homology splits (create_deterministic_splits.py)."
        )
    mapping = dict(zip(lookup["ensembl_transcript_id"].astype(str).str.strip(), lookup["homology_group"]))
    groups = []
    for tx in tx_ids:
        group = mapping.get(str(tx).strip())
        groups.append(f"tx:{tx}" if group is None or pd.isna(group) else f"hg:{group}")
    return np.array(groups)


@dataclass
class Utr3Inputs:
    """Test transcripts with an annotated 3' UTR. Tracks are truncated to max_length, `utr3` refers to the truncated track."""
    n_test: int
    n_unannotated: int
    tracks: list
    seqs: list
    utr3: list
    tx_ids: np.ndarray
    genes: np.ndarray
    groups: np.ndarray
    targets: np.ndarray
    n_without_base: int


def prepare_utr3_inputs(data: TrackData, test_idx: np.ndarray, max_length: int) -> Utr3Inputs:
    """Selects the test transcripts with an annotated 3' UTR, truncates them like the dataset and decodes the sequences."""
    if data.utr3_start is None:
        raise ValueError("[Error] The track archive has no 'utr3_start', 'full_length' and 'seq_lens' fields; the 3' UTR cannot be located.")

    # Only transcripts with an annotated 3' UTR; positions refer to the track after truncation to max_length
    annotated = np.array([i for i in test_idx if data.utr3_start[i] >= 0], dtype=np.int64)
    if len(annotated) == 0:
        raise ValueError("[Error] No test transcript has an annotated 3' UTR (utr3_start == 0 means no annotation).")

    tracks, seqs, utr3 = [], [], []
    for i in annotated:
        track = truncate_to_max_length(data.tracks[i], max_length)
        cut = len(data.tracks[i]) - len(track)
        tracks.append(track)
        seqs.append(decode_sequence(track))
        utr3.append(max(int(data.utr3_start[i]) - cut, 0))
    n_without_base = int(sum(s.count("N") for s in seqs))
    if n_without_base > 0:
        print(f"[Warning] {n_without_base} positions without any base channel (decoded as N) in the analyzed transcripts.")

    tx_ids = data.tx_ids[annotated]
    return Utr3Inputs(
        n_test=int(len(test_idx)), n_unannotated=int(len(test_idx) - len(annotated)),
        tracks=tracks, seqs=seqs, utr3=utr3, tx_ids=tx_ids, genes=data.genes[annotated],
        groups=load_homology_groups(data.dataset_name, tx_ids), targets=data.targets[annotated], n_without_base=n_without_base,
    )


# =============================================================================
# 3. Statistics
# =============================================================================

def _num(x) -> Optional[float]:
    """JSON-safe float; NaN becomes None."""
    x = float(x)
    return None if np.isnan(x) else x


def correlation_pair(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) > 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
        return {"pearson_r": float(pearsonr(a, b)[0]), "spearman_rho": float(spearmanr(a, b)[0]), "mean_abs_diff": float(np.mean(np.abs(a - b)))}
    return {"pearson_r": None, "spearman_rho": None, "mean_abs_diff": float(np.mean(np.abs(a - b))) if len(a) else None}


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> list:
    """95 % percentile interval of the mean over resampled groups."""
    means = values[rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_motif(table: pd.DataFrame, exclusions: dict, perturbations: list, rng: np.random.Generator) -> dict:
    """
    Statistics for one motif and substitution. `table` has one row per analyzed transcript: group, target, n_sites,
    pred_base, pred_motif, pred_control. Effects are averaged per homology group before any test or interval.
    """
    t = table.assign(
        effect=table.pred_motif - table.pred_control,
        motif_shift=table.pred_motif - table.pred_base,
        base_effect=table.pred_control - table.pred_base,
    )
    g = t.groupby("group")[["effect", "motif_shift", "base_effect", "n_sites"]].mean()
    effect = g["effect"].to_numpy()

    try:
        wilcoxon_p = _num(wilcoxon(effect).pvalue)
    except ValueError:  # all differences are zero
        wilcoxon_p = None

    rho = p_rho = None
    if len(g) > 2 and g["n_sites"].nunique() > 1 and g["effect"].nunique() > 1:
        rho, p_rho = (float(v) for v in spearmanr(g["n_sites"], g["effect"]))

    return {
        "exclusions": exclusions,
        "n_transcripts": int(len(t)),
        "n_groups": int(len(g)),
        "sites_mutated_per_transcript_median": float(t["n_sites"].median()),
        "sites_mutated_total": int(t["n_sites"].sum()),
        "mutation_checks": {
            "sites_skipped_total": int(sum(p.n_skipped for p in perturbations)),
            "transcripts_with_skipped_sites": int(sum(p.n_skipped > 0 for p in perturbations)),
            "original_sites_left_total": int(sum(p.original_sites_left for p in perturbations)),
            "created_sites_total": int(sum(p.created_sites for p in perturbations)),
            "transcripts_with_created_sites": int(sum(p.created_sites > 0 for p in perturbations)),
            "transcripts_where_control_changed_site_count": int(sum(p.control_sites_changed for p in perturbations)),
        },
        "effect": {
            "mean_of_group_means": float(effect.mean()),
            "median_of_group_means": float(np.median(effect)),
            "ci95_bootstrap": bootstrap_mean_ci(effect, rng),
            "wilcoxon_p": wilcoxon_p,
            "fraction_positive_groups": float(np.mean(effect > 0)),
            "transcript_level_mean": float(t["effect"].mean()),
            "transcript_level_median": float(t["effect"].median()),
        },
        "motif_shift_vs_base": {"mean_of_group_means": float(g["motif_shift"].mean()), "median_of_group_means": float(g["motif_shift"].median())},
        "base_effect_control_vs_base": {"mean_of_group_means": float(g["base_effect"].mean()), "median_of_group_means": float(g["base_effect"].median())},
        "spearman_effect_vs_n_sites": {"rho": rho, "p": p_rho},
        "correlation_of_predictions": {
            "base_vs_motif": correlation_pair(t["pred_base"].to_numpy(), t["pred_motif"].to_numpy()),
            "base_vs_control": correlation_pair(t["pred_base"].to_numpy(), t["pred_control"].to_numpy()),
            "motif_vs_control": correlation_pair(t["pred_motif"].to_numpy(), t["pred_control"].to_numpy()),
        },
        "pearson_r_with_measured_target": {
            variant: correlation_pair(t["target"].to_numpy(), t[f"pred_{variant}"].to_numpy())["pearson_r"]
            for variant in ("base", "motif", "control")
        },
    }


def substitution_agreement(tables: dict) -> dict:
    """
    Correlation of the per-transcript effects between two substitutions (transcripts analyzed in both).
    A motif effect should appear with every substitution; an effect of one substitution alone points to the base change.
    """
    result = {}
    labels = list(tables)
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            merged = tables[first][["tx_id", "effect"]].merge(tables[second][["tx_id", "effect"]], on="tx_id", suffixes=("_1", "_2"))
            pair = correlation_pair(merged["effect_1"].to_numpy(), merged["effect_2"].to_numpy())
            result[f"{first}_vs_{second}"] = {"n_transcripts": int(len(merged)), "pearson_r": pair["pearson_r"], "spearman_rho": pair["spearman_rho"]}
    return result


# =============================================================================
# 4. Plots
# =============================================================================

def plot_motif_effects(motif_names: list, series: dict, ylabel: str, title: str, output_png: Path):
    """
    Grouped bar chart: one group per motif, one bar per substitution. `series` maps the label of a substitution to
    (means, lower, upper) aligned with motif_names; lower/upper are the distances from the bar to the ends of the error bar.
    """
    n_series = len(series)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(max(5.5, len(motif_names) * 2.4), 4.8))

    for k, (label, (means, lower, upper)) in enumerate(series.items()):
        xs = np.arange(len(motif_names)) - 0.4 + width * (k + 0.5)
        means, lower, upper = np.asarray(means, float), np.asarray(lower, float), np.asarray(upper, float)
        ax.bar(xs, means, yerr=[lower, upper], width=width * 0.92, capsize=3, color=BAR_COLORS[k % len(BAR_COLORS)], label=label,
               error_kw={"linewidth": 1, "ecolor": "#333333"}, zorder=2)
        for x, m, lo, up in zip(xs, means, lower, upper):
            if np.isnan(m):
                continue
            above = m >= 0
            end = up if above else lo
            ax.annotate(f"{m:.4f}", xy=(x, m + end if above else m - end), xytext=(0, 3 if above else -3), textcoords="offset points",
                        ha="center", va="bottom" if above else "top", fontsize=7)

    ax.set_title(title, fontsize=12, loc="left")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(range(len(motif_names)))
    ax.set_xticklabels(motif_names, fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(0, color="#666666", linewidth=0.8, zorder=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[DRACH] Saved plot to: {output_png} and {output_png.with_suffix('.pdf')}")


def plot_prediction_scatter(variants: dict, summaries: dict, title: str, output_png: Path, variant_labels: tuple = ("motif mutated", "control mutated")):
    """Row per motif: original prediction against the prediction with mutated motif (left) and mutated control (right)."""
    names = list(variants)
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 4.2 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        base, motif_pred, control_pred = variants[name]
        lo, hi = float(min(base.min(), motif_pred.min(), control_pred.min())), float(max(base.max(), motif_pred.max(), control_pred.max()))
        corr = summaries[name]["correlation_of_predictions"]
        for col, (label, mutated, key) in enumerate(((variant_labels[0], motif_pred, "base_vs_motif"), (variant_labels[1], control_pred, "base_vs_control"))):
            ax = axes[row, col]
            ax.scatter(base, mutated, s=6, alpha=0.4, color=BAR_COLORS[0], linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color="#666666", linewidth=0.8)
            r = corr[key]["pearson_r"]
            ax.set_title(f"{name}, {label}: r = {r:.4f}" if r is not None else f"{name}, {label}", fontsize=10, loc="left")
            ax.set_xlabel("Predicted half-life, original", fontsize=9)
            ax.set_ylabel("Predicted half-life, mutated", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[DRACH] Saved plot to: {output_png} and {output_png.with_suffix('.pdf')}")


# =============================================================================
# 5. Analysis of one fold and aggregation across folds
# =============================================================================

def run_motif_analysis(
    model: nn.Module,
    data: TrackData,
    test_idx: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int = 12288,
    motifs: tuple = MOTIFS,
    substitutions: tuple = SUBSTITUTIONS,
) -> dict:
    """
    Runs the motif perturbation for one fold on the given test samples and writes drach_summary.json,
    drach_predictions.csv and the plots into fold_dir. Returns the summary.
    """
    model.eval()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    inputs = prepare_utr3_inputs(data, test_idx, max_length)
    tracks, seqs, utr3 = inputs.tracks, inputs.seqs, inputs.utr3
    tx_ids, genes, groups, targets = inputs.tx_ids, inputs.genes, inputs.groups, inputs.targets
    n_analyzed = len(tracks)
    print(f"[DRACH] {n_analyzed} of {inputs.n_test} test transcripts have an annotated 3' UTR, {len(np.unique(groups))} homology groups. "
          f"Precision: float32, substitutions: {[substitution_label(b) for b in substitutions]}")

    base = predict_tracks(model, tracks, tx_ids, device, batch_size, max_length)

    # The result must not depend on how the batches are put together
    invariance_diff = check_batch_invariance(model, tracks, tx_ids, base, device, batch_size, max_length)

    motif_summaries = {}
    scatter_data = {b: {} for b in substitutions}   # substitution -> motif -> (base, motif, control)
    rows = []
    for motif in motifs:
        per_substitution, tables = {}, {}
        for sub in substitutions:
            label = substitution_label(sub)
            exclusions = {"no_site": 0, "no_safe_site": 0, "not_enough_controls": 0}
            included, perturbations = [], []
            for k in tqdm(range(n_analyzed), desc=f"[DRACH] Planning {motif.name} {label}"):
                perturbation, reason = build_perturbation(seqs[k], utr3[k], motif, sub)
                if perturbation is None:
                    exclusions[reason] += 1
                else:
                    included.append(k)
                    perturbations.append(perturbation)
            exclusions["included"] = len(included)
            print(f"[DRACH] {motif.name} {label}: {len(included)} transcripts analyzed, excluded: {exclusions['no_site']} without site, "
                  f"{exclusions['no_safe_site']} without a site that can be replaced safely, {exclusions['not_enough_controls']} without enough controls")
            if not included:
                per_substitution[label] = {"exclusions": exclusions, "n_transcripts": 0}
                continue

            sub_ids = tx_ids[included]
            pred_motif = predict_tracks(model, [mutate_track(tracks[k], p.motif_positions, sub) for k, p in zip(included, perturbations)], sub_ids, device, batch_size, max_length)
            pred_control = predict_tracks(model, [mutate_track(tracks[k], p.control_positions, sub) for k, p in zip(included, perturbations)], sub_ids, device, batch_size, max_length)

            table = pd.DataFrame({
                "tx_id": sub_ids, "gene": genes[included], "group": groups[included], "target": targets[included],
                "n_sites": [p.n_sites for p in perturbations],
                "pred_base": base[included], "pred_motif": pred_motif, "pred_control": pred_control,
            })
            per_substitution[label] = summarize_motif(table, exclusions, perturbations, rng)
            tables[label] = table.assign(effect=table.pred_motif - table.pred_control)
            scatter_data[sub][motif.name] = (table["pred_base"].to_numpy(), pred_motif, pred_control)
            rows.append(table.assign(motif=motif.name, substitution=label))

            eff = per_substitution[label]["effect"]
            corr = per_substitution[label]["correlation_of_predictions"]
            print(f"[DRACH] {motif.name} {label}: effect {eff['mean_of_group_means']:+.5f} (95 % CI {eff['ci95_bootstrap'][0]:+.5f} to {eff['ci95_bootstrap'][1]:+.5f}), "
                  f"r(base, motif) = {corr['base_vs_motif']['pearson_r']}, r(base, control) = {corr['base_vs_control']['pearson_r']}")

        motif_summaries[motif.name] = {"substitutions": per_substitution, "agreement_between_substitutions": substitution_agreement(tables)}

    summary = {
        "precision": "float32",
        "batch_size": batch_size,
        "max_length": max_length,
        "n_test_transcripts": int(len(test_idx)),
        "n_without_utr3_annotation": inputs.n_unannotated,
        "n_analyzed_transcripts": n_analyzed,
        "positions_without_base": inputs.n_without_base,
        "batch_invariance_max_abs_diff": invariance_diff,
        "motifs": motif_summaries,
    }
    with open(fold_dir / "drach_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        pd.concat(rows).drop(columns="group").to_csv(fold_dir / "drach_predictions.csv", index=False)
    print(f"[DRACH] Saved motif summary to: {fold_dir / 'drach_summary.json'}")

    # Plots
    labels = [substitution_label(b) for b in substitutions]
    motif_names = [m.name for m in motifs if any(motif_summaries[m.name]["substitutions"][lab]["n_transcripts"] > 0 for lab in labels)]
    if motif_names:
        series = {}
        for lab in labels:
            means, lower, upper = [], [], []
            for name in motif_names:
                s = motif_summaries[name]["substitutions"][lab]
                if s["n_transcripts"] == 0:
                    means.append(np.nan); lower.append(np.nan); upper.append(np.nan)
                else:
                    m, (lo, hi) = s["effect"]["mean_of_group_means"], s["effect"]["ci95_bootstrap"]
                    means.append(m); lower.append(m - lo); upper.append(hi - m)
            series[lab] = (means, lower, upper)
        plot_motif_effects(
            motif_names, series,
            ylabel="Effect: prediction(motif mutated) - prediction(control mutated)",
            title=f"{fold_dir.name}: mean over homology groups, 95 % bootstrap interval",
            output_png=fold_dir / "drach_effects.png",
        )
        for sub in substitutions:
            if scatter_data[sub]:
                summaries = {n: motif_summaries[n]["substitutions"][substitution_label(sub)] for n in scatter_data[sub]}
                plot_prediction_scatter(scatter_data[sub], summaries, f"{fold_dir.name}, {substitution_label(sub)}: original against mutated prediction",
                                        fold_dir / f"drach_scatter_A_to_{sub}.png")
    return summary


def aggregate_motif_folds(fold_summaries: dict, run_dir: Path, title: str) -> dict:
    """
    Mean and standard deviation (np.std) of the per-fold results. All folds share one test set, so the spread shows
    the sensitivity to the fold and is not a confidence interval; the intervals are in the per-fold summaries.
    """
    def stats(values: list) -> Optional[dict]:
        values = [v for v in values if v is not None]
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_fold": values} if values else None

    first = fold_summaries[min(fold_summaries)]["motifs"]
    aggregated = {"folds": sorted(fold_summaries), "motifs": {}}
    for name, entry in first.items():
        aggregated["motifs"][name] = {}
        for label in entry["substitutions"]:
            per_fold = [fold_summaries[k]["motifs"][name]["substitutions"][label] for k in sorted(fold_summaries)]
            if any(m["n_transcripts"] == 0 for m in per_fold):
                continue
            aggregated["motifs"][name][label] = {
                "effect_mean_of_group_means": stats([m["effect"]["mean_of_group_means"] for m in per_fold]),
                "base_effect_median": stats([m["base_effect_control_vs_base"]["median_of_group_means"] for m in per_fold]),
                "pearson_r_base_vs_motif": stats([m["correlation_of_predictions"]["base_vs_motif"]["pearson_r"] for m in per_fold]),
                "pearson_r_base_vs_control": stats([m["correlation_of_predictions"]["base_vs_control"]["pearson_r"] for m in per_fold]),
                "pearson_r_target_base": stats([m["pearson_r_with_measured_target"]["base"] for m in per_fold]),
                "pearson_r_target_motif": stats([m["pearson_r_with_measured_target"]["motif"] for m in per_fold]),
                "pearson_r_target_control": stats([m["pearson_r_with_measured_target"]["control"] for m in per_fold]),
            }

    out = run_dir / "drach_aggregated.json"
    with open(out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[DRACH] Saved aggregated motif report across folds to: {out}")

    motif_names = [n for n, v in aggregated["motifs"].items() if v]
    labels = sorted({lab for v in aggregated["motifs"].values() for lab in v})
    if motif_names:
        series = {}
        for lab in labels:
            effects = [aggregated["motifs"][n].get(lab, {}).get("effect_mean_of_group_means") for n in motif_names]
            series[lab] = (
                [e["mean"] if e else np.nan for e in effects],
                [e["std"] if e else np.nan for e in effects],
                [e["std"] if e else np.nan for e in effects],
            )
        plot_motif_effects(
            motif_names, series,
            ylabel="Effect: prediction(motif mutated) - prediction(control mutated)",
            title=f"{title}: mean ± std over {len(fold_summaries)} folds", output_png=run_dir / "drach_aggregated.png",
        )
    return aggregated
