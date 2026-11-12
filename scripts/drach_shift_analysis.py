#!/usr/bin/env python3
"""
DRACH shift analysis of fine-tuned N-track Orthrus models: the central A of every DRACH site in the 3' UTR is moved
by one position, without changing the base composition of the transcript.

Three inputs per test transcript with an annotated 3' UTR:

  WT     unchanged input (pred_base)
  left   the central A swaps places with R:  D R A C H  ->  D A R C H
  right  the central A swaps places with C:  D R A C H  ->  D R C A H

Both shifts destroy the DRACH site. Left: for R = A (DAACH) the swap changes nothing, so only DGACH sites are
shifted (DGACH -> DAGCH); DAACH sites stay as they are and are counted. Right: every site can be shifted (DRCAH).

Any swap of two bases moves the prediction, even far away from a motif. Each direction therefore gets a control
with as many swaps of the same dinucleotide (GA -> AG for left, AC -> CA for right) at positions in the 3' UTR outside
all DRACH sites, as close as possible to one shifted site each. Effect = pred_shifted - pred_control; an effect > 0
means that moving the A away raises the predicted half-life (the model treats the intact motif as destabilizing).

A shift can form a new DRACH site with the neighbouring bases. The most frequent case is the right shift of DAACH:
U GAACU -> U GACAU contains the new site UGACA one position upstream, so the motif would only move. Such sites are
left unchanged and counted, so that every shifted site is a lost site. Control swaps that would create a site are
not used as controls.

All predictions run in float32 (see drach_analysis.py). Only the four base channels are swapped; all other channels
(CDS, splice, m6A, RBP) keep their values, so in the 7-track the m6A mark stays at the original position of the A.

Not meant to be run directly: xai_analysis.py calls run_shift_analysis() and aggregate_shift_folds().
Usage: python xai_analysis.py --data_path <tracks.npz> --analyses drach_shift
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from drach_analysis import (
    BOOTSTRAP_SEED, check_batch_invariance, plot_motif_effects, plot_prediction_scatter,
    predict_tracks, prepare_utr3_inputs, substitution_agreement, summarize_motif,
)
# Sequence logic lives in drach_motifs.py, shared with create_saluki_drach_shift_files.py
from drach_motifs import DRACH, SHIFT_NAMES, SHIFTS, Motif, Shift, match_swap_controls, plan_shifts, swap_creates_site, swap_pairs  # noqa: F401
from finetune_ntrack import TrackData


FILE_PREFIX = "drach_shift"
# All outputs go into this subdirectory of fold_k/ and of the run directory, so they do not mix with the other analyses
OUTPUT_SUBDIR = "drach_shift"
EFFECT_YLABEL = "Effect: prediction(A shifted) - prediction(control swapped)"


# =============================================================================
# 1. Swaps on tracks and the perturbation of one transcript
# =============================================================================

def swap_track(track: np.ndarray, positions: list) -> np.ndarray:
    """Copy of the track with the base channels of p and p + 1 swapped for every p; all other channels stay."""
    positions = np.asarray(positions, dtype=np.int64)
    if len(np.unique(np.concatenate([positions, positions + 1]))) != 2 * len(positions):
        raise ValueError("[Error] Swapped dinucleotides overlap!")
    swapped = track.copy()
    swapped[positions, :4] = track[positions + 1, :4]
    swapped[positions + 1, :4] = track[positions, :4]
    return swapped


@dataclass
class ShiftPerturbation:
    # Field names as in drach_analysis.Perturbation, so that summarize_motif() can read them
    motif_positions: list         # first position of every swapped site dinucleotide
    control_positions: list       # first position of every swapped control dinucleotide
    n_sites: int                  # shifted sites (= swaps in each variant)
    n_skipped: int                # sites left unchanged (n_unshiftable + n_would_create)
    n_unshiftable: int            # sites whose swap changes nothing (left: DAACH)
    n_would_create: int           # sites whose swap would create a new site
    original_sites_left: int      # original sites that still exist after the shift (expected: n_skipped)
    created_sites: int            # sites that exist only after the shift (expected: 0)
    control_sites_changed: bool   # the control swaps changed the number of sites (expected False)


def build_shift(seq: str, utr3_start: int, shift: Shift, motif: Motif = DRACH):
    """
    Returns (ShiftPerturbation, None) or (None, reason) with reason 'no_site', 'no_shiftable_site', 'no_safe_site' or
    'not_enough_controls'. Which sites are shifted is decided by drach_motifs.plan_shifts().
    """
    plan = plan_shifts(seq, utr3_start, shift, motif)
    sites, swaps, n_unshiftable, n_would_create = plan.sites, plan.swaps, plan.n_unshiftable, plan.n_would_create
    if not sites:
        return None, "no_site"
    if not swaps:
        return None, "no_safe_site" if n_would_create else "no_shiftable_site"

    controls = match_swap_controls(seq, utr3_start, motif, sites, shift.pair, swaps)
    if controls is None:
        return None, "not_enough_controls"

    after = set(motif.sites(swap_pairs(seq, swaps, shift.pair), utr3_start))
    before = set(sites)
    control_changed = len(motif.sites(swap_pairs(seq, controls, shift.pair), utr3_start)) != len(sites)
    return ShiftPerturbation(
        swaps, controls, len(swaps), n_unshiftable + n_would_create, n_unshiftable, n_would_create,
        len(after & before), len(after - before), control_changed,
    ), None


# =============================================================================
# 2. Analysis of one fold and aggregation across folds
# =============================================================================

def run_shift_analysis(
    model: nn.Module,
    data: TrackData,
    test_idx: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int = 12288,
    shifts: tuple = SHIFTS,
) -> dict:
    """
    Runs the DRACH shift for one fold on the given test samples and writes drach_shift_summary.json,
    drach_shift_predictions.csv and the plots into fold_dir/drach_shift/. Returns the summary.
    """
    model.eval()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out_dir = fold_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = prepare_utr3_inputs(data, test_idx, max_length)
    n_analyzed = len(inputs.tracks)
    print(f"[Shift] {n_analyzed} of {inputs.n_test} test transcripts have an annotated 3' UTR, {len(np.unique(inputs.groups))} homology groups. "
          f"Precision: float32, shifts: {[s.name for s in shifts]}")

    base = predict_tracks(model, inputs.tracks, inputs.tx_ids, device, batch_size, max_length)
    invariance_diff = check_batch_invariance(model, inputs.tracks, inputs.tx_ids, base, device, batch_size, max_length)

    directions, tables, scatter_data, rows = {}, {}, {}, []
    for shift in shifts:
        exclusions = {"no_site": 0, "no_shiftable_site": 0, "no_safe_site": 0, "not_enough_controls": 0}
        included, perturbations = [], []
        for k in tqdm(range(n_analyzed), desc=f"[Shift] Planning {shift.name}"):
            perturbation, reason = build_shift(inputs.seqs[k], inputs.utr3[k], shift)
            if perturbation is None:
                exclusions[reason] += 1
            else:
                included.append(k)
                perturbations.append(perturbation)
        exclusions["included"] = len(included)
        print(f"[Shift] {shift.name}: {len(included)} transcripts analyzed, excluded: {exclusions['no_site']} without site, "
              f"{exclusions['no_shiftable_site']} without a shiftable site, {exclusions['no_safe_site']} where every shift creates a new site, "
              f"{exclusions['not_enough_controls']} without enough controls")
        if not included:
            directions[shift.name] = {"label": shift.label, "exclusions": exclusions, "n_transcripts": 0}
            continue

        sub_ids = inputs.tx_ids[included]
        pred_shifted = predict_tracks(model, [swap_track(inputs.tracks[k], p.motif_positions) for k, p in zip(included, perturbations)], sub_ids, device, batch_size, max_length)
        pred_control = predict_tracks(model, [swap_track(inputs.tracks[k], p.control_positions) for k, p in zip(included, perturbations)], sub_ids, device, batch_size, max_length)

        # summarize_motif() expects the shifted variant as pred_motif
        table = pd.DataFrame({
            "tx_id": sub_ids, "gene": inputs.genes[included], "group": inputs.groups[included], "target": inputs.targets[included],
            "n_sites": [p.n_sites for p in perturbations],
            "pred_base": base[included], "pred_motif": pred_shifted, "pred_control": pred_control,
        })
        summary = summarize_motif(table, exclusions, perturbations, rng)
        summary["mutation_checks"].update({
            "sites_unshiftable_total": int(sum(p.n_unshiftable for p in perturbations)),
            "sites_skipped_new_site_total": int(sum(p.n_would_create for p in perturbations)),
        })
        directions[shift.name] = {"label": shift.label, **summary}
        tables[shift.name] = table.assign(effect=table.pred_motif - table.pred_control)
        scatter_data[shift.label] = (table["pred_base"].to_numpy(), pred_shifted, pred_control)
        rows.append(table.assign(shift=shift.name))

        eff = summary["effect"]
        corr = summary["correlation_of_predictions"]
        print(f"[Shift] {shift.name}: effect {eff['mean_of_group_means']:+.5f} (95 % CI {eff['ci95_bootstrap'][0]:+.5f} to {eff['ci95_bootstrap'][1]:+.5f}), "
              f"r(WT, shifted) = {corr['base_vs_motif']['pearson_r']}, r(WT, control) = {corr['base_vs_control']['pearson_r']}")

    summary = {
        "motif": DRACH.name,
        "precision": "float32",
        "batch_size": batch_size,
        "max_length": max_length,
        "n_test_transcripts": inputs.n_test,
        "n_without_utr3_annotation": inputs.n_unannotated,
        "n_analyzed_transcripts": n_analyzed,
        "positions_without_base": inputs.n_without_base,
        "batch_invariance_max_abs_diff": invariance_diff,
        "variant_names": {"pred_base": "WT", "pred_motif": "A shifted", "pred_control": "control dinucleotides swapped"},
        "directions": directions,
        "agreement_between_directions": substitution_agreement(tables),
    }
    with open(out_dir / f"{FILE_PREFIX}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        pd.concat(rows).drop(columns="group").rename(columns={"pred_base": "pred_wt", "pred_motif": "pred_shifted"}) \
            .to_csv(out_dir / f"{FILE_PREFIX}_predictions.csv", index=False)
    print(f"[Shift] Saved shift summary to: {out_dir / f'{FILE_PREFIX}_summary.json'}")

    analyzed = [s for s in shifts if directions[s.name]["n_transcripts"] > 0]
    if analyzed:
        means, lower, upper = [], [], []
        for s in analyzed:
            m, (lo, hi) = directions[s.name]["effect"]["mean_of_group_means"], directions[s.name]["effect"]["ci95_bootstrap"]
            means.append(m); lower.append(m - lo); upper.append(hi - m)
        plot_motif_effects(
            [s.label.replace(": ", "\n") for s in analyzed], {DRACH.name: (means, lower, upper)}, ylabel=EFFECT_YLABEL,
            title=f"{fold_dir.name}: group mean, 95 % bootstrap CI",
            output_png=out_dir / f"{FILE_PREFIX}_effects.png",
        )
        plot_prediction_scatter(
            scatter_data, {s.label: directions[s.name] for s in analyzed},
            f"{fold_dir.name}: WT against shifted prediction", out_dir / f"{FILE_PREFIX}_scatter.png",
            variant_labels=("A shifted", "control swapped"),
        )
    return summary


def aggregate_shift_folds(fold_summaries: dict, run_dir: Path, title: str) -> dict:
    """
    Mean and standard deviation (np.std) of the per-fold results. All folds share one test set, so the spread shows
    the sensitivity to the fold and is not a confidence interval; the intervals are in the per-fold summaries.
    Writes drach_shift_aggregated.{json,png,pdf} into run_dir/drach_shift/.
    """
    def stats(values: list) -> Optional[dict]:
        values = [v for v in values if v is not None]
        return {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_fold": values} if values else None

    out_dir = run_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = sorted(fold_summaries)
    aggregated = {"folds": folds, "motif": DRACH.name, "directions": {}}
    for name, entry in fold_summaries[folds[0]]["directions"].items():
        per_fold = [fold_summaries[k]["directions"][name] for k in folds]
        if any(d["n_transcripts"] == 0 for d in per_fold):
            continue
        aggregated["directions"][name] = {
            "label": entry["label"],
            "effect_mean_of_group_means": stats([d["effect"]["mean_of_group_means"] for d in per_fold]),
            "shift_vs_wt_median": stats([d["motif_shift_vs_base"]["median_of_group_means"] for d in per_fold]),
            "control_vs_wt_median": stats([d["base_effect_control_vs_base"]["median_of_group_means"] for d in per_fold]),
            "pearson_r_wt_vs_shifted": stats([d["correlation_of_predictions"]["base_vs_motif"]["pearson_r"] for d in per_fold]),
            "pearson_r_wt_vs_control": stats([d["correlation_of_predictions"]["base_vs_control"]["pearson_r"] for d in per_fold]),
            "pearson_r_target_wt": stats([d["pearson_r_with_measured_target"]["base"] for d in per_fold]),
            "pearson_r_target_shifted": stats([d["pearson_r_with_measured_target"]["motif"] for d in per_fold]),
            "pearson_r_target_control": stats([d["pearson_r_with_measured_target"]["control"] for d in per_fold]),
        }

    out = out_dir / f"{FILE_PREFIX}_aggregated.json"
    with open(out, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"[Shift] Saved aggregated shift report across folds to: {out}")

    entries = list(aggregated["directions"].values())
    if entries:
        effects = [e["effect_mean_of_group_means"] for e in entries]
        plot_motif_effects(
            [e["label"].replace(": ", "\n") for e in entries],
            {DRACH.name: ([e["mean"] for e in effects], [e["std"] for e in effects], [e["std"] for e in effects])},
            ylabel=EFFECT_YLABEL, title=f"{title}: mean ± std over {len(folds)} folds",
            output_png=out_dir / f"{FILE_PREFIX}_aggregated.png",
        )
    return aggregated
