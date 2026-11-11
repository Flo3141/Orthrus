#!/usr/bin/env python
"""
Measurement noise of the half-life targets, compared with the model error.

Question: how precisely are the targets measured, and is the model error already within that noise?

mrna_bench (per species; reads the outputs of reproduce_saluki_hl_dataset.py)
  The target is the PC1 over many half-life experiments. Per gene the experiments disagree; this spread is the
  measurement variance (technical noise plus lab, method and cell-type effects).
  1. Per-gene spread of the per-sample z-scores: between replicates (same lab, method and cell type), between
     labs (lab means) and between all samples.
  2. Standard error of the consensus per gene (mean over samples, lab-clustered): how precisely each target
     value is known, in units of the target.
  3. Each experiment as a predictor: correlation of one sample with the consensus of all OTHER labs. The model
     is "within the measurement variance" if it reaches the correlation of a typical single experiment.
  4. Noise ceiling: split-half reliability of the consensus over labs (Spearman-Brown corrected). sqrt(R) is the
     highest correlation any predictor can reach with the noisy target.

hipsc_cm (one experiment, transcript level; uses only the target half_life_transformed)
  There are no replicates, so no measurement variance can be computed. Instead:
  1. Isoform variance: how much the half-life differs between transcripts of the same gene (variance
     decomposition, ICC; all biotypes and protein-coding only), and the shift of non-coding isoforms against
     the protein-coding isoforms of the same gene.
  2. Sibling isoforms as a predictor: each transcript predicted by the mean of the other isoforms of its gene.
  3. Saluki assumption: Saluki uses one representative transcript per gene (longest ORF, then longest 5'UTR,
     then longest 3'UTR) with a gene-level half-life (duplicated IDs averaged). Compared here: representative
     transcript against the mean of the other isoforms (all biotypes and protein-coding only).

Model reference
  metrics_<dataset>.json under --results_dir/<dataset>/<track>/<variant>/ (test Pearson r, RMSE). If
  predictions_<dataset>.csv lies next to it (written by train_ridge_regression.py, not synced by sync_results.sh),
  per-gene / per-transcript comparisons are added.

Outputs (default: mrna_bench -> <data_dir>/<species>/noise_analysis/, hipsc_cm -> <dir of --data>/noise_analysis/)
  summary.json, models.tsv, figures fig*.png/.pdf, per-gene / per-sample tables (*.tsv)

Usage (requires numpy, pandas, matplotlib, scipy; locally: WSL venv /home/flo/.venvs/orthrus)
  python analyze_measurement_noise.py mrna_bench --data_dir /path/to/saluki_hl_reproduction [--species human]
  python analyze_measurement_noise.py hipsc_cm --data /path/to/hIPSC_CM_ej_cds_transformed.txt
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colors: reference palette of the dataviz method (light mode). Slots 1-3 validate all-pairs.
C_DATA, C_MODEL, C_CEIL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlesize": 10, "axes.titlecolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def zscore(s):
    return (s - s.mean()) / s.std()


def _finite_pair(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def pearson(a, b):
    a, b = _finite_pair(a, b)
    return float(stats.pearsonr(a, b).statistic) if len(a) > 2 else np.nan


def spearman(a, b):
    a, b = _finite_pair(a, b)
    return float(stats.spearmanr(a, b).statistic) if len(a) > 2 else np.nan


def rmse(a, b):
    a, b = _finite_pair(a, b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------------------------------------------
# Model reference
# ---------------------------------------------------------------------------------------------------------------
def collect_models(results_dir, dataset):
    rows = []
    for f in sorted(Path(results_dir).glob(f"{dataset}/*/*/metrics_{dataset}.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        pred = f.parent / f"predictions_{dataset}.csv"
        rows.append({"model": f"{f.parent.parent.name} ({f.parent.name})", "test_pearson_r": m.get("test_pearson_r"),
                     "test_spearman_rho": m.get("test_spearman_rho"), "test_rmse": m.get("test_rmse"),
                     "test_size": m.get("test_size"), "metrics_file": str(f),
                     "predictions_file": str(pred) if pred.exists() else ""})
    models = pd.DataFrame(rows)
    if len(models):
        models = models.sort_values("test_pearson_r", ascending=False).reset_index(drop=True)
        print(f"[{dataset}] {len(models)} models, {int((models.predictions_file != '').sum())} with predictions; "
              f"best: {models.model[0]} (test r = {models.test_pearson_r[0]:.3f})", flush=True)
    else:
        print(f"[{dataset}] [Warning] no metrics_{dataset}.json under {results_dir}; figures without model lines",
              flush=True)
    return models


def model_lines(ax, models, orient="v", label_best=True):
    """Model test r as reference lines; only the best is labeled (the others are listed in models.tsv)."""
    for i, row in models.iterrows():
        kw = dict(color=C_MODEL, lw=1.6 if i == 0 else 0.8, alpha=1 if i == 0 else 0.55, zorder=4)
        (ax.axvline if orient == "v" else ax.axhline)(row.test_pearson_r, **kw)
    if len(models) and label_best:
        r = models.test_pearson_r[0]
        txt = f"best model\n{models.model[0]}\nr = {r:.2f}"
        if orient == "v":
            ax.text(r, 1.01, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                    color=INK_2)


# ---------------------------------------------------------------------------------------------------------------
# mRNABench (Saluki compendium)
# ---------------------------------------------------------------------------------------------------------------
def load_mrnabench(data_dir, sp):
    d = Path(data_dir) / sp
    meta = pd.read_csv(d / "sample_metadata.tsv", sep="\t").set_index("sample")
    meta = meta.loc[meta.used_for_pc1].copy()
    raw = pd.read_csv(d / "halflife_matrix_filtered.tsv.gz", sep="\t", index_col=0)
    pc1 = pd.read_csv(d / "pc1.tsv", sep="\t", index_col=0)
    genes = pc1.index[pc1.in_output]
    rho = raw.loc[genes, meta.index].rank().corrwith(pc1.pc1.loc[genes].rank())
    anti = rho[rho < 0]
    if len(anti):  # degradation rates must enter with half-life orientation (see saluki_hl_reproduktion.md, 4.1)
        print(f"[{sp}] [Warning] samples anti-correlated with PC1, negated: {anti.round(2).to_dict()}", flush=True)
        raw = raw.copy()
        raw[anti.index] = -raw[anti.index]
    Z = raw.loc[genes, meta.index].apply(zscore)  # per-sample z-scores over the dataset genes
    target = pc1.loc[genes, "target"]
    return meta, Z, target, d


def gene_spread(Z, meta):
    """Per gene: SD over all samples, pooled SD within replicate groups, SD of lab means, lab-clustered SE."""
    M = Z.notna()
    rep_key = meta.lab + "|" + meta.method + "|" + meta.cell_group.astype(str)
    out = pd.DataFrame(index=Z.index)
    out["n_samples"] = M.sum(1)
    out["n_labs"] = M.T.groupby(meta.lab.values).any().T.sum(1)
    out["mean_z"] = Z.mean(1)
    out["sd_samples"] = Z.std(1, ddof=1)

    # replicates: pooled within-group variance over groups with >= 2 measured samples
    ss, dof = pd.Series(0.0, index=Z.index), pd.Series(0, index=Z.index)
    for _, cols in rep_key.groupby(rep_key).groups.items():
        if len(cols) < 2:
            continue
        sub = Z[cols]
        k = sub.notna().sum(1)
        dev = sub.sub(sub.mean(1), axis=0).pow(2).sum(1)
        ok = k >= 2
        ss[ok] += dev[ok]
        dof[ok] += k[ok] - 1
    out["sd_replicates"] = np.sqrt(ss / dof.replace(0, np.nan))
    out["df_replicates"] = dof

    # lab means and lab-clustered standard error of the sample mean (CR1 small-sample factor)
    lab_mean = Z.T.groupby(meta.lab.values).mean().T  # NaN-skipping
    lab_n = M.T.groupby(meta.lab.values).sum().T
    out["sd_labs"] = lab_mean.std(1, ddof=1)
    n = out.n_samples
    K = out.n_labs
    w = lab_n.div(n, axis=0)
    var_se = (w.pow(2) * lab_mean.sub(out.mean_z, axis=0).pow(2)).sum(1) * K / (K - 1)
    out["se_consensus_z"] = np.sqrt(var_se.where(K >= 2))
    return out


def sample_as_predictor(Z, meta, min_measured):
    """Each sample against the consensus (mean z) of all samples from OTHER labs."""
    rows = []
    for s in meta.index:
        others = meta.index[meta.lab != meta.loc[s, "lab"]]
        cons = Z[others].mean(1)
        n_o = Z[others].notna().sum(1)
        m = Z[s].notna() & (n_o >= min_measured)
        x, y = Z.loc[m, s].values, cons[m].values
        rows.append({"sample": s, "lab": meta.loc[s, "lab"], "method": meta.loc[s, "method"],
                     "cell_group": meta.loc[s, "cell_group"], "n_genes": int(m.sum()),
                     "pearson_vs_other_labs": pearson(x, y), "spearman_vs_other_labs": spearman(x, y)})
    return pd.DataFrame(rows).sort_values("pearson_vs_other_labs").reset_index(drop=True)


def split_half(Z, meta, n_splits, seed):
    """Split-half reliability over labs, Spearman-Brown corrected; returns r_half and R per split."""
    rng = np.random.default_rng(seed)
    labs = np.array(sorted(meta.lab.unique()))
    res = []
    for _ in range(n_splits):
        perm = rng.permutation(labs)
        a = meta.index[meta.lab.isin(perm[: len(labs) // 2])]
        b = meta.index[meta.lab.isin(perm[len(labs) // 2:])]
        ma, mb = Z[a].mean(1), Z[b].mean(1)
        r = pearson(ma.values, mb.values)
        res.append({"n_samples_a": len(a), "n_samples_b": len(b), "n_genes": int((ma.notna() & mb.notna()).sum()),
                    "r_half": r, "reliability": 2 * r / (1 + r)})
    return pd.DataFrame(res)


def map_mrnabench_predictions(pred, d, sp):
    """Maps the 'gene' column of predictions_*.csv (symbol or Ensembl ID) to Ensembl gene IDs."""
    g = pred["gene"].astype(str)
    if g.str.startswith("ENS").mean() > 0.9:
        return g.str.split(".").str[0]
    pkl = d / f"rnahl-{sp}_reproduced.pkl"
    if not pkl.exists():
        print(f"[{sp}] [Warning] {pkl.name} missing, cannot map gene symbols of the predictions", flush=True)
        return pd.Series(np.nan, index=pred.index)
    ds = pd.read_pickle(pkl)[["gene", "ensembl_gene_id"]].dropna()
    sym = dict(zip(ds.gene.str.upper(), ds.ensembl_gene_id))
    return g.str.upper().map(sym)


def run_mrnabench(sp, args):
    style()
    dataset = f"mrna_bench_{sp}"
    meta, Z, target, d = load_mrnabench(args.data_dir, sp)
    out = Path(args.out_dir or d / "noise_analysis")
    out.mkdir(parents=True, exist_ok=True)
    models = collect_models(args.results_dir, dataset)
    models.to_csv(out / "models.tsv", sep="\t", index=False)
    print(f"[{sp}] {len(meta)} samples from {meta.lab.nunique()} labs, {len(Z)} genes", flush=True)

    # ---- 1./2. per-gene spread and standard error of the consensus
    spread = gene_spread(Z, meta)
    sd_cons = spread.mean_z.std()  # SD of the consensus over genes: converts z units into target units
    spread["se_consensus_target"] = spread.se_consensus_z / sd_cons
    spread["target"] = target
    r_mean_target = pearson(spread.mean_z.values, target.values)

    def rms(s):
        s = s.dropna()
        return float(np.sqrt(np.mean(s ** 2))) if len(s) else np.nan

    summ = {
        "dataset": dataset, "n_samples": int(len(meta)), "n_labs": int(meta.lab.nunique()), "n_genes": int(len(Z)),
        "pearson_mean_z_vs_target": r_mean_target,
        "rms_sd_replicates": rms(spread.sd_replicates), "n_genes_with_replicates": int(spread.sd_replicates.notna().sum()),
        "rms_sd_labs": rms(spread.sd_labs), "rms_sd_samples": rms(spread.sd_samples),
        "sd_consensus_over_genes": float(sd_cons),
        "rms_se_consensus_target": rms(spread.se_consensus_target),
        "median_se_consensus_target": float(spread.se_consensus_target.median()),
        "reliability_from_se": float(1 - np.nanmean(spread.se_consensus_target ** 2)),
    }

    # ---- 3. single experiments as predictors of the consensus
    smp = sample_as_predictor(Z, meta, args.min_measured)
    smp.to_csv(out / "sample_vs_consensus.tsv", sep="\t", index=False)
    summ["median_pearson_single_sample"] = float(smp.pearson_vs_other_labs.median())
    summ["range_pearson_single_sample"] = [float(smp.pearson_vs_other_labs.min()),
                                           float(smp.pearson_vs_other_labs.max())]

    # ---- 4. noise ceiling
    sh = split_half(Z, meta, args.n_splits, args.seed)
    sh.to_csv(out / "split_half.tsv", sep="\t", index=False)
    R = sh.reliability
    summ["split_half_r_median"] = float(sh.r_half.median())
    summ["reliability_split_half_median"] = float(R.median())
    summ["reliability_split_half_95"] = [float(R.quantile(0.025)), float(R.quantile(0.975))]
    summ["noise_ceiling_r"] = float(np.sqrt(R.median()))
    summ["noise_ceiling_r_95"] = [float(np.sqrt(R.quantile(0.025))), float(np.sqrt(R.quantile(0.975)))]
    summ["noise_ceiling_r_from_se"] = float(np.sqrt(summ["reliability_from_se"]))

    if len(models):
        models["share_of_single_samples_below_model"] = [
            float((smp.pearson_vs_other_labs < r).mean()) for r in models.test_pearson_r]
        models["r_relative_to_ceiling"] = models.test_pearson_r / summ["noise_ceiling_r"]
        models.to_csv(out / "models.tsv", sep="\t", index=False)

    # ---- per-gene comparison with model predictions (if available)
    pred_rows = []
    for _, mrow in models.iterrows():
        if not mrow.predictions_file:
            continue
        p = pd.read_csv(mrow.predictions_file)
        p["ensembl_gene_id"] = map_mrnabench_predictions(p, d, sp)
        p = p.dropna(subset=["ensembl_gene_id"]).drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")
        j = p.join(spread, how="inner")
        absres = (j.true_target - j.predicted_target).abs()
        q = pd.qcut(j.sd_labs, 5, labels=False, duplicates="drop")
        pred_rows.append({
            "model": mrow.model, "n_genes_matched": int(len(j)), "n_predictions": int(len(p)),
            "rmse": rmse(j.true_target.values, j.predicted_target.values),
            "share_abs_residual_within_1.96_se": float((absres <= 1.96 * j.se_consensus_target).mean()),
            "median_abs_residual_over_se": float((absres / j.se_consensus_target).median()),
            "spearman_abs_residual_vs_sd_labs": spearman(absres.values, j.sd_labs.values),
            **{f"rmse_sd_labs_quintile_{int(k) + 1}": rmse(j.true_target[q == k].values,
                                                           j.predicted_target[q == k].values)
               for k in sorted(q.dropna().unique())},
        })
        spread[f"abs_residual__{mrow.model}"] = absres.reindex(spread.index)
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(out / "model_vs_gene_noise.tsv", sep="\t", index=False)
        summ["model_vs_gene_noise"] = pred_rows
    spread.to_csv(out / "gene_noise.tsv", sep="\t", index_label="ensembl_gene_id")

    # ---- figures
    best_rmse = models.test_rmse[0] if len(models) else None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 2.0, 61)
    for col, label, ls in (("sd_replicates", "replicates (same lab, method, cell type)", ":"),
                           ("sd_labs", "lab means", "--"), ("sd_samples", "all samples", "-")):
        v = spread[col].dropna()
        ax.hist(v.clip(upper=bins[-1]), bins=bins, histtype="step", lw=1.6, color=C_DATA, ls=ls,
                label=f"{label}: RMS {rms(v):.2f} (n={len(v):,})")
    v = spread.se_consensus_z.dropna()
    ax.hist(v.clip(upper=bins[-1]), bins=bins, histtype="stepfilled", color=C_CEIL, alpha=0.35, lw=0,
            label=f"SE of the consensus: RMS {rms(v):.2f}")
    if best_rmse is not None:
        # the target is the standardized consensus; one target unit = sd_cons single-sample units
        mz = best_rmse * sd_cons
        ax.axvline(mz, color=C_MODEL, lw=1.6, zorder=4)
        ax.text(mz, 1.01, f"model test RMSE {mz:.2f}\n({best_rmse:.2f} target units x {sd_cons:.2f})\n"
                f"{models.model[0]}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                color=INK_2)
    ax.set_xlabel("Per-gene SD of the z-scored half-life (units of one sample's z-score)")
    ax.set_ylabel("Genes")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title(f"{sp}: spread of half-life measurements per gene", loc="left", pad=30)
    save(fig, out, "fig1_gene_spread")

    fig, ax = plt.subplots(figsize=(6.5, 0.17 * len(smp) + 1.6))
    y = np.arange(len(smp))
    ax.scatter(smp.pearson_vs_other_labs, y, s=22, color=C_DATA, zorder=3, linewidths=0)
    ax.set_yticks(y, smp["sample"], fontsize=6.5)
    ax.grid(axis="y", visible=False)
    lo, hi = summ["noise_ceiling_r_95"]
    ax.axvspan(lo, hi, color=C_CEIL, alpha=0.25, lw=0)
    ax.axvline(summ["noise_ceiling_r"], color=C_CEIL, lw=1.6)
    ax.text(summ["noise_ceiling_r"], 1.01, f"noise ceiling\nr = {summ['noise_ceiling_r']:.2f}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK_2)
    model_lines(ax, models)
    ax.set_xlabel("Pearson r with the consensus of all other labs (single sample)\n"
                  "model: test Pearson r with the target")
    ax.set_xlim(min(0.2, smp.pearson_vs_other_labs.min() - 0.05), 1.0)
    ax.set_title(f"{sp}: single experiments vs. model as predictors of the consensus", loc="left", pad=40)
    save(fig, out, "fig2_samples_vs_model")

    if pred_rows:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        pr = pd.DataFrame(pred_rows).iloc[0]
        cols = [c for c in pr.index if c.startswith("rmse_sd_labs_quintile_")]
        ax.bar(range(len(cols)), [pr[c] for c in cols], color=C_MODEL, width=0.7)
        ax.set_xticks(range(len(cols)), [f"Q{i + 1}" for i in range(len(cols))])
        ax.set_xlabel("Quintile of the between-lab SD of the gene (Q1: labs agree)")
        ax.set_ylabel("Model test RMSE")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{sp}: model error by measurement disagreement\n{pr.model}", loc="left")
        save(fig, out, "fig3_model_error_by_gene_noise")

    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "model_vs_gene_noise"}, indent=2), flush=True)
    if len(models):
        print(models[["model", "test_pearson_r", "test_rmse", "share_of_single_samples_below_model",
                      "r_relative_to_ceiling"]].round(3).to_string(), flush=True)
    print(f"[{sp}] written to {out}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# hIPSC_CM (isoforms)
# ---------------------------------------------------------------------------------------------------------------
def orf_utr_lengths(seq):
    """5'UTR, ORF and 3'UTR length from the ej_cds sequence (uppercase = first nucleotide of a codon)."""
    tok = seq.split(",")
    up = [i for i, t in enumerate(tok) if t[:1].isupper()]
    if not up:
        return np.nan, np.nan, np.nan
    start, last = up[0], up[-1]
    return start, last - start + 3, len(tok) - (last + 3)


def variance_components(df, col="z", group="ensembl_gene_id"):
    """One-way random-effects ANOVA over genes with >= 2 isoforms: within, between, ICC."""
    g = df.groupby(group)[col]
    k = g.transform("size")
    d = df[k >= 2]
    g = d.groupby(group)[col]
    n_i = g.size()
    N, G = len(d), len(n_i)
    ss_w = ((d[col] - g.transform("mean")) ** 2).sum()
    ss_b = (n_i * (g.mean() - d[col].mean()) ** 2).sum()
    ms_w, ms_b = ss_w / (N - G), ss_b / (G - 1)
    n0 = (N - (n_i ** 2).sum() / N) / (G - 1)
    var_b = max((ms_b - ms_w) / n0, 0.0)
    return {"n_transcripts": int(N), "n_genes": int(G), "var_within": float(ms_w), "var_between": float(var_b),
            "sd_within": float(np.sqrt(ms_w)), "icc_gene": float(var_b / (var_b + ms_w)),
            "var_total": float(d[col].var())}


def loo_sibling_mean(df, col="z", group="ensembl_gene_id"):
    """Unweighted mean of the OTHER isoforms of the same gene (NaN if there is none)."""
    ok = df[col].notna()
    s = df[col].where(ok, 0)
    n = ok.astype(int)
    s_other = s.groupby(df[group]).transform("sum") - s
    n_other = n.groupby(df[group]).transform("sum") - n
    return (s_other / n_other).where(n_other > 0)


HIPSC_COLUMNS = ["ensembl_transcript_id", "ensembl_gene_id", "hgnc_symbol", "transcript_biotype", "sequence",
                 "half_life_transformed"]


def run_hipsc(args):
    style()
    dataset = "hipsc_cm"
    path = Path(args.data)
    out = Path(args.out_dir or path.parent / "noise_analysis")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path, sep="\t", usecols=HIPSC_COLUMNS)
    df["z"] = df.half_life_transformed  # = target: (ln(hl + 0.1) - mu) / sigma, see target_zscore_gesamtdatensatz.md
    L = np.array([orf_utr_lengths(s) for s in df.sequence])
    df["utr5_len"], df["orf_len"], df["utr3_len"] = L[:, 0], L[:, 1], L[:, 2]
    df = df.drop(columns="sequence")
    df["n_isoforms"] = df.groupby("ensembl_gene_id").z.transform("size")
    pc = df.transcript_biotype.eq("protein_coding") & df.orf_len.notna()
    models = collect_models(args.results_dir, dataset)
    print(f"[{dataset}] {len(df):,} transcripts of {df.ensembl_gene_id.nunique():,} genes; "
          f"{int((df.n_isoforms >= 2).sum()):,} transcripts in genes with >= 2 isoforms", flush=True)
    summ = {"dataset": dataset, "n_transcripts": int(len(df)), "n_genes": int(df.ensembl_gene_id.nunique()),
            "var_target": float(df.z.var())}

    # ---- 1. isoform variance
    summ["isoforms_all_biotypes"] = variance_components(df)
    summ["isoforms_protein_coding"] = variance_components(df[pc])
    gs = df.groupby("ensembl_gene_id").agg(n_isoforms=("z", "size"), sd_isoforms=("z", "std"),
                                           mean_z=("z", "mean"), gene_name=("hgnc_symbol", "first"))
    gs = gs.join(df[pc].groupby("ensembl_gene_id").z.agg(["size", "std"]).rename(
        columns={"size": "n_isoforms_pc", "std": "sd_isoforms_pc"}))

    # biotype effect: difference of a non-coding isoform to the protein-coding isoforms of the same gene
    df["pc_gene_mean"] = df.ensembl_gene_id.map(df[pc].groupby("ensembl_gene_id").z.mean())
    bt = {}
    for b, sub in df[~pc & df.pc_gene_mean.notna()].groupby("transcript_biotype"):
        if len(sub) >= 30:
            dz = sub.z - sub.pc_gene_mean
            bt[b] = {"n": int(len(sub)), "median_delta_z_vs_protein_coding": float(dz.median()),
                     "mean_delta_z_vs_protein_coding": float(dz.mean()),
                     "wilcoxon_p": float(stats.wilcoxon(dz).pvalue)}
    summ["biotype_vs_protein_coding_same_gene"] = bt

    # ---- 2. sibling isoforms as predictor (leave-one-out gene mean)
    df["sibling_mean"] = loo_sibling_mean(df)
    df["sibling_mean_pc"] = loo_sibling_mean(df.assign(z=df.z.where(pc)))  # mean of the other protein-coding ones
    for key, col, sel in (("sibling_predictor_all", "sibling_mean", df.sibling_mean.notna()),
                          ("sibling_predictor_protein_coding", "sibling_mean_pc", pc & df.sibling_mean_pc.notna())):
        summ[key] = {"n": int(sel.sum()), "pearson": pearson(df.z[sel], df[col][sel]),
                     "spearman": spearman(df.z[sel], df[col][sel]), "rmse": rmse(df.z[sel], df[col][sel])}

    # ---- 3. Saluki representative transcript (longest ORF, then 5'UTR, then 3'UTR), protein-coding only
    rep_idx = (df[pc].sort_values(["ensembl_gene_id", "orf_len", "utr5_len", "utr3_len"],
                                  ascending=[True, False, False, False])
               .groupby("ensembl_gene_id").head(1).index)
    df["saluki_representative"] = df.index.isin(rep_idx)
    rep = df.loc[rep_idx]
    rep = rep[rep.sibling_mean.notna()]
    sal = {"n_genes": int(len(rep))}
    for col, name in (("sibling_mean", "mean_other_isoforms"),
                      ("sibling_mean_pc", "mean_other_protein_coding")):
        mm = rep[col].notna()
        d = rep.z[mm] - rep[col][mm]
        sal[name] = {"n": int(mm.sum()), "pearson": pearson(rep.z[mm], rep[col][mm]),
                     "rmse": rmse(rep.z[mm], rep[col][mm]), "median_difference": float(d.median()),
                     "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
    # rank of the representative among the isoforms of its gene (is the longest transcript the most stable one?)
    rk = df[df.n_isoforms >= 2].groupby("ensembl_gene_id").z.rank(pct=True)
    sal["median_percentile_rank_of_representative"] = float(rk.reindex(rep.index).median())
    summ["saluki_representative"] = sal

    # ---- model reference and per-transcript comparison
    if len(models):
        models["sibling_predictor_r_all"] = summ["sibling_predictor_all"]["pearson"]
        models["sibling_predictor_r_protein_coding"] = summ["sibling_predictor_protein_coding"]["pearson"]
    pred_rows = []
    for _, mrow in models.iterrows():
        if not mrow.predictions_file:
            continue
        p = pd.read_csv(mrow.predictions_file).drop_duplicates("transcript_id").set_index("transcript_id")
        j = df.set_index("ensembl_transcript_id").join(p[["predicted_target"]], how="inner")
        js = j[j.sibling_mean.notna()]
        # within-gene: does the model separate isoforms of the same gene? (test genes with >= 2 test isoforms)
        jw = j[j.groupby("ensembl_gene_id").z.transform("size") >= 2]
        dz = jw.z - jw.groupby("ensembl_gene_id").z.transform("mean")
        dp = jw.predicted_target - jw.groupby("ensembl_gene_id").predicted_target.transform("mean")
        pred_rows.append({
            "model": mrow.model, "n_test_transcripts": int(len(j)),
            "pearson_all": pearson(j.z, j.predicted_target), "rmse_all": rmse(j.z, j.predicted_target),
            "n_multi_isoform": int(len(js)),
            "pearson_multi_isoform": pearson(js.z, js.predicted_target),
            "pearson_sibling_mean_same_transcripts": pearson(js.z, js.sibling_mean),
            "n_within_gene": int(len(jw)), "pearson_within_gene_deviation": pearson(dz, dp),
        })
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(out / "model_vs_isoforms.tsv", sep="\t", index=False)
        summ["model_vs_isoforms"] = pred_rows
    models.to_csv(out / "models.tsv", sep="\t", index=False)
    gs.to_csv(out / "gene_isoform_noise.tsv", sep="\t")
    df.to_csv(out / "transcript_isoforms.tsv", sep="\t", index=False)

    # ---- figures
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 2.5, 51)
    for col, label, ls in (("sd_isoforms", "all biotypes", "-"), ("sd_isoforms_pc", "protein-coding only", "--")):
        v = gs[col].dropna()
        ax.hist(v.clip(upper=bins[-1]), bins=bins, histtype="step", lw=1.6, color=C_DATA, ls=ls, density=True,
                label=f"{label}: RMS {np.sqrt(np.mean(v ** 2)):.2f} (n={len(v):,} genes)")
    if len(models):
        r0 = models.test_rmse[0]
        ax.axvline(r0, color=C_MODEL, lw=1.6, zorder=4)
        ax.text(r0, 1.01, f"model test RMSE {r0:.2f}\n{models.model[0]}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xlabel("Per-gene SD of the half-life between isoforms (target units, z-scored log half-life)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("hIPSC-CM: half-life spread between isoforms of the same gene", loc="left", pad=30)
    save(fig, out, "fig1_isoform_spread")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes, (("sibling_mean", "mean of the other isoforms (all biotypes)"),
                                        ("sibling_mean_pc", "mean of the other protein-coding isoforms"))):
        mm = rep[col].notna()
        ax.scatter(rep[col][mm], rep.z[mm], s=4, color=C_DATA, alpha=0.35, linewidths=0, rasterized=True)
        ax.plot([-4, 4], [-4, 4], color=INK_2, lw=1)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_title(title, loc="left", fontsize=9)
        ax.text(0.03, 0.97, f"r = {pearson(rep.z[mm], rep[col][mm]):.2f}\n{int(mm.sum()):,} genes",
                transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
        ax.set_xlabel("Gene-level half-life without the representative (z)")
    axes[0].set_ylabel("Saluki representative transcript (z)")
    fig.suptitle("hIPSC-CM: does the representative transcript carry the gene's half-life?\n"
                 "(longest ORF, then 5'UTR, then 3'UTR; line: identity)", x=0.01, ha="left", y=1.04, fontsize=11,
                 color=INK)
    save(fig, out, "fig2_saluki_representative")

    bars = [("sibling isoforms, all biotypes", summ["sibling_predictor_all"]["pearson"], C_DATA),
            ("sibling isoforms, protein-coding", summ["sibling_predictor_protein_coding"]["pearson"], C_DATA)]
    bars += [(f"model: {r.model}", r.test_pearson_r, C_MODEL) for _, r in models.iterrows()]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(bars) + 1.2))
    for i, (lab, val, c) in enumerate(bars):
        ax.barh(i, val, color=c, height=0.62)
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pearson r with the measured half-life (z)")
    ax.set_title("hIPSC-CM: model against the mean of the sibling isoforms\n"
                 "(siblings: transcripts of multi-isoform genes; model: test split)", loc="left")
    save(fig, out, "fig3_r_comparison")

    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "model_vs_isoforms"}, indent=2), flush=True)
    if pred_rows:
        print(pd.DataFrame(pred_rows).round(3).to_string(), flush=True)
    print(f"[{dataset}] written to {out}", flush=True)


def main(argv=None):
    base = Path(os.environ.get("ORTHRUS_BASE", "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="dataset", required=True)
    a = sub.add_parser("mrna_bench", help="Saluki compendium: variance between experiments")
    a.add_argument("--data_dir", default=str(base / "data" / "saluki_hl_reproduction"),
                   help="--out_dir of reproduce_saluki_hl_dataset.py")
    a.add_argument("--species", nargs="+", default=["human", "mouse"], choices=["human", "mouse"])
    a.add_argument("--min_measured", type=int, default=5,
                   help="single-sample comparison: minimum number of samples from other labs per gene")
    a.add_argument("--n_splits", type=int, default=200, help="random lab splits for the split-half reliability")
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("hipsc_cm", help="hIPSC-CM: variance between isoforms of the same gene")
    b.add_argument("--data", default=str(base / "data" / "hIPSC_CM" / "hIPSC_CM_ej_cds_transformed.txt"))
    for p in (a, b):
        p.add_argument("--results_dir", default=str(base / "results" / "Orthrus"),
                       help="contains <dataset>/<track>/<variant>/metrics_<dataset>.json")
        p.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)
    if args.dataset == "mrna_bench":
        if args.out_dir and len(args.species) > 1:
            ap.error("--out_dir only with a single --species")
        for sp in args.species:
            run_mrnabench(sp, args)
    else:
        run_hipsc(args)


if __name__ == "__main__":
    main()
