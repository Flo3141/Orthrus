#!/usr/bin/env python
"""
Cell-type view of the Saluki half-life compendium: is mRNA half-life context dependent?

Reads the outputs of reproduce_saluki_hl_dataset.py (per species):
  halflife_matrix_filtered.tsv.gz    measured log half-lives (NaN = not measured)
  sample_metadata.tsv                lab, method, cell type, cell group, used_for_pc1
  pc1.tsv                            global PC1 per gene (only used for the orientation check and the gene set)
  ../raw/all_HLs_<species>.txt.gz    gene names

Cell type = column cell_group of sample_metadata.tsv (cell line, with the LCL lines pooled).

Analyses
  1. Cell-type mean vs. rest mean: per gene the mean z-scored half-life of the cell type's samples against the
     mean z-scored half-life of all OTHER samples. Both axes use measured values only (no imputation, which would
     carry information from the cell type into the rest and vice versa). Genes measured in fewer than
     --min_measured of the other samples are left out.
  2. Mean Spearman correlation between the samples of two cell types (diagonal: different samples of the same
     cell type).
  3. Sample pairs by lab, method and cell type: the cleanest test, since lab and method are confounded with cell
     type. Within one lab and method, context dependence shows up as "other cell type" falling below "all equal"
     (replicates); "same lab, other method" measures the method effect at constant cell type.

Outputs in --out_dir (default <data_dir>/<species>/celltype_analysis/):
  fig1_celltype_vs_rest.png/.pdf, fig2_celltype_correlation.png/.pdf, fig3_sample_pairs.png/.pdf,
  celltype_means.tsv, celltype_stats.tsv, celltype_correlation.tsv, sample_pairs.tsv, top_residual_genes.tsv

Usage
  python analyze_halflife_by_celltype.py --data_dir /path/to/saluki_hl_reproduction [--species human mouse]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Colors (reference palette of the dataviz method, light mode; validated: slot 1 passes all checks)
HIGHLIGHT = "#2a78d6"     # categorical slot 1: genes / sample pairs
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PAIR_CATEGORIES = [  # within a lab, method and cell type can be separated; across labs both vary together
    ("same_lab_same_method_same_cell_type", "all equal"),
    ("same_lab_same_method_other_cell_type", "other\ncell type"),
    ("same_lab_other_method_same_cell_type", "other\nmethod"),
    ("same_lab_other_method_other_cell_type", "other method\nand cell type"),
    ("other_lab_same_cell_type", "same\ncell type"),
    ("other_lab_other_cell_type", "other\ncell type"),
]


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


def panel_grid(n, ncols):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.5 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    axes = list(axes.flat)  # a list, not the flat iterator: the label loops iterate the axes again
    for ax in axes[n:]:
        ax.set_visible(False)
    bottom = [axes[i] for i in range(n) if i + ncols >= n]  # lowest visible panel of each column
    for ax in bottom:
        ax.xaxis.set_tick_params(labelbottom=True)  # sharex hides them above an empty panel
    left = axes[:n:ncols]
    return fig, axes[:n], bottom, left


def load(data_dir, sp, args):
    d = Path(data_dir) / sp
    meta = pd.read_csv(d / "sample_metadata.tsv", sep="\t").set_index("sample")
    raw = pd.read_csv(d / "halflife_matrix_filtered.tsv.gz", sep="\t", index_col=0)
    pc1 = pd.read_csv(d / "pc1.tsv", sep="\t", index_col=0)
    names = pd.read_csv(Path(data_dir) / "raw" / f"all_HLs_{sp}.txt.gz", sep="\t",
                        usecols=[0, 1], index_col=0).external_gene_name
    use = meta.index[meta.used_for_pc1] if not args.include_outlier else meta.index
    meta = meta.loc[use].copy()
    meta["ct"] = meta["cell_group"].astype(str)
    genes = pc1.index[pc1.in_output] if args.genes == "output" else pc1.index
    return meta, raw, pc1, names, genes


def analyze(sp, args):
    style()
    meta, raw, pc1, names, genes = load(args.data_dir, sp, args)
    out = Path(args.out_dir or Path(args.data_dir) / sp / "celltype_analysis")
    out.mkdir(parents=True, exist_ok=True)
    cts = meta.ct.value_counts().index.tolist()  # largest cell type first, order fixed for all figures
    ncols = min(5, len(cts))
    print(f"[{sp}] {len(meta)} samples, {len(cts)} cell types, {len(genes)} genes", flush=True)

    # ---- orientation check: degradation-rate datasets must enter with half-life orientation
    rho = raw[meta.index].rank().corrwith(pc1.pc1.reindex(raw.index).rank())
    anti = rho[rho < 0]
    if len(anti):
        print(f"[{sp}] [Warning] samples anti-correlated with PC1 (degradation rates not negated?): "
              f"{anti.round(2).to_dict()}" + ("" if args.no_flip else " -> negated"), flush=True)
        if not args.no_flip:
            raw = raw.copy()
            raw[anti.index] = -raw[anti.index]

    # ---- per-sample z-scores of the measured values (Saluki standardizes each sample the same way)
    Z = raw[meta.index].apply(zscore)

    # ---- 1. cell-type mean vs rest mean, both on measured values (no imputation on either axis)
    k = args.min_measured
    measured = Z.loc[genes, meta.index].notna()
    means, ncnt, rows, resid = {}, {}, [], {}
    for c in cts:
        smp = meta.index[meta.ct == c]
        rest = meta.index[meta.ct != c]
        cm = Z.loc[genes, smp].mean(1)  # NaN-skipping: each gene over the cell type's samples that measured it
        rm = Z.loc[genes, rest].mean(1)
        n_ct = measured[smp].sum(1)
        n_rest = measured[rest].sum(1)
        n_rest_labs = measured[rest].T.groupby(meta.loc[rest, "lab"].values).any().T.sum(1)
        in_ct = cm.notna()
        m = in_ct & (n_rest >= k)
        cz, rz = zscore(cm[m]), zscore(rm[m])
        if m.sum() >= 3:
            b, a = np.polyfit(rz, cz, 1)
            resid[c] = cz - (a + b * rz)
        else:
            print(f"[{sp}] [Warning] {c}: {int(m.sum())} genes measured in >= {k} other samples, "
                  f"no comparison (lower --min_measured?)", flush=True)
            cz, rz = cz.iloc[:0], rz.iloc[:0]
            resid[c] = cz
        means[c] = cz
        means[f"{c}__rest"] = rz
        ncnt[c], ncnt[f"{c}__rest"], ncnt[f"{c}__rest_labs"] = n_ct, n_rest, n_rest_labs
        rows.append({"cell_type": c, "n_samples": len(smp),
                     "labs": ",".join(sorted(meta.loc[smp, "lab"].unique())),
                     "methods": ",".join(sorted(meta.loc[smp, "method"].unique())),
                     "n_rest_samples": len(rest), "n_rest_labs": int(meta.loc[rest, "lab"].nunique()),
                     "n_genes": int(m.sum()), "n_genes_excluded_min_measured": int((in_ct & ~m).sum()),
                     "pearson_vs_rest": float(cz.corr(rz)) if len(cz) else np.nan,
                     "spearman_vs_rest": float(cz.rank().corr(rz.rank())) if len(cz) else np.nan})
    stats = pd.DataFrame(rows).set_index("cell_type")
    ct_means = pd.DataFrame(means).join(pd.DataFrame(ncnt).add_prefix("n_"))
    ct_means.insert(0, "gene_name", names.reindex(ct_means.index))
    ct_means.to_csv(out / "celltype_means.tsv", sep="\t", index_label="ensembl_gene_id")

    fig, axes, bottom, left = panel_grid(len(cts), ncols)
    for ax, c in zip(axes, cts):
        x, y = means[f"{c}__rest"], means[c]
        ax.scatter(x, y, s=2, color=HIGHLIGHT, alpha=0.25, linewidths=0, rasterized=True)
        lim = [-4, 4]
        ax.plot(lim, lim, color=INK_2, lw=1, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        st = stats.loc[c]
        ax.set_title(f"{c} (n={st.n_samples})", loc="left")
        ax.text(0.03, 0.97, f"r = {st.pearson_vs_rest:.2f}\n{st.n_genes:,} genes",
                transform=ax.transAxes, va="top", color=INK_2, fontsize=8)
    for ax in bottom:
        ax.set_xlabel("Mean half-life, other samples (z)")
    for ax in left:
        ax.set_ylabel("Mean half-life, this cell type (z)")
    fig.suptitle(f"{sp}: half-life per gene, one cell type against all other samples\n"
                 f"(measured values only, genes measured in >= {k} other samples; line: identity)", y=1.08,
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, out, "fig1_celltype_vs_rest")

    # residual genes: largest deviations from the regression line per cell type (candidates for context dependence)
    top = []
    for c in cts:
        r = resid[c].sort_values()
        for side, sel in (("shorter", r.head(args.top_n)), ("longer", r.tail(args.top_n)[::-1])):
            top.append(pd.DataFrame({"cell_type": c, "direction": side, "residual_z": sel,
                                     "gene_name": names.reindex(sel.index),
                                     "n_samples_measured": ncnt[c].reindex(sel.index),
                                     "n_rest_samples_measured": ncnt[f"{c}__rest"].reindex(sel.index),
                                     "n_rest_labs_measured": ncnt[f"{c}__rest_labs"].reindex(sel.index)}))
    pd.concat(top).to_csv(out / "top_residual_genes.tsv", sep="\t", index_label="ensembl_gene_id")

    # ---- 2./3. sample-pair correlations on measured values
    C = raw.loc[genes, meta.index].corr(method="spearman", min_periods=args.min_common_genes)
    pairs = []
    idx = list(meta.index)
    for i, a in enumerate(idx):
        for b in idx[i + 1:]:
            if pd.isna(C.loc[a, b]):
                continue
            same_lab = meta.loc[a, "lab"] == meta.loc[b, "lab"]
            same_ct = meta.loc[a, "ct"] == meta.loc[b, "ct"]
            same_meth = meta.loc[a, "method"] == meta.loc[b, "method"]
            cat = ("same_lab_" + ("same" if same_meth else "other") + "_method_") if same_lab else "other_lab_"
            cat += ("same" if same_ct else "other") + "_cell_type"
            pairs.append({"sample_a": a, "sample_b": b, "spearman": C.loc[a, b],
                          "n_common_genes": int(raw.loc[genes, [a, b]].notna().all(1).sum()),
                          "same_lab": same_lab, "same_method": same_meth, "same_cell_type": same_ct,
                          "cell_type_a": meta.loc[a, "ct"], "cell_type_b": meta.loc[b, "ct"],
                          "category": cat})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "sample_pairs.tsv", sep="\t", index=False)

    G = pd.DataFrame(np.nan, index=cts, columns=cts)
    for c1 in cts:
        for c2 in cts:
            sel = pairs[((pairs.cell_type_a == c1) & (pairs.cell_type_b == c2))
                        | ((pairs.cell_type_a == c2) & (pairs.cell_type_b == c1))]
            if len(sel):
                G.loc[c1, c2] = sel.spearman.mean()
    G.to_csv(out / "celltype_correlation.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(0.55 * len(cts) + 2, 0.55 * len(cts) + 1.2))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmin = np.floor(np.nanmin(G.values) * 10) / 10
    im = ax.imshow(G.values, cmap=cmap, vmin=vmin, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(cts)), cts, rotation=45, ha="right")
    ax.set_yticks(range(len(cts)), cts)
    for i in range(len(cts)):
        for j in range(len(cts)):
            v = G.values[i, j]
            if np.isfinite(v):
                dark = (v - vmin) / (1 - vmin) > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if dark else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("mean Spearman correlation", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(f"{sp}: correlation between samples per cell-type pair\n(diagonal: different samples of the "
                 f"same cell type; empty: single sample)", loc="left")
    save(fig, out, "fig2_celltype_correlation")

    fig, ax = plt.subplots(figsize=(9, 4))
    rng = np.random.default_rng(0)
    labels = []
    for pos, (cat, label) in enumerate(PAIR_CATEGORIES):
        v = pairs.loc[pairs.category == cat, "spearman"].values
        labels.append(f"{label}\n(n={len(v)})")
        if len(v) == 0:
            continue
        ax.scatter(pos + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=HIGHLIGHT, alpha=0.5, linewidths=0)
        ax.plot([pos - 0.28, pos + 0.28], [np.median(v)] * 2, color=INK, lw=2)
        ax.text(pos + 0.3, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(PAIR_CATEGORIES)), labels)
    ax.set_ylabel("Spearman correlation of sample pairs")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{sp}: sample pairs by lab, method and cell type (bar: median)", loc="left", pad=24)
    ax.axvline(3.5, color=AXIS, lw=0.8)
    for x, text in ((1.5, "same lab (clean comparison at equal method)"), (4.5, "other lab")):
        ax.text(x, 1.02, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color=INK_2)
    save(fig, out, "fig3_sample_pairs")

    summary = pairs.groupby("category").spearman.agg(["count", "median", "mean"])
    stats.to_csv(out / "celltype_stats.tsv", sep="\t")
    print(stats[["n_samples", "n_rest_samples", "n_genes", "n_genes_excluded_min_measured",
                 "pearson_vs_rest"]].round(3).to_string(), flush=True)
    print(summary.round(3).to_string(), flush=True)
    print(f"[{sp}] written to {out}", flush=True)


def main(argv=None):
    base = Path(os.environ.get("ORTHRUS_BASE", "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(base / "data" / "saluki_hl_reproduction"),
                    help="--out_dir of reproduce_saluki_hl_dataset.py")
    ap.add_argument("--species", nargs="+", default=["human", "mouse"], choices=["human", "mouse"])
    ap.add_argument("--genes", default="output", choices=["output", "all"],
                    help="output: genes of the reproduced dataset; all: every gene passing the sample filter")
    ap.add_argument("--include_outlier", action="store_true",
                    help="include the samples Saluki excluded from PC1 (human: Gejman, 15 LCL samples)")
    ap.add_argument("--no_flip", action="store_true",
                    help="do not negate samples that are anti-correlated with PC1")
    ap.add_argument("--min_measured", type=int, default=5,
                    help="Fig. 1: minimum number of OTHER samples in which a gene must be measured")
    ap.add_argument("--min_common_genes", type=int, default=500, help="minimum overlap for a sample-pair correlation")
    ap.add_argument("--top_n", type=int, default=50,
                    help="genes per cell type and direction in top_residual_genes.tsv")
    ap.add_argument("--out_dir", default=None, help="default: <data_dir>/<species>/celltype_analysis")
    args = ap.parse_args(argv)
    if args.out_dir and len(args.species) > 1:
        ap.error("--out_dir only with a single --species")
    for sp in args.species:
        analyze(sp, args)


if __name__ == "__main__":
    main()
