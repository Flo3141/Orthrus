#!/usr/bin/env python
"""
Cell-type view of the Saluki half-life compendium: is mRNA half-life context dependent?

Reads the outputs of reproduce_saluki_hl_dataset.py (per species):
  halflife_matrix_filtered.tsv.gz    measured log half-lives (NaN = not measured)
  halflife_matrix_imputed_qn.tsv.gz  z-scored, imputed, quantile-normalized matrix (input of the Saluki PC1)
  sample_metadata.tsv                lab, method, cell type, cell group, used_for_pc1
  pc1.tsv                            global PC1 / target per gene
  ../raw/all_HLs_<species>.txt.gz    gene names

Analyses (grouping by --group_by, default cell_group = cell type with the LCL lines pooled)
  1. Sample PCA (Saluki Fig. 2a/d): PC1/PC2 of the samples, one panel per group, marker = method.
  2. Group consensus vs. leave-group-out PC1: per gene the mean z-scored half-life of the group's samples
     (measured values only) against the PC1 computed WITHOUT that group's samples. Using the global PC1 instead
     would be circular, because each group's own samples are part of it.
  3. Mean Spearman correlation between the samples of two groups (diagonal: different samples of the same group).
  4. Sample pairs by lab, method and group: the cleanest test, since lab and method are confounded with cell
     type. Within one lab and method, context dependence shows up as "other group" falling below "same group"
     (replicates); "same lab, other method" measures the method effect at constant cell type.

Outputs in --out_dir (default <data_dir>/<species>/celltype_analysis/):
  fig1_sample_pca.png/.pdf, fig2_consensus_vs_pc1.png/.pdf, fig3_group_correlation.png/.pdf,
  fig4_sample_pairs.png/.pdf, group_consensus.tsv, group_stats.tsv, sample_pca.tsv, sample_pairs.tsv,
  top_residual_genes.tsv

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
from matplotlib.lines import Line2D  # noqa: E402

# Colors (reference palette of the dataviz method, light mode; validated: slot 1 passes all checks)
HIGHLIGHT = "#2a78d6"     # categorical slot 1: the group of the panel
CONTEXT = "#c3c2b7"       # all other samples / genes (recessive)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
METHOD_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
PAIR_CATEGORIES = [  # within a lab, method and group can be separated; across labs both vary together
    ("same_lab_same_method_same_group", "alles gleich"),
    ("same_lab_same_method_other_group", "andere\nGruppe"),
    ("same_lab_other_method_same_group", "andere\nMethode"),
    ("same_lab_other_method_other_group", "Methode und\nGruppe anders"),
    ("other_lab_same_group", "gleiche\nGruppe"),
    ("other_lab_other_group", "andere\nGruppe"),
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


def pc1_scores(X, sign_ref=None):
    """PC1 scores of prcomp(X) (genes x samples); sign aligned to sign_ref (or to the row mean)."""
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    sc = U[:, 0] * s[0]
    ref = X.mean(1) if sign_ref is None else sign_ref
    return -sc if np.corrcoef(sc, ref)[0, 1] < 0 else sc


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
    imp = pd.read_csv(d / "halflife_matrix_imputed_qn.tsv.gz", sep="\t", index_col=0)
    pc1 = pd.read_csv(d / "pc1.tsv", sep="\t", index_col=0)
    names = pd.read_csv(Path(data_dir) / "raw" / f"all_HLs_{sp}.txt.gz", sep="\t",
                        usecols=[0, 1], index_col=0).external_gene_name
    use = meta.index[meta.used_for_pc1] if not args.include_outlier else meta.index
    meta = meta.loc[use]
    meta["group"] = meta[args.group_by].astype(str)
    genes = pc1.index[pc1.in_output] if args.genes == "output" else pc1.index
    return meta, raw, imp, pc1, names, genes


def analyze(sp, args):
    style()
    meta, raw, imp, pc1, names, genes = load(args.data_dir, sp, args)
    out = Path(args.out_dir or Path(args.data_dir) / sp / "celltype_analysis")
    out.mkdir(parents=True, exist_ok=True)
    pc_samples = [s for s in meta.index[meta.used_for_pc1] if s in imp.columns]  # PC1 never uses the outlier
    groups = meta.group.value_counts().index.tolist()  # largest group first, order fixed for all figures
    methods = sorted(meta.method.unique())
    marker = {m: METHOD_MARKERS[i % len(METHOD_MARKERS)] for i, m in enumerate(methods)}
    ncols = min(5, len(groups))
    print(f"[{sp}] {len(meta)} samples, {len(groups)} groups ({args.group_by}), {len(genes)} genes", flush=True)

    # ---- orientation check: degradation-rate datasets must enter with half-life orientation
    rho = raw[meta.index].rank().corrwith(pc1.pc1.reindex(raw.index).rank())
    anti = rho[rho < 0]
    if len(anti):
        print(f"[{sp}] [Warning] samples anti-correlated with PC1 (degradation rates not negated?): "
              f"{anti.round(2).to_dict()}" + ("" if args.no_flip else " -> negated"), flush=True)
        if not args.no_flip:
            raw, imp = raw.copy(), imp.copy()
            raw[anti.index] = -raw[anti.index]
            flip_imp = [s for s in anti.index if s in imp.columns]
            imp[flip_imp] = -imp[flip_imp]

    # ---- per-sample z-scores of the measured values (Saluki standardizes each sample the same way)
    Z = raw[meta.index].apply(zscore)

    # ---- 1. sample PCA
    X = imp.loc[:, [s for s in meta.index if s in imp.columns]]
    Xt = X.T.values - X.T.values.mean(0)
    U, s, _ = np.linalg.svd(Xt, full_matrices=False)
    var = s ** 2 / np.sum(s ** 2)
    pca = pd.DataFrame({"PC1": U[:, 0] * s[0], "PC2": U[:, 1] * s[1]}, index=X.columns).join(meta)
    pca.to_csv(out / "sample_pca.tsv", sep="\t")
    fig, axes, bottom, left = panel_grid(len(groups), ncols)
    for ax, g in zip(axes, groups):
        for m in methods:
            o = pca[(pca.group != g) & (pca.method == m)]
            ax.scatter(o.PC1, o.PC2, s=22, marker=marker[m], color=CONTEXT, linewidths=0)
        for m in methods:
            h = pca[(pca.group == g) & (pca.method == m)]
            ax.scatter(h.PC1, h.PC2, s=40, marker=marker[m], color=HIGHLIGHT, edgecolors=SURFACE, linewidths=1.5)
        ax.set_title(f"{g} (n={int((pca.group == g).sum())})", loc="left")
    for ax in bottom:
        ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    for ax in left:
        ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    handles = [Line2D([], [], marker=marker[m], ls="", color=INK_2, label=m) for m in methods]
    fig.legend(handles=handles, title="Methode", loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.04), title_fontsize=9)
    fig.suptitle(f"{sp}: PCA der Proben, je Panel eine Gruppe hervorgehoben", y=1.09, x=0.01, ha="left",
                 color=INK, fontsize=11)
    save(fig, out, "fig1_sample_pca")

    # ---- 2. group consensus vs leave-group-out PC1
    cons, ncnt, rows, resid = {}, {}, [], {}
    for g in groups:
        smp = meta.index[meta.group == g]
        c = Z.loc[genes, smp].mean(1)
        ncnt[g] = Z.loc[genes, smp].notna().sum(1)
        keep = [x for x in pc_samples if x not in smp]
        ref = pd.Series(pc1_scores(imp.loc[:, keep].values, imp.loc[:, pc_samples].mean(1).values),
                        index=imp.index).loc[genes]
        m = c.notna()
        cz, rz = zscore(c[m]), zscore(ref[m])
        b, a = np.polyfit(rz, cz, 1)
        resid[g] = cz - (a + b * rz)
        cons[g] = cz
        cons[f"{g}__loo_pc1"] = rz
        rows.append({"group": g, "n_samples": len(smp), "labs": ",".join(sorted(meta.loc[smp, "lab"].unique())),
                     "methods": ",".join(sorted(meta.loc[smp, "method"].unique())),
                     "n_genes": int(m.sum()), "pearson_vs_loo_pc1": float(np.corrcoef(cz, rz)[0, 1]),
                     "spearman_vs_loo_pc1": float(cz.rank().corr(rz.rank())),
                     "n_samples_loo_pc1": len(keep)})
    stats = pd.DataFrame(rows).set_index("group")
    consensus = pd.DataFrame(cons).join(pd.DataFrame(ncnt).add_prefix("n_"))
    consensus.insert(0, "gene_name", names.reindex(consensus.index))
    consensus["global_target"] = zscore(pc1.loc[consensus.index, "target"])
    consensus.to_csv(out / "group_consensus.tsv", sep="\t", index_label="ensembl_gene_id")

    fig, axes, bottom, left = panel_grid(len(groups), ncols)
    for ax, g in zip(axes, groups):
        x, y = cons[f"{g}__loo_pc1"], cons[g]
        ax.scatter(x, y, s=2, color=HIGHLIGHT, alpha=0.25, linewidths=0, rasterized=True)
        lim = [-4, 4]
        ax.plot(lim, lim, color=INK_2, lw=1, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        st = stats.loc[g]
        ax.set_title(f"{g} (n={st.n_samples})", loc="left")
        ax.text(0.03, 0.97, f"r = {st.pearson_vs_loo_pc1:.2f}\n{st.n_genes:,} Gene".replace(",", "."),
                transform=ax.transAxes, va="top", color=INK_2, fontsize=8)
    for ax in bottom:
        ax.set_xlabel("PC1 ohne diese Gruppe (z)")
    for ax in left:
        ax.set_ylabel("HWZ der Gruppe (z)")
    fig.suptitle(f"{sp}: Gruppen-HWZ gegen Konsens-PC1 der übrigen Proben (Linie: Identität)", y=1.02,
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, out, "fig2_consensus_vs_pc1")

    # residual genes: largest deviations from the consensus per group (candidates for context dependence)
    top = []
    for g in groups:
        r = resid[g].sort_values()
        for side, sel in (("kürzer", r.head(args.top_n)), ("länger", r.tail(args.top_n)[::-1])):
            top.append(pd.DataFrame({"group": g, "direction": side, "residual_z": sel,
                                     "gene_name": names.reindex(sel.index), "n_samples_measured":
                                     ncnt[g].reindex(sel.index)}))
    pd.concat(top).to_csv(out / "top_residual_genes.tsv", sep="\t", index_label="ensembl_gene_id")

    # ---- 3./4. sample-pair correlations on measured values
    C = raw.loc[genes, meta.index].corr(method="spearman", min_periods=args.min_common_genes)
    pairs = []
    idx = list(meta.index)
    for i, a in enumerate(idx):
        for b in idx[i + 1:]:
            if pd.isna(C.loc[a, b]):
                continue
            same_lab = meta.loc[a, "lab"] == meta.loc[b, "lab"]
            same_grp = meta.loc[a, "group"] == meta.loc[b, "group"]
            same_meth = meta.loc[a, "method"] == meta.loc[b, "method"]
            cat = ("same_lab_" + ("same" if same_meth else "other") + "_method_") if same_lab else "other_lab_"
            cat += ("same" if same_grp else "other") + "_group"
            pairs.append({"sample_a": a, "sample_b": b, "spearman": C.loc[a, b],
                          "n_common_genes": int(raw.loc[genes, [a, b]].notna().all(1).sum()),
                          "same_lab": same_lab, "same_method": same_meth,
                          "same_group": same_grp, "group_a": meta.loc[a, "group"], "group_b": meta.loc[b, "group"],
                          "category": cat})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / "sample_pairs.tsv", sep="\t", index=False)

    G = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            sel = pairs[((pairs.group_a == g1) & (pairs.group_b == g2)) | ((pairs.group_a == g2) & (pairs.group_b == g1))]
            if len(sel):
                G.loc[g1, g2] = sel.spearman.mean()
    G.to_csv(out / "group_correlation.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(0.55 * len(groups) + 2, 0.55 * len(groups) + 1.2))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmin = np.floor(np.nanmin(G.values) * 10) / 10
    im = ax.imshow(G.values, cmap=cmap, vmin=vmin, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    for i in range(len(groups)):
        for j in range(len(groups)):
            v = G.values[i, j]
            if np.isfinite(v):
                dark = (v - vmin) / (1 - vmin) > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if dark else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("mittlere Spearman-Korrelation", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(f"{sp}: Korrelation zwischen Proben je Gruppenpaar\n(Diagonale: verschiedene Proben derselben "
                 f"Gruppe; leer: nur 1 Probe)", loc="left")
    save(fig, out, "fig3_group_correlation")

    fig, ax = plt.subplots(figsize=(9, 4))
    rng = np.random.default_rng(0)
    labels = []
    for k, (cat, label) in enumerate(PAIR_CATEGORIES):
        v = pairs.loc[pairs.category == cat, "spearman"].values
        labels.append(f"{label}\n(n={len(v)})")
        if len(v) == 0:
            continue
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=HIGHLIGHT, alpha=0.5, linewidths=0)
        ax.plot([k - 0.28, k + 0.28], [np.median(v)] * 2, color=INK, lw=2)
        ax.text(k + 0.3, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(PAIR_CATEGORIES)), labels)
    ax.set_ylabel("Spearman-Korrelation der Probenpaare")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{sp}: Probenpaare nach Labor, Methode und Gruppe (Strich: Median)", loc="left", pad=24)
    ax.axvline(3.5, color=AXIS, lw=0.8)
    for x, text in ((1.5, "gleiches Labor (Vergleich bei gleicher Methode sauber)"), (4.5, "anderes Labor")):
        ax.text(x, 1.02, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                color=INK_2)
    save(fig, out, "fig4_sample_pairs")

    summary = pairs.groupby("category").spearman.agg(["count", "median", "mean"])
    stats.to_csv(out / "group_stats.tsv", sep="\t")
    print(stats[["n_samples", "n_genes", "pearson_vs_loo_pc1"]].round(3).to_string(), flush=True)
    print(summary.round(3).to_string(), flush=True)
    print(f"[{sp}] written to {out}", flush=True)


def main(argv=None):
    base = Path(os.environ.get("ORTHRUS_BASE", "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(base / "data" / "saluki_hl_reproduction"),
                    help="--out_dir of reproduce_saluki_hl_dataset.py")
    ap.add_argument("--species", nargs="+", default=["human", "mouse"], choices=["human", "mouse"])
    ap.add_argument("--group_by", default="cell_group", choices=["cell_group", "cell_type", "method", "lab"])
    ap.add_argument("--genes", default="output", choices=["output", "all"],
                    help="output: genes of the reproduced dataset; all: every gene passing the sample filter")
    ap.add_argument("--include_outlier", action="store_true",
                    help="include the samples Saluki excluded from PC1 (human: Gejman, 15 LCL samples)")
    ap.add_argument("--no_flip", action="store_true",
                    help="do not negate samples that are anti-correlated with PC1")
    ap.add_argument("--min_common_genes", type=int, default=500, help="minimum overlap for a sample-pair correlation")
    ap.add_argument("--top_n", type=int, default=50, help="genes per group and direction in top_residual_genes.tsv")
    ap.add_argument("--out_dir", default=None, help="default: <data_dir>/<species>/celltype_analysis")
    args = ap.parse_args(argv)
    if args.out_dir and len(args.species) > 1:
        ap.error("--out_dir only with a single --species")
    for sp in args.species:
        analyze(sp, args)


if __name__ == "__main__":
    main()
