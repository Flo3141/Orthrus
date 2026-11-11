#!/usr/bin/env python
"""
Measurement noise of the Saluki half-life measurements per gene (mRNABench human / mouse).

Input (outputs of reproduce_saluki_hl_dataset.py in --data_dir/<species>/)
  halflife_matrix_filtered.tsv.gz   measured log half-lives (NaN = not measured, degradation rates negated)
  sample_metadata.tsv               lab, method, cell group and used_for_pc1 per sample
  pc1.tsv                           column in_output marks the genes of the final dataset

Only measured values are used, no imputation. Samples: those that enter the PC1 (human: without the outlier
study Gejman). Genes: those of the final dataset. Each sample is z-scored over these genes.

Seven values per gene (gene_noise.tsv), all in units of one sample's z-score
  sd_samples      SD over all measured samples
  sd_replicates   pooled SD within replicate groups (same lab, method and cell type)
  sd_labs         SD of the lab means
  se_consensus    lab-clustered standard error of the mean over all samples
  sd_labs_4sU     SD of the lab means within 4sU
  sd_labs_ActD    SD of the lab means within ActD
  diff_4sU_ActD   mean of the 4sU lab means minus mean of the ActD lab means
plus the counts behind them (n_samples, n_labs, n_labs_4sU, n_labs_ActD).

summary.json: per value the number of genes with a value, RMS, mean and median over the genes.

Usage (requires numpy and pandas)
  python analyze_measurement_noise.py --data_dir /path/to/saluki_hl_reproduction [--species human mouse]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("4sU", "ActD")
VALUES = ["sd_samples", "sd_replicates", "sd_labs", "se_consensus", "sd_labs_4sU", "sd_labs_ActD", "diff_4sU_ActD"]


def load(data_dir, sp):
    """Per-sample z-scores of the measured half-lives (dataset genes x PC1 samples) and the sample metadata."""
    d = Path(data_dir) / sp
    meta = pd.read_csv(d / "sample_metadata.tsv", sep="\t").set_index("sample")
    meta = meta[meta.used_for_pc1]
    raw = pd.read_csv(d / "halflife_matrix_filtered.tsv.gz", sep="\t", index_col=0)
    pc1 = pd.read_csv(d / "pc1.tsv", sep="\t", index_col=0)
    genes = pc1.index[pc1.in_output]
    X = raw.loc[genes, meta.index].copy()
    # degradation rates must enter with half-life orientation (see saluki_hl_reproduktion.md, 4.1)
    rho = X.rank().corrwith(pc1.pc1[genes].rank())
    anti = rho.index[rho < 0]
    if len(anti):
        print(f"[{sp}] [Warning] samples anti-correlated with PC1, negated: {rho[anti].round(2).to_dict()}", flush=True)
        X[anti] = -X[anti]
    return (X - X.mean()) / X.std(), meta


def lab_means(Z, labs):
    """Mean z per gene and lab over the measured samples (NaN if the lab did not measure the gene)."""
    return Z.T.groupby(labs.values).mean().T


def gene_noise(Z, meta):
    out = pd.DataFrame(index=Z.index)
    out["n_samples"] = Z.notna().sum(1)

    # 1. spread over all samples
    out["sd_samples"] = Z.std(1)

    # 2. replicates: pooled within-group variance over the groups in which the gene has >= 2 measurements
    rep = meta.lab + "|" + meta.method + "|" + meta.cell_group.astype(str)
    ss, dof = pd.Series(0.0, index=Z.index), pd.Series(0, index=Z.index)
    for cols in rep.groupby(rep).groups.values():
        sub = Z[cols]
        k = sub.notna().sum(1)
        ok = k >= 2
        ss[ok] += sub.sub(sub.mean(1), axis=0).pow(2).sum(1)[ok]
        dof[ok] += k[ok] - 1
    out["sd_replicates"] = np.sqrt(ss / dof.replace(0, np.nan))

    # 3. spread between labs
    lab_mean = lab_means(Z, meta.lab)
    out["n_labs"] = lab_mean.notna().sum(1)
    out["sd_labs"] = lab_mean.std(1)

    # 4. lab-clustered standard error of the sample mean (CR1): sum_g (n_g/n)^2 (m_g - m)^2 * K/(K-1)
    lab_n = Z.notna().T.groupby(meta.lab.values).sum().T
    w = lab_n.div(out.n_samples, axis=0)
    K = out.n_labs
    var = (w.pow(2) * lab_mean.sub(Z.mean(1), axis=0).pow(2)).sum(1) * K / (K - 1)
    out["se_consensus"] = np.sqrt(var.where(K >= 2))

    # 5.-7. spread between labs within each method, and the difference between the methods
    method_mean = {}
    for m in METHODS:
        sel = (meta.method == m).values
        lm = lab_means(Z.loc[:, sel], meta.lab[sel])
        out[f"n_labs_{m}"] = lm.notna().sum(1)
        out[f"sd_labs_{m}"] = lm.std(1)
        method_mean[m] = lm.mean(1)
    out["diff_4sU_ActD"] = method_mean["4sU"] - method_mean["ActD"]
    return out


def summarize(out, meta, sp):
    summ = {"species": sp, "n_genes": int(len(out)), "n_samples": int(len(meta)), "n_labs": int(meta.lab.nunique()),
            "n_samples_per_method": meta.method.value_counts().to_dict(),
            "n_labs_per_method": meta.groupby("method").lab.nunique().to_dict()}
    for v in VALUES:
        s = out[v].dropna()
        summ[v] = {"n_genes": int(len(s)), "rms": float(np.sqrt(np.mean(s ** 2))), "mean": float(s.mean()),
                   "median": float(s.median())}
    return summ


def main(argv=None):
    base = Path(os.environ.get("ORTHRUS_BASE", "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(base / "data" / "saluki_hl_reproduction"),
                    help="--out_dir of reproduce_saluki_hl_dataset.py")
    ap.add_argument("--species", nargs="+", default=["human", "mouse"], choices=["human", "mouse"])
    ap.add_argument("--out_dir", default=None,
                    help="default: <data_dir>/<species>/noise_analysis; only with a single --species")
    args = ap.parse_args(argv)
    if args.out_dir and len(args.species) > 1:
        ap.error("--out_dir only with a single --species")

    for sp in args.species:
        Z, meta = load(args.data_dir, sp)
        out_dir = Path(args.out_dir or Path(args.data_dir) / sp / "noise_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{sp}] {len(meta)} samples from {meta.lab.nunique()} labs, {len(Z)} genes", flush=True)
        out = gene_noise(Z, meta)
        out.to_csv(out_dir / "gene_noise.tsv", sep="\t", index_label="ensembl_gene_id")
        summ = summarize(out, meta, sp)
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print(pd.DataFrame({v: summ[v] for v in VALUES}).T.round(3).to_string(), flush=True)
        print(f"[{sp}] written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
