#!/usr/bin/env python
"""
Rebuilds the mRNABench half-life datasets (rnahl-human, rnahl-mouse) from the Saluki raw data.

Chain of custody of the published dataset:
    Saluki matrix all_HLs_<species>.txt.gz (19,503 human / 22,499 mouse genes, 54 / 27 samples)
      -> Saluki preprocessing (filter, z-score, missMDA imputePCA, quantile normalization, PC1)
      -> Saluki training genes  (12,968 / 13,738 genes, datapack .../train_gru/f0_c0/data{0,1}/genes.tsv)
      -> Orthrus Zenodo release rna_hl_<species>.npz (same genes, 6-track one-hot, float16 target)
      -> mRNABench: drops rows without gene name (29 / 15) -> 12,939 / 13,723 rows

This script downloads the public sources, reproduces each step independently, writes the result in the
mRNABench column layout and compares every step against the published intermediate files.
See Orthrus/docs/saluki_hl_reproduktion.md for the reasoning behind every step.

Steps
  1. download   Saluki matrix (GitHub v1.0.0), selected members of the Saluki datapack (Zenodo 6326409, read
                via HTTP range requests, the full archive is 18.7 GB), Ensembl cDNA (v83 human, v90 mouse),
                GENCODE annotation (v41 / vM31, as used by mRNABench) and, for the comparison, the Orthrus
                npz files and the mRNABench parquet files.
  2. target     genes with >= 10 (human) / >= 5 (mouse) measured samples; z-score per sample; number of
                components by GCV (missMDA::estim_ncpPCA); regularized iterative PCA imputation
                (missMDA::imputePCA); quantile normalization (preprocessCore::normalize.quantiles);
                human: drop the outlier study Gejman (Duan et al. 2013, 15 LCL samples); PC1 (prcomp).
  3. genes      Saluki gene list (default) or the reconstructed rule: chosen protein-coding transcript with
                5'UTR, 3'UTR and stop codon, CDS length divisible by 3.
  4. sequences  spliced mRNA from Ensembl cDNA; CDS track (first nt of each codon) and splice track; length cap
                12,288 nt.
  5. names      gene names and chromosome from GENCODE (mRNABench: genome_kit first_by_name).
  6. compare    against Saluki featTable/genes.tsv, Orthrus npz and the mRNABench parquet.

Outputs (per species, in --out_dir/<species>/)
  rnahl-<species>_reproduced.parquet   mRNABench layout + ensembl ids, Saluki fold, untruncated length
  halflife_matrix_filtered.tsv.gz      log half-lives of the filtered genes (all samples, NaN = not measured)
  halflife_matrix_imputed_qn.tsv.gz    the same after z-score, imputation and quantile normalization
  sample_metadata.tsv                  lab, method, cell type, cell group (LCLs pooled), replicate, used for PC1
  pc1.tsv                              PC1 per filtered gene, Saluki reference value, final target
  report.json / report.md              counts per step and all comparisons

Usage
  python reproduce_saluki_hl_dataset.py --out_dir /path/to/out [--species human mouse] [--gene_set saluki|rule]
"""

import argparse
import collections
import gzip
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Sources
# =============================================================================

SALUKI_REPO_RAW = "https://raw.githubusercontent.com/vagarwal87/saluki_paper/v1.0.0/Fig1_S1_S2"
SALUKI_DATAPACK_URL = "https://zenodo.org/records/6326409/files/datasets.zip"
ORTHRUS_ZENODO = "https://zenodo.org/records/14708163/files"
MRNABENCH_HF = "https://huggingface.co/datasets/morrislab/rnahl-saluki-{sp}/resolve/main/rnahl-saluki-{sp}.parquet"

SPECIES = {
    "human": {
        "min_samples": 10,
        "ncp_max": 20,
        "outlier_prefix": "Gejman_",
        "datapack_gtf": "datasets/human/Homo_sapiens.GRCh38.83.chosenTranscript.gtf",
        "datapack_feat": "datasets/human/all_HLs_human_featTable.txt.gz",
        "datapack_genes": "datasets/deeplearning/train_gru/f0_c0/data0/genes.tsv",
        "cdna": "https://ftp.ensembl.org/pub/release-83/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
        "gencode": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz",
    },
    "mouse": {
        "min_samples": 5,
        "ncp_max": 10,
        "outlier_prefix": None,
        "datapack_gtf": "datasets/mouse/Mus_musculus.GRCm3.90.chosenTranscript.gtf.gz",
        "datapack_feat": "datasets/mouse/all_HLs_mouse_featTable.txt.gz",
        "datapack_genes": "datasets/deeplearning/train_gru/f0_c0/data1/genes.tsv",
        "cdna": "https://ftp.ensembl.org/pub/release-90/fasta/mus_musculus/cdna/Mus_musculus.GRCm38.cdna.all.fa.gz",
        "gencode": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M31/gencode.vM31.annotation.gtf.gz",
    },
}

MAX_LEN = 12288
LCL_PREFIX = "GM"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 1. Download
# =============================================================================

def download(url, dst, retries=5):
    """Downloads url to dst (atomic via .part file); skips existing files."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            log(f"download {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "saluki-hl-reproduction"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            tmp.replace(dst)
            return dst
        except Exception as e:  # network errors on the cluster are usually transient
            log(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(10 * attempt)
    raise RuntimeError(f"Download failed: {url}")


class RemoteZip:
    """Reads single members of a (zip64) archive over HTTP range requests, without downloading the archive."""

    def __init__(self, url):
        self.url = url
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120)
        self.size = int(head.headers["Content-Length"])
        tail = self._get(max(0, self.size - 200_000), self.size - 1)
        i = tail.rfind(b"PK\x06\x06")
        if i >= 0:
            cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
        else:
            i = tail.rfind(b"PK\x05\x06")
            cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
        self.index = self._parse_central_directory(self._get(cd_off, cd_off + cd_size - 1))

    def _get(self, a, b, retries=5):
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.read()
            except Exception as e:
                log(f"  range request attempt {attempt}/{retries} failed: {e}")
                time.sleep(10 * attempt)
        raise RuntimeError(f"Range request failed: {self.url} [{a}-{b}]")

    @staticmethod
    def _parse_central_directory(cd):
        index, p = {}, 0
        while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
            off, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + n].decode("utf-8")
            extra, q, z64 = cd[p + 46 + n:p + 46 + n + e], 0, []
            while q + 4 <= len(extra):
                hid, hl = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    z64 = [struct.unpack("<Q", extra[q + 4 + k:q + 12 + k])[0] for k in range(0, hl, 8)]
                q += 4 + hl
            it = iter(z64)
            if usize == 0xFFFFFFFF:
                usize = next(it)
            if csize == 0xFFFFFFFF:
                csize = next(it)
            if off == 0xFFFFFFFF:
                off = next(it)
            index[name] = (method, csize, off)
            p += 46 + n + e + c
        return index

    def extract(self, member, dst):
        method, csize, off = self.index[member]
        n, e = struct.unpack("<HH", self._get(off, off + 29)[26:30])
        data = self._get(off + 30 + n + e, off + 30 + n + e + csize - 1)
        if method == 8:
            data = zlib.decompress(data, -15)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)


def fetch_datapack_members(members, raw_dir, local_zip=None):
    """Extracts datapack members into raw_dir/datapack/ (from a local datasets.zip if given, else remotely)."""
    out = {}
    missing = [m for m in members if not (raw_dir / "datapack" / m).exists()]
    if missing:
        if local_zip:
            log(f"extract {len(missing)} members from local {local_zip}")
            with zipfile.ZipFile(local_zip) as zf:
                for m in missing:
                    dst = raw_dir / "datapack" / m
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(dst, "wb") as fh:
                        shutil.copyfileobj(src, fh)
        else:
            log(f"read {len(missing)} members of the Saluki datapack via HTTP range requests")
            rz = RemoteZip(SALUKI_DATAPACK_URL)
            for m in missing:
                log(f"  {m}")
                rz.extract(m, raw_dir / "datapack" / m)
    for m in members:
        out[m] = raw_dir / "datapack" / m
    return out


def opener(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


# =============================================================================
# 2. Target: port of the Saluki preprocessing (missMDA / preprocessCore / prcomp)
# =============================================================================

def impute_pca(X, ncp, threshold=1e-6, maxiter=1000):
    """
    Port of missMDA:::impute (imputePCA with method='Regularized', scale=TRUE, init=1, uniform row weights,
    coeff.ridge=1). Returns (completed matrix, reconstruction, n_iterations, converged).
    """
    X = np.asarray(X, float)
    n, p = X.shape
    miss = np.isnan(X)
    w = 1.0 / n
    mean_p = np.nanmean(X, 0)
    Xhat = X - mean_p
    et = np.sqrt(np.nanmean(Xhat ** 2, 0))
    Xhat = Xhat / et
    Xhat[miss] = 0.0
    fitted = Xhat.copy()
    old, it, converged, n_iter = np.inf, (1 if ncp > 0 else 0), True, 0
    while it > 0:
        n_iter += 1
        Xhat[miss] = fitted[miss]
        Xhat = Xhat * et + mean_p
        mean_p = Xhat.mean(0)
        Xhat = Xhat - mean_p
        et = np.sqrt((Xhat ** 2).mean(0))
        Xhat = Xhat / et
        U, s, Vt = np.linalg.svd(Xhat * np.sqrt(w), full_matrices=False)
        U = U / np.sqrt(w)
        denom = (n - 1) * p - (n - 1) * ncp - p * ncp + ncp ** 2
        sigma2 = n * p / min(p, n - 1) * np.sum(s[ncp:] ** 2 / denom)
        sigma2 = min(sigma2, s[ncp] ** 2)
        lam = (s[:ncp] ** 2 - sigma2) / s[:ncp]
        fitted = (U[:, :ncp] * w * lam) @ Vt[:ncp] / w
        diff = Xhat - fitted
        diff[miss] = 0
        obj = np.sum(diff ** 2 * w)
        crit = abs(1 - obj / old)
        old = obj
        it += 1
        if np.isfinite(crit) and crit < threshold and it > 5:
            it = 0
        if obj < threshold and it > 5:
            it = 0
        if it > maxiter:
            converged, it = False, 0
    Xhat = Xhat * et + mean_p
    rec = fitted * et + mean_p
    completed = X.copy()
    completed[miss] = Xhat[miss]
    return completed, rec, n_iter, converged


def estim_ncp_gcv(X, ncp_min=0, ncp_max=5):
    """Port of missMDA::estim_ncpPCA(method='Regularized', scale=TRUE, method.cv='gcv')."""
    n, p = X.shape
    nmiss = int(np.isnan(X).sum())
    ncp_max = min(n - 2, p - 1, ncp_max)
    crit, not_converged = [], []
    if ncp_min == 0:
        crit.append(np.nanmean((X - np.nanmean(X, 0)) ** 2))
    for q in range(max(ncp_min, 1), ncp_max + 1):
        _, rec, _, conv = impute_pca(X, q)
        if not conv:
            not_converged.append(q)
        crit.append(np.nanmean(((n * p - nmiss) * (X - rec) / ((n - 1) * p - nmiss - q * (n + p - q - 1))) ** 2))
        log(f"    GCV ncp={q:2d}: {crit[-1]:.6f}")
    crit = np.array(crit)
    d = np.diff(crit)
    ncp = (np.flatnonzero(d > 0)[0] + 1) if np.any(d > 0) else (int(np.argmin(crit)) + 1)
    return int(ncp + ncp_min - 1), crit.tolist(), not_converged


def quantile_normalize(X):
    """preprocessCore::normalize.quantiles for a complete matrix (ties get the mean of their quantiles)."""
    target = np.sort(X, 0).mean(1)
    grid = np.arange(1, len(target) + 1)
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = np.interp(pd.Series(X[:, j]).rank(method="average").values, grid, target)
    return out


def first_pc(X):
    """PC1 scores of prcomp(X) (centered, unscaled); sign chosen so that PC1 rises with the mean half-life."""
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, 0] * s[0]
    if np.corrcoef(scores, X.mean(1))[0, 1] < 0:
        scores = -scores
    return scores, float(s[0] ** 2 / np.sum(s ** 2))


def sample_metadata(columns, outlier_prefix):
    rows = []
    for c in columns:
        lab, method, cell, rep = (c.split("_", 3) + ["", "", "", ""])[:4]
        rows.append({
            "sample": c, "lab": lab, "method": method, "cell_type": cell,
            "cell_group": "LCL" if cell.startswith(LCL_PREFIX) else cell,  # as in Saluki Fig1_S1.R
            "replicate": rep,
            "used_for_pc1": not (outlier_prefix and c.startswith(outlier_prefix)),
        })
    return pd.DataFrame(rows)


def compute_target(hl, cfg, args, report, sp):
    """Saluki PC1 for all genes passing the sample filter."""
    values = hl.drop(columns="external_gene_name")
    meta = sample_metadata(values.columns, cfg["outlier_prefix"])
    n_meas = values.notna().sum(1)
    M = values[n_meas >= cfg["min_samples"]]
    report["n_genes_matrix"] = int(len(values))
    report["n_samples"] = int(values.shape[1])
    report["n_genes_after_sample_filter"] = int(len(M))
    log(f"  sample filter (>= {cfg['min_samples']} measured samples): {len(values)} -> {len(M)} genes")

    use = meta.used_for_pc1.values
    if args.outlier_handling == "drop_before_impute":
        M_imp_in = M.loc[:, use]
    else:
        M_imp_in = M
    Z = ((M_imp_in - M_imp_in.mean()) / M_imp_in.std()).values  # R scale(): sd with n-1

    if args.ncp and sp in args.ncp:
        ncp, crit, nc = args.ncp[sp], None, []
        log(f"  ncp fixed to {ncp}")
    else:
        log(f"  estimate ncp by GCV (0..{cfg['ncp_max']})")
        ncp, crit, nc = estim_ncp_gcv(Z, 0, cfg["ncp_max"])
        log(f"  ncp = {ncp}")
    completed, _, n_iter, conv = impute_pca(Z, ncp)
    log(f"  imputePCA: {n_iter} iterations, converged={conv}")
    qn = quantile_normalize(completed)
    if args.outlier_handling == "impute_then_drop":
        pc_in = qn[:, use]
    else:
        pc_in = qn
    pc1, var_expl = first_pc(pc_in)
    log(f"  PC1 explains {var_expl:.3f} of the variance ({pc_in.shape[1]} samples)")

    report["pca"] = {
        "outlier_handling": args.outlier_handling, "ncp": ncp, "gcv_criterion": crit,
        "gcv_not_converged_ncp": nc, "impute_iterations": n_iter, "impute_converged": conv,
        "n_samples_pc1": int(pc_in.shape[1]), "pc1_variance_explained": var_expl,
    }
    imputed = pd.DataFrame(qn, index=M.index, columns=M_imp_in.columns)
    return pd.Series(pc1, index=M.index, name="pc1"), M, imputed, meta


# =============================================================================
# 3./4. Gene structure and sequences
# =============================================================================

def parse_chosen_gtf(path):
    """Saluki chosen-transcript GTF (one transcript per protein-coding gene). Attribute: 'ENST:ENSG' or 'ENSG'."""
    genes = {}
    with opener(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            ids = f[8].strip().split(":")
            gid, tid = ids[-1], (ids[0] if len(ids) == 2 else None)
            d = genes.setdefault(gid, {"tx": tid, "chrom": f[0], "strand": f[6], "span": None,
                                       "features": collections.defaultdict(list)})
            if f[2] == "transcript":
                d["span"] = (f[0], int(f[3]), int(f[4]), f[6])
            d["features"][f[2]].append((int(f[3]), int(f[4])))
    return genes


def parse_cdna(path):
    """Ensembl cDNA FASTA -> {transcript: seq}, {transcript: (gene, chrom, start, end, strand)}."""
    seqs, hdr, cur, buf = {}, {}, None, []
    with opener(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs[cur] = "".join(buf)
                h = line[1:].split()
                cur, buf = h[0].split(".")[0], []
                loc = h[2].split(":")
                gene = next(x for x in h if x.startswith("gene:")).split(":")[1].split(".")[0]
                hdr[cur] = (gene, loc[2], int(loc[3]), int(loc[4]), "+" if loc[5] == "1" else "-")
            else:
                buf.append(line.strip())
    if cur:
        seqs[cur] = "".join(buf)
    return seqs, hdr


def resolve_transcripts(chosen, seqs, hdr, feat_seqs):
    """
    Mouse GTF carries no transcript id: match gene id, transcript span, strand and exon length to the cDNA.
    If several candidates remain with different sequences, take the one that contains Saluki's own 5'UTR,
    ORF and 3'UTR sequences (featTable); otherwise the lexicographically first transcript id.
    """
    by_gene = collections.defaultdict(list)
    for t, h in hdr.items():
        by_gene[h[0]].append(t)
    stats = collections.Counter()
    for gid, d in chosen.items():
        if d["tx"] is not None:
            stats["given"] += 1
            continue
        exon_len = sum(b - a + 1 for a, b in d["features"]["exon"])
        cand = sorted(t for t in by_gene[gid] if hdr[t][1:] == d["span"] and len(seqs[t]) == exon_len)
        stats[f"candidates_{len(cand)}"] += 1
        if not cand:
            continue
        d["tx"] = cand[0]
        if len({seqs[t] for t in cand}) > 1:
            fs = feat_seqs.get(gid)
            hit = [t for t in cand if fs and seqs[t].startswith(fs[0]) and fs[1] in seqs[t]
                   and seqs[t].endswith(fs[2])] if fs else []
            if hit:
                d["tx"] = hit[0]
                stats["ambiguous_resolved_by_featTable"] += 1
            else:
                stats["ambiguous_unresolved"] += 1
    return dict(stats)


def build_record(gid, d, seqs, args):
    seq = seqs.get(d["tx"])
    if seq is None:
        return None, "no_cdna"
    f = d["features"]
    exons = sorted(f["exon"], reverse=(d["strand"] == "-"))
    ex_len = np.array([b - a + 1 for a, b in exons])
    if ex_len.sum() != len(seq):
        return None, "cdna_length_mismatch"
    u5 = sum(b - a + 1 for a, b in f["five_prime_utr"])
    cds_len = sum(b - a + 1 for a, b in f["CDS"])
    has = {"utr5": u5 > 0, "utr3": len(f["three_prime_utr"]) > 0, "stop": len(f["stop_codon"]) > 0,
           "cds_mod3": cds_len % 3 == 0}
    L = len(seq)
    cds = np.zeros(L, np.int8)
    cds[u5:u5 + cds_len:3] = 1
    splice = np.zeros(L, np.int8)
    if args.splice_track == "exon_end":
        pos = np.cumsum(ex_len) - 1
    elif d["strand"] == "+":
        pos = np.cumsum(ex_len - 1) - 1
    else:
        tail = np.concatenate([np.cumsum((ex_len - 1)[::-1])[::-1][1:], [0]])
        pos = L - 1 - tail
    pos = pos[(pos >= 0) & (pos < L)]
    splice[pos] = 1
    if L > MAX_LEN and args.truncation != "none":
        sl = slice(0, MAX_LEN) if args.truncation == "orthrus" else slice(L - MAX_LEN, L)
        seq, cds, splice = seq[sl], cds[sl], splice[sl]
    rec = {"ensembl_gene_id": gid, "ensembl_transcript_id": d["tx"], "chromosome_ensembl": d["chrom"],
           "strand": d["strand"], "sequence": seq, "cds": cds, "splice": splice, "length_full": L,
           "truncated": L > MAX_LEN and args.truncation != "none", **{f"has_{k}": v for k, v in has.items()}}
    return rec, "ok"


# =============================================================================
# 5. Gene names (GENCODE, as used by mRNABench)
# =============================================================================

def parse_gencode_genes(path):
    """gene_id (unversioned) -> gene_name, and gene_name -> chromosome of its first occurrence."""
    id2name, name2chrom = {}, {}
    with opener(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
            name = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
            if "_PAR_Y" not in gid:
                id2name.setdefault(gid.split(".")[0], name)
            name2chrom.setdefault(name, f[0].replace("chr", ""))
    return id2name, name2chrom


# =============================================================================
# 6. Comparisons
# =============================================================================

def decode_onehot(X, chunk=500):
    nuc = np.frombuffer(b"ACGTN", dtype=np.uint8)
    out = []
    for a in range(0, len(X), chunk):
        x = X[a:a + chunk, :, :4]
        idx = np.where(x.sum(-1) == 0, 4, np.argmax(x, -1))
        out.extend(nuc[row].tobytes().decode().rstrip("N") for row in idx)
    return out


def compare_tracks(ref_seq, ref_cds, ref_spl, df, key_ref, key_df):
    """
    Compares rows of a reference (Orthrus/mRNABench) with the reproduction. Rows are joined by identical sequence
    first (gene names differ between annotation versions), then by gene name.
    """
    by_name = {g: i for i, g in enumerate(df[key_df]) if g}
    by_seq = {s: i for i, s in enumerate(df["sequence"])}
    res = collections.Counter()
    pairs = []
    for i, name in enumerate(key_ref):
        j = by_seq.get(ref_seq[i])
        if j is not None:
            res["matched_by_sequence"] += 1
        else:
            j = by_name.get(name) if name else None
            res["matched_by_name" if j is not None else "unmatched"] += 1
        if j is None:
            continue
        pairs.append((i, j))
        r = df.iloc[j]
        res["sequence_equal"] += ref_seq[i] == r.sequence
        L = len(ref_seq[i])
        res["cds_equal"] += np.array_equal(np.asarray(ref_cds[i])[:L], r.cds)
        res["splice_equal"] += np.array_equal(np.asarray(ref_spl[i])[:L], r.splice)
    return dict(res), pairs


def compare_orthrus(npz_path, df):
    z = np.load(npz_path, allow_pickle=False)  # genes are stored as fixed-width unicode, no pickle needed
    X, names, y = z["X"], z["genes"].astype(str), z["y"].astype(np.float32)
    seqs = decode_onehot(X)
    lens = [len(s) for s in seqs]
    cds = [X[i, :lens[i], 4] for i in range(len(X))]
    spl = [X[i, :lens[i], 5] for i in range(len(X))]
    res, pairs = compare_tracks(seqs, cds, spl, df, names, "gene_gencode")
    t_ref = np.float16(y[[i for i, _ in pairs]])
    t_rep = np.float16(df.target_saluki.values[[j for _, j in pairs]])
    res.update({
        "n_rows": int(len(X)), "n_empty_names": int((names == "").sum()),
        "target_saluki_equal_float16": int((t_ref == t_rep).sum()),
        "target_reproduced_pearson": float(np.corrcoef(y[[i for i, _ in pairs]],
                                                       df.target.values[[j for _, j in pairs]])[0, 1]),
        "reproduced_genes_absent_in_orthrus": int(len(df) - len({j for _, j in pairs})),
    })
    unnamed = [df.iloc[j].ensembl_gene_id for i, j in pairs if names[i] == ""]
    res["ensembl_ids_of_unnamed_orthrus_rows"] = unnamed
    return res


def compare_mrnabench(parquet_path, df):
    ref = pd.read_parquet(parquet_path)
    res, pairs = compare_tracks(list(ref.sequence), list(ref.cds), list(ref.splice), df, list(ref.gene), "gene")
    ii, jj = [i for i, _ in pairs], [j for _, j in pairs]
    res.update({
        "n_rows": int(len(ref)), "columns": list(ref.columns),
        "target_saluki_equal_float16": int((np.float16(ref.target.values[ii]) ==
                                            np.float16(df.target_saluki.values[jj])).sum()),
        "target_reproduced_pearson": float(np.corrcoef(ref.target.values[ii].astype(float),
                                                       df.target.values[jj])[0, 1]),
        "chromosome_equal": int((ref.chromosome.astype(str).values[ii] == df.chromosome.astype(str).values[jj]).sum()),
        "reproduced_rows_absent_in_mrnabench": int(len(df) - len(set(jj))),
    })
    return res


# =============================================================================
# Main per species
# =============================================================================

def download_inputs(sp, args):
    """Step 1: all inputs of one species (cached in <out_dir>/raw)."""
    cfg = SPECIES[sp]
    raw = Path(args.out_dir) / "raw"
    paths = {
        "hl": download(f"{SALUKI_REPO_RAW}/all_HLs_{sp}.txt.gz", raw / f"all_HLs_{sp}.txt.gz"),
        "datapack": fetch_datapack_members([cfg["datapack_gtf"], cfg["datapack_feat"], cfg["datapack_genes"]],
                                           raw, args.datapack_zip),
        "cdna": download(cfg["cdna"], raw / Path(cfg["cdna"]).name),
        "gencode": download(cfg["gencode"], raw / Path(cfg["gencode"]).name),
    }
    if not args.skip_compare:
        paths["orthrus_npz"] = download(f"{ORTHRUS_ZENODO}/rna_hl_{sp}.npz", raw / f"rna_hl_{sp}.npz")
        paths["mrnabench"] = download(MRNABENCH_HF.format(sp=sp), raw / f"rnahl-saluki-{sp}.parquet")
    return paths


def run_species(sp, args):
    cfg = SPECIES[sp]
    out = Path(args.out_dir) / sp
    out.mkdir(parents=True, exist_ok=True)
    report = {"species": sp, "settings": vars(args)}
    log(f"===== {sp} =====")

    # ---- 1. downloads
    paths = download_inputs(sp, args)
    hl_path, dp, cdna_path, gencode_path = paths["hl"], paths["datapack"], paths["cdna"], paths["gencode"]

    # ---- 2. target
    hl = pd.read_csv(hl_path, sep="\t").set_index("ensembl_gene_id")
    pc1, M, imputed, meta = compute_target(hl, cfg, args, report, sp)
    feat = pd.read_csv(dp[cfg["datapack_feat"]], sep="\t", usecols=[0, 1]).set_index("ENSID").HALFLIFE
    saluki_genes = pd.read_csv(dp[cfg["datapack_genes"]], sep="\t", index_col=0).set_index("Gene")
    common = feat.index.intersection(pc1.index)
    report["pc1_vs_saluki"] = {
        "n_common": int(len(common)), "n_saluki_featTable": int(len(feat)),
        "pearson": float(np.corrcoef(pc1[common], feat[common])[0, 1]),
        "max_abs_diff": float(np.abs(pc1[common] - feat[common]).max()),
        "sd_ratio": float(pc1[common].std() / feat[common].std()),
    }
    log(f"  PC1 vs Saluki featTable: r={report['pc1_vs_saluki']['pearson']:.6f}, "
        f"max|diff|={report['pc1_vs_saluki']['max_abs_diff']:.4f}")

    # ---- 3./4. gene structures and sequences
    chosen = parse_chosen_gtf(dp[cfg["datapack_gtf"]])
    seqs, hdr = parse_cdna(cdna_path)
    feat_seqs = {g: (r["5UTR"], r["ORF"], r["3UTR"]) for g, r in
                 pd.read_csv(dp[cfg["datapack_feat"]], sep="\t", usecols=["ENSID", "5UTR", "ORF", "3UTR"],
                             index_col="ENSID").fillna("").iterrows()}
    report["transcript_resolution"] = resolve_transcripts(chosen, seqs, hdr, feat_seqs)
    report["n_chosen_transcripts"] = len(chosen)
    records, status = {}, collections.Counter()
    for gid in pc1.index:
        if gid not in chosen:
            status["no_chosen_protein_coding_transcript"] += 1
            continue
        rec, st = build_record(gid, chosen[gid], seqs, args)
        status[st] += 1
        if rec:
            records[gid] = rec
    report["sequence_build"] = dict(status)

    # gene set and z-scoring reference (Saluki z-scores PC1 over its featTable genes, then drops CDS%3 != 0)
    struct_ok = [g for g, r in records.items() if r["has_utr5"] and r["has_utr3"] and r["has_stop"]]
    rule_final = [g for g in struct_ok if records[g]["has_cds_mod3"]]
    if args.gene_set == "saluki":
        z_ref = feat.index.intersection(pc1.index)
        final = [g for g in saluki_genes.index if g in records]
    else:
        z_ref = pd.Index(struct_ok)
        final = rule_final
    mu, sd = pc1[z_ref].mean(), pc1[z_ref].std()
    sg, ft_set, rule_set = set(saluki_genes.index), set(feat.index), set(rule_final)
    report["gene_sets"] = {
        "after_sample_filter": int(len(pc1)),
        "with_chosen_transcript": int(sum(g in chosen for g in pc1.index)),
        "rule_utr5_utr3_stop": len(struct_ok),
        "rule_plus_cds_mod3": len(rule_final),
        "saluki_featTable": len(ft_set), "saluki_genes_tsv": len(sg),
        "rule_and_saluki": len(rule_set & sg), "rule_only": len(rule_set - sg), "saluki_only": len(sg - rule_set),
        "saluki_genes_missing_here": int(sum(g not in records for g in saluki_genes.index)),
        "used": args.gene_set, "n_final": len(final), "zscore_reference_n": int(len(z_ref)),
    }
    log(f"  gene set '{args.gene_set}': {len(final)} genes (rule: {len(rule_final)}, Saluki: {len(sg)}, "
        f"overlap {len(rule_set & sg)})")

    # ---- 5. names
    id2name, name2chrom = parse_gencode_genes(gencode_path)
    rows = []
    for g in final:
        r = dict(records[g])
        r["gene_gencode"] = id2name.get(g, "")
        rows.append(r)
    df = pd.DataFrame(rows)
    df["gene"] = df["gene_gencode"]
    df["chromosome"] = df["gene"].map(name2chrom).fillna("")
    df["target"] = ((pc1[df.ensembl_gene_id] - mu) / sd).values
    df["pc1"] = pc1[df.ensembl_gene_id].values
    df["target_saluki"] = saluki_genes.Stability.reindex(df.ensembl_gene_id).values
    df["saluki_fold"] = saluki_genes.Fold.reindex(df.ensembl_gene_id).values
    report["names"] = {"without_gencode_name": int((df.gene == "").sum())}
    if not args.keep_unnamed:
        df = df[df.gene != ""].reset_index(drop=True)  # mRNABench: df = df[df["gene"] != ""]
    report["n_rows_output"] = int(len(df))
    have = df.target_saluki.notna()
    report["target_vs_saluki_genes_tsv"] = {
        "n": int(have.sum()),
        "pearson": float(np.corrcoef(df.target[have], df.target_saluki[have])[0, 1]),
        "max_abs_diff": float(np.abs(df.target[have] - df.target_saluki[have]).max()),
    }
    log(f"  output rows: {len(df)}; target vs Saluki genes.tsv r={report['target_vs_saluki_genes_tsv']['pearson']:.6f}")

    # ---- write
    cols = ["gene", "chromosome", "sequence", "cds", "splice", "target", "ensembl_gene_id", "ensembl_transcript_id",
            "chromosome_ensembl", "strand", "length_full", "truncated", "pc1", "target_saluki", "saluki_fold"]
    df[cols].to_parquet(out / f"rnahl-{sp}_reproduced.parquet", index=False)
    M.to_csv(out / "halflife_matrix_filtered.tsv.gz", sep="\t")
    imputed.to_csv(out / "halflife_matrix_imputed_qn.tsv.gz", sep="\t")
    meta.to_csv(out / "sample_metadata.tsv", sep="\t", index=False)
    pd.DataFrame({"pc1": pc1, "saluki_halflife": feat.reindex(pc1.index), "target": (pc1 - mu) / sd,
                  "n_measured_samples": M.notna().sum(1), "in_output": pc1.index.isin(df.ensembl_gene_id)}
                 ).to_csv(out / "pc1.tsv", sep="\t", index_label="ensembl_gene_id")

    # ---- 6. comparisons with the published data
    if not args.skip_compare:
        for key, fn, path in [("vs_orthrus_npz", compare_orthrus, paths["orthrus_npz"]),
                              ("vs_mrnabench", compare_mrnabench, paths["mrnabench"])]:
            log(f"  compare with {Path(path).name}")
            try:
                report[key] = fn(path, df)
            except Exception as e:  # a failed comparison must not discard the reproduced dataset
                log(f"  [Warning] comparison failed: {e!r}")
                report[key] = {"error": repr(e)}
    return report


def write_markdown(reports, path):
    lines = ["# Reproduktion des mRNABench-HWZ-Datensatzes aus Saluki", ""]
    for r in reports:
        g, p = r["gene_sets"], r["pca"]
        lines += [f"## {r['species']}", "", "| Schritt | Gene |", "|---|---|",
                  f"| Saluki-Matrix | {r['n_genes_matrix']} ({r['n_samples']} Proben) |",
                  f"| >= {SPECIES[r['species']]['min_samples']} gemessene Proben | {g['after_sample_filter']} |",
                  f"| mit gewähltem proteinkodierendem Transkript | {g['with_chosen_transcript']} |",
                  f"| Regel: 5'UTR, 3'UTR, Stoppcodon | {g['rule_utr5_utr3_stop']} (Saluki featTable: {g['saluki_featTable']}) |",
                  f"| Regel: + CDS-Länge durch 3 teilbar | {g['rule_plus_cds_mod3']} (Saluki genes.tsv: {g['saluki_genes_tsv']}) |",
                  f"| verwendet ({g['used']}) | {g['n_final']} |",
                  f"| ohne GENCODE-Namen | {r['names']['without_gencode_name']} |",
                  f"| Ausgabe | {r['n_rows_output']} |", "",
                  f"PCA: ncp={p['ncp']}, {p['n_samples_pc1']} Proben, PC1 erklärt {p['pc1_variance_explained']:.3f}; "
                  f"imputePCA konvergiert: {p['impute_converged']}.", "",
                  f"PC1 gegen Saluki-featTable: r={r['pc1_vs_saluki']['pearson']:.6f}, "
                  f"max. Abweichung {r['pc1_vs_saluki']['max_abs_diff']:.4f}.", "",
                  f"Ziel gegen Saluki genes.tsv: r={r['target_vs_saluki_genes_tsv']['pearson']:.6f}, "
                  f"max. Abweichung {r['target_vs_saluki_genes_tsv']['max_abs_diff']:.4f}.", ""]
        for key, title in [("vs_orthrus_npz", "Orthrus npz"), ("vs_mrnabench", "mRNABench parquet")]:
            if key in r:
                v = {k: x for k, x in r[key].items() if k not in ("columns", "ensembl_ids_of_unnamed_orthrus_rows")}
                lines += [f"Vergleich mit {title}: `{json.dumps(v)}`", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    base = Path(os.environ.get("ORTHRUS_BASE", "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(base / "data" / "saluki_hl_reproduction"))
    ap.add_argument("--species", nargs="+", default=["human", "mouse"], choices=list(SPECIES))
    ap.add_argument("--gene_set", default="saluki", choices=["saluki", "rule"],
                    help="saluki: Saluki training genes (genes.tsv); rule: reconstructed filter rule")
    ap.add_argument("--outlier_handling", default="impute_then_drop",
                    choices=["impute_then_drop", "drop_before_impute"],
                    help="human: remove the Gejman samples after imputation (default) or before z-scoring")
    ap.add_argument("--ncp", nargs="*", default=None, metavar="SPECIES=N",
                    help="skip the GCV estimation, e.g. --ncp human=10 mouse=4")
    ap.add_argument("--truncation", default="orthrus", choices=["orthrus", "saluki_paper", "none"],
                    help="orthrus: keep the first 12,288 nt (as in Orthrus/mRNABench); saluki_paper: keep the "
                         "last 12,288 nt; none: full length")
    ap.add_argument("--splice_track", default="orthrus", choices=["orthrus", "exon_end"])
    ap.add_argument("--keep_unnamed", action="store_true", help="keep rows without GENCODE gene name")
    ap.add_argument("--datapack_zip", default=None, help="local copy of the Saluki datasets.zip (optional)")
    ap.add_argument("--skip_compare", action="store_true", help="skip the comparison with Orthrus/mRNABench")
    ap.add_argument("--download_only", action="store_true",
                    help="only fetch the inputs (for login nodes; compute nodes often have no internet)")
    args = ap.parse_args(argv)
    if args.ncp is not None:
        args.ncp = {k: int(v) for k, v in (x.split("=") for x in args.ncp)}
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        import pyarrow  # noqa: F401  (parquet output and mRNABench comparison)
    except ImportError:
        sys.exit("[Error] pyarrow is required (pip install pyarrow).")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.download_only:
        for sp in args.species:
            download_inputs(sp, args)
        log(f"downloads complete: {Path(args.out_dir) / 'raw'}")
        return
    reports = []
    for sp in args.species:
        t = time.time()
        reports.append(run_species(sp, args))
        reports[-1]["runtime_s"] = round(time.time() - t, 1)
    with open(Path(args.out_dir) / "report.json", "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_markdown(reports, Path(args.out_dir) / "report.md")
    log(f"done: {Path(args.out_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
