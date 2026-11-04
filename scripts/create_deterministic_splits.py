#!/usr/bin/env python3
"""
Generate standardized 10-fold homology-grouped split lookup tables using SixTrackDatasetLoader.

Supports:
- hipsc_cm
- mrna_bench (species: human, mouse)

Splitting strategy (homology-aware, following the homology split of mRNABench):
- Paralog pairs are taken from Ensembl Compara (protein gene trees, release COMPARA_RELEASE).
  Only within-species paralogs with >= MIN_IDENTITY % identity in both directions are kept
  (35 % as in the mRNABench homology split, Shi et al. 2025, bioRxiv 10.1101/2025.07.05.662870).
- Homology groups are the connected components of the paralog graph over all genes of the species
  (transitive: A~B and B~C puts A, B and C into one group). Since Compara reports paralogs only within
  a gene tree, a group never spans two gene trees.
- Transcripts are assigned to groups via their Ensembl gene ID. Genes without paralogs form their own group,
  so all isoforms of a gene always stay together.
- Groups are distributed over n_splits folds: shuffled with the random seed, then largest groups first,
  each to the fold with the fewest transcripts (balanced, seed-dependent).
- Verifies that no homology group and no gene occurs in more than one split.

Fold roles (unchanged):
- Fine-tuning: splits 0-5 train, 6-7 val, 8-9 test
- Ridge regression: splits 0-7 CV pool, 8-9 test
"""

import argparse
import gzip
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from six_track_dataset import SixTrackDatasetLoader
from util import CV_FOLDS, TEST_SPLITS, DATA_DIR, SPLITS_LOOKUP_TABLES


COMPARA_RELEASE = 108  # matches the Ensembl 108 GTF used for the trans-factor tracks
COMPARA_URL = (
    "https://ftp.ensembl.org/pub/release-{release}/tsv/ensembl-compara/homologies/"
    "{species}/Compara.{release}.protein_default.homologies.tsv.gz"
)
PARALOG_CACHE_DIR = DATA_DIR / "homology"
PARALOG_COLUMNS = [
    "gene_stable_id", "identity", "homology_type",
    "homology_gene_stable_id", "homology_species", "homology_identity",
]
MIN_IDENTITY = 35.0

COMPARA_SPECIES = {"human": "homo_sapiens", "mouse": "mus_musculus"}
GENE_ID_PREFIXES = ("ENSG", "ENSMUSG")


# =============================================================================
# 1. Ensembl Compara Paralogs -> Homology Groups
# =============================================================================

def download_paralog_table(species: str, out_path: Path, release: int = COMPARA_RELEASE) -> Path:
    """
    Streams the Ensembl Compara homology dump of a species and keeps only within-species paralog pairs
    (the full file also contains orthologs to ~200 species and is ~150 MB compressed).
    """
    url = COMPARA_URL.format(release=release, species=species)
    print(f"[Compara] Downloading paralogs from {url} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "MScThesis/1.0"})
    n_kept = 0
    with urllib.request.urlopen(req) as resp, gzip.open(resp, "rt") as f_in, open(tmp_path, "w") as f_out:
        header = f_in.readline().rstrip("\n").split("\t")
        col_idx = [header.index(c) for c in PARALOG_COLUMNS]
        type_idx, species_idx = header.index("homology_type"), header.index("homology_species")
        f_out.write("\t".join(PARALOG_COLUMNS) + "\n")
        for line in f_in:
            parts = line.rstrip("\n").split("\t")
            if "paralog" in parts[type_idx] and parts[species_idx] == species:
                f_out.write("\t".join(parts[i] for i in col_idx) + "\n")
                n_kept += 1

    tmp_path.rename(out_path)
    print(f"[Compara] Saved {n_kept:,} paralog pairs to: {out_path}")
    return out_path


def load_homology_groups(paralog_table: Path, species: str, min_identity: float = MIN_IDENTITY) -> tuple:
    """
    Builds homology groups as connected components of the paralog graph over all genes of the species.

    Returns:
        gene_to_group: dict mapping Ensembl gene ID (unversioned) -> group index (genes with >= 1 paralog only)
        stats: summary of the paralog graph
    """
    par = pd.read_csv(paralog_table, sep="\t", usecols=PARALOG_COLUMNS)
    par = par[par["homology_type"].str.contains("paralog") & (par["homology_species"] == species)]
    n_pairs_total = len(par)
    par = par[(par["identity"] >= min_identity) & (par["homology_identity"] >= min_identity)]

    genes = pd.Index(pd.unique(pd.concat([par["gene_stable_id"], par["homology_gene_stable_id"]])))
    a = genes.get_indexer(par["gene_stable_id"])
    b = genes.get_indexer(par["homology_gene_stable_id"])
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(genes), len(genes)))
    n_groups, labels = connected_components(graph, directed=False)

    group_sizes = np.bincount(labels)
    stats = {
        "paralog_pairs_total": int(n_pairs_total),
        "paralog_pairs_used": int(len(par)),
        "genes_with_paralogs": int(len(genes)),
        "paralog_groups": int(n_groups),
        "largest_paralog_groups_genes": sorted(group_sizes.tolist(), reverse=True)[:5],
    }
    print(
        f"[Homology] {stats['paralog_pairs_used']:,} of {n_pairs_total:,} paralog pairs with identity >= {min_identity}% "
        f"-> {n_groups:,} groups over {len(genes):,} genes (largest: {stats['largest_paralog_groups_genes']})"
    )
    return dict(zip(genes, labels.tolist())), stats


def assign_homology_groups(df: pd.DataFrame, gene_to_group: dict) -> tuple:
    """
    Assigns every transcript a homology group label:
      - 'HG<n>'          gene has paralogs (shared label for all genes of the component)
      - 'GENE:<id>'      valid Ensembl gene ID without paralogs
      - 'SYMBOL:<name>'  no valid gene ID, grouped by gene symbol
      - 'TX:<id>'        neither gene ID nor symbol available
    """
    gene_ids = df["ensembl_gene_id"].astype(str).str.strip().str.split(".").str[0] \
        if "ensembl_gene_id" in df.columns else pd.Series([""] * len(df), index=df.index)
    symbols = df["gene"].astype(str).str.strip() if "gene" in df.columns else pd.Series([""] * len(df), index=df.index)

    valid_id = gene_ids.str.startswith(GENE_ID_PREFIXES)
    valid_symbol = ~symbols.isin(["", "nan", "None"])
    in_graph = gene_ids.map(lambda g: g in gene_to_group)

    groups = np.where(
        in_graph, "HG" + gene_ids.map(gene_to_group).astype("Int64").astype(str),
        np.where(
            valid_id, "GENE:" + gene_ids,
            np.where(valid_symbol, "SYMBOL:" + symbols, "TX:" + df["ensembl_transcript_id"].astype(str)),
        ),
    )

    # Gene key used for the gene-level leakage check (gene ID preferred over symbol)
    gene_key = np.where(valid_id, gene_ids, np.where(valid_symbol, symbols, df["ensembl_transcript_id"].astype(str)))

    stats = {
        "transcripts_with_paralog_group": int(in_graph.sum()),
        "transcripts_valid_gene_id": int(valid_id.sum()),
        "transcripts_symbol_fallback": int((~valid_id & valid_symbol).sum()),
        "transcripts_transcript_fallback": int((~valid_id & ~valid_symbol).sum()),
    }
    return pd.Series(groups, index=df.index), pd.Series(gene_key, index=df.index), stats


# =============================================================================
# 2. Group-Disjoint 10-Fold Assignment
# =============================================================================

def assign_groups_to_folds(groups: pd.Series, n_splits: int, random_seed: int) -> pd.Series:
    """
    Balanced, seed-dependent assignment of whole groups to folds:
    groups are shuffled, then sorted by size (largest first, stable), and each group is put into the fold
    that currently holds the fewest transcripts.
    """
    sizes = groups.value_counts()
    rng = np.random.default_rng(random_seed)
    shuffled = sizes.iloc[rng.permutation(len(sizes))]
    ordered = shuffled.iloc[np.argsort(-shuffled.to_numpy(), kind="stable")]

    fold_counts = np.zeros(n_splits, dtype=np.int64)
    group_to_fold = {}
    for group, size in ordered.items():
        fold = int(np.argmin(fold_counts))
        group_to_fold[group] = fold
        fold_counts[fold] += size

    return groups.map(group_to_fold).astype(int)


def generate_10fold_homology_splits(
    df: pd.DataFrame,
    gene_to_group: dict,
    n_splits: int = 10,
    random_seed: int = 42,
) -> tuple:
    """
    Partitions transcripts into n_splits homology-group-disjoint folds.
    Guarantees:
      1. Every homology group (and therefore every gene) belongs to exactly one split.
      2. Balanced transcript counts across splits.
    """
    df = df.copy()

    # Standardize transcript ID column
    tx_col = None
    for cand in ["ensembl_transcript_id", "transcript_id", "tx_id"]:
        if cand in df.columns:
            tx_col = cand
            break
    if tx_col is None:
        raise KeyError(f"Could not find transcript ID column in {list(df.columns)}")
    df["ensembl_transcript_id"] = df[tx_col].astype(str).str.strip()

    # Strict validation: Every transcript must have a verified Ensembl transcript ID (ENST... / ENSMUST...)
    invalid_tx = (
        (df["ensembl_transcript_id"] == "")
        | (df["ensembl_transcript_id"] == "nan")
        | (~df["ensembl_transcript_id"].str.startswith(("ENST", "ENSMUST")))
    )
    if invalid_tx.any():
        num_invalid = invalid_tx.sum()
        invalid_sample = df.loc[invalid_tx, "gene"].head(5).tolist() if "gene" in df.columns else []
        raise RuntimeError(
            f"[Error] {num_invalid} transcripts in dataset have invalid or missing Ensembl transcript IDs "
            f"(sample genes: {invalid_sample})! All transcripts must have verified ENST/ENSMUST IDs. "
            f"No fallbacks allowed."
        )

    # Handle duplicates if present
    initial_len = len(df)
    df = df.drop_duplicates(subset=["ensembl_transcript_id"]).reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[Warning] Removed {initial_len - len(df)} duplicate transcript entries.")

    df["homology_group"], df["gene_key"], assign_stats = assign_homology_groups(df, gene_to_group)
    print(
        f"Partitioning {len(df):,} transcripts across {df['gene_key'].nunique():,} genes "
        f"in {df['homology_group'].nunique():,} homology groups..."
    )
    print(
        f"[Homology] Transcripts in paralog groups: {assign_stats['transcripts_with_paralog_group']:,} | "
        f"valid gene ID: {assign_stats['transcripts_valid_gene_id']:,} | "
        f"symbol fallback: {assign_stats['transcripts_symbol_fallback']:,} | "
        f"transcript fallback: {assign_stats['transcripts_transcript_fallback']:,}"
    )

    df["split"] = assign_groups_to_folds(df["homology_group"], n_splits, random_seed)

    # Assign Fine-Tuning roles (as in CV fold 0):
    # 0-5: Train (60%), 6-7: Val (20%), 8-9: Test (20%)
    role_map = {s: "train" for s in CV_FOLDS[0]["train"]}
    role_map.update({s: "val" for s in CV_FOLDS[0]["val"]})
    role_map.update({s: "test" for s in TEST_SPLITS})
    df["finetune_role"] = df["split"].map(role_map)

    # Assign Ridge Regression roles:
    # 0-7: CV Pool (80%), 8-9: Test (20%)
    df["ridge_role"] = df["split"].apply(lambda s: "test" if s in TEST_SPLITS else "cv_pool")

    # Strict assertion check: every homology group and every gene must lie in exactly one split
    for key in ["homology_group", "gene_key"]:
        n_splits_per_key = df.groupby(key)["split"].nunique()
        leaking = n_splits_per_key[n_splits_per_key > 1]
        if len(leaking) > 0:
            raise RuntimeError(f"Leakage detected: {len(leaking)} values of '{key}' span multiple splits: {leaking.index[:5].tolist()}")

    print(f"\n[Verification] Successfully verified: every homology group and every gene lies in exactly one of {n_splits} splits.")

    group_tx = df["homology_group"].value_counts()
    assign_stats["homology_groups"] = int(len(group_tx))
    assign_stats["multi_gene_groups"] = int((df.groupby("homology_group")["gene_key"].nunique() > 1).sum())
    assign_stats["largest_group_transcripts"] = int(group_tx.max())
    assign_stats["largest_group_share"] = float(group_tx.max() / len(df))
    return df, assign_stats


def print_split_summary(df_splits: pd.DataFrame, n_splits: int = 10):
    """Prints diagnostic summary of sample, gene and homology group distributions across folds."""
    total_samples = len(df_splits)

    print("\n" + "=" * 100)
    print("                    10-FOLD HOMOLOGY-GROUPED SPLIT SUMMARY                    ")
    print("=" * 100)
    header = (
        f"{'Split':<7} | {'Transcripts':<12} | {'% Total':<8} | {'Genes':<9} | {'Hom. Groups':<11} | "
        f"{'FineTune Role':<14} | {'Ridge Role':<11} | {'Target Mean (±std)':<20}"
    )
    print(header)
    print("-" * len(header))

    for s in range(n_splits):
        sub = df_splits[df_splits["split"] == s]
        pct = (len(sub) / total_samples) * 100
        row = (
            f"{s:<7} | {len(sub):<12} | {pct:>6.2f}% | {sub['gene_key'].nunique():<9} | "
            f"{sub['homology_group'].nunique():<11} | {sub['finetune_role'].iloc[0]:<14} | {sub['ridge_role'].iloc[0]:<11} | "
            f"{sub['target'].mean():>7.4f} (±{sub['target'].std():.4f})"
        )
        print(row)

    print("=" * 100)

    for role_col, title in [("finetune_role", "Fine-Tuning Role Aggregates:"), ("ridge_role", "Ridge Regression Role Aggregates:")]:
        summary = df_splits.groupby(role_col).agg(
            Transcripts=("ensembl_transcript_id", "count"),
            Genes=("gene_key", "nunique"),
            Homology_Groups=("homology_group", "nunique"),
        )
        summary["% Transcripts"] = (summary["Transcripts"] / total_samples) * 100
        print(f"\n{title}")
        print(summary.to_string())
    print("=" * 100 + "\n")


# =============================================================================
# 3. Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Standardized 10-Fold Homology-Grouped Split Lookup Table using SixTrackDatasetLoader."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hipsc_cm", "mrna_bench"],
        help="Dataset name: 'hipsc_cm' or 'mrna_bench' (required)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        choices=["human", "mouse"],
        help="Species for mrna_bench: 'human' or 'mouse'. Required if dataset is 'mrna_bench'.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Number of folds (default: 10)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for shuffling homology groups before balanced fold assignment (default: 42)",
    )
    parser.add_argument(
        "--min_identity",
        type=float,
        default=MIN_IDENTITY,
        help=f"Minimum paralog protein identity in %% (both directions) to link genes (default: {MIN_IDENTITY})",
    )
    parser.add_argument(
        "--paralog_table",
        type=str,
        default=None,
        help="Path to a filtered Compara paralog table (default: cached download in "
             f"{PARALOG_CACHE_DIR}/<species>_paralogs_compara{COMPARA_RELEASE}.tsv)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional override of the dataset path passed to SixTrackDatasetLoader",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional override of the output lookup CSV path",
    )
    args = parser.parse_args()

    if args.dataset == "mrna_bench" and not args.species:
        parser.error("--species ('human' or 'mouse') is required when --dataset is 'mrna_bench'.")
    if args.dataset == "hipsc_cm" and args.species:
        print(f"[Notice] --species '{args.species}' was provided but hipsc_cm is human-specific. Ignoring --species.")
        args.species = None

    return args


def main():
    args = parse_args()

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = SPLITS_LOOKUP_TABLES["hipsc_cm" if args.dataset == "hipsc_cm" else f"mrna_bench_{args.species}"]

    compara_species = COMPARA_SPECIES[args.species or "human"]
    paralog_table = (
        Path(args.paralog_table) if args.paralog_table
        else PARALOG_CACHE_DIR / f"{compara_species}_paralogs_compara{COMPARA_RELEASE}.tsv"
    )

    print("=" * 80)
    print("        STANDARDIZED 10-FOLD HOMOLOGY-GROUPED SPLIT TABLE GENERATION        ")
    print("=" * 80)
    print(f"Dataset:         {args.dataset}")
    if args.species:
        print(f"Species:         {args.species}")
    print(f"Output path:     {output_path}")
    print(f"Number of folds: {args.n_splits}")
    print(f"Random seed:     {args.random_seed}")
    print(f"Paralog table:   {paralog_table} (Ensembl Compara {COMPARA_RELEASE}, {compara_species})")
    print(f"Min. identity:   {args.min_identity}%")

    # Homology groups from Ensembl Compara paralogs
    if not paralog_table.is_file():
        download_paralog_table(compara_species, paralog_table)
    gene_to_group, graph_stats = load_homology_groups(paralog_table, compara_species, args.min_identity)

    # Load canonical dataset via SixTrackDatasetLoader
    print(f"\n[Loader] Loading dataset via SixTrackDatasetLoader...")
    loader = SixTrackDatasetLoader(dataset=args.dataset, species=args.species, data_path=args.data_path)
    df = loader.get_data()
    print(f"[Loader] Loaded {len(df):,} transcripts.")

    # Generate splits
    df_splits, assign_stats = generate_10fold_homology_splits(
        df=df,
        gene_to_group=gene_to_group,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )

    # Print summary diagnostics
    print_split_summary(df_splits, n_splits=args.n_splits)

    # Output selection: save essential lookup columns and identifiers
    cols_to_save = ["ensembl_transcript_id", "gene", "split", "finetune_role", "ridge_role", "homology_group"]
    for extra_col in ["ensembl_gene_id", "target", "target_raw"]:
        if extra_col in df_splits.columns and extra_col not in cols_to_save:
            cols_to_save.append(extra_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_splits[cols_to_save].to_csv(output_path, index=False)
    print(f"\nLookup table saved to: {output_path}")

    # Also save metadata json
    meta_path = output_path.with_suffix(".json")
    meta = {
        "dataset": args.dataset,
        "species": args.species,
        "output_path": str(output_path),
        "split_type": "homology",
        "n_splits": args.n_splits,
        "random_seed": args.random_seed,
        "total_transcripts": len(df_splits),
        "total_genes": int(df_splits["gene_key"].nunique()),
        "homology": {
            "source": f"Ensembl Compara {COMPARA_RELEASE} protein gene trees (within-species paralogs)",
            "url": COMPARA_URL.format(release=COMPARA_RELEASE, species=compara_species),
            "paralog_table": str(paralog_table),
            "min_identity_both_directions": args.min_identity,
            "grouping": "connected components of the paralog graph over all genes of the species",
            **graph_stats,
            **assign_stats,
        },
        "fine_tuning_roles": {**CV_FOLDS[0], "test": TEST_SPLITS},
        "ridge_cv_folds": [
            {"name": f"Fold {k}", "train_splits": f["train"], "val_splits": f["val"]} for k, f in CV_FOLDS.items()
        ],
        "ridge_test_splits": TEST_SPLITS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata summary saved to: {meta_path}")


if __name__ == "__main__":
    main()
