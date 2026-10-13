#!/usr/bin/env python3
"""
Generate standardized 10-fold gene-grouped split lookup tables using SixTrackDatasetLoader.

Supports:
- hipsc_cm
- mrna_bench (species: human, mouse)

Features:
- Loads canonical transcripts directly via SixTrackDatasetLoader.
- Ensures strict zero-leakage gene grouping (all isoforms of a gene belong to exactly one fold).
- Assigns each transcript a fold index in [0, 1, ..., 9].
- Reproducible via configurable random seed (shuffles unique genes before GroupKFold assignment).
- Produces clean CSV lookup table and detailed distribution diagnostics.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from six_track_dataset import SixTrackDatasetLoader


def generate_10fold_gene_splits(
    df: pd.DataFrame,
    n_splits: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Partitions transcripts into n_splits gene-disjoint folds using shuffled GroupKFold.
    Guarantees:
      1. Every gene belongs to exactly one split.
      2. No gene overlap across splits.
      3. Balanced transcript counts across splits.
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

    # Standardize gene grouping column (prefer gene symbol, else ensembl_gene_id, else transcript_id)
    gene_series = df["gene"].astype(str).str.strip() if "gene" in df.columns else pd.Series([""] * len(df))
    if "ensembl_gene_id" in df.columns:
        gene_series = gene_series.replace("", np.nan).fillna(df["ensembl_gene_id"].astype(str).str.strip())
    gene_series = gene_series.replace("", np.nan).fillna(df["ensembl_transcript_id"])
    df["gene"] = gene_series

    # Handle duplicates if present
    initial_len = len(df)
    df = df.drop_duplicates(subset=["ensembl_transcript_id"]).reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[Warning] Removed {initial_len - len(df)} duplicate transcript entries.")

    print(f"Partitioning {len(df):,} unique transcripts across {df['gene'].nunique():,} unique genes...")

    rng = np.random.default_rng(random_seed)
    unique_genes = df["gene"].unique().copy()
    rng.shuffle(unique_genes)

    # Map genes to shuffled rank order
    gene_rank = {gene: idx for idx, gene in enumerate(unique_genes)}
    df_sorted = df.copy()
    df_sorted["_gene_rank"] = df_sorted["gene"].map(gene_rank)
    df_sorted = df_sorted.sort_values("_gene_rank").reset_index(drop=True)

    # Partition using GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    df_sorted["split"] = -1

    for fold_idx, (_, val_indices) in enumerate(gkf.split(df_sorted, groups=df_sorted["gene"])):
        df_sorted.loc[val_indices, "split"] = fold_idx

    # Assign Fine-Tuning roles:
    # 0-5: Train (60%), 6-7: Val (20%), 8-9: Test (20%)
    role_map = {
        0: "train", 1: "train", 2: "train", 3: "train", 4: "train", 5: "train",
        6: "val",   7: "val",
        8: "test",  9: "test",
    }
    df_sorted["finetune_role"] = df_sorted["split"].map(role_map)

    # Assign Ridge Regression roles:
    # 0-7: CV Pool (80%), 8-9: Test (20%)
    df_sorted["ridge_role"] = df_sorted["split"].apply(lambda s: "test" if s in [8, 9] else "cv_pool")

    # Clean helper columns
    df_sorted = df_sorted.drop(columns=["_gene_rank"])

    # Strict assertion check: verify zero gene leakage between any two splits
    split_genes = {}
    for s in range(n_splits):
        split_genes[s] = set(df_sorted[df_sorted["split"] == s]["gene"])

    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            overlap = split_genes[i].intersection(split_genes[j])
            if len(overlap) > 0:
                raise RuntimeError(
                    f"Gene leakage detected between split {i} and split {j}! Overlapping genes: {overlap}"
                )

    print(f"\n[Verification] Successfully verified: 0 gene overlap across all {n_splits} splits.")
    return df_sorted


def print_split_summary(df_splits: pd.DataFrame, n_splits: int = 10):
    """Prints diagnostic summary of sample and gene distributions across folds."""
    total_samples = len(df_splits)
    total_genes = df_splits["gene"].nunique()

    print("\n" + "=" * 80)
    print("                    10-FOLD GENE-GROUPED SPLIT SUMMARY                    ")
    print("=" * 80)
    header = f"{'Split':<7} | {'Transcripts':<12} | {'% Total':<8} | {'Genes':<9} | {'FineTune Role':<14} | {'Ridge Role':<11}"
    target_col = "target"
    header += f" | {'Target Mean (±std)':<20}"

    print(header)
    print("-" * len(header))

    for s in range(n_splits):
        sub = df_splits[df_splits["split"] == s]
        n_sub = len(sub)
        pct = (n_sub / total_samples) * 100
        n_g = sub["gene"].nunique()
        ft_role = sub["finetune_role"].iloc[0] if "finetune_role" in sub.columns else ""
        rd_role = sub["ridge_role"].iloc[0] if "ridge_role" in sub.columns else ""

        row = f"{s:<7} | {n_sub:<12} | {pct:>6.2f}% | {n_g:<9} | {ft_role:<14} | {rd_role:<11}"
        val_mean = sub[target_col].mean()
        val_std = sub[target_col].std()
        row += f" | {val_mean:>7.4f} (±{val_std:.4f})"
        print(row)

    print("=" * 80)

    # Grouped role summary
    ft_summary = df_splits.groupby("finetune_role").agg(
        Transcripts=("ensembl_transcript_id", "count"),
        Genes=("gene", "nunique"),
    )
    ft_summary["% Transcripts"] = (ft_summary["Transcripts"] / total_samples) * 100
    print("\nFine-Tuning Role Aggregates:")
    print(ft_summary.to_string())
    print("\nRidge Regression Role Aggregates:")
    rd_summary = df_splits.groupby("ridge_role").agg(
        Transcripts=("ensembl_transcript_id", "count"),
        Genes=("gene", "nunique"),
    )
    rd_summary["% Transcripts"] = (rd_summary["Transcripts"] / total_samples) * 100
    print(rd_summary.to_string())
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Standardized 10-Fold Gene-Grouped Split Lookup Table using SixTrackDatasetLoader."
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
        help="Random seed for gene shuffling before GroupKFold (default: 42)",
    )
    args = parser.parse_args()

    if args.dataset == "mrna_bench" and not args.species:
        parser.error("--species ('human' or 'mouse') is required when --dataset is 'mrna_bench'.")
    if args.dataset == "hipsc_cm" and args.species:
        print(f"[Notice] --species '{args.species}' was provided but hipsc_cm is human-specific. Ignoring --species.")

    return args


def main():
    args = parse_args()

    # Determine default output path if not specified

    if args.dataset == "hipsc_cm":
        output_path = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")
    else:
        output_path = Path(f"/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench/rnahl-{args.species}_10folds_lookup.csv")

    print("=" * 80)
    print("        STANDARDIZED 10-FOLD GENE-GROUPED SPLIT TABLE GENERATION        ")
    print("=" * 80)
    print(f"Dataset:         {args.dataset}")
    if args.species:
        print(f"Species:         {args.species}")
    print(f"Output path:     {output_path}")
    print(f"Number of folds: {args.n_splits}")
    print(f"Random seed:     {args.random_seed}")

    # Load canonical dataset via SixTrackDatasetLoader
    print(f"\n[Loader] Loading dataset via SixTrackDatasetLoader...")
    loader = SixTrackDatasetLoader(dataset=args.dataset, species=args.species)
    df = loader.get_data()
    print(f"[Loader] Loaded {len(df):,} transcripts.")

    # Generate splits
    df_splits = generate_10fold_gene_splits(
        df=df,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )

    # Print summary diagnostics
    print_split_summary(df_splits, n_splits=args.n_splits)

    # Output selection: save essential lookup columns and identifiers
    cols_to_save = ["ensembl_transcript_id", "gene", "split", "finetune_role", "ridge_role"]
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
        "n_splits": args.n_splits,
        "random_seed": args.random_seed,
        "total_transcripts": len(df_splits),
        "total_genes": int(df_splits["gene"].nunique()),
        "fine_tuning_roles": {
            "train": [0, 1, 2, 3, 4, 5],
            "val": [6, 7],
            "test": [8, 9],
        },
        "ridge_cv_folds": [
            {"name": "Fold 1", "train_splits": [0, 1, 2, 3, 4, 5], "val_splits": [6, 7]},
            {"name": "Fold 2", "train_splits": [2, 3, 4, 5, 6, 7], "val_splits": [0, 1]},
            {"name": "Fold 3", "train_splits": [0, 1, 4, 5, 6, 7], "val_splits": [2, 3]},
            {"name": "Fold 4", "train_splits": [0, 1, 2, 3, 6, 7], "val_splits": [4, 5]},
        ],
        "ridge_test_splits": [8, 9],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata summary saved to: {meta_path}")


if __name__ == "__main__":
    main()
