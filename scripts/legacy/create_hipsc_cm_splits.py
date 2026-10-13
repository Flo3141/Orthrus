#!/usr/bin/env python3
"""
Generate a standardized 10-fold gene-grouped split lookup table for the hIPSC_CM dataset.

Features:
- Ensures strict zero-leakage gene grouping (all isoforms of a gene belong to exactly one fold).
- Assigns each transcript a fold index in [0, 1, ..., 9].
- Reproducible via configurable random seed (shuffles unique genes before GroupKFold assignment).
- Formats roles:
    - Fine-Tuning: Train (0-5, 60%), Val (6-7, 20%), Test (8-9, 20%)
    - Ridge Regression: CV Pool (0-7, 80%), Test (8-9, 20%)
- Can load from either raw TSV (e.g. hIPSC_CM_ej_cds_transformed.txt) or NPZ archives.
- Produces clean CSV lookup table and detailed distribution diagnostics.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Loads transcripts and gene mappings from TSV, CSV, or NPZ file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    print(f"Loading data from: {input_path} ({suffix})")

    if suffix in [".txt", ".tsv", ".csv"]:
        sep = "\t" if suffix in [".txt", ".tsv"] else ","
        df = pd.read_csv(input_path, sep=sep)
    elif suffix == ".npz":
        npz = np.load(input_path, allow_pickle=True)
        data_dict = {}
        for key in npz.files:
            arr = npz[key]
            # Exclude large multidimensional feature matrices from the metadata table
            if arr.ndim == 1:
                data_dict[key] = arr
            elif arr.ndim == 2 and arr.shape[1] == 1:
                data_dict[key] = arr.squeeze(1)
        df = pd.DataFrame(data_dict)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .txt, .tsv, .csv, or .npz")

    # Standardize transcript ID column
    tx_col = None
    for cand in ["ensembl_transcript_id", "transcript_id", "tx_id"]:
        if cand in df.columns:
            tx_col = cand
            break
    if tx_col is None:
        raise KeyError(f"Could not find transcript ID column in {list(df.columns)}")
    df["ensembl_transcript_id"] = df[tx_col].astype(str)

    # Standardize gene grouping column (prefer hgnc_symbol, else ensembl_gene_id, else gene)
    gene_col = None
    for cand in ["hgnc_symbol", "ensembl_gene_id", "gene", "genes"]:
        if cand in df.columns:
            gene_col = cand
            break
    if gene_col is None:
        raise KeyError(f"Could not find gene column in {list(df.columns)}")
    df["gene"] = df[gene_col].astype(str)

    # Clean empty / whitespace strings
    df["ensembl_transcript_id"] = df["ensembl_transcript_id"].str.strip()
    df["gene"] = df["gene"].str.strip()

    # Handle duplicates if present
    initial_len = len(df)
    df = df.drop_duplicates(subset=["ensembl_transcript_id"]).reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[Warning] Removed {initial_len - len(df)} duplicate transcript entries.")

    print(f"Loaded {len(df)} unique transcripts across {df['gene'].nunique()} unique genes.")
    return df


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
    target_col = None
    for cand in ["half_life_transformed", "half_life"]:
        if cand in df_splits.columns:
            target_col = cand
            break

    if target_col:
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
        if target_col:
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
        description="Generate 10-Fold Gene-Grouped Split Lookup Table for hIPSC_CM dataset."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt",
        help="Path to source dataset file (TSV, CSV, or NPZ)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv",
        help="Destination CSV path for lookup table",
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
        help="Random seed for gene shuffling before GroupKFold",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Load dataset
    df = load_dataset(input_path)

    # Generate splits
    df_splits = generate_10fold_gene_splits(
        df=df,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )

    # Print summary
    print_split_summary(df_splits, n_splits=args.n_splits)

    # Output selection: save essential lookup columns and any available identifiers
    cols_to_save = ["ensembl_transcript_id", "gene", "split", "finetune_role", "ridge_role"]
    for extra_col in ["hgnc_symbol", "ensembl_gene_id", "transcript_biotype", "half_life_transformed", "half_life"]:
        if extra_col in df_splits.columns and extra_col not in cols_to_save:
            cols_to_save.append(extra_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_splits[cols_to_save].to_csv(output_path, index=False)
    print(f"Lookup table saved to: {output_path}")

    # Also save metadata json
    meta_path = output_path.with_suffix(".json")
    meta = {
        "input_path": str(input_path),
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
