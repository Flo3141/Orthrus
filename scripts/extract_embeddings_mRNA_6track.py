#!/usr/bin/env python3
"""
Extraktion von Orthrus 6-Track Embeddings fuer mrna-bench (Halbwertszeit-Datensaetze).
Laeuft auf dem GPU-Cluster.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel
import mrna_bench as mb


def seq_to_one_hot(seq: str) -> np.ndarray:
    """
    Konvertiert eine RNA/DNA-Sequenz in ein 4-Kanal One-Hot-Encoding.
    Konform mit dem Orthrus Paper:
      Kanal 0: A (Adenin)
      Kanal 1: C (Cytosin)
      Kanal 2: G (Guanin)
      Kanal 3: T / U (Thymin / Uracil)
      Alle anderen Zeichen (z.B. 'N') -> [0, 0, 0, 0]
    
    Returns:
        np.ndarray der Form (L, 4) mit dtype float32.
    """
    seq_bytes = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    oh = np.zeros((len(seq_bytes), 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0  # 'A'
    oh[seq_bytes == 67, 1] = 1.0  # 'C'
    oh[seq_bytes == 71, 2] = 1.0  # 'G'
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # 'T' (84) oder 'U' (85)
    return oh


def parse_binary_track(track, length: int) -> np.ndarray:
    """
    Wandelt Track-Spalten (CDS oder Splice) in einen 1D-Float32-Array der Laenge `length` um.
    Unterstuetzt sowohl Strings ("00100..."), Listen als auch vorhandene Numpy-Arrays.
    """
    if isinstance(track, str):
        arr = np.array([float(c) for c in track], dtype=np.float32)
    elif isinstance(track, (list, np.ndarray)):
        arr = np.asarray(track, dtype=np.float32)
    else:
        raise TypeError(f"Nicht unterstuetzter Track-Typ: {type(track)}")

    if len(arr) != length:
        # Falls Laenge abweicht, anpassen / pad / trim
        if len(arr) < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[:len(arr)] = arr
            arr = padded
        else:
            arr = arr[:length]
            
    return arr.reshape(-1, 1)


def build_six_track(seq: str, cds, splice) -> np.ndarray:
    """
    Erstellt das vollstaendige 6-Track Array (L, 6):
      Tracks 0-3: A, C, G, T/U (One-Hot)
      Track 4:    CDS (binär)
      Track 5:    Splice-Site (binär)
    """
    seq_oh = seq_to_one_hot(seq)
    l = len(seq)
    cds_track = parse_binary_track(cds, l)
    splice_track = parse_binary_track(splice, l)
    
    # Concatenate zu (L, 6)
    six_track = np.concatenate([seq_oh, cds_track, splice_track], axis=1)
    return six_track


def extract_embeddings_for_dataset(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288
) -> dict:
    """
    Extrahiert Embeddings fuer einen DataFrame mit dynamischem Laengen-Batching.
    """
    print(f"Verarbeite Datensatz mit {len(df)} Eintraegen...")
    
    # Filtern / Clamping auf max_length gemaess Paper (Orthrus schliesst >12288 aus)
    sample_data = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        seq = str(row["sequence"])
        if len(seq) > max_length:
            skipped_count += 1
            seq = seq[:max_length]
            cds = row["cds"][:max_length] if hasattr(row["cds"], "__getitem__") else row["cds"]
            splice = row["splice"][:max_length] if hasattr(row["splice"], "__getitem__") else row["splice"]
        else:
            cds = row["cds"]
            splice = row["splice"]

        six_track = build_six_track(seq, cds, splice)
        sample_data.append({
            "orig_idx": idx,
            "track": six_track,
            "length": six_track.shape[0],
            "gene": str(row.get("gene", "")),
            "chromosome": str(row.get("chromosome", "")),
            "target": float(row.get("target", np.nan))
        })
        
    if skipped_count > 0:
        print(f"Hinweis: {skipped_count} Sequenzen wurden auf {max_length} bp gekuerzt.")

    # Sortieren nach Laenge, um Padding-Overhead pro Batch zu minimieren
    sorted_samples = sorted(sample_data, key=lambda x: x["length"])
    
    embeddings_list = [None] * len(sample_data)
    
    print(f"Starte Embedding-Extraktion mit Batch-Groesse {batch_size}...")
    for i in tqdm(range(0, len(sorted_samples), batch_size), desc="Extrahiere Embeddings"):
        batch = sorted_samples[i : i + batch_size]
        b_lens = [s["length"] for s in batch]
        max_b_len = max(b_lens)
        
        # Padded Batch Tensor (Batch, max_b_len, 6)
        batch_arr = np.zeros((len(batch), max_b_len, 6), dtype=np.float32)
        for b_idx, s in enumerate(batch):
            l = s["length"]
            batch_arr[b_idx, :l, :] = s["track"]
            
        x_tensor = torch.from_numpy(batch_arr).to(device)
        lengths_tensor = torch.tensor(b_lens, dtype=torch.long, device=device)
        
        with torch.no_grad():
            # channel_last=True erwartet (B, L, C) mit C=6
            batch_emb = model.representation(x_tensor, lengths_tensor, channel_last=True)
            batch_emb_np = batch_emb.cpu().numpy()
            
        for b_idx, s in enumerate(batch):
            orig_i = s["orig_idx"]
            embeddings_list[orig_i] = batch_emb_np[b_idx]
            
    all_embeddings = np.stack(embeddings_list, axis=0)
    all_targets = df["target"].values.astype(np.float32)
    all_genes = df["gene"].astype(str).values
    all_chromosomes = df["chromosome"].astype(str).values
    seq_lens = df["sequence"].str.len().values
    
    return {
        "embeddings": all_embeddings,
        "targets": all_targets,
        "genes": all_genes,
        "chromosomes": all_chromosomes,
        "seq_lens": seq_lens
    }


def main():
    parser = argparse.ArgumentParser(description="Extrahiere Orthrus 6-Track Embeddings")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/mrna_bench",
        help="Pfad zum mrna_bench Datenverzeichnis"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optionales zentrales Ausgabeverzeichnis. Wenn None (Default), wird direkt in <data_dir>/<dataset_key>/embeddings/ gespeichert."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Hugging Face Modell-Identifier (z.B. quietflamingo/orthrus-large-6-track oder antichronology/orthrus-6-track)"
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["human", "mouse", "both"],
        default="both",
        help="Welche Spezies extrahiert werden soll (human, mouse oder both)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch-Groesse fuer die Inferenz"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximale Sequenzlaenge (gemaess Paper max 12288 bp)"
    )
    args = parser.parse_args()

    # Pfade vorbereiten
    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Registriere mrna-bench Pfad: {data_path}")
    mb.update_data_path(str(data_path))

    # Device & Modell initialisieren
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Geraet: {device}")
    
    print(f"Lade Orthrus 6-Track Modell '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Modell erfolgreich geladen.")

    species_to_process = []
    if args.species in ["human", "both"]:
        species_to_process.append(("human", "rnahl-human"))
    if args.species in ["mouse", "both"]:
        species_to_process.append(("mouse", "rnahl-mouse"))

    for spec_name, dataset_key in species_to_process:
        print(f"\n==================== {spec_name.upper()} DATASET ====================")
        print(f"Lade {dataset_key} ueber mrna-bench...")
        df = mb.load_dataset(dataset_key).data_df
        print(f"Geladene Zeilen: {len(df)}")
        print(f"Vorhandene Spalten: {list(df.columns)}")

        result = extract_embeddings_for_dataset(
            df=df,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        if args.output_dir is not None:
            spec_out_dir = Path(args.output_dir)
            save_file = spec_out_dir / f"orthrus_6track_embeddings_{spec_name}.npz"
        else:
            # Standard: Direkt in <data_dir>/<dataset_key>/embeddings/
            spec_out_dir = data_path / dataset_key / "embeddings"
            save_file = spec_out_dir / "orthrus_6track_embeddings.npz"

        spec_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Speichere Embeddings nach: {save_file}")
        np.savez_compressed(
            save_file,
            embeddings=result["embeddings"],
            targets=result["targets"],
            genes=result["genes"],
            chromosomes=result["chromosomes"],
            seq_lens=result["seq_lens"]
        )
        print(f"Erfolgreich gespeichert! Embedding Shape: {result['embeddings'].shape}")

    print("\nAlle Embeddings wurden erfolgreich extrahiert und gespeichert.")


if __name__ == "__main__":
    main()
