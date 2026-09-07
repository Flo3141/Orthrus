#!/usr/bin/env python3
"""
Extraktion von Orthrus 6-Track Embeddings fuer den Saluki-Datensatz (saluki_ej_cds.txt).
Laeuft auf dem GPU-Cluster.

Track-Konstruktion aus Saluki-Tokens:
- Kanäle 0-3: A, C, G, T/U (4-Kanal One-Hot Encoding)
- Kanal 4:    CDS Track (1.0 an Großbuchstaben wie 'A' in 'A,t,t', entspricht exakt dem Frame-0-Codon-Start cds[0::3]=1 in Orthrus)
- Kanal 5:    Splice Track (1.0 an Tokens mit 'ej' Suffix, markiert die Exon-Junction-Grenzen)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel


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
    seq_bytes = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    oh = np.zeros((len(seq_bytes), 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0  # 'A'
    oh[seq_bytes == 67, 1] = 1.0  # 'C'
    oh[seq_bytes == 71, 2] = 1.0  # 'G'
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # 'T' (84) oder 'U' (85)
    return oh


def parse_saluki_sequence_to_six_track(raw_seq: str) -> np.ndarray:
    """
    Parst die kommagetrennte Saluki-Sequenz und erzeugt ein (L, 6) Array:
      - Tracks 0-3: A, C, G, T/U One-Hot
      - Track 4:    CDS-Marker (1.0 bei Großbuchstaben, 0.0 bei Kleinbuchstaben)
      - Track 5:    Splice-Marker (1.0 bei 'ej' Tokens, 0.0 sonst)
    """
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    if not tokens:
        return np.zeros((0, 6), dtype=np.float32)

    # 1. Basenfolge extrahieren (erstes Zeichen jedes Tokens)
    clean_seq = "".join(tok[0] for tok in tokens)
    seq_oh = seq_to_one_hot(clean_seq)  # (L, 4)

    # 2. CDS-Track: Großbuchstabe = Codon-Start (1. Base des Codons)
    cds_track = np.array([1.0 if tok[0].isupper() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    # 3. Splice-Track: 'ej' im Token = Exon-Junction
    splice_track = np.array([1.0 if "ej" in tok.lower() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    six_track = np.concatenate([seq_oh, cds_track, splice_track], axis=1)
    return six_track


def extract_embeddings_for_saluki(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 12288,
) -> dict:
    """
    Extrahiert Orthrus 6-Track Embeddings fuer den Saluki DataFrame mit dynamischem Laengen-Batching.
    """
    print(f"Verarbeite Saluki-Datensatz mit {len(df)} Eintraegen...")

    sample_data = []
    skipped_count = 0

    for idx, row in df.iterrows():
        raw_seq = str(row["sequence"])
        six_track = parse_saluki_sequence_to_six_track(raw_seq)

        if six_track.shape[0] > max_length:
            skipped_count += 1
            six_track = six_track[:max_length, :]

        sample_data.append({
            "orig_idx": idx,
            "track": six_track,
            "length": six_track.shape[0],
            "transcript_id": str(row.get("ensembl_transcript_id", "")),
            "gene_id": str(row.get("ensembl_gene_id", "")),
            "gene_symbol": str(row.get("hgnc_symbol", "")),
            "biotype": str(row.get("transcript_biotype", "")),
            "half_life": float(row.get("half_life", np.nan)),
            "half_life_transformed": float(row.get("half_life_transformed", np.nan)),
            "rate": float(row.get("rate", np.nan)),
        })

    if skipped_count > 0:
        print(f"Hinweis: {skipped_count} Sequenzen wurden auf max_length={max_length} Nukleotide gekuerzt.")

    # Sortieren nach Laenge, um Padding im Batch zu minimieren
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

    # Metadaten in urspruenglicher DataFrame-Reihenfolge
    transcript_ids = np.array([s["transcript_id"] for s in sample_data])
    gene_ids = np.array([s["gene_id"] for s in sample_data])
    gene_symbols = np.array([s["gene_symbol"] for s in sample_data])
    biotypes = np.array([s["biotype"] for s in sample_data])
    half_lives = np.array([s["half_life"] for s in sample_data], dtype=np.float32)
    half_lives_transformed = np.array([s["half_life_transformed"] for s in sample_data], dtype=np.float32)
    rates = np.array([s["rate"] for s in sample_data], dtype=np.float32)
    seq_lens = np.array([s["length"] for s in sample_data], dtype=np.int32)

    return {
        "embeddings": all_embeddings,
        "half_life": half_lives,
        "half_life_transformed": half_lives_transformed,
        "rate": rates,
        "ensembl_transcript_id": transcript_ids,
        "ensembl_gene_id": gene_ids,
        "hgnc_symbol": gene_symbols,
        "transcript_biotype": biotypes,
        "seq_lens": seq_lens,
    }


def main():
    parser = argparse.ArgumentParser(description="Extrahiere Orthrus 6-Track Embeddings fuer Saluki")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/saluki/saluki_ej_cds_transformed.txt",
        help="Pfad zur Saluki Datendatei (tab-separiert)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/saluki",
        help="Verzeichnis zum Speichern der Embeddings",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="orthrus_6track_embeddings_saluki.npz",
        help="Dateiname fuer das gespeicherte NPZ-Archiv",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="quietflamingo/orthrus-large-6-track",
        help="Hugging Face Modell-Identifier",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch-Groesse fuer Inferenz",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximale Sequenzlaenge gemaess Orthrus Paper (Standard: 12288)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file = output_dir / args.output_filename

    print(f"Lade Saluki-Datensatz von: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    print(f"Geladene Zeilen: {len(df)}")
    print(f"Spalten: {list(df.columns)}")

    # Sicherstellen, dass half_life_transformed existiert (falls mit Rohdatei saluki_ej_cds.txt aufgerufen)
    if "half_life_transformed" not in df.columns and "half_life" in df.columns:
        print("Spalte 'half_life_transformed' nicht vorhanden - berechne aus 'half_life' (Log + Z-Score)...")
        y_raw = df["half_life"].astype(float)
        y_log = np.log(y_raw + 0.1)
        mu_log = float(y_log.mean())
        sigma_log = float(y_log.std(ddof=1))
        df["half_life_transformed"] = (y_log - mu_log) / sigma_log
        print(f"Transformation berechnet: mu={mu_log:.4f}, sigma={sigma_log:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Geraet: {device}")

    print(f"Lade Orthrus 6-Track Modell '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Modell erfolgreich geladen.")

    result = extract_embeddings_for_saluki(
        df=df,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"\nSpeichere Embeddings nach: {save_file}")
    np.savez_compressed(
        save_file,
        embeddings=result["embeddings"],
        half_life=result["half_life"],
        half_life_transformed=result["half_life_transformed"],
        rate=result["rate"],
        ensembl_transcript_id=result["ensembl_transcript_id"],
        ensembl_gene_id=result["ensembl_gene_id"],
        hgnc_symbol=result["hgnc_symbol"],
        transcript_biotype=result["transcript_biotype"],
        seq_lens=result["seq_lens"],
    )

    print(f"Erfolgreich gespeichert!")
    print(f"Embedding Shape: {result['embeddings'].shape}")


if __name__ == "__main__":
    main()
