#!/usr/bin/env python3
"""
Generierung von kontinuierlichen Trans-Faktor Dichte- und Affinitaets-Tracks (Strategie B)
fuer den Saluki-Datensatz.

Kombiniert:
1. GTF-SQLite-Datenbank (Homo_sapiens.GRCh38.108.gtf.db) fuer das genomische Exon-Mapping
2. TargetScan: miRNA-Bindungsaffinitaeten / Context++ Scores
3. ENCODE eCLIP: Experimentelle RBP-Peak-Signalwerte

Ergebnis:
Erweitert die 6 Basis-Tracks (A, C, G, U, CDS, Splice) um 2 kontinuierliche
Trans-Faktor-Tracks zu einem 8-kanaligen Track-Array der Form (L, 8) fuer jedes Transkript.
"""

import argparse
import os
from pathlib import Path
import gffutils
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. GTF SQLite Datenbank Zugriff & Koordinaten-Mapping (via gffutils)
# =============================================================================

class GtfDbHelper:
    """
    Kapselt den Zugriff auf die GTF-FeatureDB via gffutils.
    """
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            print(f"[Warnung] GTF-DB Datei '{self.db_path}' lokal nicht gefunden (laeuft auf Cluster).")

        print(f"[GTF-DB] Lade gffutils.FeatureDB: {self.db_path.name}")
        self.db = gffutils.FeatureDB(str(self.db_path))

    def get_transcript_exons(self, transcript_id: str) -> dict:
        """
        Gibt Chromosom, Strang und eine nach Transkriptionsrichtung (5' -> 3')
        geordnete Liste von Exon-Intervallen (start, end) zurueck.
        """
        clean_id = transcript_id.split(".")[0]

        # Suche Transkript nach ID (mit oder ohne Versionsnummer)
        tx = None
        for candidate in [transcript_id, clean_id]:
            try:
                tx = self.db[candidate]
                break
            except Exception:
                continue

        if tx is None:
            return None

        strand = tx.strand
        chrom = tx.chrom
        exons = [
            (exon.start, exon.end)
            for exon in self.db.children(tx, featuretype="exon", order_by="start")
        ]

        if strand == "-":
            # Bei Minusstrang: 5' -> 3' entspricht absteigenden genomischen Koordinaten
            exons.reverse()

        return {"chrom": chrom, "strand": strand, "exons": exons}


def map_genomic_intervals_to_transcript(
    exons: list,
    strand: str,
    peak_intervals: list,
    transcript_len: int,
) -> np.ndarray:
    """
    Mappt genomische Peaks mit Werten (start, end, score) auf ein 1D-Signal der reifen mRNA.
    
    Args:
        exons: Liste von (start, end) Tuplen in 5' -> 3' Reihenfolge.
        strand: '+' oder '-'
        peak_intervals: Liste von (p_start, p_end, score)
        transcript_len: Gesamtlaenge der reifen mRNA
    
    Returns:
        1D-Array der Laenge transcript_len mit kontinuierlichen Dichtewerten.
    """
    track = np.zeros(transcript_len, dtype=np.float32)
    if not exons or not peak_intervals:
        return track

    # Berechne relative Transkript-Offsets fuer jedes Exon
    curr_tx_pos = 0
    for ex_start, ex_end in exons:
        ex_len = ex_end - ex_start + 1

        for p_start, p_end, score in peak_intervals:
            # Pruefe Ueberlapp zwischen Exon und Peak
            overlap_start = max(ex_start, p_start)
            overlap_end = min(ex_end, p_end)

            if overlap_start <= overlap_end:
                if strand == "+":
                    # Plusstrang: Start relativ zu ex_start
                    rel_start = curr_tx_pos + (overlap_start - ex_start)
                    rel_end = curr_tx_pos + (overlap_end - ex_start) + 1
                else:
                    # Minusstrang: Start relativ zu ex_end (umgekehrt)
                    rel_start = curr_tx_pos + (ex_end - overlap_end)
                    rel_end = curr_tx_pos + (ex_end - overlap_start) + 1

                # Clamping gegen Rundungs-/Grenzfehler
                rel_start = max(0, min(rel_start, transcript_len))
                rel_end = max(0, min(rel_end, transcript_len))

                if rel_start < rel_end:
                    track[rel_start:rel_end] += float(score)

        curr_tx_pos += ex_len

    return track


# =============================================================================
# 2. Parser fuer die 3 externen Datenbanken
# =============================================================================

def load_targetscan_data(targetscan_path: Path) -> dict:
    """
    Laedt TargetScan miRNA-Vorhersagen.
    Unterstuetzt 'Predicted_Targets_Context_Scores' oder 'Conserved_Site_Context_Scores'.
    
    Gibt ein Dict zurueck: {transcript_id: [(utr_start, utr_end, score), ...]}
    """
    if not targetscan_path.exists():
        print(f"[Hinweis] TargetScan-Datei '{targetscan_path}' nicht gefunden.")
        return {}

    print(f"Lade TargetScan-Daten von: {targetscan_path}...")
    df_ts = pd.read_csv(targetscan_path, sep="\t", low_memory=False)

    # Typische Spaltennamen identifizieren
    tx_col = next((c for c in df_ts.columns if "transcript" in c.lower()), None)
    score_col = next((c for c in df_ts.columns if "context" in c.lower() or "score" in c.lower()), None)
    start_col = next((c for c in df_ts.columns if "start" in c.lower()), None)
    end_col = next((c for c in df_ts.columns if "end" in c.lower()), None)

    if not tx_col:
        print("[Warnung] Keine Transkript-Spalte in TargetScan gefunden.")
        return {}

    mapping = {}
    for _, row in df_ts.iterrows():
        raw_tx = str(row[tx_col]).split(".")[0]
        # Context++ Scores sind negativ; je negativer, desto staerker die Repression
        raw_score = float(row[score_col]) if score_col and pd.notnull(row[score_col]) else 1.0
        # Positive Affinitaet: Absoluter Wert oder -score
        score = abs(raw_score)

        start = int(row[start_col]) if start_col and pd.notnull(row[start_col]) else 0
        end = int(row[end_col]) if end_col and pd.notnull(row[end_col]) else start + 7

        mapping.setdefault(raw_tx, []).append((start, end, score))

    print(f"TargetScan: Bindungsstellen fuer {len(mapping)} einzigartige Transkripte geladen.")
    return mapping


def load_bed_intervals_by_chrom(bed_path: Path) -> dict:
    """
    Laedt eine BED / narrowPeak-Datei (z. B. ENCODE eCLIP oder POSTAR3).
    Gibt ein nach Chromosomen und Strand geschachteltes Dict zurueck:
    { (chrom, strand): [(start, end, score), ...] }
    """
    if not bed_path.exists():
        print(f"[Hinweis] Datei '{bed_path}' nicht gefunden.")
        return {}

    print(f"Lade BED-Intervalle von: {bed_path}...")
    data = {}
    with open(bed_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or line.startswith("track") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0].replace("chr", "")
            start = int(parts[1])
            end = int(parts[2])
            strand = parts[5] if len(parts) >= 6 and parts[5] in ["+", "-"] else "+"
            
            # Score: Falls narrowPeak (Spalte 7 signalValue) oder BED (Spalte 5 score)
            score = 1.0
            if len(parts) >= 7 and parts[6] not in [".", "-1"]:
                try:
                    score = float(parts[6])
                except ValueError:
                    score = 1.0
            elif len(parts) >= 5 and parts[4] not in [".", "-1"]:
                try:
                    score = float(parts[4])
                except ValueError:
                    score = 1.0

            key = (chrom, strand)
            data.setdefault(key, []).append((start, end, score))

    total_peaks = sum(len(v) for v in data.values())
    print(f"BED-Loader: {total_peaks} Intervalle geladen fuer {bed_path.name}.")
    return data


# =============================================================================
# 3. Saluki 6-Track Basis-Parser (A, C, G, U, CDS, Splice)
# =============================================================================

def parse_saluki_base_tracks(raw_seq: str) -> tuple:
    """
    Parst die kommagetrennte Saluki-Sequenz in die 6 Standard-Tracks:
    Returns:
        (six_track_array, seq_len, utr3_start_idx)
    """
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    l = len(tokens)
    if l == 0:
        return np.zeros((0, 6), dtype=np.float32), 0, 0

    clean_seq = "".join(tok[0] for tok in tokens)

    # 4-Kanal One-Hot
    seq_bytes = np.frombuffer(clean_seq.upper().encode("ascii"), dtype=np.uint8)
    oh = np.zeros((l, 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0  # A
    oh[seq_bytes == 67, 1] = 1.0  # C
    oh[seq_bytes == 71, 2] = 1.0  # G
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0  # T/U

    # CDS-Track: Grossbuchstaben markieren Frame-0 Codon-Starts
    cds_track = np.array([1.0 if tok[0].isupper() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    # Splice-Track: 'ej' markiert Exon-Junctions
    splice_track = np.array([1.0 if "ej" in tok.lower() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    six_track = np.concatenate([oh, cds_track, splice_track], axis=1)

    # Finde Beginn der 3' UTR (Position nach dem letzten Grossbuchstaben der CDS)
    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    utr3_start = (max(upper_indices) + 3) if upper_indices else 0
    utr3_start = min(utr3_start, l)

    return six_track, l, utr3_start


# =============================================================================
# 4. Haupt-Pipeline: Multi-Track Extraktion
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generiere kontinuierliche Trans-Faktor Dichte-Tracks (Strategie B) fuer Saluki"
    )
    # Eingabepfade
    parser.add_argument(
        "--saluki_data",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/saluki/saluki_ej_cds_transformed.txt",
        help="Pfad zur Saluki-Datendatei (tab-separiert)",
    )
    parser.add_argument(
        "--gtf_db",
        type=str,
        default="/beegfs/prj/RNA_NLP/AlphaGenome/data/Homo_sapiens.GRCh38.108.gtf.db",
        help="Pfad zur GTF SQLite DB",
    )
    parser.add_argument(
        "--targetscan_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/targetscan/Predicted_Targets_Context_Scores.default_predictions.txt",
        help="Pfad zur TargetScan Voraussage-Tabelle",
    )
    parser.add_argument(
        "--encode_eclip_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/all_rbp_peaks_merged.bed",
        help="Pfad zur ENCODE eCLIP BED/narrowPeak-Datei",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/saluki/saluki_multitrack_with_trans_factors.npz",
        help="Ausgabedatei fuer das erweiterte NPZ-Archiv",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=12288,
        help="Maximale Sequenzlaenge (Standard: 12288 bp)",
    )
    args = parser.parse_args()

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("   Generierung von Trans-Faktor Dichte-Tracks (TargetScan, ENCODE eCLIP)   ")
    print("=" * 75)

    # 1. Datenbanken & Lookup-Tabellen laden
    gtf_helper = GtfDbHelper(Path(args.gtf_db))
    ts_data = load_targetscan_data(Path(args.targetscan_file))
    eclip_data = load_bed_intervals_by_chrom(Path(args.encode_eclip_file))

    # 2. Saluki Datensatz laden
    saluki_path = Path(args.saluki_data)
    print(f"\nLade Saluki-Datensatz: {saluki_path}...")
    df = pd.read_csv(saluki_path, sep="\t")
    print(f"Eintraege geladen: {len(df)}")

    # 3. Iteration ueber alle Transkripte und Konstruktion der 8-Kanal-Tracks:
    # Kanäle 0-3: A, C, G, U (One-Hot)
    # Kanal 4:    CDS Track
    # Kanal 5:    Splice Track
    # Kanal 6:    miRNA Affinitaets-Track (TargetScan)
    # Kanal 7:    ENCODE eCLIP RBP Dichte-Track
    all_tracks = []
    coverage_stats = {
        "with_mirna": 0,
        "with_eclip": 0,
        "with_gtf_mapping": 0,
    }

    print("\nGeneriere 8-Kanal Multitrack-Tensoren...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transkripte verarbeiten"):
        raw_seq = str(row["sequence"])
        tx_id = str(row.get("ensembl_transcript_id", ""))
        clean_tx = tx_id.split(".")[0]

        # Basis 6-Tracks
        six_track, l, utr3_start = parse_saluki_base_tracks(raw_seq)
        if l > args.max_length:
            six_track = six_track[:args.max_length, :]
            l = args.max_length

        # -----------------------------------------------------------------
        # Kanal 6: TargetScan miRNA Track (kontinuierlich)
        # -----------------------------------------------------------------
        mirna_track = np.zeros((l, 1), dtype=np.float32)
        if clean_tx in ts_data:
            coverage_stats["with_mirna"] += 1
            for start, end, score in ts_data[clean_tx]:
                # Koordinaten in TargetScan sind relativ zum 3' UTR Start
                abs_start = utr3_start + start
                abs_end = utr3_start + end
                if abs_start < l:
                    clamped_end = min(abs_end, l)
                    mirna_track[abs_start:clamped_end, 0] += score

        # -----------------------------------------------------------------
        # GTF Exon-Mapping fuer genomische RBP-Peaks (ENCODE eCLIP)
        # -----------------------------------------------------------------
        eclip_track = np.zeros((l, 1), dtype=np.float32)

        tx_info = gtf_helper.get_transcript_exons(clean_tx)
        if tx_info is not None:
            coverage_stats["with_gtf_mapping"] += 1
            chrom = str(tx_info["chrom"]).replace("chr", "")
            strand = tx_info["strand"]
            exons = tx_info["exons"]
            key = (chrom, strand)

            # Kanal 7: ENCODE eCLIP
            if key in eclip_data:
                eclip_1d = map_genomic_intervals_to_transcript(exons, strand, eclip_data[key], l)
                if np.any(eclip_1d > 0):
                    coverage_stats["with_eclip"] += 1
                    eclip_track[:, 0] = eclip_1d

        # -----------------------------------------------------------------
        # Zusammenfuegen zu (L, 8)
        # -----------------------------------------------------------------
        multi_track = np.concatenate([six_track, mirna_track, eclip_track], axis=1)
        all_tracks.append(multi_track)

    # 4. Speichern im Ziel-Archiv
    print(f"\nSpeichere Multitrack-Archiv nach: {out_file}...")
    np.savez_compressed(
        out_file,
        tracks=np.array(all_tracks, dtype=object),
        ensembl_transcript_id=df["ensembl_transcript_id"].values.astype(str),
        ensembl_gene_id=df["ensembl_gene_id"].values.astype(str),
        hgnc_symbol=df["hgnc_symbol"].values.astype(str),
        half_life_transformed=df["half_life_transformed"].values.astype(np.float32) if "half_life_transformed" in df.columns else df["half_life"].values.astype(np.float32),
        half_life=df["half_life"].values.astype(np.float32) if "half_life" in df.columns else np.nan,
        rate=df["rate"].values.astype(np.float32) if "rate" in df.columns else np.nan,
        seq_lens=np.array([t.shape[0] for t in all_tracks], dtype=np.int32),
    )

    # 5. Zusammenfassung & Coverage Report
    n = len(df)
    print("\n" + "=" * 65)
    print("             COVERAGE & STATISTIK REPORT             ")
    print("=" * 65)
    print(f"Gesamtanzahl Transkripte:            {n}")
    print(f"Erfolgreich in GTF-DB gemappt:       {coverage_stats['with_gtf_mapping']} ({coverage_stats['with_gtf_mapping']/n*100:.2f} %)")
    print(f"Transkripte mit TargetScan miRNAs:   {coverage_stats['with_mirna']} ({coverage_stats['with_mirna']/n*100:.2f} %)")
    print(f"Transkripte mit ENCODE eCLIP Peaks:  {coverage_stats['with_eclip']} ({coverage_stats['with_eclip']/n*100:.2f} %)")
    print(f"Track-Dimensionen pro Transkript:    (L, 8)")
    print(f"Kanalkonfiguration:                  [A, C, G, U, CDS, Splice, TargetScan, eCLIP]")
    print(f"Datei erfolgreich gespeichert:       {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
