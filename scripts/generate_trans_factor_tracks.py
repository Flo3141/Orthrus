#!/usr/bin/env python3
"""
Generierung von kontinuierlichen Trans-Faktor Dichte- und Affinitaets-Tracks (Strategie B)
fuer den Saluki-Datensatz mit inkrementeller Speicherung und O(log N) Vektorisierung.

Kombiniert:
1. GTF-SQLite-Datenbank (Homo_sapiens.GRCh38.108.gtf.db via gffutils)
2. TargetScan: miRNA-Bindungsaffinitaeten / Context++ Scores
3. ENCODE eCLIP: Experimentelle RBP-Peak-Signalwerte

Features:
- Schnelle binäre Suche (np.searchsorted): Reduziert die Laufzeit drastisch
- Inkrementelle Speicherung in Chunks (fortsetzbar bei Abbruch)
- Automatisches Zusammenführen zur finalen NPZ-Datei
"""

from pandas._libs import properties
import argparse
import os
from pathlib import Path
import gffutils
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 1. Schneller Intervall-Index (O(log N) Suche statt O(N) Schleife)
# =============================================================================

class FastIntervalIndex:
    """
    Indexiert genomische Peaks eines Chromosoms/Strangs fuer extrem schnelle
    Overlap-Abfragen mittels sortierter NumPy-Arrays und np.searchsorted.
    """
    def __init__(self, intervals: list):
        if not intervals:
            self.empty = True
            return
        self.empty = False
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([x[0] for x in intervals_sorted], dtype=np.int64)
        self.ends = np.array([x[1] for x in intervals_sorted], dtype=np.int64)
        self.scores = np.array([x[2] for x in intervals_sorted], dtype=np.float32)
        self.max_peak_len = int(np.max(self.ends - self.starts)) if len(self.ends) > 0 else 500

    def get_overlaps(self, ex_start: int, ex_end: int):
        if self.empty:
            return None, None, None

        # Nur Peaks pruefen, deren Start <= ex_end und >= ex_start - max_peak_len liegt
        left_idx = np.searchsorted(self.starts, ex_start - self.max_peak_len, side="left")
        right_idx = np.searchsorted(self.starts, ex_end, side="right")

        if left_idx >= right_idx:
            return None, None, None

        sub_starts = self.starts[left_idx:right_idx]
        sub_ends = self.ends[left_idx:right_idx]
        sub_scores = self.scores[left_idx:right_idx]

        # Exakter Schnitt: Peak endet nach ex_start und beginnt vor ex_end
        mask = (sub_ends >= ex_start) & (sub_starts <= ex_end)
        if not np.any(mask):
            return None, None, None

        return sub_starts[mask], sub_ends[mask], sub_scores[mask]


# =============================================================================
# 2. GTF SQLite Datenbank Zugriff & Koordinaten-Mapping (via gffutils)
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
            exons.reverse()

        return {"chrom": chrom, "strand": strand, "exons": exons}


def map_genomic_intervals_to_transcript(
    exons: list,
    strand: str,
    peak_index: FastIntervalIndex,
    transcript_len: int,
) -> np.ndarray:
    """
    Mappt genomische Peaks hochperformant via Binärsuche auf die reife mRNA.
    """
    track = np.zeros(transcript_len, dtype=np.float32)
    if not exons or peak_index.empty:
        return track

    curr_tx_pos = 0
    for ex_start, ex_end in exons:
        ex_len = ex_end - ex_start + 1
        p_starts, p_ends, p_scores = peak_index.get_overlaps(ex_start, ex_end)

        if p_starts is not None:
            for p_start, p_end, score in zip(p_starts, p_ends, p_scores):
                overlap_start = max(ex_start, int(p_start))
                overlap_end = min(ex_end, int(p_end))

                if strand == "+":
                    rel_start = curr_tx_pos + (overlap_start - ex_start)
                    rel_end = curr_tx_pos + (overlap_end - ex_start) + 1
                else:
                    rel_start = curr_tx_pos + (ex_end - overlap_end)
                    rel_end = curr_tx_pos + (ex_end - overlap_start) + 1

                rel_start = max(0, min(rel_start, transcript_len))
                rel_end = max(0, min(rel_end, transcript_len))

                if rel_start < rel_end:
                    track[rel_start:rel_end] += float(score)

        curr_tx_pos += ex_len

    return track


# =============================================================================
# 3. Parser fuer TargetScan und ENCODE eCLIP
# =============================================================================

def load_targetscan_data(targetscan_path: Path) -> dict:
    if not targetscan_path.exists():
        print(f"[Hinweis] TargetScan-Datei '{targetscan_path}' nicht gefunden.")
        return {}

    print(f"Lade TargetScan-Daten von: {targetscan_path}...")
    df_ts = pd.read_csv(targetscan_path, sep="\t", low_memory=False)
    tx_col = "Transcript ID"
    score_col = "weighted context++ score"
    # Die beiden sind nicht gleich genamed
    start_col = "UTR_start"
    end_col = "UTR end"
    # Ungültige Zeilen ohne Start, End oder Score entfernen
    df_ts = df_ts.dropna(subset=[tx_col, start_col, end_col, score_col])

    mapping = {}
    for _, row in df_ts.iterrows():
        raw_tx = str(row[tx_col]).split(".")[0]
        score = abs(float(row[score_col]))

        start = int(row[start_col])
        end = int(row[end_col])

        if start >= end:
            continue

        mapping.setdefault(raw_tx, []).append((start, end, score))

    print(f"TargetScan: Bindungsstellen fuer {len(mapping)} einzigartige Transkripte geladen.")
    return mapping


def load_eclip_indexed(bed_path: Path) -> dict:
    """
    Laedt ENCODE eCLIP Peaks und baut fuer jedes (chrom, strand) einen FastIntervalIndex auf.
    """
    if not bed_path.exists():
        print(f"[Hinweis] Datei '{bed_path}' nicht gefunden.")
        return {}

    print(f"Lade ENCODE eCLIP Intervalle von: {bed_path}...")
    raw_data = {}
    with open(bed_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or line.startswith("track") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue

            chrom = parts[0].replace("chr", "")
            try:
                start = int(parts[1]) + 1   # BED 0-basiert -> 1-basiert (wie GTF)
                end = int(parts[2])         # BED end ist bereits exklusiv, entspricht also 1-basiert inklusiv
            except ValueError:
                continue

            if start >= end:
                continue

            strand = parts[5]
            if strand not in ["+", "-"]:
                continue

            # SignalValue (parts[6]) muss vorhanden und eine gültige Zahl sein (kein Fallback)
            if parts[6] in [".", "-1", "nan", "NaN", ""]:
                continue
            try:
                score = float(parts[6])
            except ValueError:
                continue

            key = (chrom, strand)
            raw_data.setdefault(key, []).append((start, end, score))

    # Erstelle FastIntervalIndex pro Chromosom/Strand
    indexed_data = {}
    for key, intervals in raw_data.items():
        indexed_data[key] = FastIntervalIndex(intervals)

    total_peaks = sum(len(v) for v in raw_data.values())
    print(f"eCLIP: {total_peaks} Peaks indiziert ueber {len(indexed_data)} (Chrom, Strand)-Kombinationen.")
    return indexed_data


# =============================================================================
# 4. Saluki 6-Track Basis-Parser
# =============================================================================

def parse_saluki_base_tracks(raw_seq: str) -> tuple:
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    last_idx = max(upper_indices)
    print("Letzte CDS-Basen:", "".join(tok[0] for tok in tokens[last_idx-5:last_idx+1]))
    print("Folgende Basen:", "".join(tok[0] for tok in tokens[last_idx+1:last_idx+7]))
    exit()
    l = len(tokens)
    if l == 0:
        return np.zeros((0, 6), dtype=np.float32), 0, 0

    clean_seq = "".join(tok[0] for tok in tokens)

    seq_bytes = np.frombuffer(clean_seq.upper().encode("ascii"), dtype=np.uint8)
    oh = np.zeros((l, 4), dtype=np.float32)
    oh[seq_bytes == 65, 0] = 1.0
    oh[seq_bytes == 67, 1] = 1.0
    oh[seq_bytes == 71, 2] = 1.0
    oh[(seq_bytes == 84) | (seq_bytes == 85), 3] = 1.0

    cds_track = np.array([1.0 if tok[0].isupper() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)
    splice_track = np.array([1.0 if "ej" in tok.lower() else 0.0 for tok in tokens], dtype=np.float32).reshape(-1, 1)

    six_track = np.concatenate([oh, cds_track, splice_track], axis=1)

    upper_indices = [i for i, tok in enumerate(tokens) if tok[0].isupper()]
    utr3_start = (max(upper_indices) + 3) if upper_indices else 0
    utr3_start = min(utr3_start, l)

    return six_track, l, utr3_start


# =============================================================================
# 5. Inkrementelles Chunking & Zusammenfuehrung
# =============================================================================

def save_chunk(chunk_idx: int, chunk_items: list, chunks_dir: Path):
    """Speichert einen Block von Transkripten inkrementell als NPZ."""
    chunk_file = chunks_dir / f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(
        chunk_file,
        tracks=np.array([item["track"] for item in chunk_items], dtype=object),
        ensembl_transcript_id=np.array([item["transcript_id"] for item in chunk_items]),
        ensembl_gene_id=np.array([item["gene_id"] for item in chunk_items]),
        hgnc_symbol=np.array([item["gene_symbol"] for item in chunk_items]),
        half_life_transformed=np.array([item["half_life_transformed"] for item in chunk_items], dtype=np.float32),
        half_life=np.array([item["half_life"] for item in chunk_items], dtype=np.float32),
        rate=np.array([item["rate"] for item in chunk_items], dtype=np.float32),
        seq_lens=np.array([item["length"] for item in chunk_items], dtype=np.int32),
        has_mirna=np.array([item["has_mirna"] for item in chunk_items], dtype=bool),
        has_eclip=np.array([item["has_eclip"] for item in chunk_items], dtype=bool),
        has_gtf=np.array([item["has_gtf"] for item in chunk_items], dtype=bool),
    )


def merge_all_chunks(chunks_dir: Path, output_file: Path):
    """Fuehrt alle erzeugten Chunks zu der finalen Master-NPZ zusammen."""
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    if not chunk_files:
        print("[Fehler] Keine Chunks zum Zusammenfuehren gefunden!")
        return

    print(f"\nFühre {len(chunk_files)} Chunks zu {output_file} zusammen...")
    all_tracks = []
    tx_ids = []
    gene_ids = []
    gene_symbols = []
    hl_trans = []
    hl_raw = []
    rates = []
    lens = []
    mirna_flags = []
    eclip_flags = []
    gtf_flags = []

    for cf in tqdm(chunk_files, desc="Chunks mergen"):
        data = np.load(cf, allow_pickle=True)
        all_tracks.extend(data["tracks"])
        tx_ids.extend(data["ensembl_transcript_id"])
        gene_ids.extend(data["ensembl_gene_id"])
        gene_symbols.extend(data["hgnc_symbol"])
        hl_trans.extend(data["half_life_transformed"])
        hl_raw.extend(data["half_life"])
        rates.extend(data["rate"])
        lens.extend(data["seq_lens"])
        mirna_flags.extend(data["has_mirna"])
        eclip_flags.extend(data["has_eclip"])
        gtf_flags.extend(data["has_gtf"])

    np.savez_compressed(
        output_file,
        tracks=np.array(all_tracks, dtype=object),
        ensembl_transcript_id=np.array(tx_ids),
        ensembl_gene_id=np.array(gene_ids),
        hgnc_symbol=np.array(gene_symbols),
        half_life_transformed=np.array(hl_trans, dtype=np.float32),
        half_life=np.array(hl_raw, dtype=np.float32),
        rate=np.array(rates, dtype=np.float32),
        seq_lens=np.array(lens, dtype=np.int32),
    )

    n = len(tx_ids)
    print("\n" + "=" * 65)
    print("             COVERAGE & STATISTIK REPORT             ")
    print("=" * 65)
    print(f"Gesamtanzahl Transkripte:            {n}")
    print(f"Erfolgreich in GTF-DB gemappt:       {sum(gtf_flags)} ({sum(gtf_flags)/n*100:.2f} %)")
    print(f"Transkripte mit TargetScan miRNAs:   {sum(mirna_flags)} ({sum(mirna_flags)/n*100:.2f} %)")
    print(f"Transkripte mit ENCODE eCLIP Peaks:  {sum(eclip_flags)} ({sum(eclip_flags)/n*100:.2f} %)")
    print(f"Track-Dimensionen pro Transkript:    (L, 8)")
    print(f"Kanalkonfiguration:                  [A, C, G, U, CDS, Splice, TargetScan, eCLIP]")
    print(f"Finale Datei gespeichert:            {output_file}")
    print("=" * 65)


# =============================================================================
# 6. Haupt-Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generiere kontinuierliche Trans-Faktor Dichte-Tracks fuer Saluki (mit Inkrementeller Speicherung)"
    )
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
        "--chunk_size",
        type=int,
        default=500,
        help="Anzahl der Transkripte pro inkrementellem Speicher-Chunk (Standard: 500)",
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

    chunks_dir = out_file.parent / f"{out_file.stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("   Generierung von Trans-Faktor Dichte-Tracks (TargetScan, ENCODE eCLIP)   ")
    print("=" * 75)
    print(f"Ausgabedatei:       {out_file}")
    print(f"Chunk-Verzeichnis:  {chunks_dir} (Chunk-Größe: {args.chunk_size})")

    # Pruefen auf bereits verarbeitete Transkripte fuer nahtlose Wiederaufnahme (Resume)
    completed_tx_ids = set()
    existing_chunks = sorted(chunks_dir.glob("chunk_*.npz"))
    for cf in existing_chunks:
        try:
            c_data = np.load(cf, allow_pickle=True)
            completed_tx_ids.update(c_data["ensembl_transcript_id"])
        except Exception:
            continue

    if completed_tx_ids:
        print(f"[Resume] Bereits {len(completed_tx_ids)} fertige Transkripte in {len(existing_chunks)} Chunks gefunden!")

    # 1. Datenbanken & Lookup-Tabellen laden
    gtf_helper = GtfDbHelper(Path(args.gtf_db))
    # ts_data = load_targetscan_data(Path(args.targetscan_file))
    # eclip_data = load_eclip_indexed(Path(args.encode_eclip_file))

    # 2. Saluki Datensatz laden
    saluki_path = Path(args.saluki_data)
    print(f"\nLade Saluki-Datensatz: {saluki_path}...")
    df = pd.read_csv(saluki_path, sep="\t")

        # =========================================================================
    # TEST: Stop-Codon & UTR3-Übergang prüfen
    # =========================================================================
    STOP_CODONS = {"TAA", "TAG", "TGA", "UAA", "UAG", "UGA"}
    
    stop_in_uppercase = 0
    stop_in_lowercase = 0
    neither = 0
    
    print("\n--- Analysiere Stop-Codons im Saluki-Datensatz ---")
    examples_shown = 0
    
    for idx, row in df.head(1000).iterrows():
        raw_seq = str(row["sequence"])
        tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
        chars = [tok[0] for tok in tokens]
        print(tokens)
        print(chars)
        exit()
        upper_indices = [i for i, c in enumerate(chars) if c.isupper()]
        if not upper_indices:
            continue  # Kein CDS vorhanden (z. B. lncRNA)
            
        last_idx = max(upper_indices)
        
        # Die letzten 3 Großbuchstaben
        last_3_upper = "".join(chars[last_idx - 2 : last_idx + 1]).upper()
        # Die ersten 3 Kleinbuchstaben direkt danach
        first_3_lower = "".join(chars[last_idx + 1 : last_idx + 4]).upper()
        
        is_upper = last_3_upper in STOP_CODONS
        is_lower = first_3_lower in STOP_CODONS
        
        if is_upper:
            stop_in_uppercase += 1
        elif is_lower:
            stop_in_lowercase += 1
        else:
            neither += 1
            
        # Zeige die ersten 3 konkreten Beispiele
        if examples_shown < 3 and (is_upper or is_lower):
            context = "".join(chars[max(0, last_idx - 6) : min(len(chars), last_idx + 7)])
            print(f"Beispiel {examples_shown + 1} (Tx: {row.get('ensembl_transcript_id', 'N/A')}):")
            print(f"  Ausschnitt (CDS=GROSS, UTR=klein): ...{context}...")
            print(f"  Letzte 3 CDS-Basen: '{last_3_upper}' -> Stop-Codon? {is_upper}")
            print(f"  Erste 3 UTR-Basen:  '{first_3_lower}' -> Stop-Codon? {is_lower}\n")
            examples_shown += 1

    print("Ergebnis über die ersten 1000 Transkripte:")
    print(f"  Stop-Codon GROSSGESCHRIEBEN (am Ende der CDS): {stop_in_uppercase}")
    print(f"  Stop-Codon KLEINGESCHRIEBEN (am Anfang der UTR): {stop_in_lowercase}")
    print(f"  Anderes / Unklar:                              {neither}")
    print("=" * 60)
    exit()
    # =========================================================================


    total_samples = len(df)
    print(f"Gesamteintraege in Saluki: {total_samples}")

    # Bestimme naechsten Chunk-Index
    chunk_idx = len(existing_chunks)
    current_chunk_items = []

    print("\nVerarbeite Transkripte (mit schnellem O(log N) Lookup & Inkrementellem Speichern)...")
    with tqdm(total=total_samples, desc="Fortschritt", initial=len(completed_tx_ids)) as pbar:
        for idx, row in df.iterrows():
            tx_id = str(row.get("ensembl_transcript_id", ""))
            clean_tx = tx_id.split(".")[0]

            # Bereits fertige Transkripte ueberspringen
            if tx_id in completed_tx_ids or clean_tx in completed_tx_ids:
                continue

            raw_seq = str(row["sequence"])

            # Basis 6-Tracks
            six_track, l, utr3_start = parse_saluki_base_tracks(raw_seq)
            if l > args.max_length:
                six_track = six_track[:args.max_length, :]
                l = args.max_length

            # Kanal 6: TargetScan miRNA
            mirna_track = np.zeros((l, 1), dtype=np.float32)
            has_mirna = False
            if clean_tx in ts_data:
                for start, end, score in ts_data[clean_tx]:
                    abs_start = utr3_start + (start - 1)
                    abs_end = utr3_start + end
                    if abs_start < l:
                        clamped_end = min(abs_end, l)
                        mirna_track[abs_start:clamped_end, 0] += score
                        has_mirna = True

            # Kanal 7: ENCODE eCLIP via schnellem Index
            eclip_track = np.zeros((l, 1), dtype=np.float32)
            has_eclip = False
            has_gtf = False

            tx_info = gtf_helper.get_transcript_exons(clean_tx)
            if tx_info is not None:
                has_gtf = True
                chrom = str(tx_info["chrom"]).replace("chr", "")
                strand = tx_info["strand"]
                exons = tx_info["exons"]
                key = (chrom, strand)

                if key in eclip_data:
                    eclip_1d = map_genomic_intervals_to_transcript(exons, strand, eclip_data[key], l)
                    if np.any(eclip_1d > 0):
                        has_eclip = True
                        eclip_track[:, 0] = eclip_1d

            # Zusammenfuegen zu (L, 8)
            multi_track = np.concatenate([six_track, mirna_track, eclip_track], axis=1)

            # Zu aktuellem Chunk hinzufuegen
            current_chunk_items.append({
                "track": multi_track,
                "transcript_id": tx_id,
                "gene_id": str(row.get("ensembl_gene_id", "")),
                "gene_symbol": str(row.get("hgnc_symbol", "")),
                "half_life_transformed": float(row.get("half_life_transformed", np.nan)),
                "half_life": float(row.get("half_life", np.nan)),
                "rate": float(row.get("rate", np.nan)),
                "length": l,
                "has_mirna": has_mirna,
                "has_eclip": has_eclip,
                "has_gtf": has_gtf,
            })
            pbar.update(1)

            # Inkrementelles Speichern nach jeweils chunk_size Transkripten
            if len(current_chunk_items) >= args.chunk_size:
                save_chunk(chunk_idx, current_chunk_items, chunks_dir)
                chunk_idx += 1
                current_chunk_items = []

        # Letzten unvollstaendigen Chunk speichern
        if current_chunk_items:
            save_chunk(chunk_idx, current_chunk_items, chunks_dir)

    # 3. Alle Chunks zusammenfuehren zur finalen Datei
    merge_all_chunks(chunks_dir, out_file)


if __name__ == "__main__":
    main()
