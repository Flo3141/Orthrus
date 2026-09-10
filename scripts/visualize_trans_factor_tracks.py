#!/usr/bin/env python3
"""
Interaktiver HTML-Viewer zur Verifizierung von Trans-Faktor Tracks (TargetScan, ENCODE eCLIP)
sowie der Sequenz-, CDS- und Splice-Annotationen fuer den Saluki-Datensatz.

Findet automatisch Transkripte in einer NPZ-Datei (oder einem Chunks-Ordner),
die SOWOHL TargetScan miRNA-Bindestellen ALS AUCH ENCODE eCLIP RBP-Peaks besitzen,
und erzeugt eine interaktive, farbcodierte HTML-Datei zur visuellen Positionspruefung im Browser.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import webbrowser


def load_candidates_from_npz(input_path: Path, target_tx_id: str = None, max_candidates: int = 15) -> list:
    """
    Sucht Transkripte mit sowohl TargetScan (>0) als auch eCLIP (>0) Peaks
    aus einer Master-NPZ, einem Chunk oder einem Chunks-Verzeichnis.
    """
    files_to_check = []
    if input_path.is_dir():
        files_to_check = sorted(input_path.glob("chunk_*.npz"))
        if not files_to_check:
            files_to_check = sorted(input_path.glob("*.npz"))
    elif input_path.is_file():
        files_to_check = [input_path]
    else:
        raise FileNotFoundError(f"Pfad '{input_path}' existiert nicht.")

    print(f"Suche Transkripte in {len(files_to_check)} Datei(en)...")
    candidates = []

    bases = np.array(["A", "C", "G", "U"])

    for file_path in files_to_check:
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"[Warnung] Konnte {file_path.name} nicht laden: {e}")
            continue

        tracks = data["tracks"]
        tx_ids = data["ensembl_transcript_id"]
        gene_ids = data["ensembl_gene_id"] if "ensembl_gene_id" in data else [""] * len(tx_ids)
        symbols = data["hgnc_symbol"] if "hgnc_symbol" in data else [""] * len(tx_ids)
        half_lives = data["half_life"] if "half_life" in data else [np.nan] * len(tx_ids)

        for i, track in enumerate(tracks):
            if track is None or len(track) == 0:
                continue

            tx = str(tx_ids[i])
            clean_tx = tx.split(".")[0]

            if target_tx_id and target_tx_id not in [tx, clean_tx]:
                continue

            # Kanalkonfiguration: [A, C, G, U, CDS, Splice, TargetScan, eCLIP]
            has_mirna = bool(np.any(track[:, 6] > 0))
            has_eclip = bool(np.any(track[:, 7] > 0))

            # Nur Transkripte mit BEIDEN trans-Faktor Signalen auswaehlen (oder wenn gezielt gesucht)
            if target_tx_id or (has_mirna and has_eclip):
                l = len(track)

                # Sequenz rekonstruieren aus One-Hot (Kanäle 0..3)
                oh = track[:, :4]
                has_base = oh.any(axis=1)
                base_idx = np.argmax(oh, axis=1)
                seq_arr = np.where(has_base, bases[base_idx], "N")
                seq_str = "".join(seq_arr)

                # CDS & Stop-Codon analysieren
                cds_track = track[:, 4]
                upper_indices = np.where(cds_track > 0)[0].tolist()
                
                cds_start = int(upper_indices[0]) if upper_indices else None
                last_upper = int(upper_indices[-1]) if upper_indices else None
                
                # In Saluki ist das Stop-Codon das letzte 3er-Codon (beginnend bei last_upper)
                if last_upper is not None and last_upper + 3 <= l:
                    stop_codon_seq = seq_str[last_upper : last_upper + 3]
                    utr3_start = last_upper + 3
                else:
                    stop_codon_seq = "N/A"
                    utr3_start = (last_upper + 3) if last_upper is not None else 0

                # Splice Junctions
                splice_indices = np.where(track[:, 5] > 0)[0].tolist()

                # TargetScan miRNA Intervalle extrahieren
                ts_track = track[:, 6]
                ts_intervals = []
                in_peak = False
                p_start = 0
                for pos, score in enumerate(ts_track):
                    if score > 0 and not in_peak:
                        in_peak = True
                        p_start = pos
                    elif score == 0 and in_peak:
                        in_peak = False
                        peak_scores = ts_track[p_start:pos]
                        ts_intervals.append({
                            "start": p_start,
                            "end": pos,
                            "max_score": round(float(np.max(peak_scores)), 4),
                            "mean_score": round(float(np.mean(peak_scores)), 4),
                            "score": round(float(np.max(peak_scores)), 4),
                            "length": pos - p_start,
                            "rel_utr_start": p_start - utr3_start if utr3_start else None,
                            "rel_utr_end": pos - utr3_start if utr3_start else None,
                        })
                if in_peak:
                    peak_scores = ts_track[p_start:l]
                    ts_intervals.append({
                        "start": p_start,
                        "end": l,
                        "max_score": round(float(np.max(peak_scores)), 4),
                        "mean_score": round(float(np.mean(peak_scores)), 4),
                        "score": round(float(np.max(peak_scores)), 4),
                        "length": l - p_start,
                        "rel_utr_start": p_start - utr3_start if utr3_start else None,
                        "rel_utr_end": l - utr3_start if utr3_start else None,
                    })

                # ENCODE eCLIP Intervalle extrahieren
                eclip_track = track[:, 7]
                eclip_intervals = []
                in_peak = False
                p_start = 0
                for pos, score in enumerate(eclip_track):
                    if score > 0 and not in_peak:
                        in_peak = True
                        p_start = pos
                    elif score == 0 and in_peak:
                        in_peak = False
                        peak_scores = eclip_track[p_start:pos]
                        eclip_intervals.append({
                            "start": p_start,
                            "end": pos,
                            "max_score": round(float(np.max(peak_scores)), 4),
                            "mean_score": round(float(np.mean(peak_scores)), 4),
                            "length": pos - p_start,
                        })
                if in_peak:
                    peak_scores = eclip_track[p_start:l]
                    eclip_intervals.append({
                        "start": p_start,
                        "end": l,
                        "max_score": round(float(np.max(peak_scores)), 4),
                        "mean_score": round(float(np.mean(peak_scores)), 4),
                        "length": l - p_start,
                    })

                candidates.append({
                    "transcript_id": tx,
                    "gene_id": str(gene_ids[i]),
                    "gene_symbol": str(symbols[i]),
                    "half_life": float(half_lives[i]) if (half_lives[i] is not None and not np.isnan(half_lives[i])) else None,
                    "length": l,
                    "sequence": seq_str,
                    "cds_track": cds_track.astype(int).tolist(),
                    "splice_track": track[:, 5].astype(int).tolist(),
                    "ts_track": np.round(track[:, 6], 4).tolist(),
                    "eclip_track": np.round(track[:, 7], 4).tolist(),
                    "cds_start": cds_start,
                    "last_upper": last_upper,
                    "stop_codon_seq": stop_codon_seq,
                    "utr3_start": utr3_start,
                    "splice_indices": splice_indices,
                    "ts_intervals": ts_intervals,
                    "eclip_intervals": eclip_intervals,
                })

                if len(candidates) >= max_candidates:
                    print(f"Limit von {max_candidates} Kandidaten erreicht.")
                    return candidates

    print(f"Insgesamt {len(candidates)} passende Transkripte mit beiden Signalen gefunden.")
    return candidates


def generate_html_viewer(candidates: list, output_html_path: Path):
    """
    Generiert eine interaktive, autarke HTML-Seite mit ansprechender visueller
    Hervorhebung fuer Sequenz, CDS, Spleissstellen, TargetScan und eCLIP Tracks.
    """
    json_data = json.dumps(candidates)

    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trans-Faktor Tracks Verifikations-Viewer</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            
            /* Farben fuer Nukleotide */
            --color-a: #10b981; /* Gruen */
            --color-c: #0284c7; /* Blau */
            --color-g: #f59e0b; /* Orange/Gelb */
            --color-u: #ef4444; /* Rot */
            --color-n: #64748b;
            
            /* Farben fuer Tracks */
            --color-cds: #8b5cf6;       /* Violett */
            --color-splice: #ec4899;    /* Pink/Magenta */
            --color-ts: #f43f5e;        /* Rose / TargetScan */
            --color-eclip: #06b6d4;     /* Cyan / eCLIP */
            --color-utr: #eab308;       /* Amber */
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }}

        h1 {{
            font-size: 1.6rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .badge {{
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 9999px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-ts {{ background-color: rgba(244, 63, 94, 0.2); color: var(--color-ts); border: 1px solid var(--color-ts); }}
        .badge-eclip {{ background-color: rgba(6, 182, 212, 0.2); color: var(--color-eclip); border: 1px solid var(--color-eclip); }}
        .badge-cds {{ background-color: rgba(139, 92, 246, 0.2); color: var(--color-cds); border: 1px solid var(--color-cds); }}

        .selector-box {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        select {{
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.95rem;
            cursor: pointer;
            outline: none;
        }}
        select:focus {{
            border-color: var(--color-eclip);
        }}

        /* Summary Cards */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}

        .card h3 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .info-row:last-child {{
            border-bottom: none;
        }}
        .info-label {{
            color: var(--text-secondary);
        }}
        .info-value {{
            font-weight: 600;
            font-family: monospace;
        }}

        /* Legende */
        .legend {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            padding: 12px 16px;
            background-color: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.85rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .color-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}

        /* Mini-Map / Uebersichts-Track */
        .overview-panel {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .overview-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .track-canvas-container {{
            position: relative;
            width: 100%;
            height: 120px;
            background-color: #0b1120;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            cursor: pointer;
        }}

        canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        /* Navigation Buttons */
        .nav-bar {{
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .nav-btn {{
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .nav-btn:hover {{
            background-color: #334155;
            border-color: #64748b;
        }}
        .nav-btn.active {{
            background-color: var(--color-eclip);
            color: #000;
            font-weight: 600;
        }}

        /* Sequenz- und Track-Inspektor */
        .inspector-panel {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}

        .sequence-scroll-wrapper {{
            overflow-x: auto;
            background-color: #0b1120;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 16px;
            max-height: 520px;
            overflow-y: auto;
        }}

        .seq-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(28px, 1fr));
            gap: 4px;
            font-family: monospace;
        }}

        .nucleotide-cell {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4px 2px;
            border-radius: 4px;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid transparent;
            font-size: 0.85rem;
            cursor: pointer;
            transition: transform 0.1s, border-color 0.1s;
            position: relative;
        }}
        .nucleotide-cell:hover {{
            transform: scale(1.15);
            z-index: 10;
            border-color: #fff;
        }}

        .nt-pos {{
            font-size: 0.6rem;
            color: var(--text-secondary);
            margin-bottom: 2px;
        }}

        .nt-char {{
            font-weight: 700;
            font-size: 0.95rem;
        }}
        .nt-char.A {{ color: var(--color-a); }}
        .nt-char.C {{ color: var(--color-c); }}
        .nt-char.G {{ color: var(--color-g); }}
        .nt-char.U {{ color: var(--color-u); }}
        .nt-char.T {{ color: var(--color-u); }}

        /* Track Indikatoren auf der Nukleotid-Zelle */
        .indicators {{
            display: flex;
            gap: 2px;
            margin-top: 4px;
            height: 4px;
        }}
        .dot {{
            width: 4px;
            height: 4px;
            border-radius: 50%;
        }}
        .dot-cds {{ background-color: var(--color-cds); }}
        .dot-splice {{ background-color: var(--color-splice); }}
        .dot-ts {{ background-color: var(--color-ts); }}
        .dot-eclip {{ background-color: var(--color-eclip); }}

        /* Spezielle Rahmen */
        .cell-ts {{
            background-color: rgba(244, 63, 94, 0.18) !important;
            border-color: rgba(244, 63, 94, 0.6) !important;
        }}
        .cell-eclip {{
            background-color: rgba(6, 182, 212, 0.18) !important;
            border-color: rgba(6, 182, 212, 0.6) !important;
        }}
        .cell-both {{
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.25), rgba(6, 182, 212, 0.25)) !important;
            border-color: #facc15 !important;
        }}
        .cell-stop {{
            border: 2px solid #ef4444 !important;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
        }}

        /* Tooltip */
        #tooltip {{
            position: fixed;
            display: none;
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: #fff;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
        }}

        .alert-box {{
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .alert-success {{
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid #10b981;
            color: #34d399;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🧬 Trans-Faktor Tracks Verifikations-Viewer</h1>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Visuelle Prüfung von Sequenz, CDS, Splice-Sites, TargetScan & ENCODE eCLIP Tracks</p>
            </div>
            <div class="selector-box">
                <label for="transcriptSelect" style="font-size: 0.9rem; color: var(--text-secondary);">Transkript auswählen:</label>
                <select id="transcriptSelect" onchange="renderTranscript(this.value)"></select>
            </div>
        </header>

        <!-- Legende -->
        <div class="legend">
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-a);"></div> A (Adenin)</div>
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-c);"></div> C (Cytosin)</div>
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-g);"></div> G (Guanin)</div>
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-u);"></div> U/T (Uracil/Thymin)</div>
            <div class="legend-item" style="margin-left: 20px;"><div class="color-dot" style="background-color: var(--color-cds);"></div> CDS (Codon Start)</div>
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-splice);"></div> Splice Junction</div>
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-ts);"></div> TargetScan miRNA (Kanal 6)</div>
            <div class="legend-item"><div class="color-dot" style="background-color: var(--color-eclip);"></div> ENCODE eCLIP (Kanal 7)</div>
        </div>

        <!-- Meta Information Cards -->
        <div class="cards-grid">
            <div class="card">
                <h3>Transkript & Gen Info <span class="badge badge-cds">Metadata</span></h3>
                <div class="info-row"><span class="info-label">Ensembl ID:</span><span class="info-value" id="infoTxId">-</span></div>
                <div class="info-row"><span class="info-label">Gene Symbol:</span><span class="info-value" id="infoSymbol">-</span></div>
                <div class="info-row"><span class="info-label">Ensembl Gene:</span><span class="info-value" id="infoGeneId">-</span></div>
                <div class="info-row"><span class="info-label">Sequenzlänge (nt):</span><span class="info-value" id="infoLen">-</span></div>
                <div class="info-row"><span class="info-label">Half-Life (h):</span><span class="info-value" id="infoHalfLife">-</span></div>
            </div>

            <div class="card">
                <h3>CDS & 3'-UTR Architektur <span class="badge badge-cds">Leseraster</span></h3>
                <div class="info-row"><span class="info-label">CDS Start (Codon 1):</span><span class="info-value" id="infoCdsStart">-</span></div>
                <div class="info-row"><span class="info-label">Letztes Codon (Stop):</span><span class="info-value" id="infoStopCodon">-</span></div>
                <div class="info-row"><span class="info-label">Stop-Codon Sequenz:</span><span class="info-value" id="infoStopSeq">-</span></div>
                <div class="info-row"><span class="info-label">3'-UTR Start Position:</span><span class="info-value" id="infoUtr3Start">-</span></div>
                <div class="info-row"><span class="info-label">Anzahl Spleißstellen:</span><span class="info-value" id="infoSpliceCount">-</span></div>
            </div>

            <div class="card">
                <h3>TargetScan miRNA <span class="badge badge-ts">Kanal 6</span></h3>
                <div class="info-row"><span class="info-label">Bindestellen gesamt:</span><span class="info-value" id="infoTsCount">-</span></div>
                <div class="info-row"><span class="info-label">Alle in 3'-UTR?</span><span class="info-value" id="infoTsUtrCheck">-</span></div>
                <div class="info-row"><span class="info-label">Max. Score (abs):</span><span class="info-value" id="infoTsMaxScore">-</span></div>
                <div class="info-row"><span class="info-label">Erste Bindestelle:</span><span class="info-value" id="infoTsFirst">-</span></div>
            </div>

            <div class="card">
                <h3>ENCODE eCLIP Peaks <span class="badge badge-eclip">Kanal 7</span></h3>
                <div class="info-row"><span class="info-label">Peaks gesamt:</span><span class="info-value" id="infoEclipCount">-</span></div>
                <div class="info-row"><span class="info-label">Max. Signalwert (L2FC):</span><span class="info-value" id="infoEclipMaxScore">-</span></div>
                <div class="info-row"><span class="info-label">Breitester Peak (nt):</span><span class="info-value" id="infoEclipMaxLen">-</span></div>
                <div class="info-row"><span class="info-label">Erster Peak:</span><span class="info-value" id="infoEclipFirst">-</span></div>
            </div>
        </div>

        <!-- Mini-Map / Gesamtuebersicht -->
        <div class="overview-panel">
            <div class="overview-title">
                <span>Transkript-Architektur & Dichte-Profile (Gesamtübersicht 0 .. L)</span>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">Klicke auf die Übersicht, um direkt zur Position zu springen</span>
            </div>
            <div class="track-canvas-container" id="canvasContainer" onclick="handleCanvasClick(event)">
                <canvas id="overviewCanvas"></canvas>
            </div>
        </div>

        <!-- Schnellnavigation Buttons -->
        <div class="nav-bar" id="quickNav">
            <span style="font-size: 0.85rem; color: var(--text-secondary); margin-right: 8px;">Schnellnavigation:</span>
            <!-- Dynamisch generierte Jump-Buttons -->
        </div>

        <!-- Detail Sequenz Inspektor -->
        <div class="inspector-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="font-size: 1.1rem; font-weight: 600;">Positionsgenauer Nukleotid- & Track-Inspektor</h3>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">
                    Scroll horizontal/vertikal durch alle Basen
                </div>
            </div>
            <div class="sequence-scroll-wrapper" id="seqWrapper">
                <div class="seq-grid" id="seqGrid"></div>
            </div>
        </div>
    </div>

    <div id="tooltip"></div>

    <script>
        const candidates = {json_data};
        let currentCandidate = null;

        // Dropdown initialisieren
        const selectEl = document.getElementById("transcriptSelect");
        candidates.forEach((c, idx) => {{
            const opt = document.createElement("option");
            opt.value = idx;
            opt.textContent = `${{c.gene_symbol || 'Unknown'}} (${{c.transcript_id}}) | ${{c.ts_intervals.length}} miRNA, ${{c.eclip_intervals.length}} eCLIP | L=${{c.length}}`;
            selectEl.appendChild(opt);
        }});

        function renderTranscript(index) {{
            const c = candidates[index];
            currentCandidate = c;

            // Metadata füllen
            document.getElementById("infoTxId").textContent = c.transcript_id;
            document.getElementById("infoSymbol").textContent = c.gene_symbol || "N/A";
            document.getElementById("infoGeneId").textContent = c.gene_id || "N/A";
            document.getElementById("infoLen").textContent = c.length;
            document.getElementById("infoHalfLife").textContent = c.half_life !== null ? c.half_life.toFixed(2) : "N/A";

            document.getElementById("infoCdsStart").textContent = c.cds_start !== null ? c.cds_start : "Keine CDS";
            document.getElementById("infoStopCodon").textContent = c.last_upper !== null ? `${{c.last_upper}} .. ${{c.last_upper+2}}` : "N/A";
            document.getElementById("infoStopSeq").textContent = c.stop_codon_seq;
            document.getElementById("infoUtr3Start").textContent = c.utr3_start;
            document.getElementById("infoSpliceCount").textContent = c.splice_indices.length;

            // TargetScan Info
            document.getElementById("infoTsCount").textContent = c.ts_intervals.length;
            const allInUtr = c.ts_intervals.every(ts => ts.start >= c.utr3_start);
            const tsUtrCheckEl = document.getElementById("infoTsUtrCheck");
            if (c.ts_intervals.length === 0) {{
                tsUtrCheckEl.textContent = "Keine";
                tsUtrCheckEl.style.color = "var(--text-secondary)";
            }} else if (allInUtr) {{
                tsUtrCheckEl.textContent = "JA (100% Valid)";
                tsUtrCheckEl.style.color = "#34d399";
            }} else {{
                tsUtrCheckEl.textContent = "Nein (vor 3' UTR!)";
                tsUtrCheckEl.style.color = "#f87171";
            }}

            const maxTsScore = c.ts_intervals.length > 0 ? Math.max(...c.ts_intervals.map(t => t.max_score || t.score)) : 0;
            document.getElementById("infoTsMaxScore").textContent = maxTsScore.toFixed(4);
            document.getElementById("infoTsFirst").textContent = c.ts_intervals.length > 0 ? `Pos ${{c.ts_intervals[0].start}}..${{c.ts_intervals[0].end}}` : "Keine";

            // eCLIP Info
            document.getElementById("infoEclipCount").textContent = c.eclip_intervals.length;
            const maxEclipScore = c.eclip_intervals.length > 0 ? Math.max(...c.eclip_intervals.map(e => e.max_score)) : 0;
            const maxEclipLen = c.eclip_intervals.length > 0 ? Math.max(...c.eclip_intervals.map(e => e.length)) : 0;
            document.getElementById("infoEclipMaxScore").textContent = maxEclipScore.toFixed(4);
            document.getElementById("infoEclipMaxLen").textContent = `${{maxEclipLen}} nt`;
            document.getElementById("infoEclipFirst").textContent = c.eclip_intervals.length > 0 ? `Pos ${{c.eclip_intervals[0].start}}..${{c.eclip_intervals[0].end}}` : "Keine";

            // Quick Nav erstellen
            const navEl = document.getElementById("quickNav");
            navEl.innerHTML = '<span style="font-size: 0.85rem; color: var(--text-secondary); margin-right: 8px;">Schnellnavigation:</span>';
            
            if (c.cds_start !== null) {{
                navEl.innerHTML += `<button class="nav-btn" onclick="scrollToPos(${{c.cds_start}})">▶ Start-Codon (${{c.cds_start}})</button>`;
            }}
            if (c.last_upper !== null) {{
                navEl.innerHTML += `<button class="nav-btn" style="border-color: var(--color-u);" onclick="scrollToPos(${{c.last_upper}})">🛑 Stop-Codon (${{c.last_upper}})</button>`;
            }}
            c.ts_intervals.forEach((ts, i) => {{
                const sInfo = ts.max_score !== undefined ? `Max: ${{ts.max_score}}` : ts.score;
                navEl.innerHTML += `<button class="nav-btn" style="border-color: var(--color-ts); color: var(--color-ts);" onclick="scrollToPos(${{ts.start}})">🎯 TargetScan #${{i+1}} (${{ts.start}}, ${{sInfo}})</button>`;
            }});
            c.eclip_intervals.forEach((ec, i) => {{
                navEl.innerHTML += `<button class="nav-btn" style="border-color: var(--color-eclip); color: var(--color-eclip);" onclick="scrollToPos(${{ec.start}})">⚡ eCLIP #${{i+1}} (${{ec.start}})</button>`;
            }});

            // Canvas rendern
            drawOverview(c);

            // Detail-Grid rendern
            renderGrid(c);
        }}

        function drawOverview(c) {{
            const canvas = document.getElementById("overviewCanvas");
            const container = document.getElementById("canvasContainer");
            canvas.width = container.clientWidth * window.devicePixelRatio;
            canvas.height = 120 * window.devicePixelRatio;
            const ctx = canvas.getContext("2d");
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            const w = container.clientWidth;
            const h = 120;
            const l = c.length;

            ctx.clearRect(0, 0, w, h);

            // 1. Hintergrund & 5' UTR / CDS / 3' UTR Balken
            const scaleX = (pos) => (pos / l) * w;

            // Transkript Backbone
            ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
            ctx.fillRect(0, 45, w, 14);

            // CDS Region
            if (c.cds_start !== null && c.utr3_start !== null) {{
                const xStart = scaleX(c.cds_start);
                const xEnd = scaleX(c.utr3_start);
                ctx.fillStyle = "rgba(139, 92, 246, 0.4)";
                ctx.fillRect(xStart, 42, Math.max(2, xEnd - xStart), 20);
                
                // CDS Label
                ctx.fillStyle = "#a78bfa";
                ctx.font = "10px sans-serif";
                ctx.fillText("CDS", (xStart + xEnd) / 2 - 10, 36);
            }}

            // Stop-Codon Markierung
            if (c.last_upper !== null) {{
                const xStop = scaleX(c.last_upper);
                ctx.fillStyle = "#ef4444";
                ctx.fillRect(xStop, 38, 3, 28);
                ctx.fillText("Stop", xStop - 10, 78);
            }}

            // Splice-Stellen
            ctx.fillStyle = "rgba(236, 72, 153, 0.7)";
            c.splice_indices.forEach(idx => {{
                ctx.fillRect(scaleX(idx), 40, 2, 24);
            }});

            // 2. TargetScan miRNA Peaks (oben, rot/pink)
            ctx.fillStyle = "rgba(244, 63, 94, 0.8)";
            c.ts_intervals.forEach(ts => {{
                const x1 = scaleX(ts.start);
                const x2 = scaleX(ts.end);
                const barW = Math.max(3, x2 - x1);
                ctx.fillRect(x1, 12, barW, 24);
            }});
            ctx.fillStyle = "#f43f5e";
            ctx.font = "10px sans-serif";
            ctx.fillText("TargetScan miRNA", 10, 20);

            // 3. ENCODE eCLIP Peaks (unten, cyan)
            ctx.fillStyle = "rgba(6, 182, 212, 0.8)";
            c.eclip_intervals.forEach(ec => {{
                const x1 = scaleX(ec.start);
                const x2 = scaleX(ec.end);
                const barW = Math.max(3, x2 - x1);
                ctx.fillRect(x1, 80, barW, 24);
            }});
            ctx.fillStyle = "#06b6d4";
            ctx.font = "10px sans-serif";
            ctx.fillText("ENCODE eCLIP Peaks", 10, 110);
        }}

        function handleCanvasClick(e) {{
            if (!currentCandidate) return;
            const container = document.getElementById("canvasContainer");
            const rect = container.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const ratio = clickX / rect.width;
            const targetPos = Math.floor(ratio * currentCandidate.length);
            scrollToPos(targetPos);
        }}

        function renderGrid(c) {{
            const gridEl = document.getElementById("seqGrid");
            gridEl.innerHTML = "";

            const frag = document.createDocumentFragment();
            const seq = c.sequence;
            const l = c.length;
            const spliceSet = new Set(c.splice_indices);

            for (let i = 0; i < l; i++) {{
                const nt = seq[i] || 'N';
                const cell = document.createElement("div");
                cell.id = `nt_${{i}}`;
                cell.className = "nucleotide-cell";

                const tsVal = (c.ts_track && c.ts_track[i] !== undefined) ? c.ts_track[i] : 0;
                const eclipVal = (c.eclip_track && c.eclip_track[i] !== undefined) ? c.eclip_track[i] : 0;
                const isTs = tsVal > 0;
                const isEclip = eclipVal > 0;
                const isStop = (c.last_upper !== null && i >= c.last_upper && i < c.last_upper + 3);

                if (isTs && isEclip) cell.classList.add("cell-both");
                else if (isTs) cell.classList.add("cell-ts");
                else if (isEclip) cell.classList.add("cell-eclip");

                if (isStop) cell.classList.add("cell-stop");

                // Content
                cell.innerHTML = `
                    <span class="nt-pos">${{i}}</span>
                    <span class="nt-char ${{nt}}">${{nt}}</span>
                    <div class="indicators">
                        ${{c.cds_track[i] ? '<div class="dot dot-cds" title="Codon Start"></div>' : ''}}
                        ${{spliceSet.has(i) ? '<div class="dot dot-splice" title="Splice Junction"></div>' : ''}}
                        ${{isTs ? '<div class="dot dot-ts" title="TargetScan"></div>' : ''}}
                        ${{isEclip ? '<div class="dot dot-eclip" title="eCLIP"></div>' : ''}}
                    </div>
                `;

                // Hover Tooltip: Zeigt den tatsächlichen basengenaue Score
                cell.onmouseenter = (e) => showTooltip(e, i, nt, c, isTs ? tsVal : undefined, isEclip ? eclipVal : undefined, spliceSet.has(i));
                cell.onmouseleave = hideTooltip;

                frag.appendChild(cell);
            }}

            gridEl.appendChild(frag);
        }}

        function scrollToPos(pos) {{
            const el = document.getElementById(`nt_${{pos}}`);
            if (el) {{
                el.scrollIntoView({{ behavior: "smooth", block: "center", inline: "center" }});
                el.style.transform = "scale(1.4)";
                el.style.borderColor = "#facc15";
                setTimeout(() => {{
                    el.style.transform = "";
                    el.style.borderColor = "";
                }}, 1500);
            }}
        }}

        const tooltip = document.getElementById("tooltip");
        function showTooltip(e, pos, nt, c, tsScore, eclipScore, isSplice) {{
            let region = "5' UTR";
            if (c.cds_start !== null && pos >= c.cds_start && pos < c.utr3_start) {{
                region = (c.last_upper !== null && pos >= c.last_upper) ? "🛑 STOP-CODON" : "CDS (Coding Sequence)";
            }} else if (pos >= c.utr3_start) {{
                region = `3' UTR (Offset +${{pos - c.utr3_start}} nt)`;
            }}

            let html = `
                <div style="font-weight: 700; margin-bottom: 4px; color: #facc15;">Position: ${{pos}} (1-basiert: ${{pos + 1}})</div>
                <div>Nukleotid: <b>${{nt}}</b></div>
                <div>Region: <b>${{region}}</b></div>
            `;
            if (c.cds_track[pos]) html += `<div style="color: var(--color-cds);">● Codon-Start (Reading Frame)</div>`;
            if (isSplice) html += `<div style="color: var(--color-splice);">● Exon Junction (Splice Site)</div>`;
            if (tsScore !== undefined) html += `<div style="color: var(--color-ts);">● TargetScan miRNA Score: <b>${{tsScore}}</b></div>`;
            if (eclipScore !== undefined) html += `<div style="color: var(--color-eclip);">● ENCODE eCLIP Signal: <b>${{eclipScore}}</b></div>`;

            tooltip.innerHTML = html;
            tooltip.style.display = "block";
            tooltip.style.left = `${{e.clientX + 14}}px`;
            tooltip.style.top = `${{e.clientY + 14}}px`;
        }}

        function hideTooltip() {{
            tooltip.style.display = "none";
        }}

        window.addEventListener("resize", () => {{
            if (currentCandidate) drawOverview(currentCandidate);
        }});

        // Erstes Transkript rendern
        if (candidates.length > 0) {{
            renderTranscript(0);
        }}
    </script>
</body>
</html>
"""

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[Erfolg] Interaktiver Track-Viewer gespeichert: {output_html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualisierung und Verifikation von TargetScan- und ENCODE eCLIP Tracks im Browser"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/saluki/saluki_multitrack_with_trans_factors_chunks",
        help="Pfad zur Master-NPZ, einem Chunk (.npz) oder dem Chunks-Ordner",
    )
    parser.add_argument(
        "--transcript_id",
        type=str,
        default=None,
        help="Gezielte Ensembl-Transkript-ID (z. B. ENST00000331001)",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=15,
        help="Maximale Anzahl an Transkripten, die in den HTML-Viewer geladen werden (Standard: 15)",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="trans_factor_track_verification.html",
        help="Pfad der auszugebenden HTML-Datei",
    )
    parser.add_argument(
        "--open_browser",
        action="store_true",
        help="Öffnet die generierte HTML-Datei nach Erstellung automatisch im Standard-Browser",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_html = Path(args.output_html)

    candidates = load_candidates_from_npz(
        input_path, target_tx_id=args.transcript_id, max_candidates=args.max_candidates
    )

    if not candidates:
        print("[Warnung] Keine Transkripte mit beiden Merkmalen gefunden.")
        return

    generate_html_viewer(candidates, out_html)

    if args.open_browser:
        webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
