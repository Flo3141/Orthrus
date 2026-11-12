#!/usr/bin/env python3
"""
Writes one Saluki input file in which every transcript appears in several variants: unchanged and with the central A
of every DRACH site in the 3' UTR moved by one position. Saluki predicts the file once; the column 'ID' separates the
variants afterwards.

Input is a file in the format of hIPSC_CM_ej_cds.txt: tab-separated, one transcript per line, column 'sequence' holds
comma-separated tokens, one per base. Lowercase = outside the CDS, uppercase = first base of a codon, suffix 'ej' /
'EJ' = exon junction.

  <stem>_DRACH_shift.txt         new first column ID = <ensembl_transcript_id>_drach_shift_<variant>, then all input
                                 columns unchanged except 'sequence'. Rows per transcript, in this order:
                                   wt     unchanged
                                   left   A swaps places with R:  D G A C H -> D A G C H  (only DGACH; DAACH stays)
                                   right  A swaps places with C:  D R A C H -> D R C A H
  <stem>_DRACH_shift_report.tsv  per transcript and direction: sites, shifted sites, skipped sites, checks

With --controls additionally the variants left_control and right_control: as many swaps of the same dinucleotide
(GA -> AG, AC -> CA) in the 3' UTR outside all DRACH sites, as close as possible to one shifted site each. Any swap
moves a prediction, so only the difference shifted - control belongs to the motif (see docs/drach_shift_analyse.md).
Every transcript gets every variant, also when nothing was shifted; the report tells which rows equal WT.

Which sites are shifted is decided by drach_motifs.plan_shifts(), the same function the Orthrus analysis
(xai_analysis.py --analyses drach_shift) uses, so both models see the same shifts. Only the bases are swapped: the
CDS case and the 'ej' suffix stay at their positions, like the CDS and splice channels in the Orthrus tracks.
The 3' UTR starts 3 positions after the last uppercase token, as in six_track_dataset.py; transcripts without
uppercase tokens have no annotated 3' UTR and stay unchanged.

Needs only numpy (no torch), so it runs in any environment.

Usage:
  python create_saluki_drach_shift_files.py
  python create_saluki_drach_shift_files.py --input <hIPSC_CM_ej_cds.txt> --output_dir <dir> --controls
"""

import argparse
import re
import sys
from contextlib import ExitStack
from pathlib import Path

from drach_motifs import DRACH, SHIFT_NAMES, SHIFTS, Shift, match_swap_controls, plan_shifts, swap_pairs
from util import DATASET_DIRS


DEFAULT_INPUT = DATASET_DIRS["hipsc_cm"] / "hIPSC_CM_ej_cds.txt"
SEQUENCE_COLUMN = "sequence"
TX_COLUMN = "ensembl_transcript_id"
VARIANT_ID_COLUMN = "ID"
WT_VARIANT = "wt"
CONTROL_SUFFIX = "_control"
TOKEN_RE = re.compile(r"^[acgtnACGTN](ej|EJ)?$")  # n = unknown base, never part of a motif
REPORT_COLUMNS = [
    "ensembl_transcript_id", "direction", "length", "utr3_start", "n_sites", "n_shifted", "n_unshiftable",
    "n_skipped_new_site", "sites_after_shift", "created_sites", "control_found",
]


def variant_id(tx_id: str, variant: str) -> str:
    """Row ID in the output file; split it with rsplit('_drach_shift_', 1) to get transcript and variant back."""
    return f"{tx_id}_drach_shift_{variant}"


def parse_tokens(field: str, tx_id: str) -> list:
    tokens = field.split(",")
    bad = [t for t in tokens if not TOKEN_RE.match(t)]
    if bad:
        raise ValueError(f"[Error] Transcript {tx_id}: unexpected sequence tokens {sorted(set(bad))[:10]}")
    return tokens


def tokens_to_rna(tokens: list) -> str:
    """Bases only, uppercase RNA alphabet (T -> U), as used by the motif patterns."""
    return "".join(t[0] for t in tokens).upper().replace("T", "U")


def utr3_start_of(tokens: list) -> int:
    """First 3' UTR position: last uppercase token (first base of the stop codon) + 3; -1 = no annotation."""
    last = max((i for i, t in enumerate(tokens) if t[0].isupper()), default=None)
    return -1 if last is None else min(last + 3, len(tokens))


def swap_tokens(tokens: list, positions: list) -> list:
    """Swaps the bases of the tokens at p and p + 1; case (CDS) and suffix (exon junction) stay at their position."""
    out = list(tokens)

    def put(base: str, target: str) -> str:
        return (base.upper() if target[0].isupper() else base.lower()) + target[1:]

    for p in positions:
        a, b = out[p], out[p + 1]
        out[p], out[p + 1] = put(b[0], a), put(a[0], b)
    return out


def process_transcript(tokens: list, shift: Shift, with_controls: bool) -> tuple:
    """Returns (shifted tokens, control tokens or None, report dict) for one transcript and one direction."""
    seq = tokens_to_rna(tokens)
    utr3 = utr3_start_of(tokens)
    report = {"length": len(tokens), "utr3_start": utr3}
    if utr3 < 0:
        report.update(n_sites=0, n_shifted=0, n_unshiftable=0, n_skipped_new_site=0, sites_after_shift=0, created_sites=0, control_found="")
        return tokens, (tokens if with_controls else None), report

    plan = plan_shifts(seq, utr3, shift)
    shifted = swap_tokens(tokens, plan.swaps)
    expected = swap_pairs(seq, plan.swaps, shift.pair)   # also checks that every swap starts the expected dinucleotide
    if tokens_to_rna(shifted) != expected:
        raise RuntimeError(f"[Error] Token swap does not match the sequence swap ({shift.name})!")
    after = DRACH.sites(expected, utr3)
    report.update(
        n_sites=len(plan.sites), n_shifted=len(plan.swaps), n_unshiftable=plan.n_unshiftable,
        n_skipped_new_site=plan.n_would_create, sites_after_shift=len(after), created_sites=len(set(after) - set(plan.sites)),
    )

    control = None
    if with_controls:
        controls = match_swap_controls(seq, utr3, DRACH, plan.sites, shift.pair, plan.swaps) if plan.swaps else []
        report["control_found"] = controls is not None
        # Without enough control dinucleotides the control stays WT; the report marks these transcripts
        control = swap_tokens(tokens, controls) if controls else tokens
    else:
        report["control_found"] = ""
    return shifted, control, report


def main():
    parser = argparse.ArgumentParser(description="Saluki input files with the central A of DRACH shifted by one position", allow_abbrev=False)
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help=f"Input file in the ej_cds format (default: {DEFAULT_INPUT})")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: directory of --input)")
    parser.add_argument("--directions", nargs="+", default=list(SHIFT_NAMES), choices=list(SHIFT_NAMES), help="Directions to write (default: left right)")
    parser.add_argument("--controls", action="store_true", help="Also write the control variants (swapped dinucleotides outside DRACH)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"[Error] Input file not found: {in_path}")
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts = [s for s in SHIFTS if s.name in args.directions]

    out_path = out_dir / f"{in_path.stem}_DRACH_shift.txt"
    report_path = out_dir / f"{in_path.stem}_DRACH_shift_report.tsv"
    if in_path.resolve() in {out_path.resolve(), report_path.resolve()}:
        raise ValueError("[Error] An output file would overwrite the input file!")

    variants = [WT_VARIANT, *(s.name for s in shifts), *(f"{s.name}{CONTROL_SUFFIX}" for s in shifts if args.controls)]
    print(f"[Saluki DRACH] Input:    {in_path}")
    print(f"[Saluki DRACH] Variants: {variants} (one row each per transcript, ID = <transcript>_drach_shift_<variant>)")

    totals = {s.name: dict(transcripts=0, unannotated=0, with_shift=0, sites=0, shifted=0, unshiftable=0, skipped_new=0, created=0, no_control=0) for s in shifts}
    n_rows = 0
    with ExitStack() as stack:
        # newline="" keeps the line endings of the input
        def open_file(path: Path, mode: str):
            return stack.enter_context(open(path, mode, encoding="utf-8", newline=""))

        src = open_file(in_path, "r")
        out = open_file(out_path, "w")
        report = open_file(report_path, "w")

        header_line = src.readline()
        header_ending = header_line[len(header_line.rstrip("\r\n")):]
        header = header_line.rstrip("\r\n").split("\t")
        if SEQUENCE_COLUMN not in header or TX_COLUMN not in header:
            raise KeyError(f"[Error] Columns '{SEQUENCE_COLUMN}' and '{TX_COLUMN}' required, found: {header}")
        if VARIANT_ID_COLUMN in header:
            raise KeyError(f"[Error] The input already has a column '{VARIANT_ID_COLUMN}'!")
        seq_col, id_col = header.index(SEQUENCE_COLUMN), header.index(TX_COLUMN)
        out.write("\t".join([VARIANT_ID_COLUMN, *header]) + header_ending)
        report.write("\t".join(REPORT_COLUMNS) + "\n")

        for n_line, line in enumerate(src, start=2):
            ending = line[len(line.rstrip("\r\n")):]
            fields = line[:len(line) - len(ending)].split("\t")
            if len(fields) != len(header):
                raise ValueError(f"[Error] Line {n_line}: {len(fields)} fields, header has {len(header)}")
            tx_id = fields[id_col]
            tokens = parse_tokens(fields[seq_col], tx_id)

            rows = {WT_VARIANT: tokens}
            for shift in shifts:
                shifted, control, rep = process_transcript(tokens, shift, args.controls)
                rows[shift.name] = shifted
                if args.controls:
                    rows[f"{shift.name}{CONTROL_SUFFIX}"] = control
                report.write("\t".join(str(v) for v in [tx_id, shift.name, *(rep[c] for c in REPORT_COLUMNS[2:])]) + "\n")

                t = totals[shift.name]
                t["transcripts"] += 1
                t["unannotated"] += rep["utr3_start"] < 0
                t["with_shift"] += rep["n_shifted"] > 0
                t["sites"] += rep["n_sites"]; t["shifted"] += rep["n_shifted"]
                t["unshiftable"] += rep["n_unshiftable"]; t["skipped_new"] += rep["n_skipped_new_site"]
                t["created"] += rep["created_sites"]
                t["no_control"] += rep["control_found"] is False

            for variant in variants:
                fields[seq_col] = ",".join(rows[variant])
                out.write("\t".join([variant_id(tx_id, variant), *fields]) + ending)
                n_rows += 1

    print("\n" + "=" * 70)
    print("             SALUKI DRACH SHIFT FILE             ")
    print("=" * 70)
    print(f"File:                              {out_path}")
    print(f"Rows:                              {n_rows:,} ({len(variants)} per transcript: {variants})")
    for s in shifts:
        t = totals[s.name]
        print(f"{s.label}")
        print(f"  Transcripts:                     {t['transcripts']:,} ({t['unannotated']:,} without 3' UTR annotation, unchanged)")
        print(f"  Transcripts with >= 1 shift:     {t['with_shift']:,}")
        print(f"  DRACH sites in 3' UTRs:          {t['sites']:,}")
        print(f"  Shifted:                         {t['shifted']:,}")
        print(f"  Not shiftable (DAACH, left):     {t['unshiftable']:,}")
        print(f"  Skipped (would form a new site): {t['skipped_new']:,}")
        print(f"  Created sites (expected 0):      {t['created']:,}")
        if args.controls:
            print(f"  Without enough controls:         {t['no_control']:,} (control row keeps WT, see report)")
    print(f"Report:                            {report_path}")
    print("=" * 70)

    if any(t["created"] for t in totals.values()):
        print("[Warning] Some shifts created new DRACH sites, check the report (created_sites).", file=sys.stderr)


if __name__ == "__main__":
    main()
