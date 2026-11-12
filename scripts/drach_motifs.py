#!/usr/bin/env python3
"""
Sequence logic of the motif analyses, without model dependencies (only numpy and re).

Shared by drach_analysis.py / drach_shift_analysis.py (Orthrus models) and create_saluki_drach_shift_files.py
(Saluki input files), so that every consumer finds and shifts exactly the same sites.

Sequences are strings over A, C, G, U (other characters never match a motif). Positions are 0-based; utr3_start is
the first position of the 3' UTR, and only sites that start at or after it count.
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


# =============================================================================
# 1. Motifs
# =============================================================================

@dataclass(frozen=True)
class Motif:
    name: str
    pattern: str
    length: int
    mutated_offset: int               # position of the A that is replaced within the pattern
    control_next_base: Optional[str]  # base that has to follow a control A; None = any

    def sites(self, seq: str, utr3_start: int) -> list:
        """Start positions of all (also overlapping) sites that begin in the 3' UTR."""
        return [m.start() for m in re.finditer(f"(?={self.pattern})", seq) if m.start() >= utr3_start]


MOTIFS = (
    Motif("DRACH", "[AGU][AG]AC[ACU]", length=5, mutated_offset=2, control_next_base="C"),
    Motif("AUUUA", "AUUUA", length=5, mutated_offset=0, control_next_base="U"),
    Motif("UGUANAUA", "UGUA[ACGU]AUA", length=8, mutated_offset=3, control_next_base=None),
)
DRACH = next(m for m in MOTIFS if m.name == "DRACH")


# =============================================================================
# 2. Shifting the central A of DRACH by one position (swap with a neighbour)
# =============================================================================

@dataclass(frozen=True)
class Shift:
    name: str
    label: str
    swap_offset: int   # first of the two swapped positions within the DRACH site
    pair: str          # dinucleotide that is swapped (at the site and at the controls)


SHIFTS = (
    Shift("left", "left: DGACH > DAGCH", swap_offset=1, pair="GA"),
    Shift("right", "right: DRACH > DRCAH", swap_offset=2, pair="AC"),
)
SHIFT_NAMES = tuple(s.name for s in SHIFTS)


def swap_pairs(seq: str, positions: list, pair: str) -> str:
    """Swaps the bases at p and p + 1 for every p; each p must start the dinucleotide `pair`."""
    chars = list(seq)
    for p in positions:
        if chars[p] + chars[p + 1] != pair:
            raise ValueError(f"[Error] Position {p} starts {chars[p] + chars[p + 1]!r}, not {pair!r}!")
        chars[p], chars[p + 1] = chars[p + 1], chars[p]
    return "".join(chars)


def swap_creates_site(seq, position: int, utr3_start: int, motif: Motif) -> bool:
    """
    True if swapping the bases at `position` and `position + 1` creates a site that did not exist (local window).
    `seq` is a string or a list of characters (the sequence with the swaps applied so far).
    """
    lo = max(0, position - motif.length + 1)
    window = "".join(seq[lo:position + 1 + motif.length])
    k = position - lo
    swapped = window[:k] + window[k + 1] + window[k] + window[k + 2:]

    def starts(w: str) -> set:
        return {lo + s for s in motif.sites(w, utr3_start - lo)}

    return bool(starts(swapped) - starts(window))


@dataclass
class ShiftPlan:
    sites: list          # start positions of all motif sites in the 3' UTR
    swaps: list          # first position of every swapped dinucleotide (one per shifted site)
    n_unshiftable: int   # sites whose swap changes nothing (left: DAACH)
    n_would_create: int  # sites whose swap would create a new site; left unchanged


def plan_shifts(seq: str, utr3_start: int, shift: Shift, motif: Motif = DRACH) -> ShiftPlan:
    """
    Decides which sites are shifted. Sites are processed from 5' to 3'; each shift is checked against the sequence
    with the shifts so far, because two nearby shifts can form a site together that neither forms alone.
    """
    sites = motif.sites(seq, utr3_start)
    chars = list(seq)
    swaps, n_unshiftable, n_would_create = [], 0, 0
    for site in sites:
        # The swapped positions lie inside this site and in no other one (overlapping DRACH sites share only D/H),
        # so earlier shifts never touch them
        p = site + shift.swap_offset
        if seq[p:p + 2] != shift.pair:
            n_unshiftable += 1
        elif swap_creates_site(chars, p, utr3_start, motif):
            n_would_create += 1
        else:
            chars[p], chars[p + 1] = chars[p + 1], chars[p]
            swaps.append(p)
    return ShiftPlan(sites, swaps, n_unshiftable, n_would_create)


def match_swap_controls(seq: str, utr3_start: int, motif: Motif, sites: list, pair: str, anchors: list) -> Optional[list]:
    """
    One control dinucleotide per anchor, greedily by distance and without reuse (ties: the upstream candidate).
    Candidates start with `pair` in the 3' UTR and lie completely outside all motif sites, so a control swap never
    destroys a site. A candidate is taken only if its swap creates no site in the sequence with the controls chosen
    so far, so the control variant has exactly the original sites. None if an anchor finds no candidate.
    Two occurrences of a dinucleotide with two different bases never overlap, so the chosen controls are disjoint.
    """
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    mask = np.zeros(len(arr), dtype=bool)
    mask[:-1] = (arr[:-1] == ord(pair[0])) & (arr[1:] == ord(pair[1]))
    mask[:utr3_start] = False
    covered = np.zeros(len(arr), dtype=bool)
    for site in sites:
        covered[site:site + motif.length] = True
    mask[:-1] &= ~covered[:-1] & ~covered[1:]

    candidates = np.flatnonzero(mask)
    if len(candidates) < len(anchors):
        return None

    chars = list(seq)
    free = np.ones(len(candidates), dtype=bool)
    controls = []
    for anchor in anchors:
        for j in np.argsort(np.abs(candidates - anchor), kind="stable"):
            c = int(candidates[j])
            if free[j] and not swap_creates_site(chars, c, utr3_start, motif):
                free[j] = False
                chars[c], chars[c + 1] = chars[c + 1], chars[c]
                controls.append(c)
                break
        else:
            return None
    return controls
