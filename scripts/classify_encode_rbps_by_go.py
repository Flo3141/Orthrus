#!/usr/bin/env python3
"""
Filter and Classify 150 ENCODE eCLIP RBPs using the Master Node GO:0006402 (mRNA catabolic process).

Step 1 (Inclusion Filter):
- Live queries QuickGO API for all official descendant terms of GO:0006402 (mRNA catabolic process).
- Filters the 150 ENCODE RBPs to only those that possess an annotation within this subtree.

Step 2 (Directional Classification):
- Evaluates the matched GO terms of each included RBP:
  * Stabilizer: Annotated with stabilization or negative regulation of decay/catabolism.
  * Destabilizer: Annotated with destabilization, direct catabolic decay, decapping, or positive regulation of decay.
  * Undefined: Annotated only with generic parent terms (e.g. regulation of mRNA stability) or with conflicting terms.

Output:
- For Stabilizer / Destabilizer: Displays the specific GO Term (ID and Name) that triggered the classification.
- For Undefined: Displays ALL matched GO Terms within the GO:0006402 subtree.
- Exports complete CSV and TXT summary reports.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import urllib.request
import pandas as pd


# The 150 RBPs from the ENCODE eCLIP dataset (all_present_rbps.txt)
ALL_ENCODE_RBPS = [
    "AARS", "AATF", "ABCF1", "AGGF1", "AKAP1", "AKAP8L", "APOBEC3C", "AQR", "BCCIP", "BCLAF1",
    "BUD13", "CDC40", "CPEB4", "CPSF6", "CSTF2", "CSTF2T", "DDX21", "DDX24", "DDX3X", "DDX42",
    "DDX51", "DDX52", "DDX55", "DDX59", "DDX6", "DGCR8", "DHX30", "DKC1", "DROSHA", "EFTUD2",
    "EIF3D", "EIF3G", "EIF3H", "EIF4G2", "EWSR1", "EXOSC5", "FAM120A", "FASTKD2", "FKBP4", "FMR1",
    "FTO", "FUBP3", "FUS", "FXR1", "FXR2", "G3BP1", "GEMIN5", "GNL3", "GPKOW", "GRSF1", "GRWD1",
    "GTF2F1", "HLTF", "HNRNPA1", "HNRNPC", "HNRNPK", "HNRNPL", "HNRNPM", "HNRNPU", "HNRNPUL1",
    "IGF2BP1", "IGF2BP2", "IGF2BP3", "ILF3", "KHDRBS1", "KHSRP", "LARP4", "LARP7", "LIN28B", "LSM11",
    "MATR3", "METAP2", "MTPAP", "NCBP2", "NIP7", "NIPBL", "NKRF", "NOL12", "NOLC1", "NONO", "NPM1",
    "NSUN2", "PABPC4", "PABPN1", "PCBP1", "PCBP2", "PHF6", "POLR2G", "PPIG", "PPIL4", "PRPF4", "PRPF8",
    "PTBP1", "PUM1", "PUM2", "PUS1", "QKI", "RBFOX2", "RBM15", "RBM22", "RBM5", "RPS11", "RPS3",
    "SAFB", "SAFB2", "SBDS", "SDAD1", "SERBP1", "SF3A3", "SF3B1", "SF3B4", "SFPQ", "SLBP", "SLTM",
    "SMNDC1", "SND1", "SRSF1", "SRSF7", "SRSF9", "SSB", "STAU2", "SUB1", "SUGP2", "SUPV3L1", "TAF15",
    "TARDBP", "TBRG4", "TIA1", "TIAL1", "TRA2A", "TROVE2", "U2AF1", "U2AF2", "UCHL5", "UPF1", "UTP18",
    "UTP3", "WDR3", "WDR43", "WRN", "XPO5", "XRCC6", "XRN2", "YBX3", "YWHAG", "ZC3H11A", "ZC3H8",
    "ZNF622", "ZNF800", "ZRANB2"
]

MASTER_NODE_ID = "GO:0006402"
MASTER_NODE_NAME = "mRNA catabolic process"


def fetch_quickgo_descendants(parent_id: str = MASTER_NODE_ID) -> set:
    """Queries official EMBL-EBI QuickGO REST API for all descendant GO IDs of a parent node."""
    url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{parent_id}/descendants"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "Bioinformatics Pipeline; MasterThesis"},
    )
    print(f"[QuickGO] Querying official descendants for master node {parent_id} ({MASTER_NODE_NAME})...")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                descendants = set(results[0].get("descendants", []))
                print(f"[QuickGO] Found {len(descendants)} official descendant GO terms under {parent_id}.")
                return descendants
    except Exception as e:
        print(f"[Warning] Failed to fetch QuickGO descendants live ({e}). Falling back to cached core set.")

    # Fallback set if offline
    return {MASTER_NODE_ID, "GO:0043488", "GO:0061157", "GO:0048255", "GO:0000184", "GO:0000956"}


def fetch_live_gene_ontology(symbols: list, chunk_size: int = 50) -> dict:
    """Queries live NCBI Gene2GO / Ensembl GO annotations via MyGene.info API."""
    url = "https://mygene.info/v3/query"
    annotations = {}

    print(f"[MyGene.info] Fetching annotations for {len(symbols)} RBPs in chunks...")
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        q_str = ",".join(chunk)
        data = (
            f"q={q_str}&scopes=symbol&fields=name,summary,go.BP&species=human&size=100"
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"User-Agent": "Bioinformatics Pipeline; MasterThesis"},
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                results = json.loads(resp.read().decode("utf-8"))

            for entry in results:
                sym = entry.get("query")
                if not sym or sym in annotations:
                    continue

                full_name = entry.get("name", "")
                summary = entry.get("summary", "")
                go_bp = entry.get("go", {}).get("BP", [])
                if isinstance(go_bp, dict):
                    go_bp = [go_bp]

                terms = []
                for bp in go_bp:
                    if isinstance(bp, dict):
                        g_id = bp.get("id", "")
                        g_term = bp.get("term", "")
                        g_ev = bp.get("evidence", "")
                        if g_id and g_term:
                            terms.append({"id": g_id, "term": g_term, "evidence": g_ev})

                annotations[sym] = {
                    "symbol": sym,
                    "name": full_name,
                    "summary": summary,
                    "go_bp": terms,
                }
        except Exception as e:
            print(f"[Warning] Failed to fetch chunk {i}-{i+chunk_size}: {e}")

        time.sleep(0.3)

    return annotations


def classify_go_term_direction(term_name: str) -> str:
    """
    Evaluates semantic direction of an mRNA catabolism/stability GO term.
    Returns: 'Stabilizer', 'Destabilizer', or 'Neutral'
    """
    t = term_name.lower()

    # 1. Check for Stabilization
    # Note: 'destabiliz' must be excluded because 'stabilization' is a substring of 'destabilization'
    is_stab = False
    if "stabilization" in t and "destabiliz" not in t and "negative regulation" not in t:
        is_stab = True
    elif "negative regulation of" in t and any(k in t for k in ["catabolic", "decay", "destabilization", "shortening"]):
        is_stab = True
    elif "positive regulation" in t and "stabilization" in t and "destabiliz" not in t:
        is_stab = True

    # 2. Check for Destabilization
    is_destab = False
    if "destabilization" in t and "negative regulation" not in t:
        is_destab = True
    elif any(k in t for k in ["catabolic process", "decay", "decapping", "poly(a) tail shortening"]) and "negative regulation" not in t:
        is_destab = True
    elif "positive regulation of" in t and any(k in t for k in ["catabolic", "decay", "destabilization"]):
        is_destab = True
    elif "negative regulation of" in t and "stabilization" in t and "destabiliz" not in t:
        is_destab = True

    if is_stab and not is_destab:
        return "Stabilizer"
    if is_destab and not is_stab:
        return "Destabilizer"
    return "Neutral"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and Classify ENCODE eCLIP RBPs via Master Node GO:0006402 and Keyword Rules."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/encode_150_rbps_go_annotated.csv",
        help="Path to save full annotated CSV table",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/eclip/encode_rbp_stability_candidates_report.txt",
        help="Path to save filtered text summary report",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_out = Path(args.output_csv)
    txt_out = Path(args.output_txt)

    # 1. Fetch official descendants of Master Node GO:0006402
    descendant_ids = fetch_quickgo_descendants(MASTER_NODE_ID)

    # 2. Fetch live Gene Ontology annotations for the 150 RBPs
    anno_data = fetch_live_gene_ontology(ALL_ENCODE_RBPS)

    records = []
    filtered_rbps = []

    print("\n[Processing] Filtering 150 RBPs against GO:0006402 subtree and determining direction...")

    for sym in ALL_ENCODE_RBPS:
        info = anno_data.get(sym, {"name": "", "summary": "", "go_bp": []})
        name = info["name"]
        summary = info["summary"]
        go_bp = info["go_bp"]

        # Step 1: Filter terms within the GO:0006402 descendant tree
        matched_tree_terms = []
        seen_term_ids = set()
        for term_obj in go_bp:
            t_id = term_obj["id"]
            t_name = term_obj["term"]
            if t_id in descendant_ids and t_id not in seen_term_ids:
                seen_term_ids.add(t_id)
                direction = classify_go_term_direction(t_name)
                matched_tree_terms.append({
                    "id": t_id,
                    "term": t_name,
                    "direction": direction,
                })

        is_in_tree = len(matched_tree_terms) > 0

        # Step 2: Directional Classification
        if is_in_tree:
            stabs = [t for t in matched_tree_terms if t["direction"] == "Stabilizer"]
            destabs = [t for t in matched_tree_terms if t["direction"] == "Destabilizer"]

            if stabs and not destabs:
                classification = "Stabilizer"
                trigger_str = f"{stabs[0]['id']}: {stabs[0]['term']}"
            elif destabs and not stabs:
                classification = "Destabilizer"
                trigger_str = f"{destabs[0]['id']}: {destabs[0]['term']}"
            else:
                # Undefined (either generic parent terms like 'regulation of mRNA stability' or conflicting terms)
                classification = "Undefined"
                # Output ALL matched GO terms
                trigger_str = " | ".join(f"{t['id']}: {t['term']}" for t in matched_tree_terms)

            filtered_rbps.append({
                "Symbol": sym,
                "Classification": classification,
                "Trigger_GO_Terms": trigger_str,
                "Full_Name": name,
                "Num_Matched_Terms": len(matched_tree_terms),
            })
        else:
            classification = "Excluded (Non-Catabolic)"
            trigger_str = "-"

        records.append({
            "Symbol": sym,
            "Full_Name": name,
            "In_GO_0006402_Tree": is_in_tree,
            "Classification": classification,
            "Trigger_or_Matched_Terms": trigger_str,
            "Summary": summary[:120] + "..." if len(summary) > 120 else summary,
        })

    # Save full CSV
    df_all = pd.DataFrame(records)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_out, index=False)
    print(f"[Saved] Full annotation table saved to: {csv_out}")

    # Build and print comprehensive text report
    lines = []
    lines.append("=" * 110)
    lines.append("     SYSTEMATIC CLASSIFICATION OF ENCODE eCLIP RBPs VIA MASTER NODE GO:0006402")
    lines.append("     Master Node: GO:0006402 (mRNA catabolic process) & QuickGO Descendant Tree")
    lines.append("=" * 110)
    lines.append(f"\nTotal ENCODE RBPs Analyzed:                   {len(ALL_ENCODE_RBPS)}")
    lines.append(f"RBPs in GO:0006402 Catabolic/Stability Tree: {len(filtered_rbps)} ({len(filtered_rbps)/len(ALL_ENCODE_RBPS)*100:.1f}%)")
    lines.append(f"Excluded Non-Catabolic RBPs:                  {len(ALL_ENCODE_RBPS) - len(filtered_rbps)} (Splicing, Ribosome, Translation)\n")

    lines.append("-" * 110)
    lines.append(f"{'Symbol':<10} {'Classification':<15} {'Triggering GO Term (or ALL terms if Undefined)'}")
    lines.append("-" * 110)

    # Sort order: Stabilizers first, then Destabilizers, then Undefined
    order_map = {"Stabilizer": 1, "Destabilizer": 2, "Undefined": 3}
    for item in sorted(filtered_rbps, key=lambda x: (order_map.get(x["Classification"], 4), x["Symbol"])):
        sym = item["Symbol"]
        cls = item["Classification"]
        trig = item["Trigger_GO_Terms"]
        lines.append(f"{sym:<10} {cls:<15} {trig}")

    lines.append("-" * 110)

    # Breakdown counts
    stab_list = [x["Symbol"] for x in filtered_rbps if x["Classification"] == "Stabilizer"]
    destab_list = [x["Symbol"] for x in filtered_rbps if x["Classification"] == "Destabilizer"]
    undef_list = [x["Symbol"] for x in filtered_rbps if x["Classification"] == "Undefined"]

    lines.append(f"\n1. Stabilizers ({len(stab_list)}):    {', '.join(stab_list)}")
    lines.append(f"2. Destabilizers ({len(destab_list)}):  {', '.join(destab_list)}")
    lines.append(f"3. Undefined ({len(undef_list)}):     {', '.join(undef_list)}")
    lines.append("\nNote for Undefined RBPs:")
    lines.append("  Undefined RBPs possess annotations within the GO:0006402 tree, but their terms either")
    lines.append("  represent generic parent terms (e.g. GO:0043488 regulation of mRNA stability)")
    lines.append("  or feature conflicting regulatory evidence across different contexts.")
    lines.append("=" * 110)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    txt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n[Saved] Summary report successfully saved to: {txt_out}\n")


if __name__ == "__main__":
    main()
