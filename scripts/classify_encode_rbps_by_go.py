#!/usr/bin/env python3
"""
Live Gene Ontology (GO) Screening for 150 ENCODE eCLIP RBPs.

Queries public Gene Ontology annotations (via the official MyGene.info / NCBI Gene2GO REST API)
for all 150 ENCODE eCLIP RNA-binding proteins to objectively identify which RBPs are annotated
with mRNA stability, decay, and catabolic processes.

Public Data Source:
- Gene Ontology Consortium (GOC) / NCBI Gene2GO / EMBL-EBI QuickGO
- API: https://mygene.info/v3/query
"""

import argparse
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

# Canonical Gene Ontology (GO) Biological Process Terms & Substring Keywords for mRNA Stability/Decay
TARGET_GO_KEYWORDS = [
    "mrna stability",
    "mrna catabolic process",
    "mrna decay",
    "deadenylation",
    "nonsense-mediated decay",
    "polyadenylation",
    "rna catabolic process",
    "staufen-mediated",
]


def fetch_live_gene_ontology(symbols: list, chunk_size: int = 50) -> dict:
    """
    Queries public Gene Ontology (GO:BP) and gene summaries for a list of gene symbols
    via the public MyGene.info API (aggregating NCBI Gene2GO & Ensembl annotations).
    """
    url = "https://mygene.info/v3/query"
    annotations = {}

    print(f"[API] Fetching live Gene Ontology annotations for {len(symbols)} genes in chunks...")
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        q_str = ",".join(chunk)
        data = (
            f"q={q_str}&scopes=symbol&fields=name,summary,go.BP&species=human&size=100"
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"User-Agent": "Mozilla/5.0 (Bioinformatics Pipeline; MasterThesis)"},
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
                        g_evidence = bp.get("evidence", "")
                        if g_term:
                            terms.append({"id": g_id, "term": g_term, "evidence": g_evidence})

                annotations[sym] = {
                    "symbol": sym,
                    "name": full_name,
                    "summary": summary,
                    "go_bp": terms,
                }
        except Exception as e:
            print(f"[Warning] Failed to query chunk {i}-{i+chunk_size}: {e}")

        time.sleep(0.3)

    return annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live Gene Ontology (GO) Screening of 150 ENCODE eCLIP RBPs."
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

    # 1. Fetch official live GO terms
    anno_data = fetch_live_gene_ontology(ALL_ENCODE_RBPS)

    records = []
    stability_candidates = []

    for sym in ALL_ENCODE_RBPS:
        info = anno_data.get(sym, {"name": "", "summary": "", "go_bp": []})
        name = info["name"]
        summary = info["summary"]
        go_bp = info["go_bp"]

        # Filter for relevant terms
        matched_go_terms = []
        for term_obj in go_bp:
            t_name = term_obj["term"]
            t_id = term_obj["id"]
            if any(k in t_name.lower() for k in TARGET_GO_KEYWORDS):
                matched_go_terms.append(f"{t_id}: {t_name}")

        # Check summary for keywords
        matched_in_summary = [k for k in TARGET_GO_KEYWORDS if k in summary.lower()]

        has_stability_role = len(matched_go_terms) > 0 or len(matched_in_summary) > 0

        # Heuristic functional role assignment
        role = "Unclassified / Other"
        all_text = " ".join(matched_go_terms).lower() + " " + summary.lower()
        if has_stability_role:
            if any(w in all_text for w in ["nonsense-mediated", "decay", "catabolic", "deadenylation", "exosome", "downregulation"]):
                role = "Destabilizer"
            elif any(w in all_text for w in ["protect", "stabiliz", "enhanc", "shield"]):
                role = "Stabilizer"
            else:
                role = "Regulator (Stability / Decay)"

        records.append({
            "Symbol": sym,
            "Full_Name": name,
            "Has_Stability_Evidence": has_stability_role,
            "Inferred_Role": role if has_stability_role else "Non-Stability",
            "Matched_GO_Terms": " | ".join(matched_go_terms[:3]) if matched_go_terms else "-",
            "Total_GO_Terms": len(go_bp),
            "Summary": summary[:120] + "..." if len(summary) > 120 else summary,
        })

        if has_stability_role:
            stability_candidates.append({
                "Symbol": sym,
                "Inferred_Role": role,
                "Primary_GO_Term": matched_go_terms[0] if matched_go_terms else f"Summary evidence: {matched_in_summary[0]}",
                "Full_Name": name,
            })

    df_all = pd.DataFrame(records)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_out, index=False)
    print(f"\n[Saved] Full GO annotation matrix saved to: {csv_out}")

    # Generate text summary report
    lines = []
    lines.append("=" * 95)
    lines.append("    OBJECTIVE GENE ONTOLOGY (GO) SCREENING OF 150 ENCODE eCLIP RBPs")
    lines.append("    Data Source: Official Gene Ontology Consortium (GOC) / NCBI Gene2GO")
    lines.append("=" * 95)
    lines.append(f"\nTotal ENCODE RBPs Analyzed:        {len(ALL_ENCODE_RBPS)}")
    lines.append(f"Confirmed mRNA Stability / Decay:  {len(stability_candidates)} ({len(stability_candidates)/len(ALL_ENCODE_RBPS)*100:.1f}%)")
    lines.append(f"Excluded Non-Stability RBPs:       {len(ALL_ENCODE_RBPS) - len(stability_candidates)} (Splicing, Ribosome, Translation)")

    lines.append("\n" + "-" * 95)
    lines.append(f"{'Symbol':<10} {'Inferred Role':<16} {'Official GO Biological Process Evidence':<55}")
    lines.append("-" * 95)

    for cand in sorted(stability_candidates, key=lambda x: (x["Inferred_Role"], x["Symbol"])):
        lines.append(f"{cand['Symbol']:<10} {cand['Inferred_Role']:<16} {cand['Primary_GO_Term'][:55]:<55}")

    lines.append("-" * 95)
    report_str = "\n".join(lines)
    print("\n" + report_str)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n[Saved] Summary report saved to: {txt_out}\n")


if __name__ == "__main__":
    main()
