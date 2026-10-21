#!/usr/bin/env python3
"""
Downloads genuine Ensembl transcript & exon structures for the local test sample
and compiles them into the expected gffutils SQLite database:
/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/Homo_sapiens.GRCh38.108.gtf.db
"""

import json
from pathlib import Path
import urllib.request
import gffutils
import pandas as pd

def main():
    data_file = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt")
    db_out = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/Homo_sapiens.GRCh38.108.gtf.db")

    if not data_file.exists():
        print(f"[Error] Dataset file not found at: {data_file}")
        return

    df = pd.read_csv(data_file, sep="\t")
    tx_ids = df["ensembl_transcript_id"].dropna().unique().tolist()
    print(f"Fetching real Ensembl annotations for {len(tx_ids)} transcripts...")

    features = []
    for tx_id in tx_ids:
        clean_id = str(tx_id).split(".")[0].strip()
        url = f"https://rest.ensembl.org/lookup/id/{clean_id}?expand=1"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                chrom = str(data["seq_region_name"])
                strand = "+" if data["strand"] == 1 else "-"
                tx_start = int(data["start"])
                tx_end = int(data["end"])
                parent_gene = str(data.get("Parent", ""))

                # Transcript feature
                tx_line = (
                    f"{chrom}\tEnsembl\ttranscript\t{tx_start}\t{tx_end}\t.\t{strand}\t.\t"
                    f"ID={clean_id};gene_id={parent_gene}\n"
                )
                features.append(gffutils.feature.feature_from_line(tx_line))

                # Exon features
                for ex in data.get("Exon", []):
                    ex_id = ex["id"]
                    ex_start = int(ex["start"])
                    ex_end = int(ex["end"])
                    ex_line = (
                        f"{chrom}\tEnsembl\texon\t{ex_start}\t{ex_end}\t.\t{strand}\t.\t"
                        f"ID={ex_id};Parent={clean_id}\n"
                    )
                    features.append(gffutils.feature.feature_from_line(ex_line))
        except Exception as e:
            print(f"Warning: Failed to fetch {clean_id}: {e}")

    print(f"Creating SQLite FeatureDB at {db_out} ({len(features)} total features)...")
    if db_out.exists():
        db_out.unlink()

    gffutils.create_db(
        features,
        str(db_out),
        force=True,
        keep_order=True,
        merge_strategy="create_unique",
    )
    print(f"Successfully generated {db_out} ({db_out.stat().st_size:,} bytes).")

    # Verify lookup via GtfDbHelper logic
    db = gffutils.FeatureDB(str(db_out))
    sample_tx = clean_id
    tx = db[sample_tx]
    exons = [(e.start, e.end) for e in db.children(tx, featuretype="exon", order_by="start")]
    print(f"Sanity Check: {sample_tx} on chr{tx.chrom} ({tx.strand}) with {len(exons)} exons.")

if __name__ == "__main__":
    main()
