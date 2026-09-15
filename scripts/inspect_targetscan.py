import zipfile
import pandas as pd
from pathlib import Path

# Path to downloaded ZIP file (e.g. in Downloads)
zip_path = Path.home() / "Downloads" / "Predicted_Targets_Context_Scores.default_predictions.txt.zip"
# If in the same directory:
# zip_path = Path("Predicted_Targets_Context_Scores.default_predictions.txt.zip")

target_tx = "ENST00000378512"
output_csv = "targetscan_ENST00000378512.csv"

print(f"Reading {zip_path.name} and searching for {target_tx}...")

matching_rows = []
header = []

with zipfile.ZipFile(zip_path, "r") as z:
    # Open the text file inside the archive
    filename = z.namelist()[0]
    with z.open(filename) as f:
        # Read header line
        header = f.readline().decode("utf-8", errors="ignore").strip().split("\t")
        
        # Filter line by line
        for line in f:
            line_str = line.decode("utf-8", errors="ignore")
            if target_tx in line_str:
                matching_rows.append(line_str.strip().split("\t"))

print(f"Found: {len(matching_rows)} entries for {target_tx}!")

df = pd.DataFrame(matching_rows, columns=header)

# Sort ascending by UTR_start
df["UTR_start"] = pd.to_numeric(df["UTR_start"], errors="coerce")
df["UTR end"] = pd.to_numeric(df["UTR end"], errors="coerce")
df = df.sort_values(by=["UTR_start", "UTR end"])

# Save as CSV
df.to_csv(output_csv, index=False)
print(f"Saved as: {output_csv}")

# Print preview of binding sites at position 44-51
relevant_cols = [c for c in ["Gene Symbol", "Transcript ID", "miRNA", "UTR_start", "UTR end", "weighted context++ score"] if c in df.columns]
print("\nFirst 20 entries sorted by UTR_start:")
print(df[relevant_cols].head(20).to_string())
