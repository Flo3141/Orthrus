import zipfile
import pandas as pd
from pathlib import Path

# Pfad zur heruntergeladenen ZIP-Datei (z. B. in Downloads)
zip_path = Path.home() / "Downloads" / "Predicted_Targets_Context_Scores.default_predictions.txt.zip"
# Falls im selben Ordner:
# zip_path = Path("Predicted_Targets_Context_Scores.default_predictions.txt.zip")

target_tx = "ENST00000378512"
output_csv = "targetscan_ENST00000378512.csv"

print(f"Lese {zip_path.name} und suche nach {target_tx}...")

matching_rows = []
header = []

with zipfile.ZipFile(zip_path, "r") as z:
    # Die Textdatei innerhalb des Archivs öffnen
    filename = z.namelist()[0]
    with z.open(filename) as f:
        # Header-Zeile einlesen
        header = f.readline().decode("utf-8", errors="ignore").strip().split("\t")
        
        # Zeilenweise filtern
        for line in f:
            line_str = line.decode("utf-8", errors="ignore")
            if target_tx in line_str:
                matching_rows.append(line_str.strip().split("\t"))

print(f"Gefunden: {len(matching_rows)} Einträge für {target_tx}!")

df = pd.DataFrame(matching_rows, columns=header)

# Nach UTR_start aufsteigend sortieren
df["UTR_start"] = pd.to_numeric(df["UTR_start"], errors="coerce")
df["UTR end"] = pd.to_numeric(df["UTR end"], errors="coerce")
df = df.sort_values(by=["UTR_start", "UTR end"])

# Als übersichtliche CSV speichern
df.to_csv(output_csv, index=False)
print(f"Gespeichert als: {output_csv}")

# Vorschau der Bindestellen an Position 44-51 ausgeben
relevant_cols = [c for c in ["Gene Symbol", "Transcript ID", "miRNA", "UTR_start", "UTR end", "weighted context++ score"] if c in df.columns]
print("\nErste 20 Einträge nach UTR_start sortiert:")
print(df[relevant_cols].head(20).to_string())

