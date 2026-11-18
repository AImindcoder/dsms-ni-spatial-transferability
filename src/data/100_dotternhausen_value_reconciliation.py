from pathlib import Path
import pandas as pd

root = Path(".")
source_file = root / "01_Data" / "external" / "Dotternhausen_Table_1_extracted.txt"
csv_file = root / "01_Data" / "raw" / "DSMSNI_v0.3.csv"

print("=" * 85)
print("ORIGINAL PUBLICATION TABLE 1")
print("=" * 85)
print(source_file.read_text(encoding="utf-8", errors="replace"))

df = pd.read_csv(csv_file, encoding="gbk", low_memory=False)
df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

doi_col = next(c for c in df.columns if c.lower() == "doi")
doi_rows = df[df[doi_col].astype(str).str.strip().str.lower() == "10.1029/2020pa004207"].copy()

wanted = [
    "SampleID", "RelativeDepth", "Age", "δ15Nbulk ‰",
    "TN %", "TOC %", "δ13Corg ‰", "δ15Nsp ‰",
    "Techniqueδ15N"
]
available = [c for c in wanted if c in doi_rows.columns]

print("\n" + "=" * 85)
print("MATCHING DSMS-NI MEASUREMENTS — SORTED BY DEPTH")
print("=" * 85)
print(doi_rows[available].sort_values("RelativeDepth").to_string(index=False))

print("\nRecords:", len(doi_rows))
print("Unique depths:", doi_rows["RelativeDepth"].nunique())
print("Available comparison columns:", available)
