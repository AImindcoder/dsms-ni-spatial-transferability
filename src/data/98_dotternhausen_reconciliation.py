import pandas as pd
from pathlib import Path

root = Path(".")
raw = root / "01_Data" / "raw" / "DSMSNI_v0.3.csv"
source = root / "01_Data" / "external" / "Dotternhausen_Table_1_extracted.txt"

print("DOTTERNHAUSEN SOURCE-TO-DSMS RECONCILIATION")
print("=" * 75)

df = pd.read_csv(raw, encoding="gbk", low_memory=False)

doi = "10.1029/2020pa004207"
subset = df[df["DOI"].astype(str).str.lower().str.strip() == doi].copy()

print("DSMS records with exact DOI:", len(subset))
print("Source Table 1 file exists:", source.exists())

if len(subset) == 0:
    print("No exact DOI records found. Check the DOI column and raw file.")
    raise SystemExit

print("\nDSMS columns potentially relevant:")
for col in df.columns:
    if any(x in col.lower() for x in [
        "depth", "relative", "15n", "toc", "tn", "13c", "sample",
        "site", "author", "doi"
    ]):
        print(" -", col)

print("\nDSMS records, selected fields:")
preferred = [
    "SampleID", "SampleName", "SiteName", "DOI",
    "RelativeDepth", "Age", "δ15Nbulk", "δ15Nker",
    "TOC", "TNbulk", "δ13Corg"
]
available = [c for c in preferred if c in subset.columns]
print(subset[available].sort_values("RelativeDepth").to_string(index=False))

print("\nRECONCILIATION NOTE")
print("The original article Table 1 has 31 depth-labelled samples.")
print("The dissertation identifies three potentially modified samples:")
print("DH-729, DH-825, DH-836.")
print("Depth agreement alone does not establish aliquot identity.")
