from pathlib import Path
import pandas as pd

raw = Path("01_Data/raw/DSMSNI_v0.3.csv")
df = pd.read_csv(raw, encoding="gbk", low_memory=False)
df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

doi = "10.1016/j.gloplacha.2015.05.002"
g = df[
    (df["DOI"].astype(str).str.strip().str.lower() == doi.lower()) &
    (df["SiteName"].astype(str).str.strip().str.lower() == "peniche")
].copy()

print("=" * 85)
print("PENICHE SOURCE-LEVEL CHRONOLOGY INVESTIGATION")
print("=" * 85)
print("Records:", len(g))

for col in ["Age", "RelativeDepth", "δ15Nbulk ‰", "TN %", "TOC %",
            "δ13Corg ‰", "δ13Ccarb ‰"]:
    if col in g.columns:
        g[col] = pd.to_numeric(g[col], errors="coerce")

for col in ["Age", "RelativeDepth", "Era", "Period", "Epoch", "Stage",
            "Techniqueδ15N", "Note", "DataSource", "Reference", "Title"]:
    if col in g.columns:
        s = g[col]
        print(f"\n{col}:")
        print("  non-missing:", int(s.notna().sum()))
        print("  unique:", int(s.nunique(dropna=True)))
        print("  examples:", s.dropna().astype(str).unique()[:12].tolist())

if "Age" in g.columns:
    print("\nAge distribution:")
    print(g["Age"].value_counts(dropna=False).head(15).to_string())

if "RelativeDepth" in g.columns:
    print("\nDepth summary:")
    print(g["RelativeDepth"].describe().to_string())

fields = [
    c for c in [
        "SampleID", "SampleName", "SiteName", "DOI", "Age",
        "RelativeDepth", "Era", "Period", "Epoch", "Stage",
        "δ15Nbulk ‰", "TN %", "TOC %", "δ13Corg ‰",
        "Techniqueδ15N", "Note", "DataSource"
    ] if c in g.columns
]

print("\nFIRST 40 RECORDS")
print(g[fields].sort_values("RelativeDepth").head(40).to_string(index=False))

out = Path("01_Data/interim/106_peniche_source_audit_records.csv")
g[fields].sort_values("RelativeDepth").to_csv(
    out, index=False, encoding="utf-8-sig"
)
print("\nSaved:", out)
