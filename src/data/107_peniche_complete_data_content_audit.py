from pathlib import Path
import pandas as pd

raw = Path("01_Data/raw/DSMSNI_v0.3.csv")
df = pd.read_csv(raw, encoding="gbk", low_memory=False)
df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

doi = "10.1016/j.gloplacha.2015.05.002"
g = df[df["DOI"].astype(str).str.strip().str.lower() == doi.lower()].copy()

fields = [
    "SampleID", "SampleName", "RelativeDepth", "Age",
    "δ15Nbulk ‰", "δ15Nsp ‰", "TN %", "TOC %",
    "δ13Corg ‰", "δ13Ccarb ‰", "Techniqueδ15N",
    "Note", "DataSource", "Reference", "Title"
]

print("=" * 80)
print("PENICHE COMPLETE DATA-CONTENT AUDIT")
print("=" * 80)
print("Records:", len(g))

print("\nFIELD COMPLETENESS")
print("-" * 80)
for col in fields:
    if col not in g.columns:
        continue
    nonmissing = g[col].notna().sum()
    unique = g[col].nunique(dropna=True)
    print(
        f"{col:25s} nonmissing={nonmissing:4d}/{len(g)} "
        f"({100*nonmissing/max(len(g),1):6.2f}%) "
        f"unique={unique}"
    )

print("\nNUMERIC GEOCHEMICAL FIELDS")
print("-" * 80)
numeric_summary = []
for col in g.columns:
    if col in ["SampleID", "RelativeDepth", "Age"]:
        continue
    values = pd.to_numeric(g[col], errors="coerce")
    n = values.notna().sum()
    if n:
        numeric_summary.append({
            "column": col,
            "numeric_values": int(n),
            "mean": values.mean(),
            "minimum": values.min(),
            "maximum": values.max()
        })

if numeric_summary:
    print(pd.DataFrame(numeric_summary).to_string(index=False))
else:
    print("No additional numeric fields detected.")

print("\nSAMPLE-NAME AND DEPTH CHECK")
print("-" * 80)
print("Unique sample names:", g["SampleName"].nunique(dropna=True))
print("Missing sample names:", g["SampleName"].isna().sum())
print("Unique depths:", g["RelativeDepth"].nunique(dropna=True))
print("Depth range:",
      g["RelativeDepth"].min(), "to", g["RelativeDepth"].max())

print("\nFIRST AND LAST TEN RECORDS")
print("-" * 80)
display = [c for c in fields if c in g.columns]
print("\nFIRST TEN")
print(g[display].head(10).to_string(index=False))
print("\nLAST TEN")
print(g[display].tail(10).to_string(index=False))

out = Path("01_Data/interim/107_peniche_complete_data_content_audit.csv")
g[display].to_csv(out, index=False, encoding="utf-8-sig")
print("\nSaved:", out)
