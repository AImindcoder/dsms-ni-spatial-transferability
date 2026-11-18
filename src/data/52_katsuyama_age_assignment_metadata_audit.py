import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
path = base / "44_katsuyama_original_52_full_records.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

doi16 = "10.1016/j.palaeo.2016.01.039"
doi20 = "10.1016/j.earscirev.2020.103173"

cols = [
    "SampleName", "DOI", "Age", "Era", "Period",
    "Epoch", "Stage", "RelativeDepth", "Note",
    "Formation", "Unit", "Material",
    "Techniqueδ15N", "δ15Nbulk\n‰"
]

available = [c for c in cols if c in df.columns]

d16 = df[df["DOI"] == doi16].copy()
d20 = df[df["DOI"] == doi20].copy()

print("=" * 72)
print("EXPERIMENT 52: ORIGINAL AGE-ASSIGNMENT METADATA AUDIT")
print("=" * 72)

print("\n2016 RECORDS")
print(d16[available].sort_values("SampleName").to_string(index=False))

print("\n2016 AGE-STAGE GROUPS")
print(
    d16.groupby(["Age", "Stage"], dropna=False)
    .agg(
        records=("SampleName", "size"),
        first_sample=("SampleName", "min"),
        last_sample=("SampleName", "max"),
        depth_values=("RelativeDepth", "count"),
        note_values=("Note", "count")
    )
    .to_string()
)

print("\n2020 RECORDS")
print(d20[available].sort_values("RelativeDepth").to_string(index=False))

print("\nAGE ASSIGNMENT EVIDENCE")
print("2016 distinct ages:", d16["Age"].nunique())
print("2016 distinct stages:", d16["Stage"].nunique())
print("2016 records with depth:", d16["RelativeDepth"].notna().sum())
print("2016 records with bed notes:", d16["Note"].notna().sum())
print("2020 distinct ages:", d20["Age"].nunique())
print("2020 records with depth:", d20["RelativeDepth"].notna().sum())
print("2020 records with bed notes:", d20["Note"].notna().sum())

print("\nNo original data modified.")
