import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
register_path = base / "42_katsuyama_cross_publication_lineage_register.csv"
raw_path = Path("01_Data/raw/DSMSNI_v0.3.csv")

register = pd.read_csv(register_path, encoding="utf-8-sig")
raw = pd.read_csv(raw_path, encoding="gbk", low_memory=False)

ids = set(register["SampleID_2016"].astype(str)) | set(register["SampleID_2020"].astype(str))
raw["SampleID"] = raw["SampleID"].astype(str)

matched = raw[raw["SampleID"].isin(ids)].copy()

full_path = base / "44_katsuyama_original_52_full_records.csv"
matched.to_csv(full_path, index=False, encoding="utf-8-sig")

print("=" * 72)
print("EXPERIMENT 44: ORIGINAL RECORD VERIFICATION")
print("=" * 72)
print("Expected original records:", len(ids))
print("Retrieved original records:", len(matched))
print("Original dataset columns:", len(raw.columns))

fields = [
    "SampleID", "SampleName", "FirstAuthor", "DOI",
    "Age", "Era", "Period", "Epoch", "Stage",
    "RelativeDepth", "Formation", "Unit",
    "SiteName", "Material", "Note"
]
target = next((c for c in raw.columns if "15Nbulk" in c), None)
technique = next((c for c in raw.columns if "Technique" in c), None)

for col in [target, technique]:
    if col and col not in fields:
        fields.append(col)

fields = [c for c in fields if c in matched.columns]

print("\nORIGINAL RECORD COMPARISON")
print(matched[fields].sort_values(["SampleName", "DOI"]).to_string(index=False))

print("\nFIELD COVERAGE")
for col in fields:
    print(f"{col}: {matched[col].notna().sum()} present / {len(matched)}")

print("\nRETRIEVAL CHECK")
print("Unique SampleIDs retrieved:", matched["SampleID"].nunique())
print("Missing requested SampleIDs:", len(ids - set(matched["SampleID"])))

print("\nFull original records saved to:")
print(full_path)
print("\nOriginal raw dataset was not modified.")
