import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
path = base / "44_katsuyama_original_52_full_records.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

doi16 = "10.1016/j.palaeo.2016.01.039"
doi20 = "10.1016/j.earscirev.2020.103173"

a = df[df["DOI"] == doi16].copy()
b = df[df["DOI"] == doi20].copy()

columns = [
    "SampleName", "SampleID", "Age", "Stage",
    "RelativeDepth", "Note", "Material"
]

a = a[columns].rename(columns={
    "SampleID": "SampleID_2016",
    "Age": "Age_2016",
    "Stage": "Stage_2016",
    "RelativeDepth": "Depth_2016",
    "Note": "Note_2016",
    "Material": "Material_2016"
})

b = b[columns].rename(columns={
    "SampleID": "SampleID_2020",
    "Age": "Age_2020",
    "Stage": "Stage_2020",
    "RelativeDepth": "Depth_2020",
    "Note": "Note_2020",
    "Material": "Material_2020"
})

comparison = a.merge(b, on="SampleName", how="inner", validate="one_to_one")

comparison["Age_difference_Ma"] = (
    comparison["Age_2016"] - comparison["Age_2020"]
).abs()

comparison["Primary_source_check"] = "Required"

output = base / "45_katsuyama_primary_source_comparison.csv"
comparison.to_csv(output, index=False, encoding="utf-8-sig")

print("EXPERIMENT 45: PRIMARY-SOURCE COMPARISON REGISTER")
print("=" * 65)
print("Matched sample identifiers:", len(comparison))
print("\nSample-level comparison:")
print(comparison.to_string(index=False))
print("\nSaved:", output)
print("\nNo age corrections or record merges performed.")
