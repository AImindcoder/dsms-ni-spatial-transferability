import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
path = base / "45_katsuyama_primary_source_comparison.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

# Sample-to-bed mapping transcribed from the 2015 source table.
source_2015 = {
    "IYF53E-1": 8,
    "IYF53E-0": 9,
    "IYF53D-3": 11,
    "IYF53D-2": 12,
    "IYF53D-1": 13,
    "IYF53D-0": 14,
    "IYF53B-0": 15,
}

df["2015_source_bed_corrected"] = df["SampleName"].map(source_2015)
df["independent_2015_bed_match"] = (
    df["2015_source_bed_corrected"].notna()
)

df["chronology_review_priority"] = "not_yet_independently_checked"

df.loc[
    df["SampleName"] == "IYF53D-0",
    "chronology_review_priority"
] = "high_priority_boundary_chronology_review"

df.loc[
    df["SampleName"].isin(
        ["IYF53D-1", "IYF53D-2", "IYF53D-3"]
    ),
    "chronology_review_priority"
] = "review_against_2015_bed_sequence"

df.loc[
    df["SampleName"] == "IYF53B-0",
    "chronology_review_priority"
] = "review_against_2015_bed_sequence"

out = base / "53b_katsuyama_corrected_2015_bed_crosswalk.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print("=" * 72)
print("EXPERIMENT 53B: CORRECTED 2015 BED CROSSWALK")
print("=" * 72)

print("Matched records:", len(df))
print("Records matched to 2015 bed table:",
      int(df["independent_2015_bed_match"].sum()))

cols = [
    "SampleName",
    "Age_2016",
    "Stage_2016",
    "Age_2020",
    "Stage_2020",
    "Depth_2020",
    "Note_2020",
    "2015_source_bed_corrected",
    "chronology_review_priority"
]

available = [c for c in cols if c in df.columns]

print("\nCorrected crosswalk:")
print(
    df[df["independent_2015_bed_match"]]
    [available]
    .sort_values("2015_source_bed_corrected")
    .to_string(index=False)
)

print("\nImportant:")
print("- 2015 and 2020 bed numbers are separate numbering systems.")
print("- No sample ages have been corrected.")
print("- No records have been merged or deleted.")
print("- Original dataset remains unchanged.")

print("\nOutput:", out)
