import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")

path = base / "45_katsuyama_primary_source_comparison.csv"
df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

# Independently documented sample-to-bed information
# from the 2015 Inuyama/Katsuyama study.
source_2015 = {
    "IYF53E-1": ("Bed 9", "Jurassic"),
    "IYF53E-0": ("Bed 10", "Jurassic"),
    "IYF53D-4": ("Bed 11", "Jurassic"),
    "IYF53D-3": ("Bed 12", "Jurassic"),
    "IYF53D-2": ("Bed 13", "Jurassic"),
    "IYF53D-1": ("Bed 14", "Jurassic"),
    "IYF53D-0": ("Bed 15", "Triassic-Jurassic"),
    "IYF53B-0": ("Bed 16", "Triassic"),
}

df["2015_source_bed"] = df["SampleName"].map(
    lambda x: source_2015.get(x, (None, None))[0]
)
df["2015_source_period"] = df["SampleName"].map(
    lambda x: source_2015.get(x, (None, None))[1]
)

df["independent_2015_evidence"] = df["SampleName"].isin(source_2015)

df["chronology_review_priority"] = "not_yet_independently_checked"

df.loc[
    df["SampleName"] == "IYF53D-0",
    "chronology_review_priority"
] = "high_priority_independent_boundary_evidence"

df.loc[
    df["SampleName"].isin(["IYF53D-1", "IYF53D-2", "IYF53D-3"]),
    "chronology_review_priority"
] = "review_against_jurassic_bed_evidence"

df.loc[
    df["SampleName"] == "IYF53B-0",
    "chronology_review_priority"
] = "review_against_triassic_bed_evidence"

out = base / "53_katsuyama_independent_chronology_evidence_register.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print("=" * 72)
print("EXPERIMENT 53: INDEPENDENT CHRONOLOGY EVIDENCE REGISTER")
print("=" * 72)

print("Matched records:", len(df))
print("Records with independent 2015 bed evidence:",
      df["independent_2015_evidence"].sum())

cols = [
    "SampleName",
    "Age_2016",
    "Stage_2016",
    "Age_2020",
    "Stage_2020",
    "2015_source_bed",
    "2015_source_period",
    "chronology_review_priority"
]

available = [c for c in cols if c in df.columns]

if "SampleName" not in df.columns:
    print("\nExpected SampleName column was not found.")
    print("Available columns:", df.columns.tolist())
else:
    print("\nIndependent evidence matches:")
    print(
        df[df["independent_2015_evidence"]]
        .to_string(index=False)
    )

print("\nOutput:", out)
print("Original dataset unchanged.")
