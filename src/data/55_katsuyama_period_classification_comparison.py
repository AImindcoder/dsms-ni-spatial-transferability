import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
path = base / "53b_katsuyama_corrected_2015_bed_crosswalk.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
df = df[df["2015_source_bed_corrected"].notna()].copy()

period_2015 = {
    "IYF53E-1": "Jurassic",
    "IYF53E-0": "Jurassic",
    "IYF53D-3": "Jurassic",
    "IYF53D-2": "Jurassic",
    "IYF53D-1": "Jurassic",
    "IYF53D-0": "Triassic-Jurassic",
    "IYF53B-0": "Triassic",
}

df["Period_2015_source"] = df["SampleName"].map(period_2015)

def normalize_period(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip().lower()
    if "triassic" in value and "jurassic" in value:
        return "Triassic-Jurassic"
    if "triassic" in value:
        return "Triassic"
    if "jurassic" in value:
        return "Jurassic"
    return "Other"

df["Period_2016_normalized"] = df["Period_2016"].apply(normalize_period)
df["Period_2020_normalized"] = df["Period_2020"].apply(normalize_period)

df["2015_vs_2016_period_match"] = (
    df["Period_2015_source"] == df["Period_2016_normalized"]
)
df["2015_vs_2020_period_match"] = (
    df["Period_2015_source"] == df["Period_2020_normalized"]
)

cols = [
    "SampleName",
    "Period_2015_source",
    "Period_2016",
    "Stage_2016",
    "Period_2020",
    "Stage_2020",
    "2015_vs_2016_period_match",
    "2015_vs_2020_period_match"
]

print("=" * 72)
print("EXPERIMENT 55: INDEPENDENT GEOLOGICAL PERIOD COMPARISON")
print("=" * 72)

print(df[cols].sort_values("2015_source_bed_corrected").to_string(index=False))

print("\n2015 vs 2016 period agreement:",
      int(df["2015_vs_2016_period_match"].sum()), "/", len(df))

print("2015 vs 2020 period agreement:",
      int(df["2015_vs_2020_period_match"].sum()), "/", len(df))

print("\nDisagreement cases:")
print(
    df[
        ~df["2015_vs_2016_period_match"]
    ][cols].to_string(index=False)
)

print("\nNo age corrections, sample merges, or deletions.")
