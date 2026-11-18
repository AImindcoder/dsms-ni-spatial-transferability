import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
path = base / "53b_katsuyama_corrected_2015_bed_crosswalk.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

required = [
    "SampleName", "Stage_2016", "Stage_2020",
    "2015_source_bed_corrected"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[df["2015_source_bed_corrected"].notna()].copy()

period_2015 = {
    "IYF53E-1": "Jurassic",
    "IYF53E-0": "Jurassic",
    "IYF53D-3": "Jurassic",
    "IYF53D-2": "Jurassic",
    "IYF53D-1": "Jurassic",
    "IYF53D-0": "Boundary",
    "IYF53B-0": "Triassic",
}

triassic = {
    "induan", "olenekian", "anisian",
    "ladinian", "carnian", "norian", "rhaetian"
}

jurassic = {
    "hettangian", "sinemurian", "pliensbachian",
    "toarcian", "aalenian", "bajocian",
    "bath onian", "bathonian", "callovian",
    "oxfordian", "kimmeridgian", "tithonian"
}

def stage_to_period(value):
    if pd.isna(value):
        return "Unknown"
    stage = str(value).strip().lower()
    if stage in triassic:
        return "Triassic"
    if stage in jurassic:
        return "Jurassic"
    return "Unknown"

df["Period_2015_source"] = df["SampleName"].map(period_2015)
df["Period_2016_derived"] = df["Stage_2016"].apply(stage_to_period)
df["Period_2020_derived"] = df["Stage_2020"].apply(stage_to_period)

def compare(source, derived):
    if source == "Boundary":
        return "boundary_compatible" if derived in ["Triassic", "Jurassic"] else "unknown"
    if derived == "Unknown":
        return "unknown"
    return "match" if source == derived else "mismatch"

df["2015_vs_2016_comparison"] = df.apply(
    lambda r: compare(r["Period_2015_source"], r["Period_2016_derived"]),
    axis=1
)

df["2015_vs_2020_comparison"] = df.apply(
    lambda r: compare(r["Period_2015_source"], r["Period_2020_derived"]),
    axis=1
)

df = df.sort_values("2015_source_bed_corrected")

cols = [
    "SampleName", "Period_2015_source",
    "Stage_2016", "Period_2016_derived",
    "Stage_2020", "Period_2020_derived",
    "2015_vs_2016_comparison",
    "2015_vs_2020_comparison"
]

print("=" * 72)
print("EXPERIMENT 55C: GEOLOGICAL PERIOD COMPARISON")
print("=" * 72)

print(df[cols].to_string(index=False))

for column in [
    "2015_vs_2016_comparison",
    "2015_vs_2020_comparison"
]:
    print(f"\n{column}")
    print(df[column].value_counts().to_string())

out = base / "55c_katsuyama_period_comparison.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:", out)
print("Original dataset unchanged.")
