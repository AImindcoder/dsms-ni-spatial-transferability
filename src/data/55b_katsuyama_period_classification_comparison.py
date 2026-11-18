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
    "IYF53D-0": "Boundary",
    "IYF53B-0": "Triassic",
}

df["Period_2015_source"] = df["SampleName"].map(period_2015)

def stage_to_period(stage):
    if pd.isna(stage):
        return "Unknown"

    stage = str(stage).strip().lower()

    triassic = {
        "induan", "olenekian", "anisian",
        "ladinian", "carnian", "norian", "rhaetian"
    }

    jurassic = {
        "hettangian", "sinemurian", "pliensbachian",
        "toarcian", "aalenian", "bajocian",
        "bath onian", "callovian", "oxfordian",
        "kimmeridgian", "tithonian"
    }

    if stage in triassic:
        return "Triassic"
    if stage in jurassic:
        return "Jurassic"
    return "Unknown"

df["Period_2016_derived"] = df["Stage_2016"].apply(stage_to_period)
df["Period_2020_derived"] = df["Stage_2020"].apply(stage_to_period)

df["2015_vs_2016_comparison"] = "not_comparable"
df["2015_vs_2020_comparison"] = "not_comparable"

for idx, row in df.iterrows():
    source = row["Period_2015_source"]

    for column, result_column in [
        ("Period_2016_derived", "2015_vs_2016_comparison"),
        ("Period_2020_derived", "2015_vs_2020_comparison")
    ]:
        derived = row[column]

        if source == "Boundary":
            result = (
                "boundary_compatible"
                if derived in ["Triassic", "Jurassic"]
                else "unknown"
            )
        elif derived == "Unknown":
            result = "unknown"
        else:
            result = "match" if source == derived else "mismatch"

        df.at[idx, result_column] = result

cols = [
    "SampleName",
    "Period_2015_source",
    "Stage_2016",
    "Period_2016_derived",
    "Stage_2020",
    "Period_2020_derived",
    "2015_vs_2016_comparison",
    "2015_vs_2020_comparison"
]

print("=" * 72)
print("EXPERIMENT 55B: GEOLOGICAL PERIOD COMPARISON")
print("=" * 72)

print(
    df[cols]
    .sort_values("2015_source_bed_corrected")
    .to_string(index=False)
)

print("\nComparison counts:")

for column in [
    "2015_vs_2016_comparison",
    "2015_vs_2020_comparison"
]:
    print("\n", column)
    print(df[column].value_counts(dropna=False).to_string())

out = base / "55b_katsuyama_period_comparison.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")

print("\nOutput:", out)
print("No original ages or records modified.")
