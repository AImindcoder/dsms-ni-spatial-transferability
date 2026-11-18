import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

base = Path("02_Data_Audit/provenance_analysis")
path = base / "53b_katsuyama_corrected_2015_bed_crosswalk.csv"

df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

df = df[df["2015_source_bed_corrected"].notna()].copy()

def extract_bed(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"Bed\s*(\d+)", str(value), re.I)
    return int(match.group(1)) if match else np.nan

df["2020_bed_number"] = df["Note_2020"].apply(extract_bed)

df = df.dropna(
    subset=[
        "2015_source_bed_corrected",
        "2020_bed_number",
        "Depth_2020",
        "Age_2020"
    ]
)

print("=" * 72)
print("EXPERIMENT 54: STRATIGRAPHIC ORDER AGREEMENT")
print("=" * 72)

pairs = [
    ("2015 bed", "2020 bed", "2015_source_bed_corrected", "2020_bed_number"),
    ("2015 bed", "2020 depth", "2015_source_bed_corrected", "Depth_2020"),
    ("2020 bed", "2020 depth", "2020_bed_number", "Depth_2020"),
    ("2015 bed", "2020 age", "2015_source_bed_corrected", "Age_2020"),
]

for label_x, label_y, x, y in pairs:
    rho, p = spearmanr(df[x], df[y])
    tau, _ = kendalltau(df[x], df[y])

    print(f"\n{label_x} vs {label_y}")
    print("Spearman rho:", round(rho, 6))
    print("Kendall tau:", round(tau, 6))

print("\n2016 AGE ASSIGNMENTS ALONG THE SAMPLE SEQUENCE")

print(
    df[
        [
            "SampleName",
            "2015_source_bed_corrected",
            "Age_2016",
            "Stage_2016",
            "Age_2020",
            "Stage_2020",
            "2020_bed_number",
            "Depth_2020"
        ]
    ]
    .sort_values("2015_source_bed_corrected")
    .to_string(index=False)
)

print("\n2016 age groups:")
print(df.groupby(["Age_2016", "Stage_2016"]).size().to_string())

print("\nInterpretation safeguards:")
print("- These are seven ordered samples from one section.")
print("- Correlation statistics are descriptive, not independent inference.")
print("- No age corrections or sample merges are performed.")
print("- Bed-number orientation is not assumed to be identical.")

print("\nOriginal dataset unchanged.")
