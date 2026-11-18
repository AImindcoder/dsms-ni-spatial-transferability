import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
df = pd.read_csv(
    base / "44_katsuyama_original_52_full_records.csv",
    encoding="utf-8-sig",
    low_memory=False
)

doi = "10.1016/j.earscirev.2020.103173"
d = df[df["DOI"] == doi].copy()

d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
d["RelativeDepth"] = pd.to_numeric(d["RelativeDepth"], errors="coerce")

boundary_depth = -2.2
boundary_age = 201.57
sample_name = "IYF53D-0"

sample = d[d["SampleName"].astype(str).str.strip() == sample_name].iloc[0]

boundary_result = pd.read_csv(
    base / "49_katsuyama_boundary_age_validation.csv",
    encoding="utf-8-sig"
).iloc[0]

depth_difference = sample["RelativeDepth"] - boundary_depth
age_difference = boundary_result["interpolated_2020_age_Ma"] - sample["Age"]

print("=" * 72)
print("EXPERIMENT 50: BOUNDARY-TO-POST-EXTINCTION SAMPLE COMPARISON")
print("=" * 72)

print("\nExternal boundary reference:")
print("Depth:", boundary_depth, "m")
print("Age:", boundary_age, "+/- 0.20 Ma")

print("\nInterpolated age at boundary:")
print(boundary_result["interpolated_2020_age_Ma"], "Ma")

print("\nIYF53D-0, Fujisaki 2020:")
print("Depth:", sample["RelativeDepth"], "m")
print("Age:", sample["Age"], "Ma")
print("Note:", sample["Note"])

print("\nDistance above boundary:", round(depth_difference, 5), "m")
print("Age difference from interpolated boundary:", round(age_difference, 6), "Ma")

if depth_difference > 0 and age_difference > 0:
    print("\nDirection check: sample is stratigraphically higher and younger.")
else:
    print("\nDirection check: investigate stratigraphic ordering.")

print("\nIMPORTANT:")
print("This calculation assumes compatible depth coordinates.")
print("It does not prove physical sample identity or exact boundary correlation.")
print("No original data modified.")
