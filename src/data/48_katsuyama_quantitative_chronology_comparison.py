import pandas as pd
import numpy as np
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")

path = base / "45_katsuyama_primary_source_comparison.csv"
df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

# Published external correlation reference:
# Ikeda et al. (2023), Figure 2: approximately 201.57 +/- 0.20 Ma
reference_age = 201.57
reference_uncertainty = 0.20
lower = reference_age - reference_uncertainty
upper = reference_age + reference_uncertainty

df["Age_2016"] = pd.to_numeric(df["Age_2016"], errors="coerce")
df["Age_2020"] = pd.to_numeric(df["Age_2020"], errors="coerce")

df["Age_difference_2016_minus_2020"] = (
    df["Age_2016"] - df["Age_2020"]
)

df["Age_2020_inside_external_reference"] = (
    df["Age_2020"].between(lower, upper)
)

df["Age_2016_inside_external_reference"] = (
    df["Age_2016"].between(lower, upper)
)

df["Distance_2016_from_reference_Ma"] = (
    df["Age_2016"] - reference_age
).abs()

df["Distance_2020_from_reference_Ma"] = (
    df["Age_2020"] - reference_age
).abs()

df["External_reference_lower_Ma"] = lower
df["External_reference_upper_Ma"] = upper

df["Evidence_class"] = "candidate_chronology_discrepancy"

mask = df["SampleName"].astype(str).str.strip() == "IYF53D-0"

df.loc[mask, "Evidence_class"] = (
    "independently_supported_end_triassic_chronology_conflict"
)

output = base / "48_katsuyama_quantitative_chronology_comparison.csv"
df.to_csv(output, index=False, encoding="utf-8-sig")

print("=" * 72)
print("EXPERIMENT 48: QUANTITATIVE CHRONOLOGY COMPARISON")
print("=" * 72)

print("Matched records:", len(df))
print("External reference:", reference_age, "+/-", reference_uncertainty, "Ma")
print("Reference interval:", lower, "to", upper, "Ma")

print("\n2016 ages inside external interval:",
      int(df["Age_2016_inside_external_reference"].sum()), "/", len(df))

print("2020 ages inside external interval:",
      int(df["Age_2020_inside_external_reference"].sum()), "/", len(df))

print("\n2016 age distribution:")
print(df["Age_2016"].value_counts(dropna=False).sort_index())

print("\n2020 age range:")
print("Minimum:", df["Age_2020"].min())
print("Maximum:", df["Age_2020"].max())
print("Median:", df["Age_2020"].median())

print("\nAge discrepancy:")
print("Median absolute difference:",
      df["Age_difference_2016_minus_2020"].abs().median(), "Ma")

print("Maximum absolute difference:",
      df["Age_difference_2016_minus_2020"].abs().max(), "Ma")

print("\nIYF53D-0:")
print(
    df.loc[mask, [
        "SampleName",
        "Age_2016",
        "Age_2020",
        "Age_difference_2016_minus_2020",
        "Age_2016_inside_external_reference",
        "Age_2020_inside_external_reference",
        "Evidence_class"
    ]].to_string(index=False)
)

print("\nSaved:", output)
print("No raw data modified. No ages corrected.")
