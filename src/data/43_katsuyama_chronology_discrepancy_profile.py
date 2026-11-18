import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")
source = base / "42_katsuyama_cross_publication_lineage_register.csv"

df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)

print("=" * 75)
print("EXPERIMENT 43: KATSUYAMA CHRONOLOGY DISCREPANCY PROFILE")
print("=" * 75)

print("\n1. 2016 AGE ASSIGNMENT DISTRIBUTION")
print(df["Age_2016_Ma"].value_counts(dropna=False).sort_index().to_string())

print("\n2. 2020 AGE ASSIGNMENT DISTRIBUTION")
print(df["Age_2020_Ma"].describe().to_string())

print("\n3. SAMPLE-BY-SAMPLE CHRONOLOGY COMPARISON")
cols = [
    "SampleName", "Age_2016_Ma", "Stage_2016",
    "Age_2020_Ma", "Stage_2020",
    "Absolute_Age_Difference_Ma"
]
print(df[cols].sort_values("Age_2020_Ma").to_string(index=False))

print("\n4. SPECIAL CASE: IYF53D-0")
special = df[df["SampleName"].astype(str).str.strip() == "IYF53D-0"]
if len(special):
    print(special[cols].to_string(index=False))
else:
    print("IYF53D-0 not found.")

print("\n5. SUMMARY")
print("Number of sample pairs:", len(df))
print("Distinct 2016 ages:", df["Age_2016_Ma"].nunique())
print("Distinct 2020 ages:", df["Age_2020_Ma"].nunique())
print("2016 age range:", df["Age_2016_Ma"].min(), "to", df["Age_2016_Ma"].max(), "Ma")
print("2020 age range:", df["Age_2020_Ma"].min(), "to", df["Age_2020_Ma"].max(), "Ma")
print("Median absolute difference:", df["Absolute_Age_Difference_Ma"].median(), "Ma")

output = base / "43_katsuyama_chronology_discrepancy_profile.csv"
df[cols].sort_values("SampleName").to_csv(output, index=False, encoding="utf-8-sig")

print("\nSaved:", output)
print("\nNo original records were modified.")
