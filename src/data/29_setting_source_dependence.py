import pandas as pd
import numpy as np

raw = pd.read_csv(
    "01_Data/raw/DSMSNI_v0.3.csv",
    encoding="gbk",
    low_memory=False
)

mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)

target = next(c for c in raw.columns if "15Nbulk" in c)
group_col = "provisional_source_group_id"

data = raw.merge(
    mapping[["SampleID", group_col]],
    on="SampleID",
    how="left",
    validate="one_to_one"
)

data[target] = pd.to_numeric(data[target], errors="coerce")

required = [
    target, "Age", "Latitude", "Longitude",
    "Material", "Setting", "LithType", "Lithology",
    group_col
]

data = data.dropna(subset=required).copy()
data["Setting"] = data["Setting"].astype(str).str.strip()

summary = data.groupby("Setting").agg(
    observations=(target, "size"),
    source_studies=(group_col, "nunique"),
    mean_d15N=(target, "mean"),
    std_d15N=(target, "std")
).reset_index()

summary["observations_per_study"] = (
    summary["observations"] / summary["source_studies"]
)

summary = summary.sort_values(
    ["source_studies", "observations"],
    ascending=[True, False]
)

print("\nSETTING CATEGORY PUBLICATION DEPENDENCE")
print("Observations:", len(data))
print("Provisional source-study groups:", data[group_col].nunique())
print("Setting categories:", len(summary))

print("\nCATEGORY SHARING DISTRIBUTION")
print(
    summary["source_studies"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nCATEGORIES USED BY ONLY ONE SOURCE GROUP")
print(
    summary[summary["source_studies"] == 1]
    .to_string(index=False)
)

print("\n20 MOST WIDELY SHARED CATEGORIES")
print(
    summary.sort_values("source_studies", ascending=False)
    .head(20)
    .to_string(index=False)
)

summary.to_csv(
    "02_Data_Audit/provenance_analysis/setting_source_study_dependence.csv",
    index=False
)

print("\nSaved setting_source_study_dependence.csv")
print("Raw dataset modified: False")
