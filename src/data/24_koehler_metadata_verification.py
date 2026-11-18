import pandas as pd

raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)

samples = [
    "DSMS-NI-6212", "DSMS-NI-6233", "DSMS-NI-6250",
    "DSMS-NI-6211", "DSMS-NI-6216", "DSMS-NI-6235"
]

doi = "10.1016/j.precamres.2018.11.008"
group = raw[raw["DOI"].astype(str).str.strip().str.lower() == doi.lower()].copy()

print("Koehler 2019 publication")
print("Total raw records:", len(group))

columns = [
    "SampleID", "Age", "Latitude", "Longitude",
    "SiteName", "Formation", "Unit", "LithType",
    "Lithology", "Setting", "Material", "Location1-detail",
    "Location2-country", "DataSource"
]
available = [c for c in columns if c in group.columns]

print("\nSELECTED SAMPLE RECORDS")
print(group[group["SampleID"].isin(samples)][available].to_string(index=False))

print("\nUNIQUE NON-MISSING VALUES ACROSS THE PUBLICATION")
for col in available:
    if col == "SampleID":
        continue
    values = group[col].dropna().astype(str).unique()
    print(f"\n{col}: {len(values)} unique values")
    print(" | ".join(values[:15]))

print("\nPROVENANCE MAPPING")
m = mapping[mapping["SampleID"].isin(samples)]
print(m.to_string(index=False))

print("\nRaw dataset modified: False")
