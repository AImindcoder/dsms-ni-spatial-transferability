import pandas as pd
from pathlib import Path

raw = pd.read_csv(
    "01_Data/raw/DSMSNI_v0.3.csv",
    encoding="gbk",
    low_memory=False
)

ids = ["DSMS-NI-33736", "DSMS-NI-33737", "DSMS-NI-33738"]

records = raw[raw["SampleID"].astype(str).isin(ids)].copy()

if len(records) != 3:
    raise ValueError(f"Expected 3 records, found {len(records)}")

target = next(c for c in raw.columns if "15Nbulk" in c)

result = pd.DataFrame({
    "SampleID": records["SampleID"],
    "original_DOI": records["DOI"],
    "original_FirstAuthor": records["FirstAuthor"],
    "original_Year": records["Year"],
    "proposed_Year": 2004,
    "SiteName": records["SiteName"],
    "Age_Ma": records["Age"],
    "delta15N_permil": records[target],
    "evidence": "Publisher article page confirms Higginson and Altabet (2004)",
    "source_DOI": "10.1029/2004GL020511",
    "decision_status": "externally_supported_metadata_correction",
    "raw_dataset_modified": False
})

output = Path(
    "02_Data_Audit/provenance_analysis/"
    "verified_year_correction_register.csv"
)

result.to_csv(output, index=False, encoding="utf-8-sig")

print("VERIFIED YEAR CORRECTION REGISTER")
print("=" * 75)
print(result.to_string(index=False))
print("\nRecords documented:", len(result))
print("Original dataset modified: False")
print("Saved:", output)
