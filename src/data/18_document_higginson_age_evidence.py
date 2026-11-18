import pandas as pd
from pathlib import Path

records = [
    {
        "SampleID": "DSMS-NI-33736",
        "SiteName": "W7706-37",
        "core": "W7706-37K",
        "DSMS_NI_Age_Ma": 0.0,
        "published_core_top_interval_cm": "0-5",
        "published_calendar_age_years_BP": 850
    },
    {
        "SampleID": "DSMS-NI-33737",
        "SiteName": "W7706-40",
        "core": "W7706-40K",
        "DSMS_NI_Age_Ma": 0.0,
        "published_core_top_interval_cm": "0-5",
        "published_calendar_age_years_BP": 170
    },
    {
        "SampleID": "DSMS-NI-33738",
        "SiteName": "W7706-41",
        "core": "W7706-41K",
        "DSMS_NI_Age_Ma": 0.0,
        "published_core_top_interval_cm": "8-10",
        "published_calendar_age_years_BP": 440
    }
]

df = pd.DataFrame(records)

df["published_calendar_age_Ma"] = (
    df["published_calendar_age_years_BP"] / 1_000_000
)

df["age_evidence_source"] = (
    "Higginson and Altabet (2004), supplementary age/depth table"
)

df["depth_correspondence_verified"] = False
df["proposed_exact_sample_age"] = pd.NA
df["decision_status"] = "zero_age_requires_sample_depth_verification"
df["raw_dataset_modified"] = False

output = Path(
    "02_Data_Audit/provenance_analysis/"
    "higginson_zero_age_evidence_register.csv"
)

df.to_csv(output, index=False, encoding="utf-8-sig")

print("HIGGINSON ZERO-AGE EVIDENCE REGISTER")
print("=" * 75)
print(df.to_string(index=False))
print("\nRecords:", len(df))
print("Exact sample ages assigned: 0")
print("Raw dataset modified: False")
print("\nSaved:", output)
