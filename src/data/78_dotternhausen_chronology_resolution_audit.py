from pathlib import Path
import pandas as pd
import numpy as np

folder = Path("02_Data_Audit/provenance_analysis")

source_path = folder / "77_dotternhausen_original_source_table.csv"
database_path = folder / "77_dotternhausen_dsmsni_records.csv"

source = pd.read_csv(source_path)
db = pd.read_csv(database_path, encoding="utf-8-sig")

target_col = next(c for c in db.columns if "15Nbulk" in c)

source["Source_Depth_cm"] = pd.to_numeric(
    source["Source_Depth_cm"], errors="coerce"
)
db["RelativeDepth"] = pd.to_numeric(
    db["RelativeDepth"], errors="coerce"
)

source["Source_d15Nbulk"] = pd.to_numeric(
    source["Source_d15Nbulk"], errors="coerce"
)
db[target_col] = pd.to_numeric(db[target_col], errors="coerce")
db["Age"] = pd.to_numeric(db["Age"], errors="coerce")

# Exact depth matching; no sample identities are invented.
merged = source.merge(
    db,
    left_on="Source_Depth_cm",
    right_on="RelativeDepth",
    how="outer",
    validate="one_to_one",
    indicator=True
)

merged["depth_match"] = merged["_merge"].eq("both")
merged["isotope_match"] = (
    merged["Source_d15Nbulk"] - merged[target_col]
).abs() < 1e-6

# Stratigraphic lithology documented in Wang et al. (2021).
def lithology(depth):
    if pd.isna(depth):
        return "unknown"
    if 980 <= depth <= 1050:
        return "bioturbated_light_gray_organic_lean_shale"
    if 610 <= depth < 980:
        return "laminated_organic_rich_shale_and_marlstone"
    return "outside_documented_study_interval"

merged["source_lithological_context"] = (
    merged["Source_Depth_cm"].apply(lithology)
)

merged["source_study_interval"] = (
    merged["Source_Depth_cm"].between(610, 1050)
)

merged["sample_specific_numeric_age_in_source_table"] = False
merged["dsmsni_age_is_constant_within_site"] = (
    merged["Age"].nunique(dropna=True) == 1
)

merged["chronology_interpretation"] = (
    "Original source documents high-resolution sampling and "
    "biostratigraphic context; sample-specific numerical age "
    "not present in extracted Table 1."
)

output = folder / "78_dotternhausen_chronology_resolution_register.csv"
merged.to_csv(output, index=False, encoding="utf-8-sig")

print("=" * 76)
print("EXPERIMENT 78: DOTTERNHAUSEN CHRONOLOGY RESOLUTION AUDIT")
print("=" * 76)

print("\nSAMPLE-LEVEL COVERAGE")
print("Combined rows:", len(merged))
print("Exact depth matches:", int(merged["depth_match"].sum()))
print("Exact isotope matches:", int(merged["isotope_match"].sum()))
print("Samples within documented 610-1050 cm interval:",
      int(merged["source_study_interval"].sum()))

print("\nLITHOLOGICAL CONTEXT")
print(merged["source_lithological_context"].value_counts().to_string())

print("\nDSMS-NI NUMERICAL AGE")
print("Unique ages:", merged["Age"].nunique(dropna=True))
print("Age values:", sorted(merged["Age"].dropna().unique().tolist()))
print("Records with age:", int(merged["Age"].notna().sum()))

print("\nSTRATIGRAPHIC COVERAGE")
print("Minimum depth (cm):", merged["Source_Depth_cm"].min())
print("Maximum depth (cm):", merged["Source_Depth_cm"].max())
print("Depth span (cm):",
      merged["Source_Depth_cm"].max() - merged["Source_Depth_cm"].min())

print("\nAGE RESOLUTION EVIDENCE")
print("Source table contains individual numerical ages: NO")
print("DSMS-NI contains one age for all records:",
      merged["dsmsni_age_is_constant_within_site"].all())
print("Assigned age:", merged["Age"].dropna().iloc[0]
      if merged["Age"].notna().any() else "missing")

print("\nIMPORTANT LIMITATION")
print(
    "This audit does not establish that the DSMS-NI age is erroneous. "
    "It distinguishes documented stratigraphic resolution from "
    "sample-specific numerical age resolution."
)

print("\nOUTPUT:", output)
print("No original data or age assignments modified.")
print("EXPERIMENT 78 COMPLETED")
