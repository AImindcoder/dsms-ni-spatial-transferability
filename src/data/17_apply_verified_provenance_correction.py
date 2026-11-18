import pandas as pd
from pathlib import Path

mapping_path = Path(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping.csv"
)
source_path = Path(
    "01_Data/external/Schlung_2013_U1340A_PANGAEA.txt"
)

output_path = Path(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_corrected.csv"
)
log_path = Path(
    "02_Data_Audit/provenance_analysis/provenance_correction_log.csv"
)

target_id = "DSMS-NI-46187"
original_doi = "10.1029/2004pa001114"
corrected_source_doi = "10.1029/2012pa002365"

df = pd.read_csv(mapping_path, dtype=str)

# Validate the independent source evidence file.
if not source_path.exists():
    raise FileNotFoundError(source_path)

source_text = source_path.read_text(
    encoding="utf-8", errors="replace"
)

if "10.1029/2012PA002365" not in source_text:
    raise ValueError("Expected Schlung DOI not found in source file.")

if "6.047" not in source_text or "4.46" not in source_text:
    raise ValueError("Expected age/isotope evidence not found.")

# Identify the target and destination group.
target_mask = df["SampleID"] == target_id
target = df[target_mask]

if len(target) != 1:
    raise ValueError("Expected exactly one target observation.")

row = target.iloc[0]

if row["normalized_doi"].lower() != original_doi:
    raise ValueError("Unexpected original DOI.")

if row["FirstAuthor"] != "Schlung S." or row["Year"] != "2013":
    raise ValueError("Author/year validation failed.")

destination_mask = (
    df["normalized_doi"].str.lower() == corrected_source_doi
)

if destination_mask.sum() != 163:
    raise ValueError("Expected exactly 163 existing Schlung DOI records.")

old_group = row["provisional_source_group_id"]
new_group = df.loc[destination_mask, "provisional_source_group_id"].iloc[0]

if old_group == new_group:
    raise ValueError("Source groups are already identical.")

# Preserve the original assignment explicitly.
df["original_provisional_source_group_id"] = (
    df["provisional_source_group_id"]
)

# Correct only the derived source-study group assignment.
df.loc[target_mask, "provisional_source_group_id"] = new_group
df.loc[target_mask, "group_basis"] = (
    "independently_matched_supplementary_dataset"
)
df.loc[target_mask, "identity_status"] = (
    "provisional_external_source_match"
)
df.loc[target_mask, "metadata_review_status"] = (
    "source_attribution_correction_supported"
)

# Keep the original DOI untouched.
df["proposed_corrected_source_doi"] = ""
df.loc[target_mask, "proposed_corrected_source_doi"] = corrected_source_doi

df.to_csv(output_path, index=False, encoding="utf-8-sig")

# Create a transparent correction log.
log = pd.DataFrame([{
    "SampleID": target_id,
    "original_DOI": original_doi,
    "proposed_source_DOI": corrected_source_doi,
    "original_source_group": old_group,
    "corrected_source_group": new_group,
    "evidence": (
        "Schlung et al. 2013 PANGAEA supplementary dataset "
        "10.1594/PANGAEA.824449; matching age 6.047 ka and "
        "delta15N 4.46 per mil."
    ),
    "site_evidence": "IODP U1340 / IODP Hole 323-U1340A",
    "depth_caveat": (
        "DSMS-NI RelativeDepth is 0.0; exact depth correspondence "
        "has not been established."
    ),
    "decision": "Provisional source-group reassignment",
    "raw_dataset_modified": False,
    "original_DOI_modified": False
}])

log.to_csv(log_path, index=False, encoding="utf-8-sig")

# Validate resulting counts.
print("PROVENANCE CORRECTION VALIDATION")
print("=" * 70)
print("Target record:", target_id)
print("Original DOI:", original_doi)
print("Original source group:", old_group)
print("New source group:", new_group)
print("Original DOI group count:", int((df["original_provisional_source_group_id"] == old_group).sum()))
print("Corrected DOI group count:", int((df["provisional_source_group_id"] == new_group).sum()))
print("Total observations:", len(df))
print("Unique assigned source groups:", df["provisional_source_group_id"].nunique())
print("Original DOI preserved:", df.loc[target_mask, "DOI"].iloc[0] == original_doi)

print("\nOUTPUT FILES")
print(output_path)
print(log_path)
print("\nOriginal mapping and raw dataset were not modified.")
