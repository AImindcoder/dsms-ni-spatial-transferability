import pandas as pd
from pathlib import Path

raw_path = Path("01_Data/raw/DSMSNI_v0.3.csv")
decision_path = Path("02_Data_Audit/provenance_analysis/verified_source_identity_decisions.csv")
out_dir = Path("02_Data_Audit/provenance_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path, encoding="gbk", low_memory=False)
decisions = pd.read_csv(decision_path)

def normalize_doi(series):
    result = series.astype("string").str.strip().str.lower()
    result = result.mask(
        result.isna() |
        result.eq("") |
        result.eq("not available")
    )
    return result

df["_doi_normalized"] = normalize_doi(df["DOI"])
decisions["_doi_normalized"] = normalize_doi(decisions["DOI"])

# Build DOI-level metadata summary.
valid = df[df["_doi_normalized"].notna()].copy()

summary = (
    valid.groupby("_doi_normalized")
    .agg(
        record_count=("SampleID", "size"),
        unique_titles=("Title", "nunique"),
        unique_authors=("FirstAuthor", "nunique"),
        unique_years=("Year", "nunique")
    )
    .reset_index()
)

summary["metadata_review_status"] = "no_detected_metadata_variation"

variation = (
    (summary["unique_titles"] > 1) |
    (summary["unique_authors"] > 1) |
    (summary["unique_years"] > 1)
)
summary.loc[variation, "metadata_review_status"] = (
    "metadata_variation_review_required"
)

# Default: one provisional group per DOI.
summary["provisional_source_group_id"] = (
    "PSG_DOI_" + summary["_doi_normalized"]
)
summary["group_basis"] = "DOI_conservative_proxy"
summary["identity_status"] = (
    "provisional_not_externally_verified_as_source_study"
)

# Apply our documented manual decisions.
decision_lookup = decisions.set_index("_doi_normalized")

for doi, decision in decision_lookup.iterrows():
    mask = summary["_doi_normalized"].eq(doi)
    summary.loc[mask, "provisional_source_group_id"] = (
        decision["source_study_id"]
    )
    summary.loc[mask, "group_basis"] = "documented_manual_decision"
    summary.loc[mask, "identity_status"] = decision["identity_status"]

# Retain metadata-review warnings even for manually reviewed groups.
# A review decision does not erase other metadata variations.

# Create a row-level mapping.
mapping = df[["SampleID", "DOI", "FirstAuthor", "Year", "Title"]].copy()
mapping["_doi_normalized"] = df["_doi_normalized"]

mapping = mapping.merge(
    summary[[
        "_doi_normalized",
        "provisional_source_group_id",
        "group_basis",
        "identity_status",
        "metadata_review_status"
    ]],
    on="_doi_normalized",
    how="left"
)

no_doi = mapping["_doi_normalized"].isna()
mapping.loc[no_doi, "group_basis"] = "unresolved_no_usable_DOI"
mapping.loc[no_doi, "identity_status"] = "unresolved"
mapping.loc[no_doi, "metadata_review_status"] = "source_identity_not_established"

mapping = mapping.rename(columns={
    "_doi_normalized": "normalized_doi"
})

mapping_path = out_dir / "full_provenance_mapping.csv"
summary_path = out_dir / "doi_provenance_summary.csv"

mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("FULL PROVENANCE MAPPING")
print("=" * 65)
print("Dataset observations:", len(df))
print("Mapped observations:", mapping["provisional_source_group_id"].notna().sum())
print("Usable DOI observations:", mapping["normalized_doi"].notna().sum())
print("Unresolved no-DOI observations:", no_doi.sum())
print("Unique usable DOIs:", summary["_doi_normalized"].nunique())
print("Provisional source groups:", summary["provisional_source_group_id"].nunique())
print("DOI groups requiring metadata review:", variation.sum())

print("\nMANUAL DECISIONS APPLIED:")
print(decisions[["DOI", "source_study_id", "record_count"]].to_string(index=False))

print("\nSAVED FILES:")
print(mapping_path.resolve())
print(summary_path.resolve())
