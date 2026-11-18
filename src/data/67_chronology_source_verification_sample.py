from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path(
    "02_Data_Audit/provenance_analysis/"
    "60_chronology_resolution_source_site_register.csv"
)
OUT = Path("02_Data_Audit/provenance_analysis")

df = pd.read_csv(INPUT)

required = [
    "_doi_key", "_site_key", "all_database_rows",
    "numeric_target_rows", "unique_ages", "unique_depths",
    "paired_age_depth_rows", "depth_age_spearman",
    "resolution_proxy", "source_site_group_id"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

rng = np.random.default_rng(67067)
selected = []

for category, group in df.groupby("resolution_proxy"):
    group = group.copy()

    group = group.sort_values(
        ["all_database_rows", "numeric_target_rows"],
        ascending=False
    )

    largest = group.head(4).copy()
    largest["selection_reason"] = "largest_observation_groups"

    remaining = group.iloc[4:].copy()

    if len(remaining) > 0:
        sample_n = min(4, len(remaining))
        random_indices = rng.choice(
            remaining.index.to_numpy(),
            size=sample_n,
            replace=False
        )
        random_part = remaining.loc[random_indices].copy()
        random_part["selection_reason"] = (
            "reproducible_random_category_sample"
        )
    else:
        random_part = remaining.copy()
        random_part["selection_reason"] = pd.Series(dtype=str)

    selected.append(largest)
    if len(random_part):
        selected.append(random_part)

review = pd.concat(selected, ignore_index=True)

review["source_verification_status"] = "not_yet_verified"
review["original_publication_age_method"] = ""
review["original_publication_depth_method"] = ""
review["independent_source_evidence"] = ""
review["reviewer_decision"] = ""

review = review.sort_values(
    ["resolution_proxy", "selection_reason", "all_database_rows"],
    ascending=[True, True, False]
)

review.to_csv(
    OUT / "67_chronology_source_verification_candidates.csv",
    index=False
)

summary = (
    review.groupby(["resolution_proxy", "selection_reason"])
    .agg(
        selected_groups=("source_site_group_id", "count"),
        distinct_publications=("_doi_key", "nunique"),
        observations=("all_database_rows", "sum")
    )
    .reset_index()
)

summary.to_csv(
    OUT / "67_chronology_source_verification_sample_summary.csv",
    index=False
)

print("=" * 76)
print("EXPERIMENT 67: CHRONOLOGY SOURCE-VERIFICATION SAMPLE")
print("=" * 76)

print("\nSELECTED GROUPS:", len(review))
print("DISTINCT PUBLICATIONS:", review["_doi_key"].nunique())

print("\nCATEGORY COVERAGE")
print(summary.to_string(index=False))

print("\nREVIEW CANDIDATES")
print(
    review[[
        "_doi_key", "_site_key", "all_database_rows",
        "numeric_target_rows", "unique_ages", "unique_depths",
        "paired_age_depth_rows", "depth_age_spearman",
        "resolution_proxy", "selection_reason"
    ]].to_string(index=False)
)

print("\nFILES SAVED")
print("67_chronology_source_verification_candidates.csv")
print("67_chronology_source_verification_sample_summary.csv")

print("\nNo source classifications or raw observations were changed.")
print("\nEXPERIMENT 67 COMPLETED")
