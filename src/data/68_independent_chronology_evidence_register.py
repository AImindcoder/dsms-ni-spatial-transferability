from pathlib import Path
import pandas as pd

OUT = Path("02_Data_Audit/provenance_analysis")
INPUT = OUT / "67_chronology_source_verification_candidates.csv"

df = pd.read_csv(INPUT)

cases = [
    {
        "doi": "10.1016/j.palaeo.2015.03.043",
        "site": "Rietheim",
        "status": "partially_verified_original_stratigraphy",
        "original_method": "Systematic stratigraphic sampling; ammonite biozone framework",
        "source_evidence": (
            "Montero-Serrano et al. (2015) describe a ~790 cm section, "
            "155 core samples at 2 cm spacing and 71 outcrop samples "
            "at 5-10 cm spacing. The study includes total nitrogen isotopes."
        ),
        "database_evidence": (
            "226 database rows, one unique assigned age, 225 unique depths."
        ),
        "interpretation": (
            "Potential mismatch between detailed source stratigraphy and "
            "single numerical database age. Original per-sample numerical "
            "age assignments and database conversion procedure remain unverified."
        ),
        "confidence": "moderate",
        "source_url": (
            "https://semaphore.uqar.ca/id/eprint/1465/1/"
            "Jean-Carlos_Montero-Serrano_et_al_juillet2015.pdf"
        )
    },
    {
        "doi": "10.1029/2007pa001518",
        "site": "MD2416",
        "status": "chronology_method_documented_database_depth_missing",
        "original_method": (
            "High-resolution XRF calcium correlation between MD2416 "
            "and ODP 882; alternative chronology using additional "
            "stratigraphic tie points discussed."
        ),
        "source_evidence": (
            "Original publication explicitly describes construction "
            "of the MD2416 age model using XRF Ca correlations."
        ),
        "database_evidence": (
            "444 rows, 444 unique ages, zero recorded depth values."
        ),
        "interpretation": (
            "Database proxy insufficient_information reflects missing "
            "depth data, not demonstrated lack of chronological resolution."
        ),
        "confidence": "high",
        "source_url": (
            "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2007PA001518"
        )
    },
    {
        "doi": "10.1029/2008gc002044",
        "site": "ODP 1012",
        "status": "chronology_method_documented",
        "original_method": (
            "SST correlation with ODP 846 before 1.8 Ma and benthic "
            "oxygen-isotope stratigraphy for the last 1.8 Ma."
        ),
        "source_evidence": (
            "Liu et al. (2008) document the chronology and discuss "
            "its limitations, including the single-core Pliocene interval."
        ),
        "database_evidence": (
            "2412 rows, 2334 unique ages, 2343 unique depths, "
            "2412 paired age-depth rows, Spearman correlation 1.0."
        ),
        "interpretation": (
            "Database monotonic pattern is consistent with a detailed "
            "chronological record, but does not identify the original "
            "age-modelling algorithm."
        ),
        "confidence": "high",
        "source_url": (
            "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008GC002044"
        )
    }
]

records = []

for case in cases:
    match = df[
        (df["_doi_key"].astype(str).str.lower() == case["doi"].lower())
        & (df["_site_key"].astype(str).str.lower() == case["site"].lower())
    ]

    if match.empty:
        raise ValueError(
            f"Could not locate selected candidate: {case['doi']} / {case['site']}"
        )

    for _, row in match.iterrows():
        records.append({
            **case,
            "source_site_group_id": row["source_site_group_id"],
            "database_resolution_proxy": row["resolution_proxy"],
            "database_rows": row["all_database_rows"],
            "numeric_target_rows": row["numeric_target_rows"],
            "unique_ages": row["unique_ages"],
            "unique_depths": row["unique_depths"],
            "paired_age_depth_rows": row["paired_age_depth_rows"],
            "depth_age_spearman": row["depth_age_spearman"],
            "original_age_assignment_exactly_verified": False,
            "database_age_values_corrected": False
        })

result = pd.DataFrame(records)

path = OUT / "68_independent_chronology_evidence_register.csv"
result.to_csv(path, index=False)

print("=" * 76)
print("EXPERIMENT 68: INDEPENDENT CHRONOLOGY EVIDENCE REGISTER")
print("=" * 76)

for _, row in result.iterrows():
    print("\nSITE:", row["site"])
    print("DOI:", row["doi"])
    print("DATABASE CATEGORY:", row["database_resolution_proxy"])
    print("VERIFICATION:", row["status"])
    print("SOURCE METHOD:", row["original_method"])
    print("INTERPRETATION:", row["interpretation"])

print("\nREGISTER:", path)
print("VERIFIED CASES:", len(result))
print("RAW DATA MODIFIED: No")
print("DATABASE AGES MODIFIED: No")
print("\nEXPERIMENT 68 COMPLETED")
