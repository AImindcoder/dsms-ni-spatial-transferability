from pathlib import Path
import pandas as pd

base = Path("02_Data_Audit/provenance_analysis")
source = base / "67_chronology_source_verification_candidates.csv"
previous = base / "68_independent_chronology_evidence_register.csv"
output = base / "69_independent_chronology_evidence_register.csv"

candidates = pd.read_csv(source)
old = pd.read_csv(previous)

doi = "10.1126/sciadv.abg2906"
site = "IODP U1342"

match = candidates[
    candidates["_doi_key"].astype(str).str.lower().eq(doi.lower())
    & candidates["_site_key"].astype(str).str.lower().eq(site.lower())
]

if match.empty:
    raise ValueError("U1342 candidate not found in Experiment 67.")

r = match.iloc[0]

new = {
    "doi": doi,
    "site": site,
    "status": "chronology_method_documented_secondary_source",
    "original_method": (
        "Visual correlation of benthic foraminiferal delta18O "
        "to the LR04 benthic delta18O stack; age model attributed "
        "to Knudson and Ravelo (2015)."
    ),
    "source_evidence": (
        "Knudson et al. (2021), Methods section, explicitly identifies "
        "the U1342 age model and its LR04 correlation basis. The paper "
        "reports approximately 1 ka average sampling frequency. Its "
        "bulk sediment delta15N data are cited to Knudson and Ravelo (2015)."
    ),
    "database_evidence": (
        f"{int(r['all_database_rows'])} rows; "
        f"{int(r['unique_ages'])} unique ages; "
        f"{int(r['unique_depths'])} unique depths; "
        f"{int(r['paired_age_depth_rows'])} paired age-depth rows; "
        f"Spearman={r['depth_age_spearman']}."
    ),
    "interpretation": (
        "The original publication documents an isotope-stack-based "
        "chronology, while the database proxy is variable_age_depth_unresolved. "
        "This may reflect non-monotonic or repeated depth representation, "
        "but does not establish that the original age model is unreliable. "
        "The isotope data attribution to the earlier publication must also "
        "be distinguished from the 2021 paper's chronology description."
    ),
    "confidence": "high_for_method_documentation",
    "source_url": (
        "https://www.researchgate.net/publication/352082885_"
        "Causes_and_timing_of_recurring_subarctic_Pacific_hypoxia"
    ),
    "source_site_group_id": r["source_site_group_id"],
    "database_resolution_proxy": r["resolution_proxy"],
    "database_rows": r["all_database_rows"],
    "numeric_target_rows": r["numeric_target_rows"],
    "unique_ages": r["unique_ages"],
    "unique_depths": r["unique_depths"],
    "paired_age_depth_rows": r["paired_age_depth_rows"],
    "depth_age_spearman": r["depth_age_spearman"],
    "original_age_assignment_exactly_verified": False,
    "database_age_values_corrected": False
}

result = pd.concat([old, pd.DataFrame([new])], ignore_index=True)
result.to_csv(output, index=False)

print("=" * 72)
print("EXPERIMENT 69: U1342 CHRONOLOGY SOURCE EVIDENCE")
print("=" * 72)
print("Site:", site)
print("DOI:", doi)
print("Database category:", r["resolution_proxy"])
print("Unique ages:", r["unique_ages"])
print("Unique depths:", r["unique_depths"])
print("Original chronology: LR04 benthic isotope correlation")
print("Sampling resolution reported: approximately 1 ka")
print("Evidence register records:", len(result))
print("Output:", output)
print("No original database values or classifications modified.")
print("\nEXPERIMENT 69 COMPLETED")
