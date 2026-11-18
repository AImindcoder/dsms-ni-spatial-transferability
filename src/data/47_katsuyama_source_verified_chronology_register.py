import pandas as pd
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")

source = pd.read_csv(
    base / "45_katsuyama_primary_source_comparison.csv",
    encoding="utf-8-sig",
    low_memory=False
)

out = source.copy()

out["external_evidence_2015"] = ""
out["external_evidence_2023"] = ""
out["chronology_assessment"] = "unresolved_requires_individual_source_verification"
out["sample_identity_assessment"] = "probable_shared_sample_identifier_not_confirmed_physical_identity"
out["isotope_coverage_assessment"] = "2016_target_missing_original_publication_reports_TN_isotopes"
out["proposed_action"] = "retain_original_values_no_correction"

mask = out["SampleName"].astype(str).str.strip() == "IYF53D-0"

out.loc[mask, "external_evidence_2015"] = (
    "Fujisaki et al. 2015 Table 3: sample IYF53D-0, "
    "Bed 14, age classified as Triassic-Jurassic."
)

out.loc[mask, "external_evidence_2023"] = (
    "Ikeda et al. 2023 identifies IYF53D-0 of Fujisaki et al. 2016 "
    "as the purple shale marking the base of the post-end-Triassic-extinction interval."
)

out.loc[mask, "chronology_assessment"] = (
    "strong_external_chronology_conflict_2016_Pliensbachian_assignment"
)

out.loc[mask, "proposed_action"] = (
    "flag_for_primary_source_review_do_not_correct_until_age_model_verified"
)

output = base / "46_katsuyama_source_verified_chronology_register.csv"
out.to_csv(output, index=False, encoding="utf-8-sig")

print("=" * 70)
print("EXPERIMENT 47: SOURCE-VERIFIED CHRONOLOGY REGISTER")
print("=" * 70)
print("Records:", len(out))
print("IYF53D-0 records:", mask.sum())
print("Other unresolved records:", (~mask).sum())
print("Original source values preserved.")
print("No records deleted or corrected.")
print("Saved:", output)
