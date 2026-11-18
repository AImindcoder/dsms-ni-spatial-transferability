from pathlib import Path
import pandas as pd

raw = Path("01_Data/raw/DSMSNI_v0.3.csv")
out = Path("02_Data_Audit/provenance_analysis")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw, encoding="gbk", low_memory=False)

def col(fragment):
    found = [c for c in df.columns if fragment.lower() in str(c).lower()]
    if not found:
        raise KeyError(f"Missing column: {fragment}")
    return found[0]

doi_col = col("DOI")
site_col = col("SiteName")
age_col = col("Age")
stage_col = col("Stage")
period_col = col("Period")

doi = "10.1029/2000PA000558"

dogna = df[
    df[doi_col].astype(str).str.lower().str.contains(doi.lower(), regex=False)
    & df[site_col].astype(str).str.strip().str.lower().eq("dogna")
].copy()

dogna[age_col] = pd.to_numeric(dogna[age_col], errors="coerce")

lower_bound = 174.7
upper_bound = 184.2
midpoint = (lower_bound + upper_bound) / 2

summary = {
    "DOI": doi,
    "Site": "Dogna",
    "records": len(dogna),
    "unique_numeric_ages": dogna[age_col].nunique(),
    "unique_stages": " | ".join(sorted(dogna[stage_col].dropna().astype(str).unique())),
    "unique_periods": " | ".join(sorted(dogna[period_col].dropna().astype(str).unique())),
    "database_age": dogna[age_col].median(),
    "assumed_Toarcian_lower_boundary_Ma": lower_bound,
    "assumed_Toarcian_upper_boundary_Ma": upper_bound,
    "calculated_stage_midpoint_Ma": midpoint,
    "database_age_equals_midpoint": bool(
        abs(dogna[age_col].median() - midpoint) < 1e-8
    ),
    "all_records_have_same_age": bool(dogna[age_col].nunique() == 1),
    "all_records_classified_Toarcian": bool(
        dogna[stage_col].astype(str).str.lower().eq("toarcian").all()
    ),
    "exact_database_assignment_method_verified": False,
}

result = pd.DataFrame([summary])
path = out / "71_dogna_stage_midpoint_verification.csv"
result.to_csv(path, index=False, encoding="utf-8-sig")

print("=" * 74)
print("EXPERIMENT 71: DOGNA STAGE-MIDPOINT VERIFICATION")
print("=" * 74)
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nOUTPUT:", path)
print("No original data or age assignments modified.")
print("EXPERIMENT 71 COMPLETED")
