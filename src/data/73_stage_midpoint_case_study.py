from pathlib import Path
import pandas as pd

RAW = Path("01_Data/raw/DSMSNI_v0.3.csv")
OUT = Path("02_Data_Audit/provenance_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, encoding="gbk", low_memory=False)

def col(fragment):
    found = [c for c in df.columns if fragment.lower() in str(c).lower()]
    if not found:
        raise KeyError(f"Column not found: {fragment}")
    return found[0]

doi_col = col("DOI")
site_col = col("SiteName")
age_col = col("Age")
stage_col = col("Stage")
period_col = col("Period")
depth_col = col("RelativeDepth")
sample_col = col("SampleName")
target_col = col("15Nbulk")

df["_age"] = pd.to_numeric(df[age_col], errors="coerce")
df["_depth"] = pd.to_numeric(df[depth_col], errors="coerce")
df["_target"] = pd.to_numeric(df[target_col], errors="coerce")

cases = [
    {
        "site": "Dogna",
        "doi": "10.1029/2000pa000558",
        "source_evidence": (
            "Experiment 70: all 163 sample labels matched the public "
            "PANGAEA source export; 74/74 numeric isotope pairs matched "
            "exactly; depth rank correlation was 1.0."
        ),
        "verification": "strong_sample_level_source_data_agreement",
        "age_interpretation": (
            "179.45 Ma exactly equals the tested Toarcian stage midpoint. "
            "Exact database assignment procedure remains unverified."
        )
    },
    {
        "site": "Hawsker Bottom",
        "doi": "10.1029/2000pa000558",
        "source_evidence": (
            "Shares the DOI and assigned age with Dogna. Individual "
            "sample-level source comparison has not yet been completed."
        ),
        "verification": "source_comparison_pending",
        "age_interpretation": (
            "179.45 Ma exactly equals the tested Toarcian stage midpoint. "
            "Exact database assignment procedure remains unverified."
        )
    },
    {
        "site": "Dotternhausen sedimentary",
        "doi": "10.1029/2020pa004207",
        "source_evidence": (
            "Wang et al. (2021) report 31 samples, average spacing "
            "approximately 3.5 cm, across a 1050-610 cm section. "
            "The studied interval is constrained by ammonite subzones."
        ),
        "verification": "original_stratigraphic_framework_documented",
        "age_interpretation": (
            "All 31 database records have 179.45 Ma. The source documents "
            "detailed stratigraphy, but exact per-sample numerical ages "
            "and the database assignment procedure remain unverified."
        )
    }
]

profiles = []
register = []

for case in cases:
    g = df[
        df[doi_col].astype(str).str.lower().str.contains(
            case["doi"].lower(), regex=False, na=False
        )
        & df[site_col].astype(str).str.strip().str.lower().eq(
            case["site"].lower()
        )
    ].copy()

    if g.empty:
        print("WARNING: No exact group found:", case["site"])
        continue

    stages = sorted(g[stage_col].dropna().astype(str).unique())
    periods = sorted(g[period_col].dropna().astype(str).unique())
    ages = sorted(g["_age"].dropna().unique())

    register.append({
        "site": case["site"],
        "doi": case["doi"],
        "records": len(g),
        "unique_ages": len(ages),
        "assigned_age_Ma": ages[0] if len(ages) == 1 else None,
        "unique_depths": g["_depth"].nunique(),
        "numeric_isotope_records": g["_target"].notna().sum(),
        "stages": " | ".join(stages),
        "periods": " | ".join(periods),
        "source_evidence": case["source_evidence"],
        "source_verification": case["verification"],
        "age_interpretation": case["age_interpretation"],
        "exact_age_assignment_verified": False,
        "database_age_corrected": False
    })

    keep = [
        c for c in [
            doi_col, site_col, sample_col, age_col, stage_col,
            period_col, depth_col, target_col
        ] if c in g.columns
    ]
    profile = g[keep].copy()
    profile.insert(0, "case_site", case["site"])
    profiles.append(profile)

register_df = pd.DataFrame(register)
profiles_df = pd.concat(profiles, ignore_index=True)

register_path = OUT / "73_stage_midpoint_case_register.csv"
profiles_path = OUT / "73_stage_midpoint_case_sample_profiles.csv"

register_df.to_csv(register_path, index=False, encoding="utf-8-sig")
profiles_df.to_csv(profiles_path, index=False, encoding="utf-8-sig")

print("=" * 74)
print("EXPERIMENT 73: THREE-SITE STAGE-MIDPOINT CASE STUDY")
print("=" * 74)

print(register_df[
    ["site", "records", "unique_ages", "assigned_age_Ma",
     "unique_depths", "numeric_isotope_records", "stages"]
].to_string(index=False))

print("\nSOURCE VERIFICATION")
for _, row in register_df.iterrows():
    print(f"\n{row['site']}: {row['source_verification']}")
    print(row["source_evidence"])

print("\nREGISTER:", register_path)
print("SAMPLE PROFILES:", profiles_path)
print("No raw records or assigned ages modified.")
print("EXPERIMENT 73 COMPLETED")
