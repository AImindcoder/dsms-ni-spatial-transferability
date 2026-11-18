from pathlib import Path
import pandas as pd

RAW = Path("01_Data/raw/DSMSNI_v0.3.csv")
OUT = Path("02_Data_Audit/provenance_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, encoding="gbk", low_memory=False)

def find_col(fragment):
    matches = [c for c in df.columns if fragment.lower() in str(c).lower()]
    if not matches:
        raise KeyError(f"Column not found: {fragment}")
    return matches[0]

doi_col = find_col("DOI")
site_col = find_col("SiteName")
age_col = find_col("Age")
stage_col = find_col("Stage")
depth_col = find_col("RelativeDepth")
target_col = find_col("15Nbulk")

df["_age"] = pd.to_numeric(df[age_col], errors="coerce")
df["_depth"] = pd.to_numeric(df[depth_col], errors="coerce")
df["_doi"] = df[doi_col].astype(str).str.strip()
df["_site"] = df[site_col].astype(str).str.strip()

df = df[
    df["_doi"].ne("")
    & df["_doi"].str.lower().ne("nan")
    & df["_site"].ne("")
    & df["_site"].str.lower().ne("nan")
].copy()

groups = []

for (doi, site), g in df.groupby(["_doi", "_site"], dropna=False):
    if len(g) < 2:
        continue

    ages = g["_age"].dropna().unique()
    depths = g["_depth"].dropna().nunique()
    stages = sorted(g[stage_col].dropna().astype(str).unique())
    periods = sorted(g[find_col("Period")].dropna().astype(str).unique())

    if len(ages) == 1 and depths > 1:
        groups.append({
            "DOI": doi,
            "SiteName": site,
            "records": len(g),
            "unique_ages": len(ages),
            "assigned_age_Ma": float(ages[0]),
            "unique_depths": depths,
            "stages": " | ".join(stages),
            "periods": " | ".join(periods),
            "numeric_target_records": pd.to_numeric(
                g[target_col], errors="coerce"
            ).notna().sum(),
            "age_equals_179_45": abs(float(ages[0]) - 179.45) < 1e-8,
        })

result = pd.DataFrame(groups)

if result.empty:
    raise RuntimeError("No constant-age/multiple-depth groups found.")

result = result.sort_values(
    ["records", "DOI"], ascending=[False, True]
)

# Stage-level summary
stage_rows = []
for _, row in result.iterrows():
    for stage in row["stages"].split(" | "):
        stage_rows.append({
            "stage": stage,
            "DOI": row["DOI"],
            "site": row["SiteName"],
            "records": row["records"],
            "age_Ma": row["assigned_age_Ma"],
            "unique_depths": row["unique_depths"],
        })

stage_df = pd.DataFrame(stage_rows)
stage_summary = (
    stage_df.groupby("stage")
    .agg(
        source_site_groups=("DOI", "size"),
        observations=("records", "sum"),
        distinct_assigned_ages=("age_Ma", "nunique"),
    )
    .reset_index()
    .sort_values("source_site_groups", ascending=False)
)

toarcian = result[
    result["stages"].str.contains("Toarcian", case=False, na=False)
].copy()

summary = {
    "constant_age_multiple_depth_groups": len(result),
    "observations_in_these_groups": int(result["records"].sum()),
    "groups_assigned_179_45_Ma": int(result["age_equals_179_45"].sum()),
    "observations_assigned_179_45_Ma": int(
        result.loc[result["age_equals_179_45"], "records"].sum()
    ),
    "Toarcian_constant_age_groups": len(toarcian),
    "Toarcian_groups_assigned_179_45_Ma": int(
        toarcian["age_equals_179_45"].sum()
    ),
    "Toarcian_observations": int(toarcian["records"].sum()),
}

result.to_csv(
    OUT / "72_constant_age_multiple_depth_groups.csv",
    index=False, encoding="utf-8-sig"
)
stage_summary.to_csv(
    OUT / "72_constant_age_stage_summary.csv",
    index=False, encoding="utf-8-sig"
)
toarcian.to_csv(
    OUT / "72_toarcian_constant_age_groups.csv",
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(
    [{"metric": k, "value": v} for k, v in summary.items()]
).to_csv(
    OUT / "72_constant_age_signature_summary.csv",
    index=False, encoding="utf-8-sig"
)

print("=" * 74)
print("EXPERIMENT 72: DATABASE-WIDE CONSTANT-AGE SIGNATURE AUDIT")
print("=" * 74)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nMOST FREQUENT STAGES")
print(stage_summary.head(20).to_string(index=False))

print("\nTOARCIAN GROUPS")
if len(toarcian):
    print(toarcian[
        ["DOI", "SiteName", "records", "assigned_age_Ma", "unique_depths"]
    ].to_string(index=False))
else:
    print("No Toarcian groups found.")

print("\nOUTPUT DIRECTORY:", OUT)
print("Raw dataset unchanged.")
print("EXPERIMENT 72 COMPLETED")
