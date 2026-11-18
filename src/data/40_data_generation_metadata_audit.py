import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RAW = "01_Data/raw/DSMSNI_v0.3.csv"
OUT = "02_Data_Audit/provenance_analysis"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(RAW, encoding="gbk", low_memory=False)

def find_col(fragment):
    return next((c for c in df.columns if fragment.lower() in str(c).lower()), None)

target_col = find_col("15Nbulk")
doi_col = find_col("DOI")
site_col = find_col("SiteName")
age_col = find_col("Age")
depth_col = find_col("RelativeDepth")
note_col = find_col("Note")
tech_col = find_col("Technique")
material_col = find_col("Material")
sample_col = find_col("SampleName")

if target_col is None:
    raise ValueError("Could not identify the bulk nitrogen isotope target.")

df["_target"] = pd.to_numeric(df[target_col], errors="coerce")
df["_age"] = pd.to_numeric(df[age_col], errors="coerce")
df["_depth"] = pd.to_numeric(df[depth_col], errors="coerce")

def classify_note(value):
    if pd.isna(value) or not str(value).strip():
        return "missing"
    s = str(value).strip().lower()
    if re.search(r"\bplot\b", s):
        return "figure_digitized"
    if any(x in s for x in ["piston", "gravity", "box", "multi"]):
        return "sampling_equipment_or_core_note"
    return "other_note"

def classify_technique(value):
    if pd.isna(value) or not str(value).strip():
        return "missing"
    s = str(value).lower()
    if "no mention" in s:
        return "not_reported"
    if "?" in s or re.search(r"\bor\b", s):
        return "uncertain_or_alternative"
    methods = []
    if "ea combustion" in s:
        methods.append("EA")
    if "denitrifier" in s:
        methods.append("denitrifier")
    if "offline combustion" in s:
        methods.append("offline")
    if "gc combustion" in s:
        methods.append("GC")
    if len(methods) > 1:
        return "mixed_methods"
    if len(methods) == 1:
        return methods[0]
    return "other_or_unclassified"

df["_note_class"] = df[note_col].apply(classify_note) if note_col else "not_available"
df["_tech_class"] = df[tech_col].apply(classify_technique) if tech_col else "not_available"

def category_report(column, filename):
    data = df[df["_target"].notna()].copy()
    result = data.groupby(column, dropna=False).agg(
        observations=("_target", "size"),
        publications=(doi_col, "nunique") if doi_col else ("_target", "size"),
        mean_delta15N=("_target", "mean"),
        median_delta15N=("_target", "median"),
        std_delta15N=("_target", "std")
    ).reset_index()
    result.to_csv(os.path.join(OUT, filename), index=False)
    print("\n", filename)
    print(result.to_string(index=False))

print("DATASET:", df.shape)
print("NUMERIC BULK TARGET:", df["_target"].notna().sum())

category_report("_note_class", "40_note_category_audit.csv")
category_report("_tech_class", "40_technique_category_audit.csv")

# Age-depth resolution proxies, grouped by DOI and SiteName.
group_cols = [c for c in [doi_col, site_col] if c is not None]
age_results = []

for keys, g in df[df["_target"].notna()].groupby(group_cols, dropna=False, sort=False):
    if not isinstance(keys, tuple):
        keys = (keys,)
    record = dict(zip(group_cols, keys))
    x = g["_depth"].dropna()
    paired = g[["_age", "_depth"]].dropna()
    n = len(g)
    age_n = g["_age"].nunique()
    depth_n = g["_depth"].nunique()
    rho = np.nan

    if len(paired) >= 5 and paired["_age"].nunique() > 1 and paired["_depth"].nunique() > 1:
        rho = spearmanr(paired["_depth"], paired["_age"]).statistic

    if n >= 5 and age_n == 1 and depth_n >= 3:
        status = "constant_age_multiple_depths_candidate"
    elif n >= 5 and age_n >= 3 and depth_n >= 3 and pd.notna(rho) and abs(rho) >= 0.80:
        status = "monotonic_age_depth_candidate"
    elif n >= 5 and age_n >= 3 and depth_n >= 3:
        status = "variable_age_depth_unresolved"
    else:
        status = "insufficient_information"

    record.update({
        "observations": n,
        "unique_ages": age_n,
        "unique_depths": depth_n,
        "age_min_Ma": g["_age"].min(),
        "age_max_Ma": g["_age"].max(),
        "depth_spearman": rho,
        "resolution_proxy": status
    })
    age_results.append(record)

age_df = pd.DataFrame(age_results)
age_path = os.path.join(OUT, "40_age_resolution_proxy_by_source_site.csv")
age_df.to_csv(age_path, index=False)

age_summary = age_df.groupby("resolution_proxy").agg(
    source_site_groups=("observations", "size"),
    observations=("observations", "sum")
).reset_index()
age_summary.to_csv(os.path.join(OUT, "40_age_resolution_proxy_summary.csv"), index=False)

print("\nAGE RESOLUTION PROXY SUMMARY")
print(age_summary.to_string(index=False))

# Candidate repeated positions; these are NOT confirmed replicate identities.
repeat_keys = [c for c in [doi_col, site_col, age_col, depth_col, material_col] if c]
repeat_data = df[df["_target"].notna()].copy()

repeat_df = repeat_data.groupby(repeat_keys, dropna=False).agg(
    observations=("_target", "size"),
    distinct_target_values=("_target", "nunique"),
    target_mean=("_target", "mean"),
    target_std=("_target", "std"),
    sample_names=(sample_col, lambda x: " | ".join(x.dropna().astype(str).unique()[:5])) if sample_col else ("_target", "size")
).reset_index()

repeat_df = repeat_df[repeat_df["observations"] > 1].sort_values(
    ["observations", "distinct_target_values"], ascending=False
)
repeat_df.to_csv(os.path.join(OUT, "40_repeated_position_candidates.csv"), index=False)

print("\nREPEATED-POSITION CANDIDATES:", len(repeat_df))
print("OBSERVATIONS IN CANDIDATE GROUPS:", repeat_df["observations"].sum() if len(repeat_df) else 0)
print("\nLARGEST CANDIDATE GROUPS")
print(repeat_df.head(15).to_string(index=False))

print("\nAUDIT FILES SAVED IN:", os.path.abspath(OUT))
