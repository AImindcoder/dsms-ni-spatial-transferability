from pathlib import Path
import pandas as pd
import numpy as np

raw = Path("01_Data/raw/DSMSNI_v0.3.csv")
outdir = Path("01_Data/interim")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw, encoding="gbk", low_memory=False)
df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

required = ["DOI", "SiteName", "Age", "RelativeDepth"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Required columns missing: {missing}")

for col in ["Age", "RelativeDepth", "δ15Nbulk ‰"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Exclude placeholder or missing DOI values for publication-level grouping.
doi = df["DOI"].astype("string").str.strip()
valid_doi = doi.notna() & ~doi.str.lower().isin(
    ["", "not available", "nan", "none", "n/a"]
)
work = df.loc[valid_doi & df["SiteName"].notna()].copy()
work["DOI_clean"] = doi.loc[work.index].str.lower()
work["Site_clean"] = work["SiteName"].astype(str).str.strip()

group_cols = ["DOI_clean", "Site_clean"]
rows = []

for (doi_value, site), g in work.groupby(group_cols, dropna=False):
    ages = g["Age"].dropna()
    depths = g["RelativeDepth"].dropna()

    if len(g) < 2 or ages.nunique() != 1 or depths.nunique() < 2:
        continue

    item = {
        "DOI": doi_value,
        "SiteName": site,
        "n_records": len(g),
        "n_unique_depths": depths.nunique(),
        "n_unique_ages": ages.nunique(),
        "constant_age_Ma": ages.iloc[0],
        "depth_min": depths.min(),
        "depth_max": depths.max(),
        "depth_span": depths.max() - depths.min(),
    }

    for col in ["FirstAuthor", "Reference", "Title", "Era", "Period", "Epoch", "Stage"]:
        if col in g.columns:
            vals = g[col].dropna().astype(str).unique()
            item[col + "_values"] = " | ".join(vals[:8])
            item[col + "_unique_count"] = len(vals)

    if "δ15Nbulk ‰" in g.columns:
        item["n_numeric_d15Nbulk"] = g["δ15Nbulk ‰"].notna().sum()

    rows.append(item)

result = pd.DataFrame(rows)

if result.empty:
    print("No groups met the constant-age / multiple-depth criteria.")
    raise SystemExit

result = result.sort_values(
    ["n_records", "depth_span"], ascending=[False, False]
).reset_index(drop=True)

path = outdir / "105_constant_age_multiple_depth_publication_site_candidates.csv"
result.to_csv(path, index=False, encoding="utf-8-sig")

print("=" * 85)
print("CONSTANT-AGE / MULTIPLE-DEPTH CANDIDATE AUDIT")
print("=" * 85)
print("Raw dataset rows:", len(df))
print("Usable DOI + site rows:", len(work))
print("Candidate publication-site groups:", len(result))
print("Total observations in candidates:", int(result["n_records"].sum()))
print("Groups with multiple geological stages:",
      int((result.get("Stage_unique_count", pd.Series(dtype=float)) > 1).sum())
      if "Stage_unique_count" in result else "Stage column unavailable")

print("\nTOP 30 CANDIDATES")
print("-" * 85)
display_cols = [
    c for c in [
        "DOI", "SiteName", "n_records", "n_unique_depths",
        "constant_age_Ma", "depth_span", "FirstAuthor_values",
        "Period_values", "Epoch_values", "Stage_values"
    ] if c in result.columns
]
print(result[display_cols].head(30).to_string(index=False))

print("\nSaved complete candidate list:")
print(path)
