from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("01_Data/raw/DSMSNI_v0.3.csv")
MAPPING = Path("02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv")
PRED = Path("06_Validation/random_vs_publication_heldout_predictions.csv")
CHRON = Path("02_Data_Audit/provenance_analysis/60_chronology_resolution_source_site_register.csv")
OUT = Path("06_Validation")

def find_col(df, fragment):
    return next(c for c in df.columns if fragment.lower() in str(c).lower())

print("=" * 72)
print("EXPERIMENT 61: PUBLICATION-CLUSTER UNCERTAINTY ANALYSIS")
print("=" * 72)

raw = pd.read_csv(RAW, encoding="gbk", low_memory=False)
mapping = pd.read_csv(MAPPING, low_memory=False)
pred = pd.read_csv(PRED, low_memory=False)
chron = pd.read_csv(CHRON, low_memory=False)

for frame in [raw, mapping, pred]:
    frame["SampleID"] = frame["SampleID"].astype(str).str.strip()

target = find_col(raw, "15Nbulk")
doi = find_col(raw, "DOI")
site = find_col(raw, "SiteName")

meta = raw[["SampleID", doi, site, target]].copy()
meta[target] = pd.to_numeric(meta[target], errors="coerce")
meta["_doi_key"] = meta[doi].astype("string").str.strip()
meta["_site_key"] = meta[site].astype("string").str.strip()

mapping_small = mapping[
    ["SampleID", "provisional_source_group_id"]
].copy()

meta = meta.merge(
    mapping_small, on="SampleID", how="left", validate="one_to_one"
)


chron_small = chron[
    ["_doi_key", "_site_key", "resolution_proxy"]
].copy()

meta = meta.merge(
    chron_small,
    on=["_doi_key", "_site_key"],
    how="left",
    validate="many_to_one"
)

if meta["resolution_proxy"].isna().any():
    raise ValueError("Some observations lack chronology categories.")

p = pred[
    (pred["validation"] == "Publication_held_out")
    & (pred["model"] == "Full_context")
].copy()

p = p.merge(
    meta[[
        "SampleID", "provisional_source_group_id",
        "resolution_proxy", target
    ]],
    on="SampleID",
    how="left",
    validate="one_to_one"
)

if len(p) != 24012:
    raise ValueError(f"Expected 24012 predictions; found {len(p)}.")

if p["provisional_source_group_id"].isna().any():
    raise ValueError("Missing publication groups in predictions.")

if p["resolution_proxy"].isna().any():
    raise ValueError("Missing chronology categories in predictions.")

if not np.allclose(p["actual"], p[target], equal_nan=True):
    raise ValueError("Prediction targets do not match raw dataset.")

p["absolute_error"] = (p["actual"] - p["prediction"]).abs()
p["squared_error"] = (p["actual"] - p["prediction"]) ** 2

# One row per publication within each chronology category.
publication = (
    p.groupby(
        ["resolution_proxy", "provisional_source_group_id"],
        dropna=False
    )
    .agg(
        observations=("SampleID", "size"),
        publication_MAE=("absolute_error", "mean"),
        publication_RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
        isotope_mean=("actual", "mean"),
        isotope_SD=("actual", "std"),
        isotope_min=("actual", "min"),
        isotope_max=("actual", "max"),
        folds=("fold", "nunique")
    )
    .reset_index()
)

if (publication["folds"] != 1).any():
    raise ValueError("A publication occurs in more than one held-out fold.")

rng = np.random.default_rng(61061)
B = 10000
summary_rows = []

for category, g in publication.groupby("resolution_proxy"):
    errors = g["publication_MAE"].to_numpy()
    n_groups = len(errors)

    if n_groups < 2:
        continue

    boot = np.empty(B)

    for b in range(B):
        sample = rng.choice(errors, size=n_groups, replace=True)
        boot[b] = sample.mean()

    obs = p[p["resolution_proxy"] == category]

    summary_rows.append({
        "resolution_proxy": category,
        "publications": n_groups,
        "observations": len(obs),
        "observation_weighted_MAE": obs["absolute_error"].mean(),
        "macro_publication_MAE": errors.mean(),
        "macro_MAE_95CI_lower": np.quantile(boot, 0.025),
        "macro_MAE_95CI_upper": np.quantile(boot, 0.975),
        "median_publication_MAE": np.median(errors),
        "publication_MAE_q25": np.quantile(errors, 0.25),
        "publication_MAE_q75": np.quantile(errors, 0.75),
        "mean_publication_isotope_mean": g["isotope_mean"].mean(),
        "median_publication_isotope_SD": g["isotope_SD"].median(),
        "pooled_isotope_SD": obs["actual"].std(),
        "isotope_min": obs["actual"].min(),
        "isotope_max": obs["actual"].max()
    })

summary = pd.DataFrame(summary_rows)

folds = (
    p.groupby(["resolution_proxy", "fold"])
    .agg(
        observations=("SampleID", "size"),
        publications=("provisional_source_group_id", "nunique"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", lambda x: np.sqrt(x.mean()))
    )
    .reset_index()
)

# Identify the publications contributing most to each category's error.
publication = publication.sort_values(
    ["resolution_proxy", "publication_MAE"],
    ascending=[True, False]
)

summary.to_csv(
    OUT / "61_chronology_publication_bootstrap_summary.csv",
    index=False
)
publication.to_csv(
    OUT / "61_chronology_publication_level_metrics.csv",
    index=False
)
folds.to_csv(
    OUT / "61_chronology_publication_fold_metrics.csv",
    index=False
)

print("\nPUBLICATION-LEVEL BOOTSTRAP SUMMARY")
print(summary.round(4).to_string(index=False))

print("\nFOLD-LEVEL RESULTS")
print(folds.round(4).to_string(index=False))

print("\nFIVE HIGHEST-ERROR PUBLICATIONS PER CATEGORY")
for category, g in publication.groupby("resolution_proxy"):
    print("\n", category)
    print(
        g.head(5)[
            ["provisional_source_group_id", "observations", "publication_MAE"]
        ].round(4).to_string(index=False)
    )

print("\nFILES SAVED")
print("06_Validation/61_chronology_publication_bootstrap_summary.csv")
print("06_Validation/61_chronology_publication_level_metrics.csv")
print("06_Validation/61_chronology_publication_fold_metrics.csv")
print("\nBootstrap resampling unit: publication group")
print("Bootstrap replicates:", B)
print("Random seed: 61061")
print("Raw dataset modified: False")
print("\nEXPERIMENT 61 COMPLETED")
