from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(".")
RAW = ROOT / "01_Data/raw/DSMSNI_v0.3.csv"
MAPPING = ROOT / "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv"
PREDICTIONS = ROOT / "06_Validation/random_vs_publication_heldout_predictions.csv"
OUT = ROOT / "06_Validation"
AUDIT_OUT = ROOT / "02_Data_Audit/provenance_analysis"

def find_col(df, fragment):
    matches = [c for c in df.columns if fragment.lower() in str(c).lower()]
    if not matches:
        raise ValueError(f"Cannot find column containing: {fragment}")
    return matches[0]

print("=" * 72)
print("EXPERIMENT 60: CHRONOLOGY RESOLUTION AND CROSS-PUBLICATION GENERALISATION")
print("=" * 72)

raw = pd.read_csv(RAW, encoding="gbk", low_memory=False)
mapping = pd.read_csv(MAPPING, low_memory=False)
pred = pd.read_csv(PREDICTIONS, low_memory=False)

target = find_col(raw, "15Nbulk")
age_col = find_col(raw, "Age")
depth_col = find_col(raw, "RelativeDepth")
doi_col = find_col(raw, "DOI")
site_col = find_col(raw, "SiteName")

for frame in [raw, mapping, pred]:
    frame["SampleID"] = frame["SampleID"].astype(str).str.strip()

if raw["SampleID"].duplicated().any():
    raise ValueError("Raw SampleID values are not unique.")

if mapping["SampleID"].duplicated().any():
    raise ValueError("Provenance mapping has duplicate SampleID values.")

required_mapping = ["SampleID", "provisional_source_group_id"]
if not set(required_mapping).issubset(mapping.columns):
    raise ValueError("Required provenance mapping fields are missing.")

# ------------------------------------------------------------
# A. Build chronology-resolution categories using ALL raw rows.
# ------------------------------------------------------------

raw["_doi_key"] = raw[doi_col].astype("string").str.strip()
raw["_site_key"] = raw[site_col].astype("string").str.strip()
raw["_age_numeric"] = pd.to_numeric(raw[age_col], errors="coerce")
raw["_depth_numeric"] = pd.to_numeric(raw[depth_col], errors="coerce")
raw["_target_numeric"] = pd.to_numeric(raw[target], errors="coerce")

chronology_records = []

for (doi, site), g in raw.groupby(
    ["_doi_key", "_site_key"], dropna=False, sort=False
):
    ages = g["_age_numeric"]
    depths = g["_depth_numeric"]
    paired = g[["_age_numeric", "_depth_numeric"]].dropna()

    n = len(g)
    age_n = ages.nunique()
    depth_n = depths.nunique()
    rho = np.nan

    if (
        len(paired) >= 5
        and paired["_age_numeric"].nunique() > 1
        and paired["_depth_numeric"].nunique() > 1
    ):
        rho = spearmanr(
            paired["_depth_numeric"], paired["_age_numeric"]
        ).statistic

    if n >= 5 and age_n == 1 and depth_n >= 3:
        status = "constant_age_multiple_depths_candidate"
    elif (
        n >= 5 and age_n >= 3 and depth_n >= 3
        and pd.notna(rho) and abs(rho) >= 0.80
    ):
        status = "monotonic_age_depth_candidate"
    elif n >= 5 and age_n >= 3 and depth_n >= 3:
        status = "variable_age_depth_unresolved"
    else:
        status = "insufficient_information"

    chronology_records.append({
        "_doi_key": doi,
        "_site_key": site,
        "all_database_rows": n,
        "numeric_target_rows": int(g["_target_numeric"].notna().sum()),
        "unique_ages": age_n,
        "unique_depths": depth_n,
        "paired_age_depth_rows": len(paired),
        "depth_age_spearman": rho,
        "resolution_proxy": status
    })

chronology = pd.DataFrame(chronology_records)
chronology["source_site_group_id"] = np.arange(len(chronology))

print("\nCHRONOLOGY CLASSIFICATION")
print("Source-site groups:", len(chronology))
print("Rows used to classify:", len(raw))
print("\n", chronology["resolution_proxy"].value_counts(dropna=False).to_string())

chronology_path = AUDIT_OUT / "60_chronology_resolution_source_site_register.csv"
chronology.to_csv(chronology_path, index=False)

# ------------------------------------------------------------
# B. Join sample-level predictions to chronology and provenance.
# ------------------------------------------------------------

mapping_small = mapping[
    ["SampleID", "provisional_source_group_id"]
].copy()

metadata = raw[
    ["SampleID", "_doi_key", "_site_key", "_target_numeric"]
].merge(
    mapping_small,
    on="SampleID",
    how="left",
    validate="one_to_one"
)

metadata = metadata.merge(
    chronology[
        ["_doi_key", "_site_key", "source_site_group_id", "resolution_proxy"]
    ],
    on=["_doi_key", "_site_key"],
    how="left",
    validate="many_to_one"
)

if metadata["resolution_proxy"].isna().any():
    missing = metadata["resolution_proxy"].isna().sum()
    raise ValueError(f"{missing} raw records did not receive chronology categories.")


pred = pred[
    pred["validation"].isin(["Random", "Publication_held_out"])
].copy()

pred = pred.merge(
    metadata[
        [
            "SampleID", "provisional_source_group_id",
            "source_site_group_id", "resolution_proxy",
            "_target_numeric"
        ]
    ],
    on="SampleID",
    how="left",
    validate="many_to_one"
)

if pred["resolution_proxy"].isna().any():
    raise ValueError("Some prediction records lack chronology categories.")

if pred["provisional_source_group_id"].isna().any():
    raise ValueError("A machine-learning prediction lacks a publication group.")

if pred["provisional_source_group_id"].isna().any():
    raise ValueError("Some prediction records lack publication groups.")

if not np.allclose(
    pred["actual"].to_numpy(),
    pred["_target_numeric"].to_numpy(),
    equal_nan=True
):
    raise ValueError("Prediction targets do not match the raw dataset.")

if pred.duplicated(["SampleID", "validation", "model"]).any():
    raise ValueError("Duplicate sample predictions detected.")

# Each publication must remain in one test fold in publication-held-out CV.
heldout = pred[pred["validation"] == "Publication_held_out"]

for model, g in heldout.groupby("model"):
    fold_counts = g.groupby("provisional_source_group_id")["fold"].nunique()
    if (fold_counts > 1).any():
        raise ValueError(
            f"Publication group crosses folds for model {model}."
        )

# ------------------------------------------------------------
# C. Performance by chronology category.
# ------------------------------------------------------------

pred["absolute_error"] = (pred["actual"] - pred["prediction"]).abs()
pred["squared_error"] = (pred["actual"] - pred["prediction"]) ** 2

def calculate_metrics(g):
    actual = g["actual"].to_numpy()
    predicted = g["prediction"].to_numpy()

    r2 = np.nan
    if len(g) >= 2 and np.unique(actual).size > 1:
        r2 = r2_score(actual, predicted)

    publication_mae = (
        g.groupby("provisional_source_group_id")["absolute_error"].mean()
    )

    return {
        "observations": len(g),
        "source_groups": g["provisional_source_group_id"].nunique(),
        "source_site_groups": g["source_site_group_id"].nunique(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2,
        "macro_publication_MAE": publication_mae.mean(),
        "median_publication_MAE": publication_mae.median()
    }

summary_rows = []

for (validation, model, category), g in pred.groupby(
    ["validation", "model", "resolution_proxy"]
):
    row = {
        "validation": validation,
        "model": model,
        "resolution_proxy": category
    }
    row.update(calculate_metrics(g))
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

# ------------------------------------------------------------
# D. Fold-level performance and category coverage.
# ------------------------------------------------------------

fold_rows = []

for (validation, model, fold, category), g in pred.groupby(
    ["validation", "model", "fold", "resolution_proxy"]
):
    row = {
        "validation": validation,
        "model": model,
        "fold": fold,
        "resolution_proxy": category
    }
    row.update(calculate_metrics(g))
    fold_rows.append(row)

fold_summary = pd.DataFrame(fold_rows)

coverage = (
    pred[
        (pred["validation"] == "Publication_held_out")
        & (pred["model"] == "Full_context")
    ]
    .groupby("resolution_proxy")
    .agg(
        observations=("SampleID", "size"),
        publications=("provisional_source_group_id", "nunique"),
        source_site_groups=("source_site_group_id", "nunique"),
        folds=("fold", "nunique")
    )
    .reset_index()
)

# ------------------------------------------------------------
# E. Save all outputs.
# ------------------------------------------------------------

summary_path = OUT / "60_chronology_resolution_performance.csv"
fold_path = OUT / "60_chronology_resolution_fold_metrics.csv"
coverage_path = OUT / "60_chronology_resolution_coverage.csv"

summary.to_csv(summary_path, index=False)
fold_summary.to_csv(fold_path, index=False)
coverage.to_csv(coverage_path, index=False)

print("\nPUBLICATION-HELD-OUT COVERAGE")
print(coverage.to_string(index=False))

print("\nPERFORMANCE BY CHRONOLOGY CATEGORY")
print(
    summary.sort_values(
        ["validation", "model", "resolution_proxy"]
    ).round(4).to_string(index=False)
)

print("\nFOLD-LEVEL PERFORMANCE")
print(fold_summary.round(4).to_string(index=False))

print("\nFILES SAVED")
for path in [
    chronology_path, summary_path, fold_path, coverage_path
]:
    print(path)

print("\nVALIDATION CHECKS PASSED")
print("Prediction observations:", len(pred))
print("Publication-held-out observations:", len(heldout))
print("Raw dataset modified: False")
print("Existing experiment outputs overwritten: False")
print("\nEXPERIMENT 60 COMPLETED")
