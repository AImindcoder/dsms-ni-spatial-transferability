from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

# ============================================================
# EXPERIMENT 277: REPRODUCIBLE PUBLICATION-HELD-OUT VALIDATION
# ============================================================

SEED_MODEL = 253
SEED_PARTITION = 261
N_SPLITS = 5

ROOT = Path(".")
INPUT = ROOT / "01_Data/raw/DSMSNI_v0.3.csv"
OUT = ROOT / "01_Data/interim"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(INPUT, encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in str(c))
TN_COL = "TN\n%"
TOC_COL = "TOC\n%"
required = ["SampleID", "DOI", "Age", TN_COL, TOC_COL, target]

missing = [c for c in required if c not in raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = raw[required].copy()

for col in ["Age", TN_COL, TOC_COL, target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["DOI"] = df["DOI"].astype("string").str.strip()
df = df.dropna(subset=required).copy()

# Keep publications with at least 10 complete observations.
doi_counts = df["DOI"].value_counts()
eligible_dois = doi_counts[doi_counts >= 10].index
df = df[df["DOI"].isin(eligible_dois)].copy()
df = df.reset_index(drop=True)
df["row_id"] = np.arange(len(df))

if df["SampleID"].astype(str).duplicated().any():
    raise ValueError("Duplicate SampleID detected.")

print("EXPERIMENT 277: REPRODUCIBLE VALIDATION")
print("Target:", target)
print("Complete observations:", len(df))
print("Eligible publications:", df["DOI"].nunique())
print("Minimum observations per publication:", df["DOI"].value_counts().min())
print("Maximum observations per publication:", df["DOI"].value_counts().max())

# Fixed model configuration from the earlier reported experiments.
def make_model():
    return HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED_MODEL
    )

X_age = df[["Age"]]
X_full = df[["Age", TN_COL, TOC_COL]]
y = df[target].to_numpy()
groups = df["DOI"].to_numpy()

# Entire publications are held out from model training.
cv = GroupKFold(n_splits=N_SPLITS)
prediction_rows = []
fold_rows = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(df, y, groups), 1
):
    age_model = make_model()
    full_model = make_model()

    age_model.fit(X_age.iloc[train_idx], y[train_idx])
    full_model.fit(X_full.iloc[train_idx], y[train_idx])

    age_pred = age_model.predict(X_age.iloc[test_idx])
    full_pred = full_model.predict(X_full.iloc[test_idx])

    test = df.iloc[test_idx].copy()
    actual = y[test_idx]

    fold_rows.append({
        "Fold": fold,
        "Training_N": len(train_idx),
        "Testing_N": len(test_idx),
        "Training_publications": df.iloc[train_idx]["DOI"].nunique(),
        "Testing_publications": test["DOI"].nunique(),
        "Age_only_MAE": mean_absolute_error(actual, age_pred),
        "Age_only_RMSE": np.sqrt(mean_squared_error(actual, age_pred)),
        "Age_only_R2": r2_score(actual, age_pred),
        "Geochemical_MAE": mean_absolute_error(actual, full_pred),
        "Geochemical_RMSE": np.sqrt(mean_squared_error(actual, full_pred)),
        "Geochemical_R2": r2_score(actual, full_pred)
    })

    for j, (_, row) in enumerate(test.iterrows()):
        prediction_rows.append({
            "row_id": int(row["row_id"]),
            "SampleID": str(row["SampleID"]),
            "DOI": row["DOI"],
            "Fold": fold,
            "Actual": actual[j],
            "Age_only_prediction": age_pred[j],
            "Geochemical_prediction": full_pred[j]
        })

    print(
        f"Fold {fold}: train={len(train_idx)}, "
        f"test={len(test_idx)}, "
        f"held-out publications={test['DOI'].nunique()}"
    )

pred = pd.DataFrame(prediction_rows)
fold_metrics = pd.DataFrame(fold_rows)

if pred["row_id"].duplicated().any():
    raise ValueError("A record received multiple out-of-fold predictions.")

if len(pred) != len(df):
    raise ValueError("Not all eligible observations received predictions.")

# Split each held-out publication into separate calibration/evaluation
# subsets. The same publication-level random seed is used consistently.
partition_map = {}
publication_rows = []

for doi, g in pred.groupby("DOI", sort=True):
    if g["Fold"].nunique() != 1:
        raise ValueError(f"Publication crosses validation folds: {doi}")

    positions = np.arange(len(g))
    calibration_pos, evaluation_pos = train_test_split(
        positions,
        test_size=0.5,
        random_state=SEED_PARTITION
    )

    calibration = g.iloc[calibration_pos]
    evaluation = g.iloc[evaluation_pos]

    for rid in calibration["row_id"]:
        partition_map[int(rid)] = "Calibration"

    for rid in evaluation["row_id"]:
        partition_map[int(rid)] = "Evaluation"

    calibration_mae = mean_absolute_error(
        calibration["Actual"],
        calibration["Age_only_prediction"]
    )

    age_eval_mae = mean_absolute_error(
        evaluation["Actual"],
        evaluation["Age_only_prediction"]
    )

    full_eval_mae = mean_absolute_error(
        evaluation["Actual"],
        evaluation["Geochemical_prediction"]
    )

    publication_rows.append({
        "DOI": doi,
        "Fold": int(g["Fold"].iloc[0]),
        "Calibration_N": len(calibration),
        "Evaluation_N": len(evaluation),
        "Age_only_calibration_MAE": calibration_mae,
        "Age_only_evaluation_MAE": age_eval_mae,
        "Geochemical_evaluation_MAE": full_eval_mae,
        "Geochemical_evaluation_improvement": age_eval_mae - full_eval_mae
    })

summary = pd.DataFrame(publication_rows)

summary["Difficulty_Group"] = pd.qcut(
    summary["Age_only_calibration_MAE"].rank(method="first"),
    q=4,
    labels=["Q1: Easier", "Q2", "Q3", "Q4: Harder"]
)

pred["Partition"] = pred["row_id"].map(partition_map)

if pred["Partition"].isna().any():
    raise ValueError("Some predictions were not assigned to a partition.")

# Save complete outputs.
pred.to_csv(OUT / "277_publication_heldout_predictions.csv", index=False)
fold_metrics.to_csv(OUT / "277_fold_metrics.csv", index=False)
summary.to_csv(OUT / "277_publication_evaluation_metrics.csv", index=False)

config = {
    "experiment": 277,
    "input": str(INPUT),
    "target": target,
    "features_age_only": ["Age"],
    "features_geochemical": ["Age", TN_COL, TOC_COL],
    "group_column": "DOI",
    "minimum_publication_records": 10,
    "validation": "GroupKFold",
    "folds": N_SPLITS,
    "model": "HistGradientBoostingRegressor",
    "max_iter": 150,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "model_random_state": SEED_MODEL,
    "within_publication_partition": "train_test_split(test_size=0.5)",
    "partition_random_state": SEED_PARTITION,
    "note": "Reproducibility rerun; not claimed to reproduce the missing original script exactly."
}

with open(OUT / "277_experiment_configuration.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# Summary statistics.
evaluation = pred[pred["Partition"] == "Evaluation"]

print("\nPUBLICATION-HELD-OUT POOLED EVALUATION")
print("Evaluation observations:", len(evaluation))
print(
    "Age-only MAE:",
    round(mean_absolute_error(
        evaluation["Actual"], evaluation["Age_only_prediction"]
    ), 4)
)
print(
    "Geochemical MAE:",
    round(mean_absolute_error(
        evaluation["Actual"], evaluation["Geochemical_prediction"]
    ), 4)
)

print("\nDIFFICULTY-GROUP RESULTS")
group_results = summary.groupby(
    "Difficulty_Group", observed=False
).agg(
    Publications=("DOI", "count"),
    Mean_calibration_MAE=("Age_only_calibration_MAE", "mean"),
    Mean_age_evaluation_MAE=("Age_only_evaluation_MAE", "mean"),
    Mean_geochemical_evaluation_MAE=("Geochemical_evaluation_MAE", "mean"),
    Mean_improvement=("Geochemical_evaluation_improvement", "mean"),
    Median_improvement=("Geochemical_evaluation_improvement", "median")
)
print(group_results.round(4).to_string())

rho, rho_p = spearmanr(
    summary["Age_only_calibration_MAE"],
    summary["Geochemical_evaluation_improvement"]
)
print(f"\nSpearman difficulty vs improvement: rho={rho:.4f}, p={rho_p:.6g}")

group_arrays = [
    g["Geochemical_evaluation_improvement"].to_numpy()
    for _, g in summary.groupby("Difficulty_Group", observed=True)
]
if len(group_arrays) == 4:
    H, p = kruskal(*group_arrays)
    print(f"Kruskal-Wallis: H={H:.4f}, p={p:.6g}")

print("\nOUTPUT FILES")
print("277_publication_heldout_predictions.csv")
print("277_fold_metrics.csv")
print("277_publication_evaluation_metrics.csv")
print("277_experiment_configuration.json")
print("\nEXPERIMENT 277 COMPLETE")
