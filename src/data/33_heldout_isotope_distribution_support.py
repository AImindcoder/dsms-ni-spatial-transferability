from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)
pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")

target = next(c for c in raw.columns if "15Nbulk" in c)
group_col = "provisional_source_group_id"
features = ["Age", "Latitude", "Longitude", "Material", "Setting", "LithType", "Lithology"]

df = raw[["SampleID", target] + features].merge(
    mapping[["SampleID", group_col]], on="SampleID", how="inner"
)
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target, group_col] + features).copy()

oof = pred[
    (pred.validation == "Publication_heldout") &
    (pred.model == "Full_context")
].copy()

# Accept the exact validation label used in the saved prediction file.
if oof.empty:
    oof = pred[
        (pred.validation == "Publication_held_out") &
        (pred.model == "Full_context")
    ].copy()

df["SampleID"] = df["SampleID"].astype(str)
oof["SampleID"] = oof["SampleID"].astype(str)

if df["SampleID"].duplicated().any() or oof["SampleID"].duplicated().any():
    raise ValueError("Duplicate SampleID detected; matching is not one-to-one.")

df = df.merge(
    oof[["SampleID", "fold", "actual", "prediction"]],
    on="SampleID", how="inner", validate="one_to_one"
)

if len(df) != len(oof):
    raise ValueError(
        f"Sample mismatch: matched={len(df)}, predictions={len(oof)}"
    )

if not np.allclose(df[target], df["actual"]):
    raise ValueError("Target values do not match prediction records.")

y = df[target].to_numpy()
groups = df[group_col].to_numpy()
folds = GroupKFold(n_splits=5)
rows = []

for fold, (train_idx, test_idx) in enumerate(folds.split(df, y, groups), 1):
    train_y = y[train_idx]
    train_mean = train_y.mean()
    q01, q05, q95, q99 = np.quantile(train_y, [0.01, 0.05, 0.95, 0.99])

    test = df.iloc[test_idx].copy()

    for group, g in test.groupby(group_col):
        actual = g[target].to_numpy()
        prediction = g["prediction"].to_numpy()
        group_mean = actual.mean()

        rows.append({
            "fold": fold,
            "source_group": group,
            "observations": len(g),
            "training_target_mean": train_mean,
            "training_q01": q01,
            "training_q05": q05,
            "training_q95": q95,
            "training_q99": q99,
            "group_target_mean": group_mean,
            "group_target_std": np.std(actual, ddof=1) if len(actual) > 1 else 0,
            "group_target_min": actual.min(),
            "group_target_max": actual.max(),
            "group_mean_distance_from_train": abs(group_mean - train_mean),
            "fraction_outside_train_q01_q99": np.mean((actual < q01) | (actual > q99)),
            "fraction_outside_train_q05_q95": np.mean((actual < q05) | (actual > q95)),
            "MAE": np.mean(np.abs(actual - prediction)),
            "mean_bias": np.mean(prediction - actual)
        })

result = pd.DataFrame(rows)

print("EXPERIMENT 33: HELD-OUT ISOTOPE DISTRIBUTION SUPPORT")
print("Matched observations:", len(df))
print("Held-out source groups:", len(result))

print("\nCORRELATION ANALYSIS")
for label, x, z in [
    ("Mean target distance vs MAE", "group_mean_distance_from_train", "MAE"),
    ("Outside 1-99 percentile vs MAE", "fraction_outside_train_q01_q99", "MAE"),
    ("Outside 5-95 percentile vs MAE", "fraction_outside_train_q05_q95", "MAE"),
    ("Mean target distance vs absolute bias",
     "group_mean_distance_from_train", "mean_bias")
]:
    rho, p = spearmanr(result[x], result[z])
    print(f"{label}: rho={rho:.4f}, p={p:.6g}")

print("\nGROUP-LEVEL SUPPORT SUMMARY")
print(result[[
    "group_mean_distance_from_train",
    "fraction_outside_train_q01_q99",
    "fraction_outside_train_q05_q95",
    "MAE", "mean_bias"
]].describe().round(4).to_string())

print("\nTEN LARGEST ERRORS")
print(result.sort_values("MAE", ascending=False).head(10)[[
    "fold", "source_group", "observations",
    "training_target_mean", "group_target_mean",
    "group_mean_distance_from_train",
    "fraction_outside_train_q01_q99",
    "MAE", "mean_bias"
]].round(3).to_string(index=False))

result.to_csv(
    "06_Validation/heldout_isotope_distribution_support.csv",
    index=False
)
print("\nSaved heldout_isotope_distribution_support.csv")
