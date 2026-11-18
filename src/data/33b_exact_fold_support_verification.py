from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)
pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")

target = next(c for c in raw.columns if "15Nbulk" in c)
group_col = "provisional_source_group_id"
features = ["Age", "Latitude", "Longitude", "Material", "Setting", "LithType", "Lithology"]

oof = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"] == "Full_context")
].copy()

df = raw[["SampleID", target] + features].merge(
    mapping[["SampleID", group_col]], on="SampleID", how="inner"
)

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target, group_col] + features).copy()

df["SampleID"] = df["SampleID"].astype(str)
oof["SampleID"] = oof["SampleID"].astype(str)

df = df.merge(
    oof[["SampleID", "fold", "actual", "prediction"]],
    on="SampleID", how="inner", validate="one_to_one"
)

if len(df) != len(oof):
    raise ValueError("Observation mismatch between dataset and saved predictions.")

if not np.allclose(df[target], df["actual"]):
    raise ValueError("Actual target values do not match.")

group_fold_counts = df.groupby(group_col)["fold"].nunique()
if (group_fold_counts != 1).any():
    raise ValueError("A source group appears in multiple held-out folds.")

rows = []

for fold in sorted(df["fold"].unique()):
    train = df[df["fold"] != fold]
    test = df[df["fold"] == fold]

    train_y = train[target].to_numpy()
    q01, q05, q95, q99 = np.quantile(
        train_y, [0.01, 0.05, 0.95, 0.99]
    )

    for group, g in test.groupby(group_col):
        actual = g[target].to_numpy()
        prediction = g["prediction"].to_numpy()

        rows.append({
            "fold": fold,
            "source_group": group,
            "observations": len(g),
            "training_observations": len(train),
            "training_target_mean": train_y.mean(),
            "training_q01": q01,
            "training_q05": q05,
            "training_q95": q95,
            "training_q99": q99,
            "group_target_mean": actual.mean(),
            "group_target_min": actual.min(),
            "group_target_max": actual.max(),
            "group_mean_distance_from_train": abs(actual.mean() - train_y.mean()),
            "fraction_outside_train_q01_q99": np.mean(
                (actual < q01) | (actual > q99)
            ),
            "fraction_outside_train_q05_q95": np.mean(
                (actual < q05) | (actual > q95)
            ),
            "MAE": np.mean(np.abs(actual - prediction)),
            "mean_bias": np.mean(prediction - actual)
        })

result = pd.DataFrame(rows)

print("EXPERIMENT 33B: EXACT SAVED-FOLD VERIFICATION")
print("Observations:", len(df))
print("Source groups:", result["source_group"].nunique())
print("Fold assignments taken directly from saved predictions.")

print("\nCORRELATIONS")
for label, x, z in [
    ("Mean target distance vs MAE", "group_mean_distance_from_train", "MAE"),
    ("Outside 1-99 percentile vs MAE", "fraction_outside_train_q01_q99", "MAE"),
    ("Outside 5-95 percentile vs MAE", "fraction_outside_train_q05_q95", "MAE")
]:
    rho, p = spearmanr(result[x], result[z])
    print(f"{label}: rho={rho:.4f}, p={p:.6g}")

print("\nFOLD COUNTS")
print(result.groupby("fold").agg(
    source_groups=("source_group", "nunique"),
    observations=("observations", "sum")
).to_string())

print("\nTOP 10 ERRORS")
print(result.nlargest(10, "MAE")[[
    "fold", "source_group", "observations",
    "group_mean_distance_from_train",
    "fraction_outside_train_q01_q99",
    "fraction_outside_train_q05_q95",
    "MAE", "mean_bias"
]].round(3).to_string(index=False))

result.to_csv(
    "06_Validation/heldout_isotope_distribution_support_exact_folds.csv",
    index=False
)
print("\nSaved exact-fold results.")
