from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

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

if len(df) != len(oof) or not np.allclose(df[target], df["actual"]):
    raise ValueError("Dataset and saved predictions do not match.")

if (df.groupby(group_col)["fold"].nunique() != 1).any():
    raise ValueError("A source group appears in multiple folds.")

df["ML_prediction"] = df["prediction"]
df["Mean_baseline"] = np.nan
df["Median_baseline"] = np.nan

for fold in sorted(df["fold"].unique()):
    train = df[df["fold"] != fold]
    test_mask = df["fold"] == fold

    df.loc[test_mask, "Mean_baseline"] = train[target].mean()
    df.loc[test_mask, "Median_baseline"] = train[target].median()

def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "R2": r2_score(y, p)
    }

rows = []
for model in ["ML_prediction", "Mean_baseline", "Median_baseline"]:
    m = metrics(df["actual"], df[model])
    rows.append({
        "model": model,
        "observations": len(df),
        **m
    })

summary = pd.DataFrame(rows)

print("EXPERIMENT 34: ML VS STATISTICAL BASELINES")
print("\nPOOLED OUT-OF-FOLD RESULTS")
print(summary.round(4).to_string(index=False))

print("\nFOLD-LEVEL MAE")
fold_rows = []
for fold, g in df.groupby("fold"):
    for model in ["ML_prediction", "Mean_baseline", "Median_baseline"]:
        fold_rows.append({
            "fold": fold,
            "model": model,
            "observations": len(g),
            "MAE": mean_absolute_error(g["actual"], g[model]),
            "RMSE": np.sqrt(mean_squared_error(g["actual"], g[model]))
        })

fold_results = pd.DataFrame(fold_rows)
print(fold_results.round(4).to_string(index=False))

print("\nIMPROVEMENT OVER BASELINES")
ml_mae = summary.loc[summary.model == "ML_prediction", "MAE"].iloc[0]
for baseline in ["Mean_baseline", "Median_baseline"]:
    base_mae = summary.loc[summary.model == baseline, "MAE"].iloc[0]
    improvement = (base_mae - ml_mae) / base_mae * 100
    print(f"Against {baseline}: {improvement:.2f}% MAE improvement")

out = Path("06_Validation")
summary.to_csv(out / "publication_heldout_baseline_comparison.csv", index=False)
fold_results.to_csv(out / "publication_heldout_baseline_fold_metrics.csv", index=False)

print("\nSaved baseline comparison files.")
