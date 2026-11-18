from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)

group_col = "provisional_source_group_id"
groups = mapping[["SampleID", group_col]].drop_duplicates("SampleID")
df = pred.merge(groups, on="SampleID", how="left")
df = df.dropna(subset=[group_col]).copy()
df["error"] = df["prediction"] - df["actual"]
df["absolute_error"] = df["error"].abs()
df["squared_error"] = df["error"] ** 2

group_rows = []
for (validation, model, group), g in df.groupby(
    ["validation", "model", group_col]
):
    group_rows.append({
        "validation": validation,
        "model": model,
        "source_group": group,
        "observations": len(g),
        "MAE": g["absolute_error"].mean(),
        "RMSE": np.sqrt(g["squared_error"].mean()),
        "mean_bias": g["error"].mean(),
        "actual_mean": g["actual"].mean(),
        "actual_std": g["actual"].std(),
        "prediction_mean": g["prediction"].mean()
    })

group_metrics = pd.DataFrame(group_rows)

summary_rows = []
for (validation, model), g in group_metrics.groupby(["validation", "model"]):
    rho, p = spearmanr(g["observations"], g["MAE"])
    summary_rows.append({
        "validation": validation,
        "model": model,
        "source_groups": len(g),
        "total_observations": g["observations"].sum(),
        "observation_weighted_MAE": mean_absolute_error(
            df.loc[
                (df.validation == validation) & (df.model == model), "actual"
            ],
            df.loc[
                (df.validation == validation) & (df.model == model), "prediction"
            ]
        ),
        "macro_group_MAE": g["MAE"].mean(),
        "median_group_MAE": g["MAE"].median(),
        "group_MAE_q25": g["MAE"].quantile(0.25),
        "group_MAE_q75": g["MAE"].quantile(0.75),
        "macro_group_RMSE": g["RMSE"].mean(),
        "mean_group_bias": g["mean_bias"].mean(),
        "group_size_MAE_spearman": rho,
        "group_size_MAE_pvalue": p
    })

summary = pd.DataFrame(summary_rows)

out = Path("06_Validation")
group_metrics.to_csv(out / "publication_level_error_metrics.csv", index=False)
summary.to_csv(out / "publication_level_error_summary.csv", index=False)

print("EXPERIMENT 32: PUBLICATION-LEVEL ERROR ANALYSIS")
print("\nSUMMARY")
print(summary.round(4).to_string(index=False))

for validation in ["Random", "Publication_held_out"]:
    for model in ["Full_context", "Without_Setting"]:
        subset = group_metrics[
            (group_metrics.validation == validation) &
            (group_metrics.model == model)
        ].sort_values("MAE", ascending=False)

        print(f"\nWORST 10 SOURCE GROUPS: {validation} | {model}")
        print(subset.head(10)[
            ["source_group", "observations", "MAE", "RMSE",
             "mean_bias", "actual_mean", "prediction_mean"]
        ].round(3).to_string(index=False))

print("\nSaved publication-level error metrics and summary.")
