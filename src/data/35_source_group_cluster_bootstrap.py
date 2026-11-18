from pathlib import Path
import pandas as pd
import numpy as np

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

if len(df) != len(oof) or not np.allclose(df[target], df["actual"]):
    raise ValueError("Dataset and saved predictions do not match.")

if (df.groupby(group_col)["fold"].nunique() != 1).any():
    raise ValueError("Source groups cross validation folds.")

df["ML_abs_error"] = (df["actual"] - df["prediction"]).abs()
df["Mean_abs_error"] = np.nan
df["Median_abs_error"] = np.nan

for fold in sorted(df["fold"].unique()):
    train = df[df["fold"] != fold]
    test_mask = df["fold"] == fold
    mean_value = train[target].mean()
    median_value = train[target].median()

    df.loc[test_mask, "Mean_abs_error"] = (
        df.loc[test_mask, "actual"] - mean_value
    ).abs()
    df.loc[test_mask, "Median_abs_error"] = (
        df.loc[test_mask, "actual"] - median_value
    ).abs()

grouped = df.groupby(group_col).agg(
    observations=("actual", "size"),
    ML_sum=("ML_abs_error", "sum"),
    Mean_sum=("Mean_abs_error", "sum"),
    Median_sum=("Median_abs_error", "sum"),
    ML_mae=("ML_abs_error", "mean"),
    Mean_mae=("Mean_abs_error", "mean"),
    Median_mae=("Median_abs_error", "mean")
).reset_index()

n_groups = len(grouped)
rng = np.random.default_rng(42)
B = 5000
draws = []

for b in range(B):
    selected = rng.integers(0, n_groups, size=n_groups)
    sample = grouped.iloc[selected]

    n = sample["observations"].sum()
    ml = sample["ML_sum"].sum() / n
    mean = sample["Mean_sum"].sum() / n
    median = sample["Median_sum"].sum() / n

    draws.append({
        "bootstrap": b + 1,
        "ML_MAE": ml,
        "Mean_baseline_MAE": mean,
        "Median_baseline_MAE": median,
        "improvement_vs_mean": (mean - ml) / mean * 100,
        "improvement_vs_median": (median - ml) / median * 100,
        "macro_ML_MAE": sample["ML_mae"].mean(),
        "macro_Mean_MAE": sample["Mean_mae"].mean(),
        "macro_Median_MAE": sample["Median_mae"].mean()
    })

boot = pd.DataFrame(draws)

print("EXPERIMENT 35: SOURCE-GROUP CLUSTER BOOTSTRAP")
print("Source groups:", n_groups)
print("Bootstrap repetitions:", B)
print("Observations:", len(df))

print("\nOBSERVED PAIRED IMPROVEMENTS")
ml_mae = df["ML_abs_error"].mean()
mean_mae = df["Mean_abs_error"].mean()
median_mae = df["Median_abs_error"].mean()

print(f"ML MAE: {ml_mae:.4f}")
print(f"Mean baseline MAE: {mean_mae:.4f}")
print(f"Median baseline MAE: {median_mae:.4f}")
print(f"Improvement vs mean: {(mean_mae-ml_mae)/mean_mae*100:.2f}%")
print(f"Improvement vs median: {(median_mae-ml_mae)/median_mae*100:.2f}%")

print("\n95% CLUSTER BOOTSTRAP CONFIDENCE INTERVALS")
for col in ["improvement_vs_mean", "improvement_vs_median"]:
    lo, hi = boot[col].quantile([0.025, 0.975])
    print(f"{col}: {boot[col].mean():.2f}% "
          f"(95% CI: {lo:.2f}% to {hi:.2f}%)")

    print(f"Bootstrap fraction with improvement <= 0: "
          f"{(boot[col] <= 0).mean():.4f}")

print("\nMACRO-GROUP MAE DIFFERENCES")
for baseline in ["Mean", "Median"]:
    difference = boot[f"macro_{baseline}_MAE"] - boot["macro_ML_MAE"]
    lo, hi = difference.quantile([0.025, 0.975])
    print(f"ML vs {baseline}: {difference.mean():.4f} "
          f"(95% CI: {lo:.4f} to {hi:.4f})")

out = Path("06_Validation")
boot.to_csv(out / "source_group_cluster_bootstrap_draws.csv", index=False)
grouped.to_csv(out / "source_group_bootstrap_inputs.csv", index=False)

print("\nSaved bootstrap draws and group-level inputs.")
