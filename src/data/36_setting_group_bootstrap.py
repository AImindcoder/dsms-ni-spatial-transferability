from pathlib import Path
import pandas as pd
import numpy as np

pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)

required_models = ["Full_context", "Without_Setting"]
p = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"].isin(required_models))
].copy()

p["SampleID"] = p["SampleID"].astype(str)

wide = p.pivot(
    index="SampleID",
    columns="model",
    values=["actual", "prediction", "fold"]
)

wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()

for col in ["actual_Full_context", "actual_Without_Setting"]:
    if col not in wide.columns:
        raise ValueError(f"Missing expected prediction field: {col}")

if not np.allclose(
    wide["actual_Full_context"],
    wide["actual_Without_Setting"]
):
    raise ValueError("The models do not have identical actual targets.")

if not np.array_equal(
    wide["fold_Full_context"],
    wide["fold_Without_Setting"]
):
    raise ValueError("The models do not share identical folds.")

wide = wide.merge(
    mapping[["SampleID", "provisional_source_group_id"]],
    on="SampleID",
    how="left",
    validate="one_to_one"
)

wide = wide.dropna(subset=["provisional_source_group_id"]).copy()

wide["error_full"] = (
    wide["actual_Full_context"] - wide["prediction_Full_context"]
).abs()

wide["error_without_setting"] = (
    wide["actual_Without_Setting"] - wide["prediction_Without_Setting"]
).abs()

wide["paired_improvement"] = (
    wide["error_without_setting"] - wide["error_full"]
)

grouped = wide.groupby("provisional_source_group_id").agg(
    observations=("SampleID", "size"),
    full_error_sum=("error_full", "sum"),
    without_setting_error_sum=("error_without_setting", "sum"),
    full_group_mae=("error_full", "mean"),
    without_setting_group_mae=("error_without_setting", "mean"),
    mean_paired_improvement=("paired_improvement", "mean")
).reset_index()

n_groups = len(grouped)
rng = np.random.default_rng(2026)
B = 5000
results = []

for b in range(B):
    idx = rng.integers(0, n_groups, size=n_groups)
    sample = grouped.iloc[idx]
    n = sample["observations"].sum()

    full_mae = sample["full_error_sum"].sum() / n
    without_mae = sample["without_setting_error_sum"].sum() / n

    results.append({
        "bootstrap": b + 1,
        "pooled_improvement": without_mae - full_mae,
        "relative_improvement_percent":
            (without_mae - full_mae) / without_mae * 100,
        "macro_group_improvement":
            (sample["without_setting_group_mae"] -
             sample["full_group_mae"]).mean()
    })

boot = pd.DataFrame(results)

full_mae = wide["error_full"].mean()
without_mae = wide["error_without_setting"].mean()
difference = without_mae - full_mae

print("EXPERIMENT 36: SETTING FEATURE — GROUP BOOTSTRAP")
print("Observations:", len(wide))
print("Source groups:", n_groups)

print("\nOBSERVED PAIRED PERFORMANCE")
print(f"Full context MAE: {full_mae:.4f}")
print(f"Without Setting MAE: {without_mae:.4f}")
print(f"Absolute MAE improvement: {difference:.4f}")
print(f"Relative MAE improvement: {difference / without_mae * 100:.2f}%")

print("\n95% SOURCE-GROUP BOOTSTRAP INTERVALS")
for col in [
    "pooled_improvement",
    "relative_improvement_percent",
    "macro_group_improvement"
]:
    lo, hi = boot[col].quantile([0.025, 0.975])
    print(f"{col}: {boot[col].mean():.4f} "
          f"(95% CI: {lo:.4f} to {hi:.4f})")
    print("Fraction <= 0:", round((boot[col] <= 0).mean(), 4))

print("\nSOURCE-GROUP COMPARISON")
print("Groups where full context has lower MAE:",
      int((grouped["mean_paired_improvement"] > 0).sum()))
print("Groups where removing Setting has lower MAE:",
      int((grouped["mean_paired_improvement"] < 0).sum()))
print("Tied groups:",
      int((grouped["mean_paired_improvement"] == 0).sum()))

out = Path("06_Validation")
boot.to_csv(out / "setting_group_bootstrap_draws.csv", index=False)
grouped.to_csv(out / "setting_group_bootstrap_inputs.csv", index=False)

print("\nSaved Experiment 36 outputs.")
