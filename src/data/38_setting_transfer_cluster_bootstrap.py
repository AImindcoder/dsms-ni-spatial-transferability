from pathlib import Path
import pandas as pd
import numpy as np

df = pd.read_csv("06_Validation/setting_transfer_publication_heldout_summary.csv")
# Load exact observation-level results by reconstructing them from saved predictions.
pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)

target = next(c for c in raw.columns if "15Nbulk" in c)

full = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"] == "Full_context")
][["SampleID", "fold", "actual", "prediction"]].copy()
full = full.rename(columns={"prediction": "full_prediction"})

without = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"] == "Without_Setting")
][["SampleID", "prediction"]].copy()
without = without.rename(columns={"prediction": "without_prediction"})

for frame in [raw, mapping, full, without]:
    frame["SampleID"] = frame["SampleID"].astype(str).str.strip()

data = raw[["SampleID", "Setting"]].merge(
    mapping[["SampleID", "provisional_source_group_id"]],
    on="SampleID", how="inner", validate="one_to_one"
)

data = data.merge(full, on="SampleID", how="inner", validate="one_to_one")
data = data.merge(without, on="SampleID", how="inner", validate="one_to_one")

if len(data) != 24012:
    raise ValueError("Unexpected observation count.")

data["Setting"] = data["Setting"].astype(str).str.strip()

for fold in sorted(data["fold"].unique()):
    train_settings = set(data.loc[data["fold"] != fold, "Setting"])
    mask = data["fold"] == fold
    data.loc[mask, "setting_status"] = np.where(
        data.loc[mask, "Setting"].isin(train_settings),
        "Known_Setting",
        "Unseen_Setting"
    )

data["full_error"] = (data["actual"] - data["full_prediction"]).abs()
data["without_error"] = (data["actual"] - data["without_prediction"]).abs()

group_col = "provisional_source_group_id"

known_groups = set(data.loc[data.setting_status == "Known_Setting", group_col])
unseen_groups = set(data.loc[data.setting_status == "Unseen_Setting", group_col])

print("EXPERIMENT 38: SETTING TRANSFER CLUSTER BOOTSTRAP")
print("Total source groups:", data[group_col].nunique())
print("Groups with known-setting observations:", len(known_groups))
print("Groups with unseen-setting observations:", len(unseen_groups))
print("Groups appearing in both categories:", len(known_groups & unseen_groups))

rng = np.random.default_rng(38)
B = 5000
results = []

for status in ["Known_Setting", "Unseen_Setting"]:
    subset = data[data.setting_status == status]

    grouped = subset.groupby(group_col).agg(
        n=("actual", "size"),
        full_sum=("full_error", "sum"),
        without_sum=("without_error", "sum"),
        full_group_mae=("full_error", "mean"),
        without_group_mae=("without_error", "mean")
    ).reset_index()

    G = len(grouped)
    draws = []

    for _ in range(B):
        sample = grouped.iloc[rng.integers(0, G, size=G)]
        n = sample["n"].sum()

        full_mae = sample["full_sum"].sum() / n
        without_mae = sample["without_sum"].sum() / n

        draws.append({
            "absolute_improvement": without_mae - full_mae,
            "relative_improvement_percent":
                (without_mae - full_mae) / without_mae * 100,
            "macro_group_improvement":
                (sample["without_group_mae"] -
                 sample["full_group_mae"]).mean()
        })

    boot = pd.DataFrame(draws)

    observed_full = subset["full_error"].mean()
    observed_without = subset["without_error"].mean()
    observed_improvement = observed_without - observed_full

    print(f"\n{status}")
    print("Observations:", len(subset))
    print("Source groups:", G)
    print("Full-context MAE:", round(observed_full, 4))
    print("Without-Setting MAE:", round(observed_without, 4))
    print("Observed absolute improvement:", round(observed_improvement, 4))

    for metric in [
        "absolute_improvement",
        "relative_improvement_percent",
        "macro_group_improvement"
    ]:
        lo, hi = boot[metric].quantile([0.025, 0.975])
        print(
            f"{metric}: mean={boot[metric].mean():.4f}; "
            f"95% CI=[{lo:.4f}, {hi:.4f}]; "
            f"fraction <= 0={(boot[metric] <= 0).mean():.4f}"
        )

        boot["setting_status"] = status

    results.append(boot)

out = Path("06_Validation")
pd.concat(results, ignore_index=True).to_csv(
    out / "setting_transfer_cluster_bootstrap_draws.csv",
    index=False
)

print("\nSaved Experiment 38 bootstrap results.")
