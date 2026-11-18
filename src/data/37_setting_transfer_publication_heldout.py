from pathlib import Path
import pandas as pd
import numpy as np

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

without = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"] == "Without_Setting")
][["SampleID", "prediction"]].copy()

full = full.rename(columns={"prediction": "full_prediction"})
without = without.rename(columns={"prediction": "without_setting_prediction"})

for frame in [raw, mapping, full, without]:
    frame["SampleID"] = frame["SampleID"].astype(str).str.strip()

raw_data = raw[["SampleID", "Setting"]].copy()
raw_data[target] = pd.to_numeric(raw[target], errors="coerce")

df = raw_data.merge(
    mapping[["SampleID", "provisional_source_group_id"]],
    on="SampleID", how="inner", validate="one_to_one"
)

df = df.merge(
    full, on="SampleID", how="inner", validate="one_to_one"
)

df = df.merge(
    without, on="SampleID", how="inner", validate="one_to_one"
)

if len(df) != 24012:
    raise ValueError(f"Expected 24012 observations, obtained {len(df)}.")

if not np.allclose(df[target], df["actual"], equal_nan=True):
    raise ValueError("Raw target values do not match saved predictions.")

if df["Setting"].isna().any():
    raise ValueError("Missing Setting values detected.")

if df["provisional_source_group_id"].isna().any():
    raise ValueError("Missing source group identifiers detected.")

if (df.groupby("provisional_source_group_id")["fold"].nunique() != 1).any():
    raise ValueError("A source group crosses validation folds.")

df["Setting"] = df["Setting"].astype(str).str.strip()

df["full_error"] = (df["actual"] - df["full_prediction"]).abs()
df["without_error"] = (
    df["actual"] - df["without_setting_prediction"]
).abs()

df["setting_status"] = ""

for fold in sorted(df["fold"].unique()):
    train_settings = set(
        df.loc[df["fold"] != fold, "Setting"].unique()
    )

    mask = df["fold"] == fold

    df.loc[mask, "setting_status"] = np.where(
        df.loc[mask, "Setting"].isin(train_settings),
        "Known_Setting",
        "Unseen_Setting"
    )

print("EXPERIMENT 37: CORRECTED SETTING TRANSFER AUDIT")
print("Observations:", len(df))
print("Source groups:", df["provisional_source_group_id"].nunique())
print("Validation folds:", df["fold"].nunique())

print("\nPOOLED RESULTS")

rows = []

for status, g in df.groupby("setting_status"):
    rows.append({
        "setting_status": status,
        "observations": len(g),
        "source_groups": g["provisional_source_group_id"].nunique(),
        "full_context_MAE": g["full_error"].mean(),
        "without_setting_MAE": g["without_error"].mean(),
        "MAE_improvement": g["without_error"].mean() - g["full_error"].mean()
    })

summary = pd.DataFrame(rows)
print(summary.round(4).to_string(index=False))

print("\nFOLD-LEVEL RESULTS")

fold_rows = []

for (fold, status), g in df.groupby(["fold", "setting_status"]):
    fold_rows.append({
        "fold": fold,
        "setting_status": status,
        "observations": len(g),
        "full_context_MAE": g["full_error"].mean(),
        "without_setting_MAE": g["without_error"].mean(),
        "MAE_improvement": g["without_error"].mean() - g["full_error"].mean()
    })

fold_results = pd.DataFrame(fold_rows)
print(fold_results.round(4).to_string(index=False))

out = Path("06_Validation")
summary.to_csv(out / "setting_transfer_publication_heldout_summary.csv", index=False)
fold_results.to_csv(out / "setting_transfer_publication_heldout_folds.csv", index=False)

print("\nSUCCESS: All prediction and source-group checks passed.")
print("Saved Experiment 37 outputs.")
