import pandas as pd
import numpy as np

pred = pd.read_csv("06_Validation/random_vs_publication_heldout_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)

full = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"] == "Full_context")
].copy()

without = pred[
    (pred["validation"] == "Publication_held_out") &
    (pred["model"] == "Without_Setting")
].copy()

print("FULL PREDICTION ROWS:", len(full))
print("WITHOUT SETTING ROWS:", len(without))
print("RAW DATA ROWS:", len(raw))

print("\nSAMPLE ID TYPES")
print("Raw:", raw["SampleID"].dtype)
print("Predictions:", full["SampleID"].dtype)

raw["SampleID"] = raw["SampleID"].astype(str).str.strip()
full["SampleID"] = full["SampleID"].astype(str).str.strip()
without["SampleID"] = without["SampleID"].astype(str).str.strip()

print("\nDUPLICATE SAMPLE IDS")
print("Raw:", raw["SampleID"].duplicated().sum())
print("Full predictions:", full["SampleID"].duplicated().sum())
print("Without Setting:", without["SampleID"].duplicated().sum())

raw_target = pd.to_numeric(raw[target], errors="coerce")

matched = full.merge(
    raw[["SampleID"]].assign(raw_target=raw_target),
    on="SampleID", how="left", validate="one_to_one"
)

print("\nMATCHING")
print("Matched IDs:", matched["raw_target"].notna().sum())
print("Unmatched IDs:", matched["raw_target"].isna().sum())

valid = matched["raw_target"].notna()
if valid.any():
    difference = np.abs(
        matched.loc[valid, "raw_target"].to_numpy() -
        matched.loc[valid, "actual"].to_numpy()
    )
    print("Maximum target difference:", difference.max())
    print("Exact target matches:", np.isclose(difference, 0).sum())
    print("Target mismatches:", (~np.isclose(difference, 0)).sum())

print("\nSAMPLE ID EXAMPLES")
print("Prediction IDs:", full["SampleID"].head(5).tolist())
print("Raw IDs:", raw["SampleID"].head(5).tolist())

print("\nPREDICTION COLUMNS:", full.columns.tolist())
print("RAW TARGET COLUMN:", repr(target))
