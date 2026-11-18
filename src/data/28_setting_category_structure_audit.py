import pandas as pd
import re

raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
folds = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")

target = next(c for c in raw.columns if "15Nbulk" in c)

data = folds[["SampleID", "fold"]].merge(raw, on="SampleID", how="left")
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target, "Age", "Latitude", "Longitude",
                           "Material", "Setting", "LithType", "Lithology"]).copy()

data["Setting"] = data["Setting"].astype(str).str.strip()
data["Setting_normalized"] = (
    data["Setting"].str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip(" ,;")
)

print("SETTING CATEGORY AUDIT")
print("Complete-case observations:", len(data))
print("Exact distinct categories:", data["Setting"].nunique())
print("Normalized distinct categories:", data["Setting_normalized"].nunique())
print("Categories containing commas:",
      data["Setting"].str.contains(",", regex=False).sum())
print("Categories containing semicolons:",
      data["Setting"].str.contains(";", regex=False).sum())

print("\nTOP 30 EXACT SETTING CATEGORIES")
top = data["Setting"].value_counts().head(30)
for label, count in top.items():
    print(f"{count:6d} | {label}")

print("\nRARE CATEGORIES (5 OR FEWER OBSERVATIONS)")
counts = data["Setting"].value_counts()
rare = counts[counts <= 5]
print("Number of rare categories:", len(rare))
print("Observations in rare categories:", int(rare.sum()))
for label, count in rare.head(30).items():
    print(f"{count:6d} | {label}")

print("\nNORMALIZATION COLLISIONS")
collision = (
    data.groupby("Setting_normalized")["Setting"]
    .nunique()
    .sort_values(ascending=False)
)
collision = collision[collision > 1]
print("Normalized labels with multiple original spellings:", len(collision))

for normalized, _ in collision.head(20).items():
    variants = data.loc[
        data["Setting_normalized"] == normalized, "Setting"
    ].drop_duplicates().tolist()
    print("\nNormalized:", normalized)
    for variant in variants:
        print("  -", variant)

print("\nSaved nothing; raw dataset remains unchanged.")
