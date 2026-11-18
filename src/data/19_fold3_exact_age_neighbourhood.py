import pandas as pd

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
cols = ["SampleID", "Age", "FirstAuthor", "Year", "DOI", "Formation", "Setting", target]
d = pred.merge(raw[cols], on="SampleID", how="left")

d[target] = pd.to_numeric(d[target], errors="coerce")

test = d[(d["fold"] == 3) & (d["Age"] == 2971)]
train = d[(d["fold"] != 3) & d["Age"].between(2946, 2996)].copy()

print("TEST CASE")
print("Observations:", len(test))
print(test["actual"].describe())

print("\nNEARBY TRAINING DATA")
print("Observations:", len(train))
print("Publication groups:", train["DOI"].nunique())
print("Age range:", train["Age"].min(), "to", train["Age"].max())

print("\nNUMERIC TRAINING ISOTOPE DISTRIBUTION")
print(train[target].describe())

print("\nINDIVIDUAL TRAINING OBSERVATIONS")
print(train[["SampleID", "Age", "FirstAuthor", "Year", "DOI", "Formation", target]].to_string(index=False))

print("\nPUBLICATION GROUP SUMMARY")
summary = train.groupby(["DOI", "FirstAuthor", "Year"], dropna=False).agg(
    observations=(target, "count"),
    mean_age=("Age", "mean"),
    mean_d15N=(target, "mean"),
    min_d15N=(target, "min"),
    max_d15N=(target, "max"),
    formations=("Formation", "nunique")
)
print(summary.to_string())

print("\nOriginal dataset modified: False")
