import pandas as pd
import numpy as np

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
cols = ["SampleID", "Age", "Latitude", "Longitude", "FirstAuthor",
        "Year", "DOI", "Formation", "Setting", target]

d = pred.merge(raw[cols], on="SampleID", how="left")
d[target] = pd.to_numeric(d[target], errors="coerce")

test = d[(d["fold"] == 3) & (d["Age"] == 2971)]
train = d[(d["fold"] != 3)].dropna(subset=["Latitude", "Longitude", target]).copy()

lat0 = test["Latitude"].median()
lon0 = test["Longitude"].median()

def distance_km(lat, lon):
    r = 6371.0
    p1, p2 = np.radians(lat0), np.radians(lat)
    dp = np.radians(lat - lat0)
    dl = np.radians(lon - lon0)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(a))

train["distance_km"] = distance_km(train["Latitude"], train["Longitude"])
nearest = train.nsmallest(20, "distance_km")

print("TEST LOCATION: Tumbiana Formation")
print("Test age: 2971 Ma")
print("Test mean actual:", test["actual"].mean())
print("Test mean prediction:", test["prediction"].mean())

print("\n20 CLOSEST TRAINING OBSERVATIONS")
print(nearest[["SampleID", "distance_km", "Age", target,
               "FirstAuthor", "Year", "Formation", "Setting"]].to_string(index=False))

print("\nNEAREST TRAINING DISTANCE SUMMARY")
print(train["distance_km"].describe(percentiles=[.01, .05, .10, .25]))

print("\nTRAINING OBSERVATIONS WITHIN 100 KM:", (train["distance_km"] <= 100).sum())
print("WITHIN 500 KM:", (train["distance_km"] <= 500).sum())
print("WITHIN 1000 KM:", (train["distance_km"] <= 1000).sum())

print("\nOriginal dataset modified: False")
