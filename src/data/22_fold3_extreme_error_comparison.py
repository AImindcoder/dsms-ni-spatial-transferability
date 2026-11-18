import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
raw[target] = pd.to_numeric(raw[target], errors="coerce")

data = pred.merge(
    raw[["SampleID", "Age", "Latitude", "Longitude",
         "FirstAuthor", "Year", "DOI", "Formation", target]],
    on="SampleID", how="left"
).dropna(subset=["Age", "Latitude", "Longitude", target, "fold"])

train = data[data["fold"] != 3]
test = data[data["fold"] == 3].copy()

models = {
    "Age_only": ["Age"],
    "Age_plus_geography": ["Age", "Latitude", "Longitude"]
}

for name, features in models.items():
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=2
    )
    model.fit(train[features], train[target])
    test[name + "_prediction"] = model.predict(test[features])
    test[name + "_absolute_error"] = abs(
        test[target] - test[name + "_prediction"]
    )

print("FOLD 3: LARGEST ERRORS BY MODEL")

for name in models:
    print("\n", name)
    cols = [
        "SampleID", "Age", "FirstAuthor", "Year",
        "Formation", target,
        name + "_prediction", name + "_absolute_error"
    ]
    print(
        test.nlargest(15, name + "_absolute_error")[cols]
        .to_string(index=False)
    )

print("\nSAVED RESULTS: No raw dataset modifications")
