import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
metadata = ["SampleID", "Age", "Latitude", "Longitude",
            "DOI", "FirstAuthor", "Year", "Formation", target]

data = pred.merge(raw[metadata], on="SampleID", how="left")
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=["Age", "Latitude", "Longitude", target, "fold"])

train = data[data["fold"] != 3]
test = data[data["fold"] == 3].copy()

feature_sets = {
    "Age_only": ["Age"],
    "Age_plus_geography": ["Age", "Latitude", "Longitude"]
}

for name, features in feature_sets.items():
    model = RandomForestRegressor(
        n_estimators=80, max_depth=18, min_samples_leaf=3,
        max_features=1.0, random_state=42, n_jobs=2
    )
    model.fit(train[features], train[target])
    test[name + "_prediction"] = model.predict(test[features])
    test[name + "_error"] = abs(
        test[target] - test[name + "_prediction"]
    )

for name in feature_sets:
    print("\nTOP 10 PUBLICATION GROUPS:", name)

    summary = test.groupby(
        ["DOI", "FirstAuthor", "Year"], dropna=False
    ).agg(
        observations=(target, "count"),
        mean_age=("Age", "mean"),
        actual_mean=(target, "mean"),
        actual_min=(target, "min"),
        actual_max=(target, "max"),
        predicted_mean=(name + "_prediction", "mean"),
        MAE=(name + "_error", "mean"),
        RMSE=(name + "_error", lambda x: np.sqrt(np.mean(x**2))),
        formation_values=("Formation", lambda x: ", ".join(
            sorted(x.dropna().astype(str).unique())[:5]
        )),
        missing_formation=("Formation", lambda x: x.isna().mean())
    ).sort_values("MAE", ascending=False)

    print(summary.head(10).round(4).to_string())

print("\nOriginal dataset modified: False")
