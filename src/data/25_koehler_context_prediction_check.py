import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
numeric = ["Age", "Latitude", "Longitude"]
categorical = ["Material", "Setting", "LithType", "Lithology"]
features = numeric + categorical

metadata = ["SampleID", "DOI", "FirstAuthor", "Year",
            "Age", "Latitude", "Longitude", "SiteName",
            "Location1-detail", "Location2-country",
            "Material", "Setting", "LithType", "Lithology", target]

raw[target] = pd.to_numeric(raw[target], errors="coerce")

data = pred.merge(raw[metadata], on="SampleID", how="left")
data = data.dropna(subset=[target, "fold"] + features)

train = data[data["fold"] != 3]
test = data[data["fold"] == 3].copy()

models = {
    "Age_geography": numeric,
    "Geological_context": features
}

for name, cols in models.items():
    cat_cols = [c for c in cols if c in categorical]
    num_cols = [c for c in cols if c in numeric]

    prep = ColumnTransformer([
        ("numeric", "passthrough", num_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = make_pipeline(
        prep,
        RandomForestRegressor(
            n_estimators=80, max_depth=18, min_samples_leaf=3,
            max_features=1.0, random_state=42, n_jobs=2
        )
    )
    model.fit(train[cols], train[target])
    test[name + "_prediction"] = model.predict(test[cols])

group = test[test["DOI"] == "10.1016/j.precamres.2018.11.008"].copy()

group["Age_geography_error"] = abs(
    group[target] - group["Age_geography_prediction"]
)
group["Geological_context_error"] = abs(
    group[target] - group["Geological_context_prediction"]
)
group["MAE_improvement"] = (
    group["Age_geography_error"] - group["Geological_context_error"]
)

print("KOEHLER 2019 INDIVIDUAL PREDICTIONS")
print(group[[
    "SampleID", "Age", "SiteName", "Location1-detail",
    "Material", "Setting", "LithType", "Lithology",
    target, "Age_geography_prediction",
    "Geological_context_prediction", "MAE_improvement"
]].to_string(index=False))

print("\nIMPROVEMENT SUMMARY")
print("Observations:", len(group))
print("Improved predictions:", (group["MAE_improvement"] > 0).sum())
print("Worsened predictions:", (group["MAE_improvement"] < 0).sum())
print("Mean individual MAE improvement:", group["MAE_improvement"].mean())

output = Path("08_Results/koehler_2019_context_prediction_audit.csv")
group.to_csv(output, index=False)
print("\nSaved:", output)
print("Raw dataset modified: False")
