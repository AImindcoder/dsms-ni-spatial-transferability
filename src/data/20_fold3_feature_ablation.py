import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

root = Path(".")
pred = pd.read_csv(root / "06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv(root / "01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
raw[target] = pd.to_numeric(raw[target], errors="coerce")

data = pred.merge(raw[["SampleID", "Age", "Latitude", "Longitude", target]], on="SampleID", how="left")
data = data.dropna(subset=["Age", "Latitude", "Longitude", target, "fold"])

test = data[data["fold"] == 3].copy()
train = data[data["fold"] != 3].copy()

case = test[
    (test["Age"].sub(2971).abs() < 0.001) &
    (test["Latitude"].sub(-21.7).abs() < 0.001) &
    (test["Longitude"].sub(120.62).abs() < 0.001)
]

if case.empty:
    print("Target case not found. Check fold and coordinates.")
    raise SystemExit

feature_sets = {
    "Age only": ["Age"],
    "Geography only": ["Latitude", "Longitude"],
    "Age + geography": ["Age", "Latitude", "Longitude"]
}

print("TEST OBSERVATIONS:", len(case))
print("ACTUAL MEAN:", round(case[target].mean(), 4))
print("ORIGINAL PREDICTION MEAN:", round(case["predicted"].mean(), 4) if "predicted" in case.columns else "See original prediction column")

for name, features in feature_sets.items():
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=2
    )
    model.fit(train[features], train[target])
    predictions = model.predict(case[features])

    print("\n", name)
    print("Mean prediction:", round(float(np.mean(predictions)), 4))
    print("MAE:", round(float(np.mean(np.abs(case[target].to_numpy() - predictions))), 4))
    print("Prediction range:", round(float(np.min(predictions)), 4), "to", round(float(np.max(predictions)), 4))
