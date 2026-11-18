import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)
raw[target] = pd.to_numeric(raw[target], errors="coerce")

data = pred.merge(
    raw[["SampleID", "Age", "Latitude", "Longitude", target]],
    on="SampleID",
    how="left"
).dropna(subset=["Age", "Latitude", "Longitude", target, "fold"])

feature_sets = {
    "Age_only": ["Age"],
    "Geography_only": ["Latitude", "Longitude"],
    "Age_plus_geography": ["Age", "Latitude", "Longitude"]
}

results = []
pooled = {}

for name, features in feature_sets.items():
    actual_all = []
    predicted_all = []

    for fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != fold]
        test = data[data["fold"] == fold]

        model = RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=42,
            n_jobs=2
        )

        model.fit(train[features], train[target])
        predictions = model.predict(test[features])
        actual = test[target].to_numpy()

        actual_all.extend(actual)
        predicted_all.extend(predictions)

        results.append({
            "model": name,
            "fold": int(fold),
            "observations": len(test),
            "MAE": mean_absolute_error(actual, predictions),
            "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
            "R2": r2_score(actual, predictions)
        })

    pooled[name] = {
        "MAE": mean_absolute_error(actual_all, predicted_all),
        "RMSE": np.sqrt(mean_squared_error(actual_all, predicted_all)),
        "R2": r2_score(actual_all, predicted_all)
    }

print("FIVE-FOLD COORDINATE-HELD-OUT COMPARISON")
print(pd.DataFrame(results).round(4).to_string(index=False))

print("\nPOOLED OUT-OF-FOLD RESULTS")
print(pd.DataFrame(pooled).T.round(4).to_string())

output = pd.DataFrame(results)
output.to_csv("06_Validation/coordinate_feature_ablation_fold_metrics.csv", index=False)
print("\nSaved: 06_Validation/coordinate_feature_ablation_fold_metrics.csv")
