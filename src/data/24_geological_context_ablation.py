import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)

target = next(c for c in raw.columns if "15Nbulk" in c)

numeric = ["Age", "Latitude", "Longitude"]
categorical = ["Material", "Setting", "LithType", "Lithology"]
features = numeric + categorical

raw[target] = pd.to_numeric(raw[target], errors="coerce")

data = pred.merge(
    raw[["SampleID", target] + features],
    on="SampleID",
    how="left"
)

data = data.dropna(subset=[target, "fold"] + features).copy()

print("MATCHED COMPLETE-CASE COHORT")
print("Observations:", len(data))
print("Publication groups:", data["fold"].nunique(), "folds")
print("Categorical feature levels:")
for c in categorical:
    print(c, ":", data[c].nunique())

models = {
    "Age_geography": numeric,
    "Age_geography_geological_context": features
}

results = []
pooled = {}

for name, cols in models.items():
    categorical_cols = [c for c in cols if c in categorical]
    numeric_cols = [c for c in cols if c in numeric]

    transformer = ColumnTransformer([
        ("numeric", "passthrough", numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    actual_all = []
    predicted_all = []

    for fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != fold]
        test = data[data["fold"] == fold]

        model = make_pipeline(
            transformer,
            RandomForestRegressor(
                n_estimators=80,
                max_depth=18,
                min_samples_leaf=3,
                max_features=1.0,
                random_state=42,
                n_jobs=2
            )
        )

        model.fit(train[cols], train[target])
        predictions = model.predict(test[cols])
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

print("\nFOLD RESULTS")
print(pd.DataFrame(results).round(4).to_string(index=False))

print("\nPOOLED RESULTS")
print(pd.DataFrame(pooled).T.round(4).to_string())

pd.DataFrame(results).to_csv(
    "06_Validation/geological_context_ablation_fold_metrics.csv",
    index=False
)

print("\nSaved geological_context_ablation_fold_metrics.csv")
print("Raw dataset modified: False")
