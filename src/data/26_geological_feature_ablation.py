import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
folds = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")

target = next(c for c in raw.columns if "15Nbulk" in c)
numeric = ["Age", "Latitude", "Longitude"]
categorical = ["Material", "Setting", "LithType", "Lithology"]

data = folds[["SampleID", "fold"]].merge(raw, on="SampleID", how="left")
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target, "fold"] + numeric + categorical)

experiments = {
    "Baseline_Age_Geography": numeric,
    "Full_Geological_Context": numeric + categorical,
    "Without_Material": numeric + [c for c in categorical if c != "Material"],
    "Without_Setting": numeric + [c for c in categorical if c != "Setting"],
    "Without_LithType": numeric + [c for c in categorical if c != "LithType"],
    "Without_Lithology": numeric + [c for c in categorical if c != "Lithology"]
}

fold_results = []
pooled_results = []

print("MATCHED COHORT:", len(data))
print("PUBLICATION GROUPS:", data["DOI"].nunique())

for name, cols in experiments.items():
    predictions = []

    cat_cols = [c for c in cols if c in categorical]
    num_cols = [c for c in cols if c in numeric]

    for fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != fold]
        test = data[data["fold"] == fold]

        prep = ColumnTransformer([
            ("numeric", "passthrough", num_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        model = make_pipeline(
            prep,
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
        pred = model.predict(test[cols])

        predictions.extend(zip(test.index, test[target].values, pred))

        fold_results.append({
            "experiment": name,
            "fold": int(fold),
            "observations": len(test),
            "MAE": mean_absolute_error(test[target], pred),
            "RMSE": np.sqrt(mean_squared_error(test[target], pred)),
            "R2": r2_score(test[target], pred)
        })

    actual = np.array([x[1] for x in predictions])
    predicted = np.array([x[2] for x in predictions])

    pooled_results.append({
        "experiment": name,
        "observations": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    })

print("\nPOOLED FEATURE ABLATION RESULTS")
print(pd.DataFrame(pooled_results).to_string(index=False))

print("\nFOLD RESULTS")
print(pd.DataFrame(fold_results).to_string(index=False))

pd.DataFrame(fold_results).to_csv(
    "06_Validation/geological_feature_ablation_folds.csv", index=False
)
pd.DataFrame(pooled_results).to_csv(
    "06_Validation/geological_feature_ablation_summary.csv", index=False
)

print("\nSaved both ablation result files.")
print("Raw dataset modified: False")
