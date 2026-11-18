import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
folds = pd.read_csv("06_Validation/coordinate_heldout_oof_predictions.csv")

target = next(c for c in raw.columns if "15Nbulk" in c)
numeric = ["Age", "Latitude", "Longitude"]
categorical = ["Material", "Setting", "LithType", "Lithology"]

data = folds[["SampleID", "fold"]].merge(raw, on="SampleID", how="left")
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target, "fold"] + numeric + categorical).copy()

data["Setting"] = data["Setting"].astype(str)

features = {
    "Full_context": numeric + categorical,
    "Without_Setting": numeric + ["Material", "LithType", "Lithology"]
}

records = []

for fold in sorted(data["fold"].unique()):
    train = data[data["fold"] != fold]
    test = data[data["fold"] == fold]

    known_settings = set(train["Setting"].unique())
    test = test.copy()
    test["setting_status"] = test["Setting"].apply(
        lambda x: "Known" if x in known_settings else "Unseen"
    )

    predictions = {}

    for model_name, cols in features.items():
        cat_cols = [c for c in cols if c in categorical]
        num_cols = [c for c in cols if c in numeric]

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
        predictions[model_name] = model.predict(test[cols])

    test["full_prediction"] = predictions["Full_context"]
    test["without_setting_prediction"] = predictions["Without_Setting"]

    for status, group in test.groupby("setting_status"):
        records.append({
            "fold": int(fold),
            "setting_status": status,
            "observations": len(group),
            "distinct_settings": group["Setting"].nunique(),
            "Full_context_MAE": mean_absolute_error(
                group[target], group["full_prediction"]
            ),
            "Without_Setting_MAE": mean_absolute_error(
                group[target], group["without_setting_prediction"]
            )
        })

results = pd.DataFrame(records)

print("SETTING CATEGORY GENERALIZATION AUDIT")
print(results.to_string(index=False))

print("\nPOOLED RESULTS BY SETTING STATUS")
for status, group in results.groupby("setting_status"):
    # Reconstruct observation-weighted MAE from fold-level MAEs
    n = group["observations"].sum()
    full_mae = np.average(group["Full_context_MAE"], weights=group["observations"])
    without_mae = np.average(
        group["Without_Setting_MAE"], weights=group["observations"]
    )

    print(f"\n{status}:")
    print("Observations:", n)
    print("Full context MAE:", round(full_mae, 4))
    print("Without Setting MAE:", round(without_mae, 4))
    print("MAE difference (without - full):", round(without_mae - full_mae, 4))

results.to_csv(
    "06_Validation/setting_category_generalization_audit.csv", index=False
)

print("\nSaved: 06_Validation/setting_category_generalization_audit.csv")
print("Raw dataset modified: False")
