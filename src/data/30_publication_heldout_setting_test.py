import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

raw = pd.read_csv(
    "01_Data/raw/DSMSNI_v0.3.csv",
    encoding="gbk",
    low_memory=False
)

folds = pd.read_csv(
    "06_Validation/coordinate_heldout_oof_predictions.csv"
)

mapping = pd.read_csv(
    "02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv",
    low_memory=False
)

target = next(c for c in raw.columns if "15Nbulk" in c)
group_col = "provisional_source_group_id"

data = folds[["SampleID"]].merge(raw, on="SampleID", how="left")
data = data.merge(
    mapping[["SampleID", group_col]],
    on="SampleID",
    how="left",
    validate="one_to_one"
)

numeric = ["Age", "Latitude", "Longitude"]
categorical = ["Material", "Setting", "LithType", "Lithology"]

data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(
    subset=[target, group_col] + numeric + categorical
).copy()

data["Setting"] = data["Setting"].astype(str).str.strip()

experiments = {
    "Full_context": numeric + categorical,
    "Without_Setting": numeric + [
        "Material", "LithType", "Lithology"
    ]
}

cv = GroupKFold(n_splits=5)
all_results = []
predictions = []

print("PUBLICATION-HELD-OUT EXPERIMENT")
print("Observations:", len(data))
print("Provisional source groups:", data[group_col].nunique())
print("Setting categories:", data["Setting"].nunique())

for fold, (train_idx, test_idx) in enumerate(
    cv.split(data, groups=data[group_col]), start=1
):
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    assert set(train[group_col]).isdisjoint(set(test[group_col]))

    known_settings = set(train["Setting"].unique())
    test["setting_status"] = np.where(
        test["Setting"].isin(known_settings), "Known", "Unseen"
    )

    fold_predictions = {}

    for name, cols in experiments.items():
        cat_cols = [c for c in cols if c in categorical]
        num_cols = [c for c in cols if c in numeric]

        prep = ColumnTransformer([
            ("numeric", "passthrough", num_cols),
            ("categorical",
             OneHotEncoder(handle_unknown="ignore"),
             cat_cols)
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
        fold_predictions[name] = model.predict(test[cols])

    for name, pred in fold_predictions.items():
        for status in ["Known", "Unseen"]:
            mask = test["setting_status"].values == status
            if mask.sum() == 0:
                continue

            all_results.append({
                "fold": fold,
                "model": name,
                "setting_status": status,
                "observations": int(mask.sum()),
                "source_groups": test.loc[
                    mask, group_col
                ].nunique(),
                "MAE": mean_absolute_error(
                    test.loc[mask, target], pred[mask]
                ),
                "RMSE": np.sqrt(mean_squared_error(
                    test.loc[mask, target], pred[mask]
                ))
            })

        predictions.append(pd.DataFrame({
            "fold": fold,
            "SampleID": test["SampleID"].values,
            "source_group": test[group_col].values,
            "setting_status": test["setting_status"].values,
            "actual": test[target].values,
            "model": name,
            "prediction": pred
        }))

    print(f"Completed fold {fold}: "
          f"train={len(train)}, test={len(test)}, "
          f"test source groups={test[group_col].nunique()}")

results = pd.DataFrame(all_results)
pred_df = pd.concat(predictions, ignore_index=True)

print("\nFOLD-LEVEL RESULTS")
print(results.to_string(index=False))

print("\nPOOLED PUBLICATION-HELD-OUT RESULTS")

for (model, status), group in pred_df.groupby(
    ["model", "setting_status"]
):
    print(
        f"{model:20s} | {status:6s} | "
        f"N={len(group):6d} | "
        f"MAE={mean_absolute_error(group.actual, group.prediction):.4f} | "
        f"RMSE={np.sqrt(mean_squared_error(group.actual, group.prediction)):.4f} | "
        f"R2={r2_score(group.actual, group.prediction):.4f}"
    )

results.to_csv(
    "06_Validation/publication_heldout_setting_fold_metrics.csv",
    index=False
)

pred_df.to_csv(
    "06_Validation/publication_heldout_setting_predictions.csv",
    index=False
)

print("\nSaved publication-held-out metrics and predictions.")
print("Publication overlap between train and test: 0")
print("Raw dataset modified: False")
