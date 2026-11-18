from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(".")
raw = pd.read_csv("01_Data/raw/DSMSNI_v0.3.csv", encoding="gbk", low_memory=False)
mapping = pd.read_csv("02_Data_Audit/provenance_analysis/full_provenance_mapping_final.csv")

target = next(c for c in raw.columns if "15Nbulk" in c)
required = ["SampleID", "Age", "Latitude", "Longitude", "Material", "Setting", "LithType", "Lithology"]
group_col = "provisional_source_group_id"

df = raw[required + [target]].merge(
    mapping[["SampleID", group_col]], on="SampleID", how="inner"
)
df[target] = pd.to_numeric(df[target], errors="coerce")
features = ["Age", "Latitude", "Longitude", "Material", "Setting", "LithType", "Lithology"]
df = df.dropna(subset=[target, group_col] + features).reset_index(drop=True)

print("EXPERIMENT 31: RANDOM vs PUBLICATION-HELD-OUT")
print("Observations:", len(df))
print("Source groups:", df[group_col].nunique())

numeric = ["Age", "Latitude", "Longitude"]
categorical = ["Material", "Setting", "LithType", "Lithology"]
models = {
    "Full_context": features,
    "Without_Setting": [c for c in features if c != "Setting"]
}

def make_model(cols):
    nums = [c for c in numeric if c in cols]
    cats = [c for c in categorical if c in cols]
    prep = ColumnTransformer([
        ("num", "passthrough", nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)
    ])
    return Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(
            n_estimators=80, max_depth=18, min_samples_leaf=3,
            max_features=1.0, random_state=42, n_jobs=2
        ))
    ])

splits = {
    "Random": list(KFold(n_splits=5, shuffle=True, random_state=42).split(df)),
    "Publication_held_out": list(
        GroupKFold(n_splits=5).split(df, groups=df[group_col])
    )
}

results = []
predictions = []

for validation, folds in splits.items():
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        overlap = set(train[group_col]) & set(test[group_col])

        print(f"{validation} fold {fold}: train={len(train)}, test={len(test)}, overlapping groups={len(overlap)}")

        for model_name, cols in models.items():
            model = make_model(cols)
            model.fit(train[cols], train[target])
            pred = model.predict(test[cols])
            actual = test[target].to_numpy()

            results.append({
                "validation": validation,
                "fold": fold,
                "model": model_name,
                "observations": len(test),
                "overlapping_source_groups": len(overlap),
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(mean_squared_error(actual, pred)),
                "R2": r2_score(actual, pred)
            })

            for sid, y, p in zip(test["SampleID"], actual, pred):
                predictions.append({
                    "SampleID": sid,
                    "validation": validation,
                    "fold": fold,
                    "model": model_name,
                    "actual": y,
                    "prediction": p
                })

results = pd.DataFrame(results)
predictions = pd.DataFrame(predictions)

summary = (
    predictions.groupby(["validation", "model"])
    .apply(lambda g: pd.Series({
        "observations": len(g),
        "MAE": mean_absolute_error(g.actual, g.prediction),
        "RMSE": np.sqrt(mean_squared_error(g.actual, g.prediction)),
        "R2": r2_score(g.actual, g.prediction)
    }), include_groups=False)
    .reset_index()
)

out = Path("06_Validation")
results.to_csv(out / "random_vs_publication_heldout_fold_metrics.csv", index=False)
predictions.to_csv(out / "random_vs_publication_heldout_predictions.csv", index=False)
summary.to_csv(out / "random_vs_publication_heldout_summary.csv", index=False)

print("\nPOOLED OUT-OF-FOLD RESULTS")
print(summary.to_string(index=False))
print("\nSaved all three result files.")
print("Raw dataset modified: False")
