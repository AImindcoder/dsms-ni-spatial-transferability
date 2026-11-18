"""
DSMS-NI independent benchmark reconstruction.

This is newly reconstructed code, NOT recovered original
benchmark training code.

Runs one ExtraTrees fold for implementation verification.
Does not modify original research data or benchmark outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SOURCE = Path(
    r"D:\Research\DSMS_NI_Research"
    r"\01_Data\interim"
    r"\EXP_SPATIAL_CV_CENOZOIC_COMPLETE_CASE_AUDIT.csv"
)

REPO = Path(
    r"D:\Research\DSMS_NI_GitHub_Publication"
)

OUTPUT = (
    REPO / "reconstruction_outputs"
    / "extratrees_smoke_test"
)

BLOCK = "11_0"
SEED = 20261111

def main():

    data = pd.read_csv(SOURCE)

    metrics = pd.read_csv(
        REPO / "results/MULTIMODEL_BENCHMARK"
        / "buffered_fold_metrics.csv"
    )

    eligible = set(metrics["block"].astype(str))

    lat, lon = map(int, BLOCK.split("_"))

    neighbors = {
        f"{lat-1}_{lon}",
        f"{lat+1}_{lon}",
        f"{lat}_{(lon-1)%12}",
        f"{lat}_{(lon+1)%12}"
    } & eligible

    test_mask = data["block"].astype(str) == BLOCK

    test_publications = set(
        data.loc[test_mask, "doi_key"].dropna()
    )

    train_mask = (
        ~data["block"].astype(str).isin(
            neighbors | {BLOCK}
        )
        & ~data["doi_key"].isin(test_publications)
    )

    train = data.loc[train_mask].copy()
    test = data.loc[test_mask].copy()

    assert len(train) == 44449
    assert len(test) == 75

    assert not (
        set(train["doi_key"].dropna())
        & set(test["doi_key"].dropna())
    )

    assert not (
        set(train["sequence"].dropna())
        & set(test["sequence"].dropna())
    )

    numeric = [
        "age", "latitude", "longitude", "depth"
    ]

    categorical = ["setting"]

    features = numeric + categorical

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]),
                categorical
            )
        ]
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=3,
                max_features=1.0,
                random_state=SEED,
                n_jobs=2
            )
        )
    ])

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]
    y_test = test["target"]

    print("Training observations:", len(train))
    print("Testing observations:", len(test))

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = float(
        np.mean(np.abs(
            y_test.to_numpy() - predictions
        ))
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    result = pd.DataFrame({
        "row_index": test.index,
        "block": BLOCK,
        "observed": y_test.to_numpy(),
        "reconstructed_prediction": predictions
    })

    result.to_csv(
        OUTPUT / "predictions.csv",
        index=False
    )

    print("Reconstructed ExtraTrees MAE:", mae)
    print("Output:", OUTPUT / "predictions.csv")
    print("SMOKE TEST COMPLETE")

if __name__ == "__main__":
    main()
