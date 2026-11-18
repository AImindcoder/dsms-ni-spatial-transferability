"""
DSMS-NI five-model benchmark reconstruction.

This is newly implemented code, not the recovered original
benchmark implementation.

Model settings are provisional and must not be described
as the original benchmark's recovered hyperparameters.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["age", "latitude", "longitude", "depth"]
CATEGORICAL = ["setting"]
FEATURES = NUMERIC + CATEGORICAL

SEED = 20261111

MODELS = [
    "MedianBaseline",
    "ExtraTrees",
    "RandomForest",
    "HistGradientBoosting",
    "Ridge",
    "NeuralNetwork",
]


def make_preprocessor(standardize=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if standardize:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer([
        (
            "numeric",
            Pipeline(numeric_steps),
            NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            CATEGORICAL,
        ),
    ])


def make_model(name):
    estimators = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=2,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=2,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=31,
            learning_rate=0.1,
            random_state=SEED,
        ),
        "Ridge": Ridge(alpha=1.0),
        "NeuralNetwork": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=0.001,
            max_iter=300,
            random_state=SEED,
            early_stopping=True,
        ),
    }

    standardize = name in {"Ridge", "NeuralNetwork"}

    return Pipeline([
        ("preprocessing", make_preprocessor(standardize)),
        ("model", estimators[name]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--block", default="11_0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]

    data = pd.read_csv(args.data)

    metrics = pd.read_csv(
        repo / "results/MULTIMODEL_BENCHMARK"
        / "buffered_fold_metrics.csv"
    )

    eligible = set(metrics["block"].astype(str))

    if args.block not in eligible:
        raise ValueError("Unknown test block")

    lat, lon = map(int, args.block.split("_"))

    neighbors = {
        f"{lat-1}_{lon}",
        f"{lat+1}_{lon}",
        f"{lat}_{(lon-1)%12}",
        f"{lat}_{(lon+1)%12}",
    } & eligible

    blocks = data["block"].astype(str)
    test_mask = blocks == args.block

    test_publications = set(
        data.loc[test_mask, "doi_key"].dropna()
    )

    train_mask = (
        ~blocks.isin(neighbors | {args.block})
        & ~data["doi_key"].isin(test_publications)
    )

    train = data.loc[train_mask]
    test = data.loc[test_mask]

    expected = metrics[
        (metrics["block"].astype(str) == args.block)
        & (metrics["model"] == "ExtraTrees")
    ]

    if len(expected) != 1:
        raise ValueError("Missing or duplicate benchmark fold")

    assert len(train) == int(expected.iloc[0]["n_train"])
    assert len(test) == int(expected.iloc[0]["n_test"])

    assert not set(train.index) & set(test.index)

    assert not (
        set(train["doi_key"].dropna())
        & set(test["doi_key"].dropna())
    )

    assert not (
        set(train["sequence"].dropna())
        & set(test["sequence"].dropna())
    )

    X_train = train[FEATURES]
    X_test = test[FEATURES]
    y_train = train["target"].to_numpy()
    y_test = test["target"].to_numpy()

    predictions = []
    fold_metrics = []

    for name in MODELS:
        print("Training:", name, flush=True)

        if name == "MedianBaseline":
            pred = np.full(
                len(test),
                np.median(y_train),
                dtype=float,
            )
        else:
            model = make_model(name)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(
            mean_squared_error(y_test, pred)
        )
        r2 = r2_score(y_test, pred)

        fold_metrics.append({
            "block": args.block,
            "model": name,
            "n_train": len(train),
            "n_test": len(test),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        })

        predictions.append(pd.DataFrame({
            "row_index": test.index,
            "block": args.block,
            "model": name,
            "observed": y_test,
            "predicted": pred,
        }))

        print("MAE:", round(mae, 6), flush=True)

    output = (
        Path(args.output)
        if args.output
        else repo / "reconstruction_outputs"
        / "five_model_benchmark"
        / args.block
    )

    output.mkdir(parents=True, exist_ok=True)

    pd.concat(
        predictions,
        ignore_index=True
    ).to_csv(
        output / "predictions.csv",
        index=False
    )

    pd.DataFrame(fold_metrics).to_csv(
        output / "metrics.csv",
        index=False
    )

    config = {
        "implementation": "independent_reconstruction",
        "original_training_code_recovered": False,
        "hyperparameters": "provisional",
        "seed": SEED,
        "block": args.block,
        "models": MODELS,
        "n_train": len(train),
        "n_test": len(test),
    }

    with open(
        output / "reconstruction_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\n=== RECONSTRUCTION COMPLETE ===")
    print(pd.DataFrame(fold_metrics).to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
