"""
Portable DSMS-NI ExtraTrees reconstruction.

New implementation, not recovered original benchmark code.
Uses fixed, provisional hyperparameters.

Example:
python portable_extratrees.py --data dataset.csv --block 11_0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
        raise ValueError("Block is not an eligible test block")

    lat, lon = map(int, args.block.split("_"))

    neighbors = {
        f"{lat-1}_{lon}",
        f"{lat+1}_{lon}",
        f"{lat}_{(lon-1)%12}",
        f"{lat}_{(lon+1)%12}"
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

    expected_train = int(expected.iloc[0]["n_train"])
    expected_test = int(expected.iloc[0]["n_test"])

    if len(train) != expected_train:
        raise ValueError("Training count does not match benchmark")

    if len(test) != expected_test:
        raise ValueError("Test count does not match benchmark")

    if set(train["sequence"].dropna()) & set(test["sequence"].dropna()):
        raise ValueError("Sequence leakage detected")

    if set(train["doi_key"].dropna()) & set(test["doi_key"].dropna()):
        raise ValueError("Publication leakage detected")

    numeric = ["age", "latitude", "longitude", "depth"]
    categorical = ["setting"]

    preprocessing = ColumnTransformer([
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
                    SimpleImputer(strategy="most_frequent")
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
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=3,
                max_features=1.0,
                random_state=20261111,
                n_jobs=2
            )
        )
    ])

    features = numeric + categorical

    model.fit(
        train[features],
        train["target"]
    )

    predictions = model.predict(test[features])

    mae = float(np.mean(
        np.abs(test["target"].to_numpy() - predictions)
    ))

    output = (
        Path(args.output)
        if args.output
        else repo / "reconstruction_outputs"
        / "portable_extratrees"
        / f"{args.block}_predictions.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "row_index": test.index,
        "block": args.block,
        "observed": test["target"].to_numpy(),
        "predicted": predictions
    }).to_csv(output, index=False)

    print("Block:", args.block)
    print("Training:", len(train))
    print("Testing:", len(test))
    print("MAE:", mae)
    print("Saved:", output)


if __name__ == "__main__":
    main()
