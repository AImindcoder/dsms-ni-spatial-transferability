import csv
import math
from collections import defaultdict
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
folder = repo / "results" / "HISTORICAL_EXTRATREES_37FOLD"
prefix = "EXP_SPATIAL_37FOLD_EXTRATREES_"

def read_csv(name):
    with (folder / (prefix + name)).open(
        newline="", encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))

oof = read_csv("OOF.csv")
summary = read_csv("SUMMARY.csv")

print("=== HISTORICAL BALANCED MAE VERIFICATION ===")
print("OOF observations:", len(oof))

models = {
    "ExtraTrees": "prediction",
    "Median baseline": "baseline_prediction"
}

groupings = {
    "Sequence_balanced_MAE": "sequence",
    "Publication_balanced_MAE": "doi_key",
    "Block_balanced_MAE": "Test_block"
}

errors = []

for row in summary:
    model = row["Model"]

    if model not in models:
        errors.append(f"Unknown model: {model}")
        continue

    prediction_column = models[model]

    print(f"\nMODEL: {model}")

    for metric_name, group_column in groupings.items():
        group_errors = defaultdict(list)

        for record in oof:
            group = record[group_column]

            actual = float(record["actual"])
            predicted = float(record[prediction_column])

            group_errors[group].append(abs(actual - predicted))

        group_maes = [
            sum(values) / len(values)
            for values in group_errors.values()
        ]

        calculated = sum(group_maes) / len(group_maes)
        reported = float(row[metric_name])

        passed = math.isclose(
            calculated, reported,
            rel_tol=1e-10,
            abs_tol=1e-8
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"{metric_name}: "
            f"groups={len(group_maes)} "
            f"calculated={calculated:.9f} "
            f"reported={reported:.9f} "
            f"{status}"
        )

        if not passed:
            errors.append(f"{model}: {metric_name}")

print("\n=== FINAL RESULT ===")

if errors:
    print("OVERALL: FAIL")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print("All six balanced MAE statistics independently verified.")
print("OVERALL: PASS")
