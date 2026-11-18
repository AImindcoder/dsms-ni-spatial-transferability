import csv
import math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / "results" / "MULTIMODEL_BENCHMARK"

def read_csv(name):
    with (folder / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

predictions = read_csv("buffered_oof_predictions.csv")
fold_metrics = read_csv("buffered_fold_metrics.csv")
summary = read_csv("summary.csv")

groups = defaultdict(list)
observations = defaultdict(list)

for row in predictions:
    key = (row["mode"], row["block"], row["model"])
    groups[key].append(
        (float(row["observed"]), float(row["predicted"]))
    )
    observations[(row["mode"], row["model"])].append(row)

errors = []
checked = 0

for row in fold_metrics:
    key = (row["mode"], row["block"], row["model"])

    if key not in groups:
        errors.append(f"Missing prediction group: {key}")
        continue

    pairs = groups[key]
    n = len(pairs)

    mae = sum(abs(y - p) for y, p in pairs) / n
    rmse = math.sqrt(sum((y - p) ** 2 for y, p in pairs) / n)

    if n != int(row["n_test"]):
        errors.append(f"Count mismatch: {key}")

    if not math.isclose(mae, float(row["mae"]), abs_tol=1e-8):
        errors.append(f"MAE mismatch: {key}")

    if not math.isclose(rmse, float(row["rmse"]), abs_tol=1e-8):
        errors.append(f"RMSE mismatch: {key}")

    checked += 1

fold_keys = {
    (r["mode"], r["block"], r["model"])
    for r in fold_metrics
}

if len(fold_keys) != len(fold_metrics):
    errors.append("Duplicate fold metric keys")

if set(groups) != fold_keys:
    errors.append("Prediction and fold metric group sets differ")

for row in summary:
    key = (row["mode"], row["model"])

    relevant = [
        r for r in fold_metrics
        if (r["mode"], r["model"]) == key
    ]

    if len(relevant) != 37:
        errors.append(f"Expected 37 folds for {key}, found {len(relevant)}")

    if relevant:
        equal_block_mae = sum(
            float(r["mae"]) for r in relevant
        ) / len(relevant)

        if not math.isclose(
            equal_block_mae,
            float(row["equal_block_mean_mae"]),
            abs_tol=1e-8
        ):
            errors.append(f"Equal-block MAE mismatch: {key}")

models = sorted({r["model"] for r in predictions})
blocks = sorted({r["block"] for r in predictions})

print("=== SPATIAL FOLD VERIFICATION ===")
print("Models:", len(models))
print("Unique test blocks:", len(blocks))
print("Fold metric rows:", len(fold_metrics))
print("Verified fold groups:", checked)
print("Prediction rows:", len(predictions))

print("\n=== RESULTS ===")

if errors:
    for error in errors[:30]:
        print("FAIL:", error)
    print("OVERALL: FAIL")
    raise SystemExit(1)

print("All fold counts, MAE and RMSE values match predictions.")
print("All equal-block MAE values match fold metrics.")
print("OVERALL: PASS")
