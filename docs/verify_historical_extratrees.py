import csv
import math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / "results" / "HISTORICAL_EXTRATREES_37FOLD"
prefix = "EXP_SPATIAL_37FOLD_EXTRATREES_"

def rows(name):
    with (folder / (prefix + name)).open(
        newline="", encoding="utf-8-sig"
    ) as f:
        yield from csv.DictReader(f)

errors = []
oof = list(rows("OOF.csv"))
fold_metrics = list(rows("FOLD_METRICS.csv"))
summary = list(rows("SUMMARY.csv"))

print("=== HISTORICAL EXTRATREES VERIFICATION ===")
print("OOF records:", len(oof))
print("Fold metric rows:", len(fold_metrics))
print("Summary rows:", len(summary))

by_fold = defaultdict(list)
seen_oof = set()

for r in oof:
    fold = r["Fold"]
    key = (fold, r["row_id"])

    if key in seen_oof:
        errors.append(f"Duplicate OOF row: {key}")

    seen_oof.add(key)
    by_fold[fold].append(r)

if len(by_fold) != 37:
    errors.append(f"Expected 37 folds; found {len(by_fold)}")

for r in fold_metrics:
    fold = r["Fold"]
    group = by_fold.get(fold, [])

    if len(group) != int(r["Test_records"]):
        errors.append(f"Test count mismatch: fold {fold}")

    if any(x["Test_block"] != r["Test_block"] for x in group):
        errors.append(f"Test block mismatch: fold {fold}")

    if not group:
        continue

    actual = [float(x["actual"]) for x in group]

    for prediction_col, mae_col, rmse_col in [
        ("prediction", "ExtraTrees_MAE", "ExtraTrees_RMSE"),
        ("baseline_prediction", "Median_baseline_MAE",
         "Median_baseline_RMSE")
    ]:
        pred = [float(x[prediction_col]) for x in group]
        n = len(group)

        mae = sum(abs(a-p) for a,p in zip(actual,pred)) / n
        rmse = math.sqrt(
            sum((a-p)**2 for a,p in zip(actual,pred)) / n
        )

        if not math.isclose(
            mae, float(r[mae_col]), abs_tol=1e-8
        ):
            errors.append(f"{mae_col} mismatch: fold {fold}")

        if not math.isclose(
            rmse, float(r[rmse_col]), abs_tol=1e-8
        ):
            errors.append(f"{rmse_col} mismatch: fold {fold}")

print("\n=== POOLED SUMMARY ===")

for r in summary:
    model = r["Model"]
    col = (
        "prediction"
        if model == "ExtraTrees"
        else "baseline_prediction"
    )

    actual = [float(x["actual"]) for x in oof]
    pred = [float(x[col]) for x in oof]

    n = len(actual)
    mean = sum(actual) / n

    mae = sum(abs(a-p) for a,p in zip(actual,pred)) / n
    mse = sum((a-p)**2 for a,p in zip(actual,pred)) / n
    rmse = math.sqrt(mse)

    denominator = sum((a-mean)**2 for a in actual)
    r2 = 1 - n*mse/denominator

    checks = {
        "N": n == int(r["N"]),
        "MAE": math.isclose(mae, float(r["MAE"]), abs_tol=1e-8),
        "RMSE": math.isclose(rmse, float(r["RMSE"]), abs_tol=1e-8),
        "R2": math.isclose(r2, float(r["R2"]), abs_tol=1e-8)
    }

    print(
        f"{model}: n={n} MAE={mae:.6f} "
        f"RMSE={rmse:.6f} R2={r2:.6f} "
        f"{'PASS' if all(checks.values()) else 'FAIL'}"
    )

    for metric, ok in checks.items():
        if not ok:
            errors.append(f"{model}: {metric} mismatch")

print("\n=== MANIFEST ROW SEPARATION ===")

roles = defaultdict(int)
train_ids = defaultdict(set)
test_ids = defaultdict(set)
manifest_count = 0

for r in rows("MANIFEST.csv"):
    manifest_count += 1
    fold = r["Fold"]
    role = r["Role"].strip().lower()
    rid = r["row_id"]

    roles[role] += 1

    if role == "train":
        if rid in train_ids[fold]:
            errors.append(f"Duplicate train assignment: {fold}/{rid}")
        train_ids[fold].add(rid)
    elif role == "test":
        if rid in test_ids[fold]:
            errors.append(f"Duplicate test assignment: {fold}/{rid}")
        test_ids[fold].add(rid)

print("Manifest entries:", manifest_count)
print("Role counts:", dict(roles))

for fold in by_fold:
    overlap = train_ids[fold] & test_ids[fold]

    if overlap:
        errors.append(f"Train/test row overlap: fold {fold}")

    expected = {r["row_id"] for r in by_fold[fold]}

    if expected != test_ids[fold]:
        errors.append(f"OOF/manifest test IDs mismatch: fold {fold}")

    matching = [
        r for r in fold_metrics if r["Fold"] == fold
    ]

    if matching and len(train_ids[fold]) != int(
        matching[0]["Train_records"]
    ):
        errors.append(f"Manifest train count mismatch: fold {fold}")

print("\n=== FINAL RESULT ===")

if errors:
    print("OVERALL: FAIL")
    for error in errors[:25]:
        print(" -", error)
    print("Total issues:", len(errors))
    raise SystemExit(1)

print("All tested numerical and row-separation checks passed.")
print("OVERALL: PASS")
