import csv
import math
import os
from collections import defaultdict

folder = r"D:\Research\DSMS_NI_GitHub_Publication\results\MULTIMODEL_BENCHMARK"

predictions = os.path.join(folder, "buffered_oof_predictions.csv")
summary_path = os.path.join(folder, "summary.csv")

with open(summary_path, newline="", encoding="utf-8-sig") as f:
    reported = {
        (r["mode"], r["model"]): r
        for r in csv.DictReader(f)
    }

stats = defaultdict(lambda: {
    "count": 0,
    "sum_abs": 0.0,
    "sum_sq": 0.0,
    "sum_y": 0.0,
    "sum_y_sq": 0.0
})

with open(predictions, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        key = (row["mode"], row["model"])
        y = float(row["observed"])
        p = float(row["predicted"])
        e = y - p

        if not (math.isfinite(y) and math.isfinite(p)):
            raise ValueError("Non-finite prediction detected")

        s = stats[key]
        s["count"] += 1
        s["sum_abs"] += abs(e)
        s["sum_sq"] += e * e
        s["sum_y"] += y
        s["sum_y_sq"] += y * y

print("\n=== INDEPENDENT METRIC VERIFICATION ===")

all_pass = True

for key in sorted(stats):
    s = stats[key]
    n = s["count"]

    mae = s["sum_abs"] / n
    rmse = math.sqrt(s["sum_sq"] / n)

    total_variance = s["sum_y_sq"] - s["sum_y"] ** 2 / n
    r2 = 1 - s["sum_sq"] / total_variance if total_variance > 0 else float("nan")

    expected = reported[key]

    checks = [
        n == int(expected["n_test"]),
        math.isclose(mae, float(expected["mae"]), abs_tol=1e-8),
        math.isclose(rmse, float(expected["rmse"]), abs_tol=1e-8),
        math.isclose(r2, float(expected["r2"]), abs_tol=1e-8)
    ]

    passed = all(checks)
    all_pass &= passed

    print(
        f'{key[1]:24s} '
        f'n={n:6d} '
        f'MAE={mae:.6f} '
        f'RMSE={rmse:.6f} '
        f'R2={r2:.6f} '
        f'{"PASS" if passed else "FAIL"}'
    )

if set(stats) != set(reported):
    print("FAIL: Summary and prediction model sets differ")
    all_pass = False

print("\nOVERALL:", "PASS" if all_pass else "FAIL")

if not all_pass:
    raise SystemExit(1)
