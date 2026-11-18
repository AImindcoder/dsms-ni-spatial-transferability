import csv
from collections import defaultdict
from pathlib import Path

repo = Path(__file__).resolve().parents[1]

source = Path(
    r"D:\Research\DSMS_NI_Research\01_Data\interim"
    r"\EXP_SPATIAL_CV_CENOZOIC_COMPLETE_CASE_AUDIT.csv"
)

folder = repo / "results" / "HISTORICAL_EXTRATREES_37FOLD"
prefix = "EXP_SPATIAL_37FOLD_EXTRATREES_"

def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)

data = list(read_rows(source))
metrics = list(read_rows(folder / (prefix + "FOLD_METRICS.csv")))

# The earlier independent metadata audit established that
# row_id is the zero-based position in the source CSV.
eligible = {r["Test_block"] for r in metrics}

if len(eligible) != 37:
    raise SystemExit("FAIL: Expected 37 distinct eligible blocks.")

def neighbors(block):
    lat, lon = map(int, block.split("_"))

    candidates = set()

    # Latitude: noncyclic.
    for new_lat in (lat - 1, lat + 1):
        if 0 <= new_lat < 12:
            candidates.add(f"{new_lat}_{lon}")

    # Longitude: 12 cyclic columns of 30 degrees each.
    candidates.add(f"{lat}_{(lon - 1) % 12}")
    candidates.add(f"{lat}_{(lon + 1) % 12}")

    return candidates & eligible

fold_info = {}
errors = []

print("=== RECONSTRUCTED BUFFER NEIGHBORS ===")

for r in metrics:
    fold = r["Fold"]
    block = r["Test_block"]
    buffered = neighbors(block)

    fold_info[fold] = {
        "test": block,
        "buffered": buffered,
        "train_expected": int(r["Train_records"]),
        "test_expected": int(r["Test_records"])
    }

    reported = int(r["Buffered_neighbors"])

    if len(buffered) != reported:
        errors.append(
            f"Fold {fold}: reported {reported} neighbors, "
            f"reconstructed {len(buffered)}"
        )

    print(
        f"Fold {int(fold):02d} "
        f"test={block:5s} "
        f"neighbors={','.join(sorted(buffered)) or 'none'}"
    )

print("\n=== VERIFYING MANIFEST SPATIAL ASSIGNMENTS ===")

counts = defaultdict(lambda: defaultdict(int))
violations = defaultdict(int)
manifest_count = 0

for r in read_rows(folder / (prefix + "MANIFEST.csv")):
    manifest_count += 1

    fold = r["Fold"]
    role = r["Role"].strip().lower()

    if fold not in fold_info:
        errors.append(f"Unknown fold: {fold}")
        continue

    try:
        idx = int(r["row_id"])
        source_row = data[idx]
    except (ValueError, IndexError):
        errors.append(f"Invalid source row ID: {r['row_id']}")
        continue

    block = source_row["block"]
    info = fold_info[fold]

    if r["Test_block"] != info["test"]:
        errors.append(f"Manifest test-block mismatch in fold {fold}")

    counts[fold][role] += 1

    if role == "test":
        if block != info["test"]:
            violations[fold] += 1

    elif role == "train":
        if block == info["test"] or block in info["buffered"]:
            violations[fold] += 1

    else:
        errors.append(f"Unexpected role: {role}")

for fold, info in fold_info.items():
    if counts[fold]["train"] != info["train_expected"]:
        errors.append(f"Fold {fold}: training count mismatch")

    if counts[fold]["test"] != info["test_expected"]:
        errors.append(f"Fold {fold}: test count mismatch")

    if violations[fold]:
        errors.append(
            f"Fold {fold}: {violations[fold]} spatial violations"
        )

print("Manifest assignments checked:", manifest_count)
print("Spatial violations:", sum(violations.values()))
print("Buffer count mismatches:", sum(
    "neighbors" in e for e in errors
))

print("\n=== FINAL RESULT ===")

if errors:
    print("OVERALL: FAIL")
    for error in errors[:30]:
        print(" -", error)
    print("Total issues:", len(errors))
    raise SystemExit(1)

print("All 37 reconstructed buffer counts match.")
print("All test rows belong to their designated blocks.")
print("No training rows occur in test or buffered blocks.")
print("OVERALL: PASS")
