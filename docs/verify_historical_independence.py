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
manifest = folder / (prefix + "MANIFEST.csv")
oof = list(read_rows(folder / (prefix + "OOF.csv")))

print("=== DATASET IDENTIFIER INSPECTION ===")
print("Source rows:", len(data))
print("Source columns:", list(data[0]) if data else [])
print("First source record:", data[0] if data else None)

if not data:
    raise SystemExit("FAIL: Source dataset is empty.")

# Determine the row-ID mapping using the OOF observations.
# Prefer an explicit unique row_id field if one exists.
if "row_id" in data[0]:
    keys = [str(r["row_id"]) for r in data]
    if len(keys) != len(set(keys)):
        raise SystemExit("FAIL: Source row_id values are not unique.")
    lookup = dict(zip(keys, data))
    mapping = "explicit row_id"
else:
    # Test both common positional conventions against OOF metadata.
    candidates = {}

    for offset in (0, 1):
        lookup_candidate = {
            str(i + offset): r for i, r in enumerate(data)
        }

        matches = 0
        checked = 0

        for r in oof[:min(1000, len(oof))]:
            source_row = lookup_candidate.get(r["row_id"])
            if source_row is None:
                continue

            checked += 1
            if (
                source_row.get("sequence") == r["sequence"]
                and source_row.get("doi_key") == r["doi_key"]
            ):
                matches += 1

        candidates[offset] = (matches, checked)

    print("Positional mapping candidates:", candidates)

    valid = [
        offset for offset, (matches, checked) in candidates.items()
        if checked > 0 and matches == checked
    ]

    if len(valid) != 1:
        raise SystemExit(
            "STOP: Cannot uniquely establish source row-ID mapping."
        )

    offset = valid[0]
    lookup = {
        str(i + offset): r for i, r in enumerate(data)
    }
    mapping = f"positional index with offset {offset}"

print("Selected mapping:", mapping)

# Validate every OOF record against its source metadata.
for r in oof:
    source_row = lookup.get(r["row_id"])

    if source_row is None:
        raise SystemExit(f"FAIL: Missing OOF row_id {r['row_id']}")

    for field in ("sequence", "doi_key", "block"):
        expected = (
            r["Test_block"] if field == "block" else r[field]
        )

        if source_row.get(field) != expected:
            raise SystemExit(
                f"FAIL: Source/OOF mismatch for row "
                f"{r['row_id']}, field {field}"
            )

print("OOF source metadata: PASS")

train_sequences = defaultdict(set)
test_sequences = defaultdict(set)
train_publications = defaultdict(set)
test_publications = defaultdict(set)

count = 0

for r in read_rows(manifest):
    count += 1
    source_row = lookup.get(r["row_id"])

    if source_row is None:
        raise SystemExit(
            f"FAIL: Manifest row_id {r['row_id']} missing in source"
        )

    fold = r["Fold"]
    role = r["Role"].strip().lower()
    seq = source_row["sequence"]
    pub = source_row["doi_key"]

    if role == "train":
        train_sequences[fold].add(seq)
        train_publications[fold].add(pub)
    elif role == "test":
        test_sequences[fold].add(seq)
        test_publications[fold].add(pub)
    else:
        raise SystemExit(f"FAIL: Unexpected role {role}")

print("\n=== INDEPENDENT EXCLUSION VERIFICATION ===")
print("Manifest entries checked:", count)

failures = []

for fold in sorted(
    set(train_sequences) | set(test_sequences),
    key=int
):
    shared_seq = train_sequences[fold] & test_sequences[fold]
    shared_pub = train_publications[fold] & test_publications[fold]

    print(
        f"Fold {int(fold):02d}: "
        f"shared sequences={len(shared_seq)}, "
        f"shared publications={len(shared_pub)}"
    )

    if shared_seq or shared_pub:
        failures.append(fold)

if failures:
    print("OVERALL: FAIL")
    raise SystemExit(1)

print("\nOVERALL: PASS")
print("Publication and sequence exclusions independently verified.")
