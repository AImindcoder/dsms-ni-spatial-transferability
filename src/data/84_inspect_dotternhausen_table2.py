from pathlib import Path
import re

path = Path("01_Data/external/83_dotternhausen_original_table2_extracted.txt")
text = path.read_text(encoding="utf-8")

lines = text.splitlines()

print("=" * 76)
print("EXPERIMENT 84: ORIGINAL TABLE 2 STRUCTURE INSPECTION")
print("=" * 76)

headers = [
    i for i, line in enumerate(lines)
    if "Depth [cm]" in line
]

print("\nTABLE HEADER LOCATIONS:", headers)

for header in headers:
    print(f"\n--- HEADER AT LINE {header + 1} ---")
    for i in range(header, min(header + 18, len(lines))):
        print(f"{i + 1:04d}: {lines[i]}")

print("\n--- NUMERIC ROW CANDIDATES ---")

candidate_count = 0

for i, line in enumerate(lines):
    cleaned = line.strip().replace("−", "-").replace("–", "-")
    cleaned = cleaned.replace(",", "")

    tokens = cleaned.split()

    if not tokens:
        continue

    try:
        depth = float(tokens[0])
    except ValueError:
        continue

    if not (0 <= depth <= 2000):
        continue

    numeric_count = 0
    for token in tokens[1:]:
        try:
            float(token)
            numeric_count += 1
        except ValueError:
            pass

    if numeric_count >= 7:
        candidate_count += 1
        if candidate_count <= 20:
            print(
                f"Line {i + 1}: depth={depth}, "
                f"numeric_fields={numeric_count}, "
                f"content={line.strip()}"
            )

print("\nCandidate row count (minimum 7 numeric fields):", candidate_count)
print("Total extracted text lines:", len(lines))
print("\nNo source data modified.")
print("EXPERIMENT 84 COMPLETED")
