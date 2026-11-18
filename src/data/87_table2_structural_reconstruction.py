from pathlib import Path
import re
from collections import Counter

path = Path("01_Data/external/83_dotternhausen_original_table2_extracted.txt")
lines = path.read_text(encoding="utf-8").splitlines()

def parse_cell(line):
    value = line.strip().replace("\u2003", "").replace("\xa0", "")
    value = value.replace("−", "-").replace("–", "-")
    value = value.strip()

    if value in {"—", "-", "–"}:
        return None, "missing"

    value = value.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return float(value), "numeric"

    return None, "ignore"

headers = [
    i for i, line in enumerate(lines)
    if "Depth [cm]" in line
]

print("=" * 76)
print("EXPERIMENT 87: TABLE 2 STRUCTURAL RECONSTRUCTION AUDIT")
print("=" * 76)

all_rows = []

for section_no, header in enumerate(headers, 1):
    end = headers[section_no] if section_no < len(headers) else len(lines)
    cells = []

    for i in range(header + 1, end):
        line = lines[i].strip()

        if "(Continues)" in line or "TA B LE 2" in line:
            break

        value, status = parse_cell(line)
        if status != "ignore":
            cells.append((value, status, i + 1, line))

    remainder = len(cells) % 10
    complete_count = len(cells) // 10

    print(f"\nSECTION {section_no}")
    print("-" * 50)
    print("Header line:", header + 1)
    print("Extracted data cells:", len(cells))
    print("Complete 10-column rows:", complete_count)
    print("Unassigned trailing cells:", remainder)

    if remainder:
        print("TRAILING CELLS — MUST INVESTIGATE:")
        for item in cells[-remainder:]:
            print(item[2], repr(item[3]))

    section_rows = []
    for j in range(complete_count):
        group = cells[j * 10:(j + 1) * 10]
        row = [item[0] for item in group]
        depth = row[0]

        if depth is None or not (0 <= depth <= 2000):
            print("INVALID DEPTH GROUP:", group)
            continue

        section_rows.append(row)

    all_rows.extend(section_rows)

    print("Valid reconstructed rows:", len(section_rows))
    if section_rows:
        print("First reconstructed row:", section_rows[0])
        print("Last reconstructed row:", section_rows[-1])

print("\n" + "=" * 76)
print("COMBINED STRUCTURAL SUMMARY")
print("=" * 76)

depths = [row[0] for row in all_rows]
counts = Counter(depths)

print("Total reconstructed rows:", len(all_rows))
print("Unique depths:", len(counts))
print("Repeated depth values:", sum(1 for n in counts.values() if n > 1))
print("Repeated depth frequencies:",
      {k: v for k, v in sorted(counts.items()) if v > 1})

print("\nMissing values by column:")
column_names = [
    "Depth",
    "δ13Ccarb",
    "δ13Corg",
    "Δ13Ccarb-org",
    "δ18O",
    "TOCcf",
    "CaCO3",
    "DICtotal",
    "DICorg",
    "DICorg/DICtotal"
]

for col, name in enumerate(column_names):
    missing = sum(row[col] is None for row in all_rows)
    print(f"{name}: {missing}/{len(all_rows)}")

print("\nNo source data modified.")
print("No reconstructed data file created.")
print("EXPERIMENT 87 COMPLETED")
