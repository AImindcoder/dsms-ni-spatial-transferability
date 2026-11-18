from pathlib import Path
from collections import Counter
import re
import pymupdf

pdf_path = Path("01_Data/external/Wang_2021_PhD_dissertation_original.pdf")
doc = pymupdf.open(pdf_path)

columns = [
    "Depth_cm",
    "d13Ccarb",
    "d13Corg",
    "Delta13Ccarb_org",
    "d18O",
    "TOCcf",
    "CaCO3",
    "DICtotal",
    "DICorg",
    "DICorg_DICtotal"
]

# Approximate horizontal centers, calibrated from Experiment 89.
centers = [58, 123, 179, 232, 279, 329, 370, 416, 460, 503]
boundaries = [
    (centers[i] + centers[i + 1]) / 2
    for i in range(len(centers) - 1)
]

def number(value):
    value = value.strip().replace(",", "")
    value = value.replace("−", "-").replace("–", "-")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None
    return float(value)

def column_index(x):
    for i, boundary in enumerate(boundaries):
        if x < boundary:
            return i
    return 9

print("=" * 76)
print("EXPERIMENT 90: COORDINATE-BASED TABLE 2 AUDIT")
print("=" * 76)

all_rows = []
anomalies = []
page_counts = Counter()

for page_no in [105, 106, 107]:
    page = doc[page_no]
    words = page.get_text("words")

    header_words = [
        w for w in words
        if w[4].strip() == "Depth" and w[0] < 100
    ]

    if not header_words:
        print(f"\nWARNING: No table header detected on PDF page index {page_no}")
        continue

    header_y = min(w[1] for w in header_words)

    depth_words = []
    for w in words:
        x0, y0, x1, y1, token = w[:5]
        depth = number(token)

        if (
            x0 < 80
            and depth is not None
            and 600 <= depth <= 1100
            and y0 > header_y + 8
        ):
            depth_words.append(w)

    page_rows = []

    for depth_word in depth_words:
        x0, y0, x1, y1, token = depth_word[:5]
        depth = number(token)
        row_y = (y0 + y1) / 2

        row = [None] * 10
        row[0] = depth
        assigned = {0}
        conflicts = []

        for w in words:
            wx0, wy0, wx1, wy1, value = w[:5]
            wy = (wy0 + wy1) / 2

            if abs(wy - row_y) > 2.5:
                continue

            if wx0 < 80:
                continue

            col = column_index((wx0 + wx1) / 2)
            parsed = number(value)

            if parsed is None:
                continue

            if col in assigned:
                conflicts.append((col, value, wx0, wy))
            else:
                row[col] = parsed
                assigned.add(col)

        if conflicts:
            anomalies.append((page_no, depth, conflicts))

        page_rows.append(row)
        all_rows.append(row)
        page_counts[page_no] += 1

    print(f"\nPDF PAGE INDEX {page_no}")
    print("Header y-coordinate:", round(header_y, 2))
    print("Depth candidates:", len(depth_words))
    print("Rows reconstructed:", len(page_rows))

    if page_rows:
        print("First row:", page_rows[0])
        print("Last row:", page_rows[-1])

print("\n" + "=" * 76)
print("COMBINED RESULTS")
print("=" * 76)

print("Total reconstructed rows:", len(all_rows))
print("Rows by PDF page:", dict(page_counts))

depth_counts = Counter(row[0] for row in all_rows)
duplicates = {
    depth: count for depth, count in sorted(depth_counts.items())
    if count > 1
}

print("Unique depths:", len(depth_counts))
print("Repeated depths:", duplicates)

print("\nMissing measurements by column:")
for i, name in enumerate(columns):
    missing = sum(row[i] is None for row in all_rows)
    print(f"{name}: {missing}/{len(all_rows)}")

print("\nColumn assignment conflicts:", len(anomalies))
for item in anomalies[:20]:
    print("CONFLICT:", item)

print("\nSelected rows for manual verification:")
for target in [788, 796, 801, 805, 815, 949, 993, 1003, 1012]:
    matches = [row for row in all_rows if row[0] == target]
    for row in matches:
        print(target, dict(zip(columns, row)))

print("\nNo output dataset created.")
print("No source values modified.")
print("EXPERIMENT 90 COMPLETED")
