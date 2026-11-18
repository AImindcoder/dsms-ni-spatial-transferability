from pathlib import Path
from collections import Counter
import csv
import re
import pymupdf

root = Path(".")
pdf_path = root / "01_Data/external/Wang_2021_PhD_dissertation_original.pdf"
out_dir = root / "01_Data/interim"
out_dir.mkdir(parents=True, exist_ok=True)

doc = pymupdf.open(pdf_path)

columns = [
    "Depth_cm", "d13Ccarb", "d13Corg", "Delta13Ccarb_org",
    "d18O", "TOCcf", "CaCO3", "DICtotal", "DICorg",
    "DICorg_DICtotal"
]

centers = [58, 123, 179, 232, 279, 329, 370, 416, 460, 503]
boundaries = [(centers[i] + centers[i + 1]) / 2 for i in range(9)]

def parse_number(value):
    value = value.strip().replace(",", "")
    value = value.replace("−", "-").replace("–", "-")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return float(value)
    return None

def get_column(x):
    for i, boundary in enumerate(boundaries):
        if x < boundary:
            return i
    return 9

records = []

for page_no in [105, 106, 107]:
    page = doc[page_no]
    words = page.get_text("words")

    header_words = [
        w for w in words
        if w[4].strip() == "Depth" and w[0] < 100
    ]
    if not header_words:
        raise RuntimeError(f"Missing table header on PDF page {page_no}")

    header_y = min(w[1] for w in header_words)

    depth_words = []
    for w in words:
        depth = parse_number(w[4])
        if (
            w[0] < 80 and depth is not None
            and 600 <= depth <= 1100
            and w[1] > header_y + 8
        ):
            depth_words.append(w)

    for depth_word in depth_words:
        depth = parse_number(depth_word[4])
        row_y = (depth_word[1] + depth_word[3]) / 2
        row = [None] * 10
        row[0] = depth

        for w in words:
            if w[0] <= 80:
                continue

            word_y = (w[1] + w[3]) / 2
            if abs(word_y - row_y) > 2.5:
                continue

            value = parse_number(w[4])
            if value is None:
                continue

            col = get_column((w[0] + w[2]) / 2)
            if row[col] is not None:
                raise RuntimeError(
                    f"Column collision: page={page_no}, depth={depth}, "
                    f"column={columns[col]}"
                )
            row[col] = value

        records.append({
            "Observation_ID": f"DOT_T2_P{page_no}_{len(records)+1:03d}",
            **dict(zip(columns, row)),
            "PDF_page_index": page_no,
            "PDF_row_y": round(row_y, 3),
            "Extraction_method": "PyMuPDF word coordinates; 10-column layout",
            "Source_PDF": pdf_path.name
        })

expected_by_page = {105: 43, 106: 43, 107: 12}
actual_by_page = Counter(r["PDF_page_index"] for r in records)

if len(records) != 98 or dict(actual_by_page) != expected_by_page:
    raise RuntimeError(
        f"Unexpected row counts: total={len(records)}, pages={dict(actual_by_page)}"
    )

depth_counts = Counter(r["Depth_cm"] for r in records)
duplicates = {
    depth: count for depth, count in sorted(depth_counts.items())
    if count > 1
}

if duplicates != {869.0: 2, 993.0: 2}:
    raise RuntimeError(f"Unexpected repeated-depth pattern: {duplicates}")

csv_path = out_dir / "dotternhausen_table2_coordinate_reconstruction.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

audit_path = out_dir / "dotternhausen_table2_reconstruction_audit.txt"

with audit_path.open("w", encoding="utf-8") as f:
    f.write("DOTTERNHAUSEN TABLE 2 — RECONSTRUCTION AUDIT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Source PDF: {pdf_path}\n")
    f.write(f"Total observations: {len(records)}\n")
    f.write(f"Rows by PDF page: {dict(actual_by_page)}\n")
    f.write(f"Unique depths: {len(depth_counts)}\n")
    f.write(f"Repeated depths: {duplicates}\n\n")

    f.write("Missing measurements by column:\n")
    for col in columns:
        missing = sum(r[col] is None for r in records)
        f.write(f"{col}: {missing}/{len(records)}\n")

    f.write("\nRepeated-depth observations:\n")
    for depth in duplicates:
        for r in records:
            if r["Depth_cm"] == depth:
                f.write(
                    f"Depth={depth}; ID={r['Observation_ID']}; "
                    f"d13Corg={r['d13Corg']}; TOCcf={r['TOCcf']}\n"
                )

    f.write("\nOriginal PDF and DSMS-NI raw data were not modified.\n")
    f.write("This is a derived reconstruction, not an independently "
            "verified sample-identity crosswalk.\n")

print("=" * 72)
print("EXPERIMENT 92 COMPLETED")
print("=" * 72)
print("Rows saved:", len(records))
print("Rows by PDF page:", dict(actual_by_page))
print("Unique depths:", len(depth_counts))
print("Repeated depths:", duplicates)
print("CSV:", csv_path.resolve())
print("Audit:", audit_path.resolve())
print("\nMissing values:")
for col in columns:
    missing = sum(r[col] is None for r in records)
    print(f"{col}: {missing}")
print("\nOriginal source data unchanged.")
