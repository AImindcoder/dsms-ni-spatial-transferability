from pathlib import Path
from collections import Counter
import re
import pymupdf

pdf = pymupdf.open("01_Data/external/Wang_2021_PhD_dissertation_original.pdf")

centers = [58, 123, 179, 232, 279, 329, 370, 416, 460, 503]
boundaries = [(centers[i] + centers[i+1]) / 2 for i in range(9)]

def parse_number(s):
    s = s.strip().replace(",", "").replace("−", "-").replace("–", "-")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return float(s)
    return None

targets = {788, 796, 801, 805, 815, 869, 993}

print("=" * 76)
print("EXPERIMENT 91: MISSING-CELL AND REPEATED-DEPTH VERIFICATION")
print("=" * 76)

for page_no in [105, 106, 107]:
    page = pdf[page_no]
    words = page.get_text("words")

    print(f"\nPDF PAGE INDEX {page_no}")

    # Show original header positions for column calibration.
    for label in ["Depth", "δ13Ccarb", "δ13Corg", "Δ13Ccarb-", "δ18O",
                  "TOCcf", "CaCO3", "[DIC]total", "[DIC]org"]:
        matches = [w for w in words if w[4].strip() == label]
        if matches:
            w = min(matches, key=lambda x: x[1])
            print(f"HEADER {label!r}: x={w[0]:.1f}, y={w[1]:.1f}")

    for w in words:
        depth = parse_number(w[4])
        if depth not in targets or w[0] >= 80:
            continue

        y = (w[1] + w[3]) / 2
        print(f"\nTARGET DEPTH {depth:g} cm | y={y:.2f}")

        nearby = [
            item for item in words
            if abs(((item[1] + item[3]) / 2) - y) <= 2.5
            and item[0] > 80
        ]
        nearby.sort(key=lambda item: item[0])

        for item in nearby:
            x = (item[0] + item[2]) / 2
            col = next((i for i, b in enumerate(boundaries) if x < b), 9)
            print(f"column={col:2d} | x={x:7.2f} | {item[4]!r}")

print("\nEXPERIMENT 91 COMPLETED")
