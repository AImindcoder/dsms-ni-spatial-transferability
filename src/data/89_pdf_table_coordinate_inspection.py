from pathlib import Path
import pymupdf

pdf_path = Path("01_Data/external/Wang_2021_PhD_dissertation_original.pdf")
doc = pymupdf.open(pdf_path)

print("=" * 76)
print("EXPERIMENT 89: ORIGINAL PDF TABLE COORDINATE INSPECTION")
print("=" * 76)

for page_no, page in enumerate(doc):
    words = page.get_text("words")
    page_text = page.get_text()

    if "Depth [cm]" not in page_text:
        continue

    print(f"\nPDF PAGE INDEX: {page_no} | SIZE: {page.rect}")

    for target in ["788", "796", "801", "805", "815", "949"]:
        matches = [
            w for w in words
            if w[4].replace(",", "") == target
        ]

        for match in matches:
            x0, y0, x1, y1 = match[:4]
            print(f"\nTARGET DEPTH {target} at x={x0:.1f}, y={y0:.1f}")

            nearby = [
                w for w in words
                if abs(((w[1] + w[3]) / 2) - ((y0 + y1) / 2)) < 12
            ]

            nearby.sort(key=lambda w: w[0])
            for w in nearby:
                print(
                    f"x={w[0]:7.1f} to {w[2]:7.1f} | "
                    f"{w[4]!r}"
                )

print("\nPDF TOTAL PAGES:", len(doc))
print("EXPERIMENT 89 COMPLETED")
