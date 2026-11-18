from pathlib import Path
import fitz

pdf_path = Path("01_Data/external/Wang_2021_PhD_dissertation_original.pdf")
out_dir = Path("01_Data/interim/97_table2_visual_verification")
out_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)

print("EXPERIMENT 97: ORIGINAL TABLE 2 VISUAL VERIFICATION")
print("PDF pages:", len(doc))

for page_number in [105, 106, 107]:
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
    output = out_dir / f"Table2_original_page_{page_number + 1}.png"
    pix.save(output)
    print(f"Saved: {output} | dimensions: {pix.width} x {pix.height}")

print("\nOpen these three PNG files and upload them here.")
print("We will visually verify the original printed values against our reconstruction.")
print("No source files modified.")
