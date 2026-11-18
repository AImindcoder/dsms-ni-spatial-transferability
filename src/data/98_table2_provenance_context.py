from pathlib import Path
import fitz

pdf_path = Path("01_Data/external/Wang_2021_PhD_dissertation_original.pdf")
doc = fitz.open(pdf_path)

out_dir = Path("01_Data/interim")
out_file = out_dir / "98_table2_provenance_context.txt"

lines = []
lines.append("EXPERIMENT 98: TABLE 2 PROVENANCE CONTEXT")
lines.append("=" * 78)
lines.append(f"PDF pages: {len(doc)}")

# Inspect the pages surrounding the original Table 2.
for page_no in range(101, 111):
    if page_no >= len(doc):
        continue
    page = doc[page_no]
    lines.append("\n" + "=" * 78)
    lines.append(f"PDF PAGE INDEX {page_no} | PRINTED PAGE APPROX. {page_no + 592}")
    lines.append("=" * 78)
    lines.append(page.get_text("text", sort=True))

# Search the dissertation for references to Table 2 and its data.
lines.append("\n" + "=" * 78)
lines.append("ALL TABLE 2 REFERENCES IN DISSERTATION")
lines.append("=" * 78)

for page_no, page in enumerate(doc):
    text = page.get_text("text", sort=True)
    text_lines = text.splitlines()
    hits = [
        i for i, line in enumerate(text_lines)
        if "table 2" in line.lower()
        or "table2" in line.lower()
    ]
    if hits:
        lines.append(f"\n--- PDF PAGE INDEX {page_no} ---")
        shown = set()
        for i in hits:
            for j in range(max(0, i-3), min(len(text_lines), i+5)):
                if j not in shown:
                    lines.append(text_lines[j])
                    shown.add(j)

out_file.write_text("\n".join(lines), encoding="utf-8")

print("Experiment 98 completed.")
print("Output:", out_file)
print("Pages inspected: PDF indices 101–110")
print("The original PDF was not modified.")
