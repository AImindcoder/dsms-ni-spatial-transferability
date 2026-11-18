from pathlib import Path
import fitz
import re

root = Path(r"D:\Research\DSMS_NI_Research")
pdf_path = root / "01_Data" / "external" / "Wang_2021_PhD_dissertation_original.pdf"
out_path = root / "01_Data" / "interim" / "96_dissertation_table_provenance_context.txt"

doc = fitz.open(pdf_path)

lines = []
lines.append("EXPERIMENT 96: DISSERTATION TABLE PROVENANCE CONTEXT")
lines.append("=" * 78)
lines.append(f"PDF: {pdf_path.name}")
lines.append(f"Page count: {len(doc)}")
lines.append(f"Metadata: {doc.metadata}")
lines.append("")

patterns = re.compile(
    r"table\s*[0-9]+|chapter\s*[0-9]+|"
    r"dotternhausen|2020pa004207|"
    r"nitrogen|isotope|sampling|analytical methods|"
    r"materials and methods|references|acknowledg",
    re.IGNORECASE
)

for page_index in range(94, min(111, len(doc))):
    page = doc[page_index]
    text = page.get_text("text")
    page_lines = [x.strip() for x in text.splitlines() if x.strip()]

    lines.append("")
    lines.append("=" * 78)
    lines.append(f"PDF PAGE INDEX {page_index}")
    lines.append("=" * 78)

    # Print the beginning of each page to preserve its immediate context.
    lines.extend(page_lines[:18])

    # Print matching lines and nearby context.
    matching = [
        i for i, line in enumerate(page_lines)
        if patterns.search(line)
    ]

    lines.append("")
    lines.append("--- CONTEXTUAL MATCHES ---")

    shown = set()
    for i in matching:
        for j in range(max(0, i - 1), min(len(page_lines), i + 3)):
            if j not in shown:
                lines.append(f"[line {j+1}] {page_lines[j]}")
                shown.add(j)

report = "\n".join(lines)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(report, encoding="utf-8")

print(report)
print("")
print(f"Saved report: {out_path}")
print("EXPERIMENT 96 COMPLETED — original PDF unchanged.")
