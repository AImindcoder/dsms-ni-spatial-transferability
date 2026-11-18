from pathlib import Path
import fitz
import re

pdf_path = Path(r"D:\Research\DSMS_NI_Research\01_Data\external\Wang_2021_PhD_dissertation_original.pdf")
doc = fitz.open(pdf_path)

print("=" * 78)
print("EXPERIMENT 97: IDENTIFY THE SOURCE PUBLICATION")
print("=" * 78)

for i in list(range(0, min(12, len(doc)))) + list(range(86, min(96, len(doc)))):
    text = doc[i].get_text("text")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    print(f"\n--- PDF PAGE INDEX {i} ---")
    for line in lines[:35]:
        print(line)

    for line in lines:
        if re.search(r"\bdoi\b|https?://doi\.org|10\.\d{4,9}/", line, re.I):
            print("POSSIBLE DOI/TITLE:", line)

print("\nEXPERIMENT 97 COMPLETED — PDF unchanged.")
