from pathlib import Path
import fitz
import re

pdf_path = Path(r"D:\Research\DSMS_NI_Research\01_Data\external\Wang_2021_PhD_dissertation_original.pdf")
doc = fitz.open(pdf_path)

print("=" * 78)
print("EXPERIMENT 98: LOCATE THE ORIGINAL NITROGEN-ISOTOPE DATA")
print("=" * 78)

keywords = re.compile(
    r"appendix\s*b|restricted oxygen-deficient|"
    r"10\.1029/2020pa004207|"
    r"nitrogen isotope|δ15n|δ¹⁵n|"
    r"table\s*[0-9]+.*nitrogen|"
    r"bulk.*kerogen|kerogen.*bulk",
    re.IGNORECASE
)

matches = []

for i, page in enumerate(doc):
    text = page.get_text("text")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    found = [j for j, line in enumerate(lines) if keywords.search(line)]

    if found:
        matches.append(i)
        print(f"\n--- PDF PAGE INDEX {i} ---")
        for j in found:
            for k in range(max(0, j-2), min(len(lines), j+5)):
                print(lines[k])

print("\n" + "=" * 78)
print("MATCHING PDF PAGE INDICES")
print(matches)
print("EXPERIMENT 98 COMPLETED — original PDF unchanged.")
