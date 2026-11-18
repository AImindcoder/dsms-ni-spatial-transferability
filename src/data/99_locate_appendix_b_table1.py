import fitz
import re
from pathlib import Path

pdf = Path(r"01_Data\external\Wang_2021_PhD_dissertation_original.pdf")
doc = fitz.open(pdf)

print("SEARCHING APPENDIX B FOR THE ORIGINAL NITROGEN ISOTOPE TABLE")
print("=" * 75)

for i in range(117, len(doc)):
    text = doc[i].get_text("text")
    lower = text.lower()

    sample_ids = sorted(set(re.findall(r"\bDH[-–]\s*\d+(?:\.\d+)?\b", text)))
    relevant = (
        "table 1" in lower
        or "δ15nbulk" in lower
        or "δ15nker" in lower
        or "d15nbulk" in lower
        or "d15nker" in lower
        or len(sample_ids) >= 3
    )

    if relevant:
        print(f"\nPDF PAGE INDEX {i} | PRINTED PAGE APPROX. {i - 117}")
        print("Sample IDs:", sample_ids[:60])
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        for j, line in enumerate(lines):
            if (
                "table 1" in line.lower()
                or "δ15n" in line.lower()
                or "DH-" in line
                or "DH–" in line
                or "Depth" in line
            ):
                start = max(0, j - 2)
                end = min(len(lines), j + 5)
                print(" | ".join(lines[start:end])[:1200])

print("\nSCAN COMPLETE")
