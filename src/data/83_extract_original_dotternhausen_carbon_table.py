from pathlib import Path
import urllib.request
import subprocess
import sys

try:
    import fitz
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "pymupdf"
    ])
    import fitz

folder = Path("01_Data/external")
folder.mkdir(parents=True, exist_ok=True)

pdf_path = folder / "Wang_2021_PhD_dissertation_original.pdf"
url = (
    "https://publikationen.uni-tuebingen.de/xmlui/bitstream/"
    "10900/119509/4/PhD%20dissertation_Wang%20Yunfeng.pdf"
)

if not pdf_path.exists():
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        pdf_path.write_bytes(response.read())

doc = fitz.open(pdf_path)

matches = []
for page_number, page in enumerate(doc):
    text = page.get_text("text")
    normalized = " ".join(text.lower().split())

    if (
        "stable isotope and toc" in normalized
        and "dotternhausen samples" in normalized
    ):
        matches.append(page_number)

if not matches:
    raise RuntimeError(
        "Could not locate the original Table 2 automatically. "
        "PDF downloaded, but table extraction needs manual inspection."
    )

pages_to_extract = sorted(set(
    p
    for start in matches
    for p in range(start, min(start + 3, len(doc)))
))

output_text = []
print("=" * 76)
print("EXPERIMENT 83: ORIGINAL DOTTERNHAUSEN CARBON TABLE EXTRACTION")
print("=" * 76)

print("\nPDF:", pdf_path)
print("PDF pages containing table heading:", [p + 1 for p in matches])
print("Pages extracted:", [p + 1 for p in pages_to_extract])

for page_number in pages_to_extract:
    page_text = doc[page_number].get_text("text")
    output_text.append(
        f"\n\n===== PDF PAGE {page_number + 1} =====\n\n{page_text}"
    )
    print(f"\n{'=' * 24} PDF PAGE {page_number + 1} {'=' * 24}")
    print(page_text[:12000])

text_path = folder / "83_dotternhausen_original_table2_extracted.txt"
text_path.write_text("".join(output_text), encoding="utf-8")

print("\nEXTRACTED TEXT SAVED:", text_path)
print("No source measurements or DSMS-NI records modified.")
print("EXPERIMENT 83 EXTRACTION COMPLETED")
