from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import json

base = Path("01_Data/external")
base.mkdir(parents=True, exist_ok=True)

docx_path = base / "Dotternhausen_Zenodo_Table_1.docx"
txt_path = base / "Dotternhausen_Table_1_extracted.txt"

url = "https://zenodo.org/api/records/4778565/files/Table%201.docx/content"

print("=" * 74)
print("EXPERIMENT 76: DOTTERNHAUSEN ORIGINAL TABLE EXTRACTION")
print("=" * 74)

if not docx_path.exists():
    request = Request(url, headers={"User-Agent": "DSMS-NI-research-audit/1.0"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    docx_path.write_bytes(data)

print("\nDownloaded document:", docx_path)
print("File size:", docx_path.stat().st_size, "bytes")

ns = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}

lines = []

with ZipFile(docx_path) as archive:
    root = ET.fromstring(archive.read("word/document.xml"))

    body = root.find("w:body", ns)

    for element in body:
        tag = element.tag.split("}")[-1]

        if tag == "p":
            text = "".join(
                node.text or ""
                for node in element.findall(".//w:t", ns)
            ).strip()

            if text:
                lines.append(text)

        elif tag == "tbl":
            lines.append("\n[TABLE START]")

            for row in element.findall("./w:tr", ns):
                cells = []

                for cell in row.findall("./w:tc", ns):
                    paragraphs = []

                    for paragraph in cell.findall(".//w:p", ns):
                        value = "".join(
                            node.text or ""
                            for node in paragraph.findall(".//w:t", ns)
                        )
                        if value:
                            paragraphs.append(value)

                    cells.append(" ".join(paragraphs).strip())

                lines.append(" | ".join(cells))

            lines.append("[TABLE END]\n")

text = "\n".join(lines)
txt_path.write_text(text, encoding="utf-8")

print("\nEXTRACTED DOCUMENT CONTENT")
print("-" * 74)
print(text[:18000])

print("\nOUTPUT TEXT:", txt_path)
print("TOTAL EXTRACTED LINES:", len(lines))
print("\nOriginal document preserved. No DSMS-NI records modified.")
print("EXPERIMENT 76 COMPLETED")
