from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np

docx_path = Path("01_Data/external/Dotternhausen_Zenodo_Table_1.docx")
raw_path = Path("01_Data/raw/DSMSNI_v0.3.csv")
out_dir = Path("02_Data_Audit/provenance_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("EXPERIMENT 77: CORRECTED DOTTERNHAUSEN SOURCE VERIFICATION")
print("=" * 76)

ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with ZipFile(docx_path) as archive:
    xml_root = ET.fromstring(archive.read("word/document.xml"))

tables = xml_root.findall(".//w:tbl", ns)
if not tables:
    raise RuntimeError("No Word table found.")

table_rows = []

for row in tables[0].findall("./w:tr", ns):
    cells = []
    for cell in row.findall("./w:tc", ns):
        value = "".join(
            node.text or ""
            for node in cell.findall(".//w:t", ns)
        ).strip()
        cells.append(value)
    table_rows.append(cells)

header_index = None

for i, row in enumerate(table_rows):
    if any("Depth" in cell and "cm" in cell for cell in row):
        header_index = i
        break

if header_index is None:
    print("Extracted rows for diagnosis:")
    for row in table_rows[:8]:
        print(row)
    raise RuntimeError("Could not identify the sample data header.")

headers = [x.strip() for x in table_rows[header_index]]
print("\nIDENTIFIED SOURCE HEADERS")
for i, header in enumerate(headers):
    print(i, repr(header))

source = pd.DataFrame(
    table_rows[header_index + 1:],
    columns=headers
)

depth_col = next(c for c in source.columns if "Depth" in c)
bulk_col = next(
    c for c in source.columns
    if "15Nbulk" in c.replace("δ", "").replace(" ", "")
)

source = source.rename(columns={
    depth_col: "Source_Depth_cm",
    bulk_col: "Source_d15Nbulk"
})

source["Source_Depth_cm"] = pd.to_numeric(
    source["Source_Depth_cm"], errors="coerce"
)
source["Source_d15Nbulk"] = pd.to_numeric(
    source["Source_d15Nbulk"], errors="coerce"
)

source = source.dropna(
    subset=["Source_Depth_cm", "Source_d15Nbulk"]
).copy()

# Load original DSMS-NI CSV without modifying it.
df = pd.read_csv(raw_path, encoding="gbk", low_memory=False)

doi_col = next(c for c in df.columns if c.lower() == "doi")
site_col = next(c for c in df.columns if c.lower() == "sitename")
target_col = next(c for c in df.columns if "15Nbulk" in c)

db = df[
    df[doi_col].astype(str).str.strip().str.lower().eq(
        "10.1029/2020pa004207"
    )
    & df[site_col].astype(str).str.contains(
        "dotternhausen", case=False, na=False
    )
].copy()

db[target_col] = pd.to_numeric(db[target_col], errors="coerce")
db["RelativeDepth"] = pd.to_numeric(
    db["RelativeDepth"], errors="coerce"
)

print("\nRECORD COUNTS")
print("Original source rows:", len(source))
print("DSMS-NI records:", len(db))
print("Source numeric isotope values:", source["Source_d15Nbulk"].count())
print("DSMS-NI numeric isotope values:", db[target_col].count())

# Compare isotope values as multisets, without assuming row order.
a = np.sort(source["Source_d15Nbulk"].dropna().to_numpy())
b = np.sort(db[target_col].dropna().to_numpy())

print("\nISOTOPE COMPARISON")
if len(a) == len(b):
    differences = np.abs(a - b)
    print("Same numeric count:", True)
    print("Exact sorted agreement (<1e-6):", bool(np.all(differences < 1e-6)))
    print("Median absolute difference:", float(np.median(differences)))
    print("Maximum absolute difference:", float(np.max(differences)))
else:
    print("Same numeric count:", False)
    print("Source values:", len(a), "Database values:", len(b))

# Compare depth using sorted source and database rows.
source_sorted = source.sort_values("Source_Depth_cm").reset_index(drop=True)
db_sorted = db.sort_values("RelativeDepth").reset_index(drop=True)

print("\nDEPTH COMPARISON")
if len(source_sorted) == len(db_sorted):
    sd = source_sorted["Source_Depth_cm"].to_numpy()
    dd = db_sorted["RelativeDepth"].to_numpy()

    for name, candidate in {
        "source_cm": sd,
        "source_m": sd / 100,
        "negative_source_m": -sd / 100,
        "negative_source_cm": -sd
    }.items():
        print(
            name,
            "Pearson=", round(float(np.corrcoef(candidate, dd)[0, 1]), 6),
            "Spearman=", round(float(pd.Series(candidate).corr(
                pd.Series(dd), method="spearman"
            )), 6),
            "MAE=", round(float(np.mean(np.abs(candidate - dd))), 6)
        )
else:
    print(
        "Different row counts; depth comparison not performed.",
        len(source_sorted), len(db_sorted)
    )

# Save auditable extracts.
source.to_csv(
    out_dir / "77_dotternhausen_original_source_table.csv",
    index=False, encoding="utf-8-sig"
)

db.to_csv(
    out_dir / "77_dotternhausen_dsmsni_records.csv",
    index=False, encoding="utf-8-sig"
)

print("\nSOURCE ISOTOPE VALUES")
print(source_sorted[["Source_Depth_cm", "Source_d15Nbulk"]].to_string(index=False))

print("\nDATABASE RECORDS")
cols = [
    c for c in [
        "SampleID", "SampleName", "RelativeDepth",
        target_col, "Age", "Stage"
    ] if c in db.columns
]
print(db_sorted[cols].to_string(index=False))

print("\nOUTPUTS SAVED IN:", out_dir)
print("No raw records or age assignments modified.")
print("EXPERIMENT 77 COMPLETED")
