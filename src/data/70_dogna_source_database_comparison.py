from pathlib import Path
from urllib.request import Request, urlopen
from io import StringIO
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
RAW = ROOT / "01_Data" / "raw" / "DSMSNI_v0.3.csv"
OUT = ROOT / "02_Data_Audit" / "provenance_analysis"
EXT = ROOT / "01_Data" / "external"

OUT.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)

DOI = "10.1029/2000PA000558"
URL = "https://doi.pangaea.de/10.1594/PANGAEA.846560?format=textfile"

def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def find_column(df, fragment):
    matches = [c for c in df.columns if fragment.lower() in str(c).lower()]
    if not matches:
        raise KeyError(f"Could not find column containing: {fragment}")
    return matches[0]

print("=" * 76)
print("EXPERIMENT 70: DOGNA ORIGINAL-DATASET COMPARISON")
print("=" * 76)

if not RAW.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW}")

# 1. Load DSMS-NI without modifying it.
ds = pd.read_csv(RAW, encoding="gbk", low_memory=False)

doi_col = find_column(ds, "DOI")
site_col = find_column(ds, "SiteName")
sample_col = find_column(ds, "SampleName")
depth_col = find_column(ds, "RelativeDepth")
target_col = find_column(ds, "15Nbulk")

doi_values = ds[doi_col].astype(str).str.lower()
doi_mask = doi_values.str.contains(re.escape(DOI.lower()), regex=True, na=False)
site_mask = ds[site_col].astype(str).str.strip().str.lower().eq("dogna")

database = ds.loc[doi_mask & site_mask].copy()

if database.empty:
    raise ValueError(
        "No DSMS-NI records matched both the Dogna site and target DOI. "
        "Inspect DOI and SiteName values before proceeding."
    )

database["_sample_key"] = database[sample_col].map(normalize)
database["_target_numeric"] = pd.to_numeric(
    database[target_col], errors="coerce"
)
database["_depth_numeric"] = pd.to_numeric(
    database[depth_col], errors="coerce"
)

# 2. Download and preserve the public PANGAEA source export.
request = Request(URL, headers={"User-Agent": "Mozilla/5.0 research audit"})
with urlopen(request, timeout=60) as response:
    source_bytes = response.read()

source_path = EXT / "Dogna_PANGAEA_846560_original_export.txt"
source_path.write_bytes(source_bytes)

source_text = source_bytes.decode("utf-8", errors="replace")
lines = source_text.splitlines()

header_index = None
for i, line in enumerate(lines):
    low = line.lower()
    if "sample label" in low and "section" in low and "15n" in low:
        header_index = i
        break

if header_index is None:
    raise ValueError(
        "Could not identify the PANGAEA table header. "
        f"Original download preserved at: {source_path}"
    )

source = pd.read_csv(
    StringIO("\n".join(lines[header_index:])),
    sep="\t",
    low_memory=False
)

source_sample_col = find_column(source, "Sample label")
source_depth_col = find_column(source, "Section")
source_target_col = find_column(source, "15N")

source["_sample_key"] = source[source_sample_col].map(normalize)
source["_source_target_numeric"] = pd.to_numeric(
    source[source_target_col], errors="coerce"
)
source["_source_depth_numeric"] = pd.to_numeric(
    source[source_depth_col], errors="coerce"
)

# 3. Identify ambiguous labels before joining.
db_counts = database["_sample_key"].value_counts()
src_counts = source["_sample_key"].value_counts()

db_unique = database[
    database["_sample_key"].ne("")
    & database["_sample_key"].map(db_counts).eq(1)
].copy()

src_unique = source[
    source["_sample_key"].ne("")
    & source["_sample_key"].map(src_counts).eq(1)
].copy()

matched = db_unique.merge(
    src_unique,
    on="_sample_key",
    how="inner",
    suffixes=("_DSMSNI", "_PANGAEA")
)

# 4. Compare isotope measurements and section positions.
matched["isotope_difference_DSMS_minus_source"] = (
    matched["_target_numeric"] - matched["_source_target_numeric"]
)
matched["depth_difference_DSMS_minus_source"] = (
    matched["_depth_numeric"] - matched["_source_depth_numeric"]
)

valid_iso = matched.dropna(
    subset=["_target_numeric", "_source_target_numeric"]
)
valid_depth = matched.dropna(
    subset=["_depth_numeric", "_source_depth_numeric"]
)

def correlation(a, b, method):
    if len(a) < 3:
        return np.nan
    return pd.Series(a).corr(pd.Series(b), method=method)

summary = []

def add(name, value):
    summary.append({"metric": name, "value": value})

add("DSMS-NI Dogna records", len(database))
add("PANGAEA source records", len(source))
add("Unique DSMS-NI sample labels", database["_sample_key"].nunique())
add("Unique PANGAEA sample labels", source["_sample_key"].nunique())
add("Unambiguous matched sample labels", len(matched))
add("Matched records with numeric isotope values in both sources", len(valid_iso))
add("Matched records with numeric depths in both sources", len(valid_depth))
add("DSMS-NI numeric isotope records", database["_target_numeric"].notna().sum())
add("PANGAEA numeric isotope records", source["_source_target_numeric"].notna().sum())

if len(valid_iso):
    diffs = valid_iso["isotope_difference_DSMS_minus_source"]
    add("Exact isotope matches (absolute difference <= 0.000001)", int((diffs.abs() <= 1e-6).sum()))
    add("Median absolute isotope difference", float(diffs.abs().median()))
    add("Maximum absolute isotope difference", float(diffs.abs().max()))
    add("Isotope Pearson correlation", correlation(
        valid_iso["_target_numeric"],
        valid_iso["_source_target_numeric"], "pearson"
    ))
    add("Isotope Spearman correlation", correlation(
        valid_iso["_target_numeric"],
        valid_iso["_source_target_numeric"], "spearman"
    ))

if len(valid_depth):
    add("Depth Pearson correlation", correlation(
        valid_depth["_depth_numeric"],
        valid_depth["_source_depth_numeric"], "pearson"
    ))
    add("Depth Spearman correlation", correlation(
        valid_depth["_depth_numeric"],
        valid_depth["_source_depth_numeric"], "spearman"
    ))
    add("Median absolute depth difference", float(
        valid_depth["depth_difference_DSMS_minus_source"].abs().median()
    ))
    add("Depth correlation after sign reversal", correlation(
        valid_depth["_depth_numeric"],
        -valid_depth["_source_depth_numeric"], "pearson"
    ))

# 5. Save detailed evidence and a readable summary.
matched_path = OUT / "70_dogna_sample_level_source_comparison.csv"
summary_path = OUT / "70_dogna_source_comparison_summary.csv"

matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
pd.DataFrame(summary).to_csv(
    summary_path, index=False, encoding="utf-8-sig"
)

print("\nDOGNA COMPARISON")
print("-" * 76)
print("DSMS-NI records:", len(database))
print("PANGAEA source records:", len(source))
print("Unambiguous sample-label matches:", len(matched))
print("Numeric isotope pairs:", len(valid_iso))
print("Numeric depth pairs:", len(valid_depth))

if len(valid_iso):
    print("Exact isotope matches:", int(
        (valid_iso["isotope_difference_DSMS_minus_source"].abs() <= 1e-6).sum()
    ))
    print("Median absolute isotope difference:", round(
        valid_iso["isotope_difference_DSMS_minus_source"].abs().median(), 8
    ))

if len(valid_depth):
    print("Depth Spearman correlation:", round(
        correlation(valid_depth["_depth_numeric"],
                    valid_depth["_source_depth_numeric"], "spearman"), 6
    ))

print("\nDATABASE AGE VALUES")
print(database["Age"].describe().to_string())
print("Unique database ages:", database["Age"].nunique())

print("\nOUTPUTS")
print("Matched comparison:", matched_path)
print("Summary:", summary_path)
print("Preserved PANGAEA export:", source_path)
print("\nNo raw data, original ages, or existing classifications modified.")
print("EXPERIMENT 70 COMPLETED")
