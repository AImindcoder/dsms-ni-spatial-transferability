from pathlib import Path
from urllib.request import Request, urlopen
from io import StringIO
import re
import numpy as np
import pandas as pd

RAW = Path("01_Data/raw/DSMSNI_v0.3.csv")
EXT = Path("01_Data/external")
OUT = Path("02_Data_Audit/provenance_analysis")

EXT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DOI = "10.1029/2000PA000558"
URL = "https://doi.pangaea.de/10.1594/PANGAEA.846584?format=textfile"

def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def find_column(frame, fragment):
    matches = [
        c for c in frame.columns
        if fragment.lower() in str(c).lower()
    ]
    if not matches:
        raise KeyError(
            f"Could not find '{fragment}'. Available columns: "
            + " | ".join(map(str, frame.columns))
        )
    return matches[0]

print("=" * 76)
print("EXPERIMENT 74: HAWKSER BOTTOMS SOURCE-DATABASE COMPARISON")
print("=" * 76)

# Load the unchanged DSMS-NI source file.
ds = pd.read_csv(RAW, encoding="gbk", low_memory=False)

doi_col = find_column(ds, "DOI")
site_col = find_column(ds, "SiteName")
sample_col = find_column(ds, "SampleName")
depth_col = find_column(ds, "RelativeDepth")
target_col = find_column(ds, "15Nbulk")

mask = (
    ds[doi_col].astype(str).str.lower().str.contains(
        re.escape(DOI.lower()), regex=True, na=False
    )
    & ds[site_col].astype(str).str.strip().str.lower().eq(
        "hawsker bottom"
    )
)

database = ds.loc[mask].copy()

if database.empty:
    raise ValueError(
        "No records found for SiteName 'Hawsker Bottom'. "
        "Inspect the site's exact spelling in DSMS-NI."
    )

database["_sample_key"] = database[sample_col].map(normalize)
database["_target"] = pd.to_numeric(
    database[target_col], errors="coerce"
)
database["_depth"] = pd.to_numeric(
    database[depth_col], errors="coerce"
)

# Download the independent Hawsker PANGAEA dataset.
request = Request(URL, headers={"User-Agent": "Mozilla/5.0 research audit"})

with urlopen(request, timeout=60) as response:
    source_bytes = response.read()

source_path = EXT / "Hawsker_PANGAEA_846584_original_export.txt"
source_path.write_bytes(source_bytes)

lines = source_bytes.decode(
    "utf-8", errors="replace"
).splitlines()

header_index = None

for i, line in enumerate(lines):
    low = line.lower()
    if "sample label" in low and "section" in low and "15n" in low:
        header_index = i
        break

if header_index is None:
    raise ValueError(
        "Could not locate the PANGAEA table header. "
        f"Download preserved at {source_path}"
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
source["_target"] = pd.to_numeric(
    source[source_target_col], errors="coerce"
)
source["_depth"] = pd.to_numeric(
    source[source_depth_col], errors="coerce"
)

# Identify labels that can be matched without ambiguity.
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

matched["isotope_difference"] = (
    matched["_target_DSMSNI"] - matched["_target_PANGAEA"]
)
matched["depth_difference"] = (
    matched["_depth_DSMSNI"] - matched["_depth_PANGAEA"]
)

valid_iso = matched.dropna(
    subset=["_target_DSMSNI", "_target_PANGAEA"]
)
valid_depth = matched.dropna(
    subset=["_depth_DSMSNI", "_depth_PANGAEA"]
)

def corr(a, b, method):
    if len(a) < 3:
        return np.nan
    return pd.Series(a).corr(pd.Series(b), method=method)

metrics = [
    ("DSMS-NI Hawsker records", len(database)),
    ("PANGAEA source records", len(source)),
    ("Unique DSMS-NI sample labels", database["_sample_key"].nunique()),
    ("Unique PANGAEA sample labels", source["_sample_key"].nunique()),
    ("Unambiguous matched labels", len(matched)),
    ("Matched numeric isotope pairs", len(valid_iso)),
    ("Matched numeric depth pairs", len(valid_depth)),
    ("DSMS-NI numeric isotope records", database["_target"].notna().sum()),
    ("PANGAEA numeric isotope records", source["_target"].notna().sum()),
]

if len(valid_iso):
    diff = valid_iso["isotope_difference"]
    metrics.extend([
        ("Exact isotope matches within 1e-6", int((diff.abs() <= 1e-6).sum())),
        ("Median absolute isotope difference", float(diff.abs().median())),
        ("Maximum absolute isotope difference", float(diff.abs().max())),
        ("Isotope Pearson correlation", corr(
            valid_iso["_target_DSMSNI"],
            valid_iso["_target_PANGAEA"], "pearson"
        )),
        ("Isotope Spearman correlation", corr(
            valid_iso["_target_DSMSNI"],
            valid_iso["_target_PANGAEA"], "spearman"
        )),
    ])

if len(valid_depth):
    metrics.extend([
        ("Depth Pearson correlation", corr(
            valid_depth["_depth_DSMSNI"],
            valid_depth["_depth_PANGAEA"], "pearson"
        )),
        ("Depth Spearman correlation", corr(
            valid_depth["_depth_DSMSNI"],
            valid_depth["_depth_PANGAEA"], "spearman"
        )),
        ("Depth Spearman after sign reversal", corr(
            valid_depth["_depth_DSMSNI"],
            -valid_depth["_depth_PANGAEA"], "spearman"
        )),
        ("Median absolute depth difference", float(
            valid_depth["depth_difference"].abs().median()
        )),
    ])

summary = pd.DataFrame(metrics, columns=["metric", "value"])

matched_path = OUT / "74_hawsker_sample_level_source_comparison.csv"
summary_path = OUT / "74_hawsker_source_comparison_summary.csv"

matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\nHAWSKER COMPARISON")
print("-" * 76)

for _, row in summary.iterrows():
    print(f"{row['metric']}: {row['value']}")

print("\nDATABASE AGE PROFILE")
age_col = find_column(database, "Age")
print(pd.to_numeric(database[age_col], errors="coerce").describe())
print("Unique assigned ages:", pd.to_numeric(
    database[age_col], errors="coerce"
).nunique())

print("\nOUTPUTS")
print("Matched records:", matched_path)
print("Summary:", summary_path)
print("Preserved original source:", source_path)
print("\nNo raw records or age assignments modified.")
print("EXPERIMENT 74 COMPLETED")
