from pathlib import Path
import pandas as pd
import numpy as np

root = Path(".")
source_path = root / "01_Data" / "external" / "Dotternhausen_Table_1_extracted.txt"
raw_path = root / "01_Data" / "raw" / "DSMSNI_v0.3.csv"
out_path = root / "01_Data" / "interim" / "100_dotternhausen_verified_source_crosswalk.csv"
report_path = root / "01_Data" / "interim" / "100_dotternhausen_verified_source_crosswalk_audit.txt"

# Parse the original publication's extracted Table 1.
lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
inside = False
records = []

for line in lines:
    if "[TABLE START]" in line:
        inside = True
        continue
    if "[TABLE END]" in line:
        inside = False
        continue
    if not inside or "|" not in line:
        continue

    parts = [x.strip() for x in line.split("|")]
    if len(parts) != 11 or parts[0] in ("", "Depth [cm]"):
        continue

    try:
        depth = float(parts[0].replace(",", ""))
        values = [float(x.replace(",", "").replace("−", "-")) for x in parts[1:]]
    except ValueError:
        continue

    records.append([depth] + values)

source_columns = [
    "Depth_cm", "TOC_source", "d13Corg_source", "TNbulk_source",
    "CNbulk_source", "d15Nbulk_source", "d13Cker_source",
    "TNker_source", "d15Nker_source", "CNker_source",
    "Delta15Nbulk_ker_source"
]
source = pd.DataFrame(records, columns=source_columns)

# Load the original DSMS-NI records.
df = pd.read_csv(raw_path, encoding="gbk", low_memory=False)
df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

doi_col = next(c for c in df.columns if c.lower() == "doi")
db = df[df[doi_col].astype(str).str.strip().str.lower()
        == "10.1029/2020pa004207"].copy()

db["RelativeDepth"] = pd.to_numeric(db["RelativeDepth"], errors="coerce")
db = db.rename(columns={
    "δ15Nbulk ‰": "d15Nbulk_DS​​MS",
    "δ15Nsp ‰": "d15Nsp_DS​​MS",
    "TN %": "TN_DS​​MS",
    "TOC %": "TOC_DS​​MS",
    "δ13Corg ‰": "d13Corg_DS​​MS"
})

# Match by depth, then test all available published measurements.
merged = source.merge(
    db,
    left_on="Depth_cm",
    right_on="RelativeDepth",
    how="outer",
    indicator=True
)

comparisons = {
    "d15Nbulk": ("d15Nbulk_source", "d15Nbulk_DS​​MS"),
    "TNbulk": ("TNbulk_source", "TN_DS​​MS"),
    "TOC": ("TOC_source", "TOC_DS​​MS"),
    "d13Corg": ("d13Corg_source", "d13Corg_DS​​MS"),
    "d15Nker_vs_database_d15Nsp": ("d15Nker_source", "d15Nsp_DS​​MS")
}

summary = []
for label, (left, right) in comparisons.items():
    merged[left] = pd.to_numeric(merged[left], errors="coerce")
    merged[right] = pd.to_numeric(merged[right], errors="coerce")
    difference = (merged[left] - merged[right]).abs()
    merged[f"{label}_absolute_difference"] = difference
    matched = merged["_merge"].eq("both") & difference.notna()
    within = matched & difference.le(0.0051)
    summary.append((label, int(matched.sum()), int(within.sum()),
                    float(difference[matched].max()) if matched.any() else np.nan))

# Record the three samples identified by the dissertation.
flagged_depths = {729.0, 825.0, 836.0}
merged["author_quality_flag"] = np.where(
    merged["Depth_cm"].isin(flagged_depths),
    "author_flagged_potential_post_depositional_modification",
    "not_flagged_by_authors_in_identified_discussion"
)

merged["source_identity_status"] = np.where(
    merged["_merge"].eq("both"),
    "DOI_and_depth_matched_source_table_record",
    "unmatched"
)

merged.to_csv(out_path, index=False, encoding="utf-8-sig")

report = [
    "DOTTERNHAUSEN SOURCE-TO-DSMS VERIFIED CROSSWALK",
    "=" * 75,
    f"Publication Table 1 rows parsed: {len(source)}",
    f"DSMS records with exact DOI: {len(db)}",
    f"Rows matched by depth: {(merged['_merge'] == 'both').sum()}",
    f"Rows unmatched: {(merged['_merge'] != 'both').sum()}",
    "",
    "NUMERICAL COMPARISON (tolerance <= 0.0051)",
]
for label, n, within, maxdiff in summary:
    report.append(
        f"{label}: compared={n}, within_tolerance={within}, "
        f"maximum_absolute_difference={maxdiff}"
    )

report += [
    "",
    "AUTHOR-IDENTIFIED QUALITY FLAGS",
    "Depths flagged: 729, 825, 836 cm",
    "Interpretation: potential local post-depositional modification;",
    "retain records and use the flag for sensitivity analyses.",
    "",
    "Important: depth correspondence and numerical reproduction establish",
    "source-table correspondence, not independent proof of physical aliquot identity.",
    "",
    f"Crosswalk: {out_path}",
]
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("\nSaved crosswalk and audit report.")
