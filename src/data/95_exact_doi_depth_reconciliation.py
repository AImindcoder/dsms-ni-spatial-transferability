from pathlib import Path
import pandas as pd
import re

print("=" * 78)
print("EXPERIMENT 95: EXACT DOI AND SAMPLE-DEPTH RECONCILIATION")
print("=" * 78)

doi_target = "10.1029/2020PA004207"

# ---------------------------------------------------------
# 1. Exact DOI search in DSMS-NI
# ---------------------------------------------------------
dsms_path = Path("01_Data/raw/DSMSNI_v0.3.csv")
dsms = pd.read_csv(dsms_path, encoding="gbk", low_memory=False)

doi_col = "DOI"
doi_values = dsms[doi_col].astype("string").str.strip().str.lower()
exact = dsms.loc[doi_values == doi_target.lower()].copy()

print("\n--- EXACT DOI SEARCH ---")
print("Target DOI:", doi_target)
print("Exact DOI rows:", len(exact))

if len(exact):
    cols = [
        c for c in [
            "SampleID", "FirstAuthor", "Title", "DOI",
            "SiteName", "SampleName", "Age", "RelativeDepth",
            "Stage", "δ15Nbulk\n‰"
        ] if c in exact.columns
    ]
    print(exact[cols].to_string(index=False))
else:
    print("No exact DOI records found in DSMS-NI.")

# Also show Wang-author records separately, without treating them as matches.
if "FirstAuthor" in dsms.columns:
    wang = dsms[
        dsms["FirstAuthor"].astype("string").str.contains(
            r"\bWang\b", case=False, na=False, regex=True
        )
    ]
    print("\nRows with FirstAuthor containing Wang:", len(wang))
    if len(wang):
        print(wang[["FirstAuthor", "Title", "DOI", "SiteName"]]
              .drop_duplicates().head(15).to_string(index=False))

# ---------------------------------------------------------
# 2. Parse original Table 1
# ---------------------------------------------------------
table1_path = Path("01_Data/external/Dotternhausen_Table_1_extracted.txt")
text = table1_path.read_text(encoding="utf-8")

records = []
for line in text.splitlines():
    line = line.strip()
    if not line or not re.match(r"^\d", line):
        continue

    parts = [x.strip() for x in line.split("|")]
    if len(parts) < 11:
        continue

    try:
        values = [float(x) for x in parts[:11]]
    except ValueError:
        continue

    records.append(values)

columns = [
    "Depth_cm", "TOC_wt_pct", "d13Corg",
    "TNbulk_wt_pct", "CNbulk", "d15Nbulk",
    "d13Cker", "TNker", "d15Nker", "CNker",
    "Delta15Nbulk_ker"
]

t1 = pd.DataFrame(records, columns=columns)

print("\n--- TABLE 1 PARSING ---")
print("Parsed records:", len(t1))
print("Unique Table 1 depths:", t1["Depth_cm"].nunique())

# ---------------------------------------------------------
# 3. Compare Table 1 with coordinate-reconstructed Table 2
# ---------------------------------------------------------
table2_path = Path(
    "01_Data/interim/dotternhausen_table2_coordinate_reconstruction.csv"
)
t2 = pd.read_csv(table2_path)

results = []

for _, row in t1.iterrows():
    depth = row["Depth_cm"]
    candidates = t2[t2["Depth_cm"] == depth].copy()

    if len(candidates):
        for _, candidate in candidates.iterrows():
            results.append({
                "Table1_Depth_cm": depth,
                "Table1_d13Corg": row["d13Corg"],
                "Table1_d15Nbulk": row["d15Nbulk"],
                "Table2_Observation_ID": candidate["Observation_ID"],
                "Table2_d13Corg": candidate["d13Corg"],
                "Table2_d13Corg_difference": (
                    candidate["d13Corg"] - row["d13Corg"]
                ),
                "Table2_PDF_page": candidate["PDF_page_index"],
                "Exact_depth_match": True
            })
    else:
        results.append({
            "Table1_Depth_cm": depth,
            "Table1_d13Corg": row["d13Corg"],
            "Table1_d15Nbulk": row["d15Nbulk"],
            "Table2_Observation_ID": None,
            "Table2_d13Corg": None,
            "Table2_d13Corg_difference": None,
            "Table2_PDF_page": None,
            "Exact_depth_match": False
        })

recon = pd.DataFrame(results)

print("\n--- TABLE 1 TO TABLE 2 DEPTH MATCHES ---")
print("Table 1 records:", len(t1))
print("Table 1 records with exact Table 2 depth:", 
      recon.loc[recon["Exact_depth_match"], "Table1_Depth_cm"].nunique())
print("Table 1 records without exact Table 2 depth:",
      recon.loc[~recon["Exact_depth_match"], "Table1_Depth_cm"].nunique())
print("Total candidate pairs:", int(recon["Exact_depth_match"].sum()))

print("\nDepth-by-depth comparison:")
print(recon.to_string(index=False))

# ---------------------------------------------------------
# 4. Save auditable reconciliation
# ---------------------------------------------------------
out = Path("01_Data/interim/95_dotternhausen_table1_table2_reconciliation.csv")
recon.to_csv(out, index=False)

print("\nSaved:", out)
print("\nEXPERIMENT 95 COMPLETED")
print("No source data modified.")
