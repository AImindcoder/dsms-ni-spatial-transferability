from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 78)
print("EXPERIMENT 96: DSMS-NI VALIDATION AGAINST ORIGINAL TABLE 1")
print("=" * 78)

doi = "10.1029/2020pa004207"

# Parse original Table 1
source_path = Path("01_Data/external/Dotternhausen_Table_1_extracted.txt")
source_text = source_path.read_text(encoding="utf-8")

records = []
for line in source_text.splitlines():
    parts = [x.strip() for x in line.strip().split("|")]
    if len(parts) < 11:
        continue
    try:
        records.append([float(x) for x in parts[:11]])
    except ValueError:
        continue

columns = [
    "Depth_cm", "TOC_wt_pct", "d13Corg",
    "TNbulk_wt_pct", "CNbulk", "d15Nbulk",
    "d13Cker", "TNker", "d15Nker", "CNker",
    "Delta15Nbulk_ker"
]
table1 = pd.DataFrame(records, columns=columns)

# Exact DOI records from DSMS-NI
dsms_path = Path("01_Data/raw/DSMSNI_v0.3.csv")
dsms = pd.read_csv(dsms_path, encoding="gbk", low_memory=False)

doi_values = dsms["DOI"].astype("string").str.strip().str.lower()
subset = dsms.loc[doi_values == doi].copy()

subset["RelativeDepth"] = pd.to_numeric(
    subset["RelativeDepth"], errors="coerce"
)
target_col = "δ15Nbulk\n‰"
subset[target_col] = pd.to_numeric(
    subset[target_col], errors="coerce"
)

# Sort both tables by depth
table1 = table1.sort_values("Depth_cm").reset_index(drop=True)
subset = subset.sort_values("RelativeDepth").reset_index(drop=True)

print("\n--- RECORD COUNTS ---")
print("Original Table 1:", len(table1))
print("DSMS-NI exact DOI:", len(subset))
print("Table 1 unique depths:", table1["Depth_cm"].nunique())
print("DSMS-NI unique depths:", subset["RelativeDepth"].nunique())

# One-to-one depth merge
comparison = table1.merge(
    subset[["SampleID", "RelativeDepth", target_col]],
    left_on="Depth_cm",
    right_on="RelativeDepth",
    how="outer",
    indicator=True,
    validate="one_to_one"
)

comparison["Depth_difference_cm"] = (
    comparison["RelativeDepth"] - comparison["Depth_cm"]
)
comparison["d15N_difference"] = (
    comparison[target_col] - comparison["d15Nbulk"]
)

print("\n--- DEPTH MATCHING ---")
print(comparison["_merge"].value_counts().to_string())
print("All depths identical:",
      bool((comparison.loc[comparison["_merge"] == "both",
                            "Depth_difference_cm"] == 0).all()))

matched = comparison[comparison["_merge"] == "both"].copy()

print("\n--- δ15Nbulk VALIDATION ---")
print("Matched records:", len(matched))
print("Exact δ15N matches:",
      int(np.isclose(matched["d15N_difference"], 0).sum()))
print("Nonzero δ15N differences:",
      int((~np.isclose(matched["d15N_difference"], 0)).sum()))

if len(matched):
    print("Maximum absolute δ15N difference:",
          matched["d15N_difference"].abs().max())
    print("Mean absolute δ15N difference:",
          matched["d15N_difference"].abs().mean())

print("\n--- COMPLETE SAMPLE COMPARISON ---")
display_cols = [
    "SampleID", "Depth_cm", "RelativeDepth",
    "d15Nbulk", target_col, "d15N_difference",
    "_merge"
]
print(comparison[display_cols].to_string(index=False))

# Table 1 vs Table 2 carbon-isotope difference summary
recon_path = Path(
    "01_Data/interim/95_dotternhausen_table1_table2_reconciliation.csv"
)
recon = pd.read_csv(recon_path)
carbon = recon[recon["Exact_depth_match"]].copy()
carbon["Table2_d13Corg_difference"] = pd.to_numeric(
    carbon["Table2_d13Corg_difference"], errors="coerce"
)

print("\n--- TABLE 1 / TABLE 2 δ13Corg DIFFERENCE SUMMARY ---")
print("Shared-depth pairs:", len(carbon))
print(carbon["Table2_d13Corg_difference"].describe().to_string())
print("Exact δ13Corg matches:",
      int(np.isclose(carbon["Table2_d13Corg_difference"], 0).sum()))
print("Within 0.1‰:",
      int((carbon["Table2_d13Corg_difference"].abs() <= 0.1).sum()))
print("Within 0.5‰:",
      int((carbon["Table2_d13Corg_difference"].abs() <= 0.5).sum()))

# Save full comparison
out = Path("01_Data/interim/96_dsms_vs_original_table1_validation.csv")
comparison.to_csv(out, index=False)

print("\nSaved:", out)
print("\nEXPERIMENT 96 COMPLETED")
print("Original source files unchanged.")
