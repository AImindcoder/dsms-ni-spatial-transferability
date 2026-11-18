
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Research\DSMS_NI_Research")

DATA_FILE = (
    PROJECT_ROOT
    / "01_Data"
    / "raw"
    / "DSMSNI_v0.3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "02_Data_Audit"
    / "audit_reports"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. VERIFY ORIGINAL DATASET
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

print("=" * 65)
print("DSMS-NI DATASET FORENSIC AUDIT")
print("=" * 65)

# ============================================================
# 3. FILE INTEGRITY: SHA-256
# ============================================================

print("\n[1/7] Calculating SHA-256 checksum...")

sha256 = hashlib.sha256()

with open(DATA_FILE, "rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
        sha256.update(chunk)

file_hash = sha256.hexdigest()
file_size = DATA_FILE.stat().st_size

# ============================================================
# 4. LOAD DATASET
# ============================================================

print("[2/7] Loading original CSV...")

df = pd.read_csv(
    DATA_FILE,
    encoding="latin1",
    low_memory=False
)
print("Dataset successfully loaded.")

# ============================================================
# 5. STRUCTURAL ANALYSIS
# ============================================================

print("[3/7] Inspecting dataset structure...")

rows, columns = df.shape

column_inventory = pd.DataFrame({
    "column_name": df.columns,
    "data_type": [
        str(dtype) for dtype in df.dtypes
    ],
    "non_missing_values": df.notna().sum().values,
    "missing_values": df.isna().sum().values,
    "missing_percentage": (
        df.isna().mean().values * 100
    ),
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

column_inventory.to_csv(
    OUTPUT_DIR / "column_inventory.csv",
    index=False
)

# ============================================================
# 6. MISSING VALUE ANALYSIS
# ============================================================

print("[4/7] Analysing missing values...")

missing_analysis = column_inventory[
    [
        "column_name",
        "missing_values",
        "missing_percentage"
    ]
].sort_values(
    by="missing_percentage",
    ascending=False
)

missing_analysis.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)

# ============================================================
# 7. DUPLICATE ANALYSIS
# ============================================================

print("[5/7] Checking exact duplicate rows...")

duplicate_count = int(df.duplicated().sum())

duplicate_percentage = (
    duplicate_count / rows * 100
    if rows > 0 else 0
)

# ============================================================
# 8. NUMERICAL VARIABLE ANALYSIS
# ============================================================

print("[6/7] Analysing numerical variables...")

numeric_df = df.select_dtypes(
    include=[np.number]
)

numeric_summary = numeric_df.describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).transpose()

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_summary.csv"
)

# ============================================================
# 9. SAMPLE RECORDS
# ============================================================

df.head(10).to_csv(
    OUTPUT_DIR / "sample_records.csv",
    index=False
)

# ============================================================
# 10. DATASET OVERVIEW
# ============================================================

print("[7/7] Creating final audit report...")

overview = {
    "audit_timestamp": datetime.now().isoformat(),
    "dataset_filename": DATA_FILE.name,
    "dataset_path": str(DATA_FILE),
    "file_size_bytes": file_size,
    "file_size_megabytes": round(file_size / (1024 ** 2), 3),
    "sha256": file_hash,
    "number_of_rows": int(rows),
    "number_of_columns": int(columns),
    "numeric_columns": int(len(numeric_df.columns)),
    "non_numeric_columns": int(columns - len(numeric_df.columns)),
    "total_missing_cells": int(df.isna().sum().sum()),
    "total_cells": int(rows * columns),
    "missing_cells_percentage": round(
        float(df.isna().sum().sum() / (rows * columns) * 100),
        4
    ) if rows * columns else 0,
    "exact_duplicate_rows": duplicate_count,
    "exact_duplicate_percentage": round(
        duplicate_percentage, 4
    ),
    "dataframe_memory_usage_mb": round(
        df.memory_usage(deep=True).sum() / (1024 ** 2),
        3
    )
}

with open(
    OUTPUT_DIR / "dataset_overview.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        overview,
        file,
        indent=4,
        ensure_ascii=False
    )

with open(
    OUTPUT_DIR / "dataset_overview.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("DSMS-NI DATASET FORENSIC AUDIT\n")
    file.write("=" * 55 + "\n\n")

    for key, value in overview.items():
        file.write(f"{key}: {value}\n")

# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("AUDIT COMPLETED")
print("=" * 65)

print(f"Rows: {rows:,}")
print(f"Columns: {columns:,}")
print(f"Numeric columns: {len(numeric_df.columns):,}")
print(f"Exact duplicate rows: {duplicate_count:,}")
print(f"Missing cells: {overview['total_missing_cells']:,}")
print(f"Missing percentage: {overview['missing_cells_percentage']}%")
print(f"File size: {overview['file_size_megabytes']} MB")
print(f"SHA-256: {file_hash}")

print("\nAUDIT REPORTS SAVED TO:")
print(OUTPUT_DIR)

print("\nOriginal dataset was not modified.")
