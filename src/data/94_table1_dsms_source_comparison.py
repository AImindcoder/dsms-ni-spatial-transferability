from pathlib import Path
import pandas as pd

print("=" * 76)
print("EXPERIMENT 94: TABLE 1 AND DSMS-NI SOURCE COMPARISON")
print("=" * 76)

# 1. Original extracted Table 1
table1_path = Path("01_Data/external/Dotternhausen_Table_1_extracted.txt")
print("\n--- ORIGINAL TABLE 1 TEXT ---")

if table1_path.exists():
    table1_text = table1_path.read_text(encoding="utf-8")
    print(table1_text)
else:
    print("Table 1 extraction not found.")

# 2. DSMS-NI source records
csv_path = Path("01_Data/raw/DSMSNI_v0.3.csv")
df = pd.read_csv(csv_path, encoding="gbk", low_memory=False)

print("\n--- DSMS-NI DATASET ---")
print("Dataset shape:", df.shape)
print("\nRelevant columns:")

for col in df.columns:
    if any(term in col.lower() for term in [
        "doi", "reference", "title", "author", "site",
        "sample", "age", "depth", "15nbulk", "technique",
        "note", "datasource"
    ]):
        print(repr(col))

# Find likely Wang/Dotternhausen records using source metadata.
search_columns = [
    c for c in df.columns
    if any(term in c.lower() for term in [
        "doi", "reference", "title", "author", "site"
    ])
]

mask = pd.Series(False, index=df.index)

for col in search_columns:
    values = df[col].astype("string")
    mask |= values.str.contains(
        "10.1029/2020PA004207|Dotternhausen|Wang",
        case=False, na=False, regex=True
    )

matched = df.loc[mask].copy()

print("\n--- MATCHED DSMS-NI RECORDS ---")
print("Matched rows:", len(matched))

preferred_terms = [
    "doi", "reference", "title", "firstauthor", "author",
    "sitename", "samplename", "sampleid", "age",
    "relativedepth", "15nbulk", "technique", "note"
]

display_cols = [
    c for c in df.columns
    if any(term in c.lower() for term in preferred_terms)
]

print("\nSelected metadata:")
if len(matched):
    print(matched[display_cols].head(60).to_string(index=False))
else:
    print("No matching records found using the current search terms.")

print("\nEXPERIMENT 94 COMPLETED")
print("All original files unchanged.")
