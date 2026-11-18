import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

df["DOI_clean"] = (
    df["DOI"].astype("string")
    .str.strip()
    .str.lower()
)

invalid = ["not available", "na", "n/a", "none", "unknown", ""]
df.loc[df["DOI_clean"].isin(invalid), "DOI_clean"] = pd.NA

identified = df[df["DOI_clean"].notna()].copy()

print("GLOBAL DOI CONSISTENCY AUDIT")
print("Records with usable DOI:", len(identified))
print("Distinct DOI groups:", identified["DOI_clean"].nunique())

for col in ["Title", "FirstAuthor", "Year", "Reference"]:
    counts = identified.groupby("DOI_clean")[col].nunique(dropna=True)
    conflicts = counts[counts > 1]

    print(f"\nDOIs linked to multiple {col} values:", len(conflicts))

    if len(conflicts):
        for doi in conflicts.index:
            rows = identified[identified["DOI_clean"] == doi]
            print("\nDOI:", doi)
            print(rows[col].value_counts(dropna=False).to_string())

print("\nTITLE CONSISTENCY ACROSS DIFFERENT DOIs")
title_groups = (
    identified.groupby("Title")["DOI_clean"]
    .nunique()
    .sort_values(ascending=False)
)
print("Titles linked to multiple DOIs:", (title_groups > 1).sum())
print(title_groups[title_groups > 1].to_string())

print("\nAUTHOR-YEAR CONSISTENCY")
author_year = identified.groupby(
    ["FirstAuthor", "Year"]
)["DOI_clean"].nunique()

print("Author-year combinations linked to multiple DOIs:",
      (author_year > 1).sum())
