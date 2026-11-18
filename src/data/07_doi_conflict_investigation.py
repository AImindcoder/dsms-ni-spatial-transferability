import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

doi = df["DOI"].astype("string").str.strip().str.lower()
valid = doi.notna() & ~doi.isin(
    ["not available", "na", "n/a", "none", "unknown", ""]
)

d = df.loc[valid].copy()
d["DOI_clean"] = doi[valid]

print("CASES WHERE ONE TITLE HAS MULTIPLE DOIs")
title_dois = d.groupby("Title")["DOI_clean"].nunique()
for title in title_dois[title_dois > 1].index:
    subset = d[d["Title"] == title]
    print("\nTITLE:", title)
    print(subset.groupby("DOI_clean").size().to_string())
    print("Authors:", subset["FirstAuthor"].unique().tolist())
    print("Years:", subset["Year"].unique().tolist())

print("\nCASES WHERE ONE REFERENCE HAS MULTIPLE DOIs")
ref_dois = d.groupby("Reference")["DOI_clean"].nunique()
for ref in ref_dois[ref_dois > 1].index:
    subset = d[d["Reference"] == ref]
    print("\nREFERENCE:", ref)
    print(subset.groupby("DOI_clean").size().to_string())
    print("Titles:", subset["Title"].unique().tolist())
    print("Authors:", subset["FirstAuthor"].unique().tolist())
    print("Years:", subset["Year"].unique().tolist())
