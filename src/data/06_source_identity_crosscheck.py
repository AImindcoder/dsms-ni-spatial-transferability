import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

doi = df["DOI"].astype("string").str.strip().str.lower()
valid = doi.notna() & ~doi.isin(
    ["not available", "na", "n/a", "none", "unknown", ""]
)

missing = df.loc[~valid].copy()
identified = df.loc[valid].copy()

print("CHECKING THE NINE SOURCES AGAINST DOI-IDENTIFIED RECORDS")

for _, row in missing.drop_duplicates("Title").iterrows():
    title = row["Title"]
    ref = row["Reference"]

    title_match = identified[identified["Title"] == title]
    ref_match = identified[identified["Reference"] == ref]

    print("\nSOURCE:", title)
    print("DOI records with same title:", len(title_match))
    print("DOI records with same reference:", len(ref_match))

    if len(title_match):
        print("DOIs by matching title:",
              title_match["DOI"].dropna().unique().tolist())

    if len(ref_match):
        print("DOIs by matching reference:",
              ref_match["DOI"].dropna().unique().tolist())

print("\nCHECKING DOI CONSISTENCY")
doi_title_counts = identified.groupby("Title")["DOI"].nunique()
doi_ref_counts = identified.groupby("Reference")["DOI"].nunique()

print("Titles linked to multiple DOI values:",
      (doi_title_counts > 1).sum())
print("References linked to multiple DOI values:",
      (doi_ref_counts > 1).sum())
