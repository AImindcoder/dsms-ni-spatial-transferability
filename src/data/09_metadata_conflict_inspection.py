import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

targets = [
    "10.1029/2008pa001702",
    "10.1029/2004gl020511",
    "10.1029/2004pa001114",
    "10.1016/j.chemgeo.2010.09.001",
    "10.19658/j.issn.1007-2802.2023.42.084"
]

columns = [
    "SampleID", "FirstAuthor", "Year", "Title", "DOI",
    "Reference", "Age", "Period", "Epoch", "Stage",
    "SiteName", "Formation", "Material",
    "DataSource", "Note"
]

doi_col = df["DOI"].astype("string").str.strip().str.lower()

for target in targets:
    subset = df.loc[doi_col == target, columns].copy()

    print("\n" + "=" * 90)
    print("DOI:", target)
    print("TOTAL RECORDS:", len(subset))

    for col in ["Title", "FirstAuthor", "Year", "Reference"]:
        print(f"\n{col} VALUE COUNTS:")
        print(subset[col].value_counts(dropna=False).to_string())

    print("\nSAMPLE RECORDS:")
    print(subset.drop_duplicates(
        subset=["Title", "FirstAuthor", "Year", "Reference"]
    ).head(12).to_string(index=False))
