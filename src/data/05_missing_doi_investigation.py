import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

doi = df["DOI"].astype("string").str.strip().str.lower()
valid = doi.notna() & ~doi.isin(
    ["not available", "na", "n/a", "none", "unknown", ""]
)

missing = df.loc[~valid].copy()

print("RECORDS WITHOUT USABLE DOI")
print("Total records:", len(missing))
print("Unique references:", missing["Reference"].nunique())
print("Unique titles:", missing["Title"].nunique())
print("Unique first authors:", missing["FirstAuthor"].nunique())

print("\nTOP 20 REFERENCES AMONG RECORDS WITHOUT DOI")
print(missing["Reference"].value_counts().head(20).to_string())

print("\nTOP 20 TITLES AMONG RECORDS WITHOUT DOI")
print(missing["Title"].value_counts().head(20).to_string())

print("\nRECORDS WITHOUT DOI BY AUTHOR")
print(missing["FirstAuthor"].value_counts().head(15).to_string())

print("\nDOI STATUS")
print(df.loc[~valid, "DOI"].value_counts(dropna=False).to_string())
