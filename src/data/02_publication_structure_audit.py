import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

print("TOTAL RECORDS:", len(df))

print("\nUNIQUE PUBLICATION IDENTIFIERS")
for col in ["DOI", "FirstAuthor", "Reference", "Title"]:
    print(f"{col}: {df[col].nunique(dropna=True)} unique; {df[col].isna().sum()} missing")

d = df[df["DOI"].notna()].copy()
d["DOI"] = d["DOI"].astype(str).str.strip()

counts = d.groupby("DOI").size().sort_values(ascending=False)

print("\nTOP 20 PUBLICATIONS BY NUMBER OF RECORDS")
print(counts.head(20).to_string())

print("\nPUBLICATION DISTRIBUTION")
print("Publications with 1 record:", (counts == 1).sum())
print("Publications with 2-10 records:", ((counts >= 2) & (counts <= 10)).sum())
print("Publications with 11-100 records:", ((counts >= 11) & (counts <= 100)).sum())
print("Publications with 101-500 records:", ((counts >= 101) & (counts <= 500)).sum())
print("Publications with more than 500 records:", (counts > 500).sum())

print("\nDOI COVERAGE")
print("Records linked to a DOI:", len(d))
print("Records without a DOI:", df["DOI"].isna().sum())
