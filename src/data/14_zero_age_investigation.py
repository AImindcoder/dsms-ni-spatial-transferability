import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

age = pd.to_numeric(df["Age"], errors="coerce")
zero = df.loc[age == 0].copy()
positive = df.loc[age > 0].copy()

target = "δ15Nbulk\n‰"
zero[target] = pd.to_numeric(zero[target], errors="coerce")

print("=" * 85)
print("ZERO-AGE RECORD INVESTIGATION")
print("=" * 85)

print("\n1. BASIC STATISTICS")
print("Zero-age records:", len(zero))
print("Records with bulk δ15N:", zero[target].notna().sum())
print("Records without bulk δ15N:", zero[target].isna().sum())
print("Unique DOIs:", zero["DOI"].nunique())
print("Unique authors:", zero["FirstAuthor"].nunique())
print("Unique site names:", zero["SiteName"].nunique())

for col in [
    "FirstAuthor", "Year", "DOI", "DataSource",
    "Location2-country", "SiteName", "Material",
    "Techniqueδ15N", "Period", "Epoch", "Stage"
]:
    print("\n" + "=" * 70)
    print("DISTRIBUTION:", col)
    print(zero[col].value_counts(dropna=False).head(20).to_string())

print("\n2. ZERO-AGE TARGET DISTRIBUTION")
print(zero[target].describe().to_string())

print("\n3. AGE COMPARISON")
print("Positive-age records:", len(positive))
print("Positive-age minimum:", age[age > 0].min())
print("Positive-age median:", age[age > 0].median())

print("\n4. ZERO-AGE RECORDS WITH COMPLETE IDENTIFIERS")
columns = [
    "SampleID", "FirstAuthor", "Year", "DOI",
    "SiteName", "Location2-country", "Latitude",
    "Longitude", "Age", "Period", "Epoch",
    "Material", "DataSource", target
]

print(zero[columns].head(25).to_string(index=False))

print("\n5. DATA SOURCE COMPARISON")
print("Zero-age records by source:")
print(zero["DataSource"].value_counts(dropna=False).head(15).to_string())

print("\nINSPECTION COMPLETE. RAW DATA UNCHANGED.")
