import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

df["Site_clean"] = df["SiteName"].astype("string").str.strip()

sites = ["Shangsi", "ODP 1240", "Chaotian", "Festningen", "74"]

columns = [
    "SampleID", "FirstAuthor", "Year", "DOI",
    "SiteName", "Location1-detail", "Location2-country",
    "Latitude", "Longitude", "Paleolatitude",
    "Paleolongitude", "Age", "Period", "Epoch",
    "Stage", "Formation", "SampleName",
    "Material", "DataSource", "Note"
]

for site in sites:

    sub = df[df["Site_clean"] == site].copy()

    print("\n" + "=" * 100)
    print("SITE:", site)
    print("RECORDS:", len(sub))

    if sub.empty:
        continue

    for col in [
        "Location1-detail", "Location2-country",
        "Latitude", "Longitude", "Paleolatitude",
        "Paleolongitude", "Formation", "DataSource"
    ]:
        print(f"\n{col} - DISTINCT VALUES:")
        print(sub[col].value_counts(dropna=False).head(12).to_string())

    print("\nRECORDS WITH AGE = 0:")
    zero = sub[pd.to_numeric(sub["Age"], errors="coerce") == 0]

    print("Count:", len(zero))

    if len(zero):
        print(zero[columns].head(10).to_string(index=False))

print("\n" + "=" * 100)
print("GLOBAL AGE ZERO AUDIT")

age = pd.to_numeric(df["Age"], errors="coerce")

print("Total records:", len(df))
print("Age exactly zero:", (age == 0).sum())
print("Age missing:", age.isna().sum())
print("Age greater than zero:", (age > 0).sum())
print("Age less than zero:", (age < 0).sum())

print("\nZERO-AGE RECORDS BY PERIOD:")
print(
    df.loc[age == 0, "Period"]
      .value_counts(dropna=False)
      .to_string()
)

print("\nINSPECTION COMPLETED. ORIGINAL DATA UNCHANGED.")
