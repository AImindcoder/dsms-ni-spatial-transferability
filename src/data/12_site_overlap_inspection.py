import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

sites = ["Shangsi", "ODP 1240", "Chaotian", "Festningen", "74"]

df["Site_clean"] = df["SiteName"].astype("string").str.strip()
df["DOI_clean"] = df["DOI"].astype("string").str.strip().str.lower()

invalid = df["DOI_clean"].isin(["not available", ""])
df.loc[invalid, "DOI_clean"] = pd.NA

target = "δ15Nbulk\n‰"

for site in sites:

    sub = df[df["Site_clean"] == site].copy()

    print("\n" + "=" * 95)
    print("SITE:", site)
    print("TOTAL RECORDS:", len(sub))
    print("DISTINCT DOIs:", sub["DOI_clean"].nunique())

    if sub.empty:
        continue

    print("\nPUBLICATION BREAKDOWN:")

    publication_summary = (
        sub.groupby("DOI_clean", dropna=False)
        .agg(
            records=("SampleID", "size"),
            authors=("FirstAuthor", "nunique"),
            years=("Year", lambda x: ", ".join(map(str, sorted(x.dropna().unique())))),
            titles=("Title", "nunique"),
            age_min=("Age", "min"),
            age_max=("Age", "max"),
            sample_names=("SampleName", "nunique")
        )
        .sort_values("records", ascending=False)
    )

    print(publication_summary.to_string())

    print("\nREPRESENTATIVE RECORDS:")

    columns = [
        "SampleID", "FirstAuthor", "Year", "DOI",
        "Title", "SampleName", "Age", "Formation",
        "Material", target
    ]

    print(
        sub[columns]
        .drop_duplicates()
        .head(12)
        .to_string(index=False)
    )

print("\nINSPECTION COMPLETED. RAW DATA WAS NOT MODIFIED.")
