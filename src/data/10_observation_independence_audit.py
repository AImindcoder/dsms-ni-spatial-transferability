import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

print("OBSERVATION INDEPENDENCE AUDIT")
print("Total records:", len(df))

for col in ["SampleID", "SampleName", "SiteName", "Formation",
            "Unit", "Location1-detail", "Location2-country"]:

    s = df[col].astype("string").str.strip()
    valid = s.notna() & (s != "")

    counts = s[valid].value_counts()

    print("\n" + "=" * 65)
    print("VARIABLE:", col)
    print("Nonmissing records:", valid.sum())
    print("Distinct values:", counts.size)
    print("Values occurring more than once:", (counts > 1).sum())
    print("Records belonging to repeated values:",
          counts[counts > 1].sum())
    print("Largest group size:", counts.max() if len(counts) else 0)

    if len(counts):
        print("\nTOP 10 REPEATED VALUES:")
        print(counts[counts > 1].head(10).to_string())

print("\nDATA SOURCE DISTRIBUTION")
print(df["DataSource"].value_counts(dropna=False).head(20).to_string())
