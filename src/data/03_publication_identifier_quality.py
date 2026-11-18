import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

print("PLACEHOLDER VALUES IN PUBLICATION COLUMNS")
for col in ["DOI", "Reference", "Title", "FirstAuthor"]:
    s = df[col].astype("string").str.strip()
    placeholders = s.str.lower().isin(["not available", "na", "n/a", "none", "unknown", ""])
    print(f"\n{col}:")
    print("Placeholder records:", placeholders.sum())
    print("Placeholder values:", s[placeholders].value_counts().to_dict())

doi = df["DOI"].astype("string").str.strip()
valid = doi.notna() & ~doi.str.lower().isin(
    ["not available", "na", "n/a", "none", "unknown", ""]
)

counts = doi[valid].value_counts()

print("\nCORRECTED DOI SUMMARY")
print("Records with usable DOI:", valid.sum())
print("Distinct usable DOI identifiers:", counts.size)
print("Records without usable DOI:", (~valid).sum())

print("\nTOP 20 VALID PUBLICATIONS")
print(counts.head(20).to_string())

print("\nVALID PUBLICATION SIZE DISTRIBUTION")
print("1 record:", (counts == 1).sum())
print("2-10 records:", ((counts >= 2) & (counts <= 10)).sum())
print("11-100 records:", ((counts >= 11) & (counts <= 100)).sum())
print("101-500 records:", ((counts >= 101) & (counts <= 500)).sum())
print("More than 500 records:", (counts > 500).sum())
