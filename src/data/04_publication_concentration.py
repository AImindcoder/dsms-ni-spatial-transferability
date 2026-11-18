import pandas as pd
import numpy as np

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

doi = df["DOI"].astype("string").str.strip().str.lower()
valid = doi.notna() & ~doi.isin(
    ["not available", "na", "n/a", "none", "unknown", ""]
)

counts = doi[valid].value_counts()
total = counts.sum()
shares = counts / total

print("PUBLICATION CONCENTRATION ANALYSIS")
print("Total DOI-linked records:", total)
print("Number of valid DOI groups:", len(counts))

print("\nTOP PUBLICATION CONTRIBUTIONS")
for n in [1, 5, 10, 20, 35]:
    records = counts.head(n).sum()
    print(f"Top {n} publications: {records:,} records ({records/total*100:.2f}%)")

print("\nGROUP SIZE STATISTICS")
print("Mean records per DOI:", round(counts.mean(), 2))
print("Median records per DOI:", round(counts.median(), 2))
print("Maximum records per DOI:", counts.max())
print("Minimum records per DOI:", counts.min())

print("\nPUBLICATION CONCENTRATION INDICES")
hhi = (shares ** 2).sum()
effective_groups = 1 / hhi
print("Herfindahl-Hirschman Index:", round(hhi, 5))
print("Effective number of equally-sized publication groups:", round(effective_groups, 2))

print("\nRECORD SHARE BY PUBLICATION SIZE")
for label, mask in [
    ("1-10 records", counts <= 10),
    ("11-100 records", (counts >= 11) & (counts <= 100)),
    ("101-500 records", (counts >= 101) & (counts <= 500)),
    ("More than 500 records", counts > 500)
]:
    group_records = counts[mask].sum()
    print(f"{label}: {group_records:,} records ({group_records/total*100:.2f}%)")
