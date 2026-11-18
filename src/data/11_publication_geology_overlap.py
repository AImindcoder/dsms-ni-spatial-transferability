import pandas as pd

path = r"01_Data/raw/DSMSNI_v0.3.csv"
df = pd.read_csv(path, encoding="gbk", low_memory=False)

def clean(series):
    s = series.astype("string").str.strip()
    return s.mask(s.eq(""))

df["DOI_clean"] = clean(df["DOI"]).str.lower()
df["Site_clean"] = clean(df["SiteName"])
df["Formation_clean"] = clean(df["Formation"])
df["SampleName_clean"] = clean(df["SampleName"])

invalid_doi = df["DOI_clean"].isin(["not available", "nan", "none"])
df.loc[invalid_doi, "DOI_clean"] = pd.NA

print("=" * 75)
print("PUBLICATION-GEOLOGICAL GROUP OVERLAP AUDIT")
print("=" * 75)

# 1. Sites associated with multiple DOI identifiers
site_doi = (
    df.dropna(subset=["Site_clean", "DOI_clean"])
      .groupby("Site_clean")
      .agg(
          records=("SampleID", "size"),
          distinct_dois=("DOI_clean", "nunique"),
          distinct_formations=("Formation_clean", "nunique")
      )
      .sort_values(["distinct_dois", "records"], ascending=False)
)

multi_doi_sites = site_doi[site_doi["distinct_dois"] > 1]

print("\n1. SITE NAMES ASSOCIATED WITH MULTIPLE DOIs")
print("Sites with usable DOI information:", len(site_doi))
print("Sites linked to multiple DOIs:", len(multi_doi_sites))
print("\nTOP 20:")
print(multi_doi_sites.head(20).to_string())

# 2. DOI groups associated with multiple site names
doi_site = (
    df.dropna(subset=["DOI_clean", "Site_clean"])
      .groupby("DOI_clean")
      .agg(
          records=("SampleID", "size"),
          distinct_sites=("Site_clean", "nunique"),
          distinct_formations=("Formation_clean", "nunique")
      )
      .sort_values("distinct_sites", ascending=False)
)

print("\n2. PUBLICATIONS ASSOCIATED WITH MULTIPLE SITES")
print("DOI groups with site information:", len(doi_site))
print("\nTOP 15:")
print(doi_site.head(15).to_string())

# 3. Formations associated with multiple DOIs
formation_doi = (
    df.dropna(subset=["Formation_clean", "DOI_clean"])
      .groupby("Formation_clean")
      .agg(
          records=("SampleID", "size"),
          distinct_dois=("DOI_clean", "nunique"),
          distinct_sites=("Site_clean", "nunique")
      )
      .sort_values(["distinct_dois", "records"], ascending=False)
)

print("\n3. FORMATIONS ASSOCIATED WITH MULTIPLE DOIs")
print("Formations with usable DOI information:", len(formation_doi))
print("\nTOP 20:")
print(formation_doi.head(20).to_string())

# 4. Repeated sample names across DOI groups
sample_doi = (
    df.dropna(subset=["SampleName_clean", "DOI_clean"])
      .groupby("SampleName_clean")
      .agg(
          records=("SampleID", "size"),
          distinct_dois=("DOI_clean", "nunique"),
          distinct_sites=("Site_clean", "nunique")
      )
      .sort_values(["distinct_dois", "records"], ascending=False)
)

multi_doi_samples = sample_doi[sample_doi["distinct_dois"] > 1]

print("\n4. SAMPLE NAMES ASSOCIATED WITH MULTIPLE DOIs")
print("Repeated sample names across DOI groups:", len(multi_doi_samples))
print("\nTOP 20:")
print(multi_doi_samples.head(20).to_string())

# Save audit tables without changing the original dataset
out = r"02_Data_Audit\geological_groups"

site_doi.to_csv(out + r"\site_publication_relationships.csv", encoding="utf-8-sig")
doi_site.to_csv(out + r"\publication_site_relationships.csv", encoding="utf-8-sig")
formation_doi.to_csv(out + r"\formation_publication_relationships.csv", encoding="utf-8-sig")
sample_doi.to_csv(out + r"\sample_publication_relationships.csv", encoding="utf-8-sig")

print("\nAUDIT TABLES SAVED.")
