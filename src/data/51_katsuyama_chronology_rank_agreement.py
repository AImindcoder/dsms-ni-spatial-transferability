import pandas as pd
from scipy.stats import spearmanr, kendalltau
from pathlib import Path

base = Path("02_Data_Audit/provenance_analysis")

path = base / "45_katsuyama_primary_source_comparison.csv"
df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

df["Age_2016"] = pd.to_numeric(df["Age_2016"], errors="coerce")
df["Age_2020"] = pd.to_numeric(df["Age_2020"], errors="coerce")

d = df.dropna(subset=["Age_2016", "Age_2020"]).copy()
d = d.sort_values("Age_2020", ascending=False)

rho, sp = spearmanr(d["Age_2016"], d["Age_2020"])
tau, kp = kendalltau(d["Age_2016"], d["Age_2020"])

print("=" * 72)
print("EXPERIMENT 51: CROSS-PUBLICATION CHRONOLOGY RANK AGREEMENT")
print("=" * 72)

print("\nMatched records:", len(d))
print("Unique 2016 ages:", d["Age_2016"].nunique())
print("Unique 2020 ages:", d["Age_2020"].nunique())

print("\n2016 age distribution:")
print(d["Age_2016"].value_counts().sort_index(ascending=False))

print("\nSpearman rank correlation:", rho)
print("Spearman p-value:", sp)
print("Kendall rank correlation:", tau)
print("Kendall p-value:", kp)

print("\nORDERED RECORDS: OLDEST TO YOUNGEST IN 2020")
print(
    d[["SampleName", "Age_2016", "Age_2020"]]
    .to_string(index=False)
)

print("\n2016 CHRONOLOGY TRANSITIONS")
print(
    d.groupby("Age_2016", sort=False)["SampleName"]
    .agg(["count", "first", "last"])
    .to_string()
)

print("\nINTERPRETATION LIMIT:")
print("Rank agreement does not validate absolute ages.")
print("Repeated 2016 ages represent ties and reduced resolution.")
print("No source data modified.")

output = base / "51_katsuyama_chronology_rank_agreement.csv"
d.to_csv(output, index=False, encoding="utf-8-sig")
print("\nSaved:", output)
