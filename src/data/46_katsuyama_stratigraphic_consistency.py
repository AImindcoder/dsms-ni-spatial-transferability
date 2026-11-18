import pandas as pd
from scipy.stats import spearmanr, kendalltau
from pathlib import Path

path = Path("02_Data_Audit/provenance_analysis/44_katsuyama_original_52_full_records.csv")
df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

doi = "10.1016/j.earscirev.2020.103173"
d = df[df["DOI"] == doi].copy()

d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
d["RelativeDepth"] = pd.to_numeric(d["RelativeDepth"], errors="coerce")
d["BedNumber"] = pd.to_numeric(
    d["Note"].astype(str).str.extract(r"Bed\s+(\d+)", expand=False),
    errors="coerce"
)

print("=" * 70)
print("EXPERIMENT 46: STRATIGRAPHIC CONSISTENCY ANALYSIS")
print("=" * 70)

def report(label, x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) >= 3:
        rho, p = spearmanr(valid["x"], valid["y"])
        tau, kp = kendalltau(valid["x"], valid["y"])
        print(f"\n{label}")
        print("Valid pairs:", len(valid))
        print("Spearman correlation:", round(rho, 5))
        print("Spearman p-value:", p)
        print("Kendall correlation:", round(tau, 5))
        print("Kendall p-value:", kp)
    else:
        print(f"\n{label}: insufficient data")

report("AGE vs RELATIVE DEPTH", d["RelativeDepth"], d["Age"])
report("AGE vs BED NUMBER", d["BedNumber"], d["Age"])
report("BED NUMBER vs RELATIVE DEPTH", d["BedNumber"], d["RelativeDepth"])

print("\nORDERED SAMPLE SEQUENCE")
print(
    d[["SampleName", "Age", "RelativeDepth", "Note"]]
    .sort_values("RelativeDepth")
    .to_string(index=False)
)

print("\nMISSINGNESS")
for c in ["Age", "RelativeDepth", "Note", "δ15Nbulk\n‰"]:
    if c in d.columns:
        print(c, "missing:", d[c].isna().sum(), "/", len(d))

print("\nNo original data modified.")
