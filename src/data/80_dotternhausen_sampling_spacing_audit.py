from pathlib import Path
import pandas as pd
import numpy as np

path = Path("02_Data_Audit/provenance_analysis/79_dotternhausen_refined_lithology_register.csv")
df = pd.read_csv(path, encoding="utf-8-sig")

df = df.sort_values("Source_Depth_cm").reset_index(drop=True)
depth = df["Source_Depth_cm"].dropna().to_numpy()
gaps = np.diff(depth)

print("=" * 72)
print("EXPERIMENT 80: DOTTERNHAUSEN SAMPLE SPACING AUDIT")
print("=" * 72)

print("\nSAMPLE DEPTH COVERAGE")
print("Samples:", len(depth))
print("Minimum depth (cm):", depth.min())
print("Maximum depth (cm):", depth.max())
print("Total depth span (cm):", depth.max() - depth.min())

print("\nADJACENT SAMPLE SPACING")
print("Number of gaps:", len(gaps))
print("Mean gap (cm):", gaps.mean())
print("Median gap (cm):", np.median(gaps))
print("Minimum gap (cm):", gaps.min())
print("Maximum gap (cm):", gaps.max())
print("Standard deviation (cm):", gaps.std(ddof=1))

print("\nGAP DISTRIBUTION")
for threshold in [5, 10, 15, 20]:
    print(f"Gaps <= {threshold} cm:", int((gaps <= threshold).sum()))

print("\nALL ADJACENT GAPS")
for i, gap in enumerate(gaps):
    print(f"{depth[i]:7.1f} -> {depth[i+1]:7.1f} cm : {gap:5.1f} cm")

print("\nCOMPARISON WITH PUBLISHED METHOD")
print("Published reported spacing: approximately 3.5 cm")
print("Spacing calculated from Table 1:", round(float(gaps.mean()), 3), "cm")
print("Ratio of calculated to reported spacing:", round(float(gaps.mean()/3.5), 3))

print("\nINTERPRETATION")
print(
    "This is a discrepancy between the reported average and the spacing "
    "of the 31 tabulated geochemical samples. It does not establish an "
    "error until the authors' sampling definition and any omitted samples "
    "or sampling subsets are checked."
)

print("\nNo source data modified.")
print("EXPERIMENT 80 COMPLETED")
