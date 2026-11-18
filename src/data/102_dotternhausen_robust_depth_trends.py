from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import theilslopes, kendalltau, spearmanr
from statsmodels.stats.stattools import durbin_watson

path = Path("01_Data/interim/100_dotternhausen_verified_source_crosswalk.csv")
df = pd.read_csv(path)

df["Depth_cm"] = pd.to_numeric(df["Depth_cm"], errors="coerce")
df["bulk"] = pd.to_numeric(df["d15Nbulk_source"], errors="coerce")
df["ker"] = pd.to_numeric(df["d15Nker_source"], errors="coerce")

flagged_depths = {729.0, 825.0, 836.0}
df["author_flagged"] = df["Depth_cm"].isin(flagged_depths)
df = df.sort_values("Depth_cm").reset_index(drop=True)

def analyse(label, data, target):
    x = data["Depth_cm"].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)

    slope, intercept, low, high = theilslopes(y, x, method="separate")
    tau, kp = kendalltau(x, y)
    rho, sp = spearmanr(x, y)

    residuals = y - (slope * x + intercept)
    dw = durbin_watson(residuals)

    loo_slopes = []
    for i in range(len(data)):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        loo_slopes.append(theilslopes(yy, xx, method="separate")[0])

    print(f"\n{label} — {target}")
    print("-" * 65)
    print(f"n: {len(data)}")
    print(f"Theil-Sen slope: {slope:.6f} ‰/cm")
    print(f"Theil-Sen 95% CI: [{low:.6f}, {high:.6f}]")
    print(f"Kendall tau: {tau:.4f}, p={kp:.5g}")
    print(f"Spearman rho: {rho:.4f}, p={sp:.5g}")
    print(f"Leave-one-out slope range: {min(loo_slopes):.6f} to {max(loo_slopes):.6f}")
    print(f"Leave-one-out signs: {sum(s < 0 for s in loo_slopes)} negative / "
          f"{sum(s > 0 for s in loo_slopes)} positive")
    print(f"Durbin-Watson on fitted residuals: {dw:.4f}")

print("=" * 75)
print("DOTTERNHAUSEN ROBUST DEPTH-TREND ANALYSIS")
print("=" * 75)

for label, subset in [
    ("ALL OBSERVATIONS", df),
    ("EXCLUDING AUTHOR-FLAGGED SAMPLES", df[~df["author_flagged"]].copy())
]:
    analyse(label, subset, "bulk")
    analyse(label, subset, "ker")

print("\nCAUTION")
print("-" * 65)
print("Theil-Sen confidence intervals assume independent observations.")
print("Durbin-Watson is a diagnostic, not a correction for serial dependence.")
print("This is a single-section exploratory analysis, not global evidence.")
