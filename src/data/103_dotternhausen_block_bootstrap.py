from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import theilslopes

path = Path("01_Data/interim/100_dotternhausen_verified_source_crosswalk.csv")
df = pd.read_csv(path)

data = pd.DataFrame({
    "depth": pd.to_numeric(df["Depth_cm"], errors="coerce"),
    "bulk": pd.to_numeric(df["d15Nbulk_source"], errors="coerce"),
    "ker": pd.to_numeric(df["d15Nker_source"], errors="coerce")
}).dropna().sort_values("depth").reset_index(drop=True)

flagged_depths = {729.0, 825.0, 836.0}
clean = data[~data["depth"].isin(flagged_depths)].reset_index(drop=True)

def circular_block_resample(residuals, block_length, rng):
    n = len(residuals)
    sampled = []
    while len(sampled) < n:
        start = rng.integers(0, n)
        block = [residuals[(start + j) % n] for j in range(block_length)]
        sampled.extend(block)
    return np.asarray(sampled[:n])

def run_bootstrap(label, subset, target, block_lengths=(2, 3, 4), B=2000):
    x = subset["depth"].to_numpy(dtype=float)
    y = subset[target].to_numpy(dtype=float)

    fit = theilslopes(y, x, method="separate")
    base_slope = fit.slope
    base_intercept = fit.intercept
    fitted = base_intercept + base_slope * x
    residuals = y - fitted

    print(f"\n{label} — {target}")
    print("-" * 70)
    print(f"n={len(x)} | Original Theil-Sen slope={base_slope:.7f}‰/cm")

    for block_length in block_lengths:
        rng = np.random.default_rng(20261108 + block_length)
        slopes = []

        for _ in range(B):
            resampled_residuals = circular_block_resample(
                residuals, block_length, rng
            )
            synthetic_y = fitted + resampled_residuals
            slope = theilslopes(
                synthetic_y, x, method="separate"
            ).slope
            slopes.append(slope)

        slopes = np.asarray(slopes)
        low, high = np.percentile(slopes, [2.5, 97.5])
        negative_fraction = np.mean(slopes < 0)

        print(
            f"Block={block_length}: "
            f"bootstrap 95% interval=[{low:.7f}, {high:.7f}], "
            f"P(slope<0)={negative_fraction:.3f}"
        )

print("=" * 75)
print("DOTTERNHAUSEN CIRCULAR MOVING-BLOCK RESIDUAL BOOTSTRAP")
print("=" * 75)

for label, subset in [
    ("ALL 31 SAMPLES", data),
    ("28 SAMPLES, AUTHOR-FLAGGED OBSERVATIONS EXCLUDED", clean)
]:
    run_bootstrap(label, subset, "bulk")
    run_bootstrap(label, subset, "ker")

print("\nINTERPRETATION")
print("-" * 70)
print("These are residual-bootstrap sensitivity intervals, not exact")
print("confidence intervals under an established geological dependence model.")
print("Conclusions should be compared across block lengths.")
print("A single section cannot establish a global marine isotope trend.")
