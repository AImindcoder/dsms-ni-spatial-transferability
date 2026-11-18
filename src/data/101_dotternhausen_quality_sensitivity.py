from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

path = Path("01_Data/interim/100_dotternhausen_verified_source_crosswalk.csv")
df = pd.read_csv(path)

depth = pd.to_numeric(df["Depth_cm"], errors="coerce")
bulk = pd.to_numeric(df["d15Nbulk_source"], errors="coerce")
ker = pd.to_numeric(df["d15Nker_source"], errors="coerce")
offset = pd.to_numeric(df["Delta15Nbulk_ker_source"], errors="coerce")

flagged_depths = {729.0, 825.0, 836.0}
flagged = depth.isin(flagged_depths)

def describe(label, mask):
    print("\n" + label)
    print("-" * 65)
    for name, values in [
        ("δ15Nbulk", bulk[mask]),
        ("δ15Nker", ker[mask]),
        ("Published Δ15Nbulk-ker", offset[mask])
    ]:
        values = values.dropna()
        print(
            f"{name}: n={len(values)}, mean={values.mean():.4f}, "
            f"median={values.median():.4f}, SD={values.std():.4f}, "
            f"min={values.min():.2f}, max={values.max():.2f}"
        )

    for name, values in [("δ15Nbulk", bulk), ("δ15Nker", ker)]:
        valid = mask & depth.notna() & values.notna()
        rho, p = spearmanr(depth[valid], values[valid])
        print(f"Depth vs {name}: Spearman rho={rho:.4f}, p={p:.5g}")

print("=" * 75)
print("DOTTERNHAUSEN ISOTOPE QUALITY SENSITIVITY ANALYSIS")
print("=" * 75)
print("Total observations:", len(df))
print("Author-flagged observations:", int(flagged.sum()))
print("Remaining observations:", int((~flagged).sum()))

describe("A. ALL 31 PUBLISHED OBSERVATIONS", pd.Series(True, index=df.index))
describe("B. 28 OBSERVATIONS EXCLUDING THE THREE AUTHOR-FLAGGED SAMPLES", ~flagged)

print("\nAUTHOR-FLAGGED OBSERVATIONS")
print("-" * 65)
print(pd.DataFrame({
    "Depth_cm": depth[flagged],
    "d15Nbulk": bulk[flagged],
    "d15Nker": ker[flagged],
    "Published_delta": offset[flagged]
}).to_string(index=False))

print("\nFLAGGED VS NON-FLAGGED OFFSET")
print("-" * 65)
print("Flagged mean absolute offset:",
      f"{offset[flagged].abs().mean():.4f}‰")
print("Other samples mean absolute offset:",
      f"{offset[~flagged].abs().mean():.4f}‰")

print("\nINTERPRETATION LIMITS")
print("-" * 65)
print("This is one geological section, not an independent validation set.")
print("Depth observations may be serially dependent.")
print("The three samples were flagged by the original authors;")
print("do not interpret this selected-group comparison as a causal test.")
