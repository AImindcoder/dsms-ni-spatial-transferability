from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, theilslopes

base = Path("02_Data_Audit/provenance_analysis")
source = base / "57_katsuyama_2020_isotope_sequence.csv"

output = base / "58_katsuyama_trend_robustness.csv"
summary_file = base / "58_katsuyama_trend_robustness_summary.txt"

df = pd.read_csv(source, encoding="utf-8-sig")

df["Depth_2020"] = pd.to_numeric(df["Depth_2020"], errors="coerce")
df["DSMSNI_δ15Nbulk_2020"] = pd.to_numeric(
    df["DSMSNI_δ15Nbulk_2020"], errors="coerce"
)

valid = df.dropna(
    subset=["Depth_2020", "DSMSNI_δ15Nbulk_2020"]
).copy().reset_index(drop=True)

x = valid["Depth_2020"].to_numpy()
y = valid["DSMSNI_δ15Nbulk_2020"].to_numpy()

rho, p = spearmanr(x, y)
slope, intercept, low_slope, high_slope = theilslopes(y, x)

loo = []

for i in range(len(valid)):
    reduced = valid.drop(index=i)
    r, _ = spearmanr(
        reduced["Depth_2020"],
        reduced["DSMSNI_δ15Nbulk_2020"]
    )

    loo.append({
        "removed_sample": valid.loc[i, "SampleName"],
        "removed_value": y[i],
        "spearman_without_sample": r,
        "direction_preserved": np.sign(r) == np.sign(rho)
    })

loo_df = pd.DataFrame(loo)

# Consecutive measured observations in stratigraphic order.
# A skipped position indicates at least one missing measurement
# between two numeric observations.
changes = []

for i in range(1, len(valid)):
    previous = valid.iloc[i - 1]
    current = valid.iloc[i]

    changes.append({
        "upper_sample": previous["SampleName"],
        "lower_sample": current["SampleName"],
        "upper_bed": previous["Note_2020"],
        "lower_bed": current["Note_2020"],
        "upper_depth": previous["Depth_2020"],
        "lower_depth": current["Depth_2020"],
        "upper_isotope": previous["DSMSNI_δ15Nbulk_2020"],
        "lower_isotope": current["DSMSNI_δ15Nbulk_2020"],
        "isotope_change_per_step": (
            current["DSMSNI_δ15Nbulk_2020"]
            - previous["DSMSNI_δ15Nbulk_2020"]
        ),
        "absolute_change": abs(
            current["DSMSNI_δ15Nbulk_2020"]
            - previous["DSMSNI_δ15Nbulk_2020"]
        )
    })

changes_df = pd.DataFrame(changes)

missing = df[df["DSMSNI_δ15Nbulk_2020"].isna()]

summary = [
    "EXPERIMENT 58 — KATSUYAMA TREND ROBUSTNESS",
    "=" * 65,
    "",
    f"Numeric observations: {len(valid)}",
    f"Missing observations: {len(missing)}",
    "",
    "ROBUST TREND ESTIMATE",
    "-" * 40,
    f"Spearman rho: {rho:.6f}",
    f"Spearman p-value (descriptive only): {p:.6g}",
    f"Theil-Sen slope: {slope:.6f} per metre",
    f"Theil-Sen lower slope estimate: {low_slope:.6f}",
    f"Theil-Sen upper slope estimate: {high_slope:.6f}",
    f"Theil-Sen intercept: {intercept:.6f}",
    "",
    "LEAVE-ONE-OUT SENSITIVITY",
    "-" * 40,
    f"Minimum Spearman rho: {loo_df['spearman_without_sample'].min():.6f}",
    f"Maximum Spearman rho: {loo_df['spearman_without_sample'].max():.6f}",
    f"Mean Spearman rho: {loo_df['spearman_without_sample'].mean():.6f}",
    f"Direction preserved in all removals: "
    f"{loo_df['direction_preserved'].all()}",
    "",
    "MOST INFLUENTIAL REMOVALS",
    "-" * 40,
]

for _, r in loo_df.reindex(
    loo_df["spearman_without_sample"].sub(rho).abs().sort_values(
        ascending=False
    ).index
).head(5).iterrows():
    summary.append(
        f"Remove {r['removed_sample']} "
        f"(δ15N={r['removed_value']:.3f}‰): "
        f"rho={r['spearman_without_sample']:.6f}"
    )

summary.extend([
    "",
    "LARGEST ADJACENT ISOTOPE CHANGES",
    "-" * 40,
])

for _, r in changes_df.nlargest(5, "absolute_change").iterrows():
    summary.append(
        f"{r['upper_sample']} → {r['lower_sample']}: "
        f"{r['isotope_change_per_step']:+.3f}‰; "
        f"{r['upper_bed']} → {r['lower_bed']}"
    )

summary.extend([
    "",
    "MISSING MEASUREMENT POSITIONS",
    "-" * 40,
])

for _, r in missing.iterrows():
    summary.append(
        f"{r['SampleName']}: {r['Note_2020']}, "
        f"depth={r['Depth_2020']}, age={r['Age_2020']}"
    )

summary.extend([
    "",
    "INTERPRETATION LIMITATIONS",
    "-" * 40,
    "1. Leave-one-out analysis assesses individual-point influence only.",
    "2. It does not account for serial dependence between adjacent samples.",
    "3. Theil-Sen slope is descriptive, not a causal environmental estimate.",
    "4. Missing isotope values were not reconstructed.",
    "5. Results apply only to this Katsuyama sequence.",
    "",
    f"Detailed sensitivity results: {output}",
])

loo_df.to_csv(output, index=False, encoding="utf-8-sig")
summary_file.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nSaved:", output)
print("Saved:", summary_file)
