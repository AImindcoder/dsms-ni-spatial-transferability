from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import theilslopes, pearsonr

base = Path("02_Data_Audit/provenance_analysis")
source = base / "57_katsuyama_2020_isotope_sequence.csv"

output = base / "59_katsuyama_serial_dependence.csv"
summary_file = base / "59_katsuyama_serial_dependence_summary.txt"

df = pd.read_csv(source, encoding="utf-8-sig")

df["Depth_2020"] = pd.to_numeric(df["Depth_2020"], errors="coerce")
df["DSMSNI_δ15Nbulk_2020"] = pd.to_numeric(
    df["DSMSNI_δ15Nbulk_2020"], errors="coerce"
)

df = df.sort_values(
    "Depth_2020", ascending=False
).reset_index(drop=True)

valid = df.dropna(
    subset=["Depth_2020", "DSMSNI_δ15Nbulk_2020"]
).copy()

x = valid["Depth_2020"].to_numpy()
y = valid["DSMSNI_δ15Nbulk_2020"].to_numpy()

slope, intercept, low, high = theilslopes(y, x)

valid["fitted_trend"] = intercept + slope * valid["Depth_2020"]
valid["trend_residual"] = (
    valid["DSMSNI_δ15Nbulk_2020"] - valid["fitted_trend"]
)

# Preserve original sequence positions. Only use adjacent pairs
# where neither observation is missing.
df["trend_residual"] = np.nan

for idx, row in valid.iterrows():
    df.loc[idx, "trend_residual"] = (
        row["DSMSNI_δ15Nbulk_2020"] - row["fitted_trend"]
    )

pairs = []

for i in range(1, len(df)):
    previous = df.iloc[i - 1]
    current = df.iloc[i]

    if (
        pd.notna(previous["DSMSNI_δ15Nbulk_2020"])
        and pd.notna(current["DSMSNI_δ15Nbulk_2020"])
    ):
        pairs.append({
            "upper_sample": previous["SampleName"],
            "lower_sample": current["SampleName"],
            "upper_isotope": previous["DSMSNI_δ15Nbulk_2020"],
            "lower_isotope": current["DSMSNI_δ15Nbulk_2020"],
            "upper_residual": previous["trend_residual"],
            "lower_residual": current["trend_residual"],
            "depth_gap_m": abs(
                current["Depth_2020"] - previous["Depth_2020"]
            )
        })

pairs_df = pd.DataFrame(pairs)

def correlation(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]

raw_lag1 = correlation(
    pairs_df["upper_isotope"],
    pairs_df["lower_isotope"]
)

residual_lag1 = correlation(
    pairs_df["upper_residual"],
    pairs_df["lower_residual"]
)

residual_sd = valid["trend_residual"].std()
isotope_sd = valid["DSMSNI_δ15Nbulk_2020"].std()

summary = [
    "EXPERIMENT 59 — KATSUYAMA SERIAL-DEPENDENCE AUDIT",
    "=" * 65,
    "",
    f"Total sequence records: {len(df)}",
    f"Numeric isotope observations: {len(valid)}",
    f"Missing isotope observations: {df['DSMSNI_δ15Nbulk_2020'].isna().sum()}",
    f"Strictly adjacent complete pairs: {len(pairs_df)}",
    "",
    "NEIGHBOURING-SAMPLE ASSOCIATION",
    "-" * 40,
    f"Raw isotope lag-1 Pearson correlation: {raw_lag1:.6f}",
    f"Detrended residual lag-1 Pearson correlation: {residual_lag1:.6f}",
    "",
    "TREND-RESIDUAL VARIATION",
    "-" * 40,
    f"Observed isotope standard deviation: {isotope_sd:.6f}‰",
    f"Residual standard deviation: {residual_sd:.6f}‰",
    "",
    "INTERPRETATION",
    "-" * 40,
    "The raw lag-1 correlation measures similarity between neighbouring measured samples.",
    "The residual lag-1 correlation measures neighbouring similarity after removing a fitted linear Theil-Sen trend.",
    "Only directly adjacent records with two available isotope measurements were included.",
    "These diagnostics are descriptive and do not establish a formal dependence model.",
    "The small number of observations limits reliable estimation of serial dependence.",
    "No p-values are used to claim independent geological evidence.",
    "",
    "NEIGHBOURING PAIRS",
    "-" * 40,
]

for _, r in pairs_df.iterrows():
    summary.append(
        f"{r['upper_sample']} -> {r['lower_sample']}: "
        f"depth gap={r['depth_gap_m']:.3f} m; "
        f"isotope={r['upper_isotope']:.3f}->{r['lower_isotope']:.3f}‰"
    )

summary.extend([
    "",
    f"Output: {output}"
])

pairs_df.to_csv(output, index=False, encoding="utf-8-sig")
summary_file.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
print("\nSaved:", output)
print("Saved:", summary_file)
