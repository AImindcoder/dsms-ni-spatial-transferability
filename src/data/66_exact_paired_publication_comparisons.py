from pathlib import Path
from itertools import combinations
from scipy.stats import binomtest, wilcoxon
import numpy as np
import pandas as pd

INPUT = Path("06_Validation/61_chronology_publication_level_metrics.csv")
OUT = Path("06_Validation")
B = 20000
SEED = 66066
rng = np.random.default_rng(SEED)

df = pd.read_csv(INPUT)
df = df[[
    "resolution_proxy",
    "provisional_source_group_id",
    "publication_MAE"
]].copy()

df["publication_MAE"] = pd.to_numeric(
    df["publication_MAE"], errors="coerce"
)
df = df.dropna()

if df.duplicated(
    ["resolution_proxy", "provisional_source_group_id"]
).any():
    raise ValueError("Duplicate publication-category records detected.")

wide = df.pivot(
    index="provisional_source_group_id",
    columns="resolution_proxy",
    values="publication_MAE"
)

categories = sorted(df["resolution_proxy"].unique())
detail_rows = []
summary_rows = []

print("=" * 76)
print("EXPERIMENT 66: EXACT PAIRED PUBLICATION COMPARISONS")
print("=" * 76)

for a, b in combinations(categories, 2):
    paired = wide[[a, b]].dropna()
    differences = (
        paired[a].to_numpy(dtype=float)
        - paired[b].to_numpy(dtype=float)
    )

    n = len(differences)

    if n == 0:
        continue

    for doi, difference in zip(paired.index, differences):
        detail_rows.append({
            "category_A": a,
            "category_B": b,
            "publication": doi,
            "MAE_A_minus_B": difference
        })

    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    zero = int(np.sum(differences == 0))
    nonzero = positive + negative

    sign_p = (
        binomtest(
            positive, nonzero, p=0.5, alternative="two-sided"
        ).pvalue
        if nonzero else np.nan
    )

    if n >= 2 and not np.all(differences == differences[0]):
        try:
            wilcoxon_p = wilcoxon(
                differences,
                alternative="two-sided",
                method="auto"
            ).pvalue
        except ValueError:
            wilcoxon_p = np.nan
    else:
        wilcoxon_p = np.nan

    bootstrap_means = np.empty(B)

    for r in range(B):
        sample = rng.choice(differences, size=n, replace=True)
        bootstrap_means[r] = sample.mean()

    ci_low, ci_high = np.quantile(
        bootstrap_means, [0.025, 0.975]
    )

    summary_rows.append({
        "category_A": a,
        "category_B": b,
        "shared_publications": n,
        "mean_paired_difference_A_minus_B": differences.mean(),
        "median_paired_difference_A_minus_B": np.median(differences),
        "positive_differences": positive,
        "negative_differences": negative,
        "zero_differences": zero,
        "exact_sign_test_p": sign_p,
        "wilcoxon_signed_rank_p": wilcoxon_p,
        "paired_bootstrap_95CI_lower": ci_low,
        "paired_bootstrap_95CI_upper": ci_high
    })

summary = pd.DataFrame(summary_rows)
details = pd.DataFrame(detail_rows)

summary.to_csv(
    OUT / "66_exact_paired_publication_summary.csv", index=False
)
details.to_csv(
    OUT / "66_exact_paired_publication_details.csv", index=False
)

print("\nPAIRED PUBLICATION SUMMARY")
print(summary.round(5).to_string(index=False))

print("\nINDIVIDUAL PUBLICATION DIFFERENCES")
print(details.round(5).to_string(index=False))

print("\nFILES SAVED")
print("06_Validation/66_exact_paired_publication_summary.csv")
print("06_Validation/66_exact_paired_publication_details.csv")

print("\nBootstrap unit: paired publication-level difference.")
print("Bootstrap replicates:", B)
print("Random seed:", SEED)
print("Sign test excludes exact zero differences.")
print("These are exploratory comparisons; small paired samples")
print("and multiple comparisons limit inferential strength.")
print("\nEXPERIMENT 66 COMPLETED")
