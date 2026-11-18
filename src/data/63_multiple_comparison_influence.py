from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

INPUT = Path("06_Validation/61_chronology_publication_level_metrics.csv")
OUT = Path("06_Validation")
B = 10000
SEED = 63063
rng = np.random.default_rng(SEED)

print("=" * 76)
print("EXPERIMENT 63: MULTIPLE-COMPARISON AND PUBLICATION-INFLUENCE ANALYSIS")
print("=" * 76)

pub = pd.read_csv(INPUT)
pub = pub[[
    "resolution_proxy",
    "provisional_source_group_id",
    "publication_MAE"
]].copy()

pub["publication_MAE"] = pd.to_numeric(
    pub["publication_MAE"], errors="coerce"
)
pub = pub.dropna()

if pub.duplicated(
    ["resolution_proxy", "provisional_source_group_id"]
).any():
    raise ValueError("Duplicate publication-category records detected.")

categories = sorted(pub["resolution_proxy"].unique())
ids = sorted(pub["provisional_source_group_id"].unique())

matrix = pub.pivot(
    index="provisional_source_group_id",
    columns="resolution_proxy",
    values="publication_MAE"
).reindex(index=ids, columns=categories)

X = matrix.to_numpy(dtype=float)
n = len(ids)
pairs = list(combinations(categories, 2))

def category_mean(x, j):
    v = x[:, j]
    v = v[np.isfinite(v)]
    return v.mean() if len(v) else np.nan

observed = np.array([
    category_mean(X, j) for j in range(len(categories))
])

observed_diffs = np.array([
    observed[categories.index(a)] - observed[categories.index(b)]
    for a, b in pairs
])

# Publication-cluster bootstrap: entire publication rows are resampled.
boot = np.full((B, len(pairs)), np.nan)

for r in range(B):
    sampled = rng.integers(0, n, size=n)
    xb = X[sampled, :]

    means = np.array([
        category_mean(xb, j) for j in range(len(categories))
    ])

    for k, (a, b) in enumerate(pairs):
        ia, ib = categories.index(a), categories.index(b)
        if np.isfinite(means[ia]) and np.isfinite(means[ib]):
            boot[r, k] = means[ia] - means[ib]

valid_rows = np.isfinite(boot).all(axis=1)
boot_valid = boot[valid_rows]

if len(boot_valid) < B * 0.95:
    raise ValueError(
        f"Only {len(boot_valid)} valid simultaneous bootstrap replicates."
    )

# Family-wise simultaneous max-t confidence intervals.
boot_sd = boot_valid.std(axis=0, ddof=1)
if np.any(boot_sd <= 0):
    raise ValueError("A comparison has zero bootstrap standard deviation.")

standardized = np.abs(
    (boot_valid - observed_diffs) / boot_sd
)
max_statistics = standardized.max(axis=1)
critical_value = np.quantile(max_statistics, 0.95)

comparison_rows = []

for k, (a, b) in enumerate(pairs):
    d = observed_diffs[k]
    lower = d - critical_value * boot_sd[k]
    upper = d + critical_value * boot_sd[k]

    comparison_rows.append({
        "category_A": a,
        "category_B": b,
        "difference_A_minus_B": d,
        "simultaneous_95CI_lower": lower,
        "simultaneous_95CI_upper": upper,
        "bootstrap_probability_positive": np.mean(
            boot_valid[:, k] > 0
        ),
        "bootstrap_SD": boot_sd[k],
        "familywise_critical_value": critical_value,
        "valid_replicates": len(boot_valid)
    })

simultaneous = pd.DataFrame(comparison_rows)

# Leave-one-publication-out influence analysis.
influence_rows = []

key_pairs = [
    ("constant_age_multiple_depths_candidate",
     "monotonic_age_depth_candidate"),
    ("insufficient_information",
     "monotonic_age_depth_candidate")
]

for a, b in key_pairs:
    ia, ib = categories.index(a), categories.index(b)

    full_difference = (
        category_mean(X, ia) - category_mean(X, ib)
    )

    loo_differences = []

    for removed in range(n):
        keep = np.arange(n) != removed
        reduced = X[keep, :]

        ma = category_mean(reduced, ia)
        mb = category_mean(reduced, ib)

        if np.isfinite(ma) and np.isfinite(mb):
            loo_differences.append({
                "removed_publication": ids[removed],
                "difference": ma - mb
            })

    loo = pd.DataFrame(loo_differences)

    influence_rows.append({
        "category_A": a,
        "category_B": b,
        "full_difference": full_difference,
        "minimum_LOO_difference": loo["difference"].min(),
        "maximum_LOO_difference": loo["difference"].max(),
        "LOO_positive_count": int((loo["difference"] > 0).sum()),
        "LOO_total": len(loo),
        "LOO_sign_reversal_count": int((loo["difference"] <= 0).sum()),
        "most_influential_publication": loo.loc[
            (loo["difference"] - full_difference).abs().idxmax(),
            "removed_publication"
        ],
        "largest_absolute_change": (
            loo["difference"] - full_difference
        ).abs().max()
    })

influence = pd.DataFrame(influence_rows)

# Robust descriptive sensitivity using 10% trimmed publication MAE.
trimmed_rows = []

for category, g in pub.groupby("resolution_proxy"):
    values = g["publication_MAE"].to_numpy()

    trimmed_rows.append({
        "resolution_proxy": category,
        "publications": len(values),
        "ordinary_macro_MAE": values.mean(),
        "trimmed_macro_MAE_10pct": trim_mean(values, proportiontocut=0.10),
        "maximum_publication_MAE": values.max(),
        "minimum_publication_MAE": values.min()
    })

trimmed = pd.DataFrame(trimmed_rows)

trimmed_comparisons = []

for a, b in key_pairs:
    ma = trimmed.loc[
        trimmed["resolution_proxy"] == a, "trimmed_macro_MAE_10pct"
    ].iloc[0]
    mb = trimmed.loc[
        trimmed["resolution_proxy"] == b, "trimmed_macro_MAE_10pct"
    ].iloc[0]

    trimmed_comparisons.append({
        "category_A": a,
        "category_B": b,
        "trimmed_difference_A_minus_B": ma - mb
    })

trimmed_comparisons = pd.DataFrame(trimmed_comparisons)

simultaneous.to_csv(
    OUT / "63_simultaneous_bootstrap_comparisons.csv", index=False
)
influence.to_csv(
    OUT / "63_leave_one_publication_out_influence.csv", index=False
)
trimmed.to_csv(
    OUT / "63_trimmed_publication_error_summary.csv", index=False
)
trimmed_comparisons.to_csv(
    OUT / "63_trimmed_key_category_comparisons.csv", index=False
)

print("\nSIMULTANEOUS FAMILY-WISE 95% CONFIDENCE INTERVALS")
print(simultaneous.round(4).to_string(index=False))

print("\nLEAVE-ONE-PUBLICATION-OUT INFLUENCE")
print(influence.round(4).to_string(index=False))

print("\nORDINARY VS 10% TRIMMED PUBLICATION MAE")
print(trimmed.round(4).to_string(index=False))

print("\nTRIMMED-MEAN KEY COMPARISONS")
print(trimmed_comparisons.round(4).to_string(index=False))

print("\nFILES SAVED")
print("06_Validation/63_simultaneous_bootstrap_comparisons.csv")
print("06_Validation/63_leave_one_publication_out_influence.csv")
print("06_Validation/63_trimmed_publication_error_summary.csv")
print("06_Validation/63_trimmed_key_category_comparisons.csv")

print("\nBootstrap unit: complete publication cluster")
print("Bootstrap replicates:", B)
print("Valid simultaneous replicates:", len(boot_valid))
print("Random seed:", SEED)
print("No raw data or previous experiment outputs modified.")
print("\nEXPERIMENT 63 COMPLETED")
