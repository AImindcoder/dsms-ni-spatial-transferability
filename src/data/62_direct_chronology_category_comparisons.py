from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

BASE = Path("06_Validation")
INPUT = BASE / "61_chronology_publication_level_metrics.csv"

B = 10000
SEED = 62062
rng = np.random.default_rng(SEED)

print("=" * 76)
print("EXPERIMENT 62: DIRECT PUBLICATION-CLUSTER CHRONOLOGY COMPARISONS")
print("=" * 76)

pub = pd.read_csv(INPUT)

required = {
    "resolution_proxy",
    "provisional_source_group_id",
    "publication_MAE",
    "observations"
}
missing = required - set(pub.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

pub = pub[list(required)].copy()
pub["provisional_source_group_id"] = (
    pub["provisional_source_group_id"].astype(str).str.strip()
)
pub["publication_MAE"] = pd.to_numeric(
    pub["publication_MAE"], errors="coerce"
)
pub = pub.dropna(subset=[
    "resolution_proxy", "provisional_source_group_id", "publication_MAE"
])

if pub.duplicated(
    ["resolution_proxy", "provisional_source_group_id"]
).any():
    raise ValueError("Duplicate publication-category rows detected.")

categories = sorted(pub["resolution_proxy"].unique())
publication_ids = sorted(pub["provisional_source_group_id"].unique())

# A publication may contribute to more than one chronology category.
category_counts = pub.groupby("resolution_proxy")[
    "provisional_source_group_id"
].nunique()

overlap_rows = []
for a, b in combinations(categories, 2):
    ids_a = set(pub.loc[
        pub["resolution_proxy"] == a, "provisional_source_group_id"
    ])
    ids_b = set(pub.loc[
        pub["resolution_proxy"] == b, "provisional_source_group_id"
    ])
    overlap_rows.append({
        "category_A": a,
        "category_B": b,
        "publications_A": len(ids_a),
        "publications_B": len(ids_b),
        "shared_publications": len(ids_a & ids_b)
    })

overlap = pd.DataFrame(overlap_rows)

# One publication is resampled as a complete cluster.
# All category contributions belonging to that publication are retained.
matrix = pub.pivot(
    index="provisional_source_group_id",
    columns="resolution_proxy",
    values="publication_MAE"
).reindex(index=publication_ids, columns=categories)

values = matrix.to_numpy(dtype=float)
n_publications = len(publication_ids)
category_index = {c: i for i, c in enumerate(categories)}

observed_means = {}
for c in categories:
    observed_means[c] = float(pub.loc[
        pub["resolution_proxy"] == c, "publication_MAE"
    ].mean())

comparison_rows = []

for a, b in combinations(categories, 2):
    ia = category_index[a]
    ib = category_index[b]
    observed_difference = observed_means[a] - observed_means[b]

    boot_differences = np.empty(B)
    valid = 0

    for r in range(B):
        sampled = rng.integers(0, n_publications, size=n_publications)
        sampled_values = values[sampled, :]

        va = sampled_values[:, ia]
        vb = sampled_values[:, ib]

        ma = va[np.isfinite(va)].mean() if np.isfinite(va).any() else np.nan
        mb = vb[np.isfinite(vb)].mean() if np.isfinite(vb).any() else np.nan

        if np.isfinite(ma) and np.isfinite(mb):
            boot_differences[valid] = ma - mb
            valid += 1

    if valid < B * 0.95:
        raise ValueError(
            f"Too many invalid bootstrap replicates for {a} vs {b}: {valid}/{B}"
        )

    boot_differences = boot_differences[:valid]

    comparison_rows.append({
        "category_A": a,
        "category_B": b,
        "publications_A": int(category_counts[a]),
        "publications_B": int(category_counts[b]),
        "observed_macro_MAE_A": observed_means[a],
        "observed_macro_MAE_B": observed_means[b],
        "difference_A_minus_B": observed_difference,
        "bootstrap_95CI_lower": np.quantile(boot_differences, 0.025),
        "bootstrap_95CI_upper": np.quantile(boot_differences, 0.975),
        "bootstrap_probability_difference_positive": np.mean(
            boot_differences > 0
        ),
        "valid_bootstrap_replicates": valid
    })

comparisons = pd.DataFrame(comparison_rows)

# Sensitivity analysis: paired comparisons among publications represented
# in both categories. Resampling is by shared publication.
paired_rows = []

for a, b in combinations(categories, 2):
    ia = category_index[a]
    ib = category_index[b]

    common = matrix[[a, b]].dropna()
    n_common = len(common)

    if n_common < 2:
        paired_rows.append({
            "category_A": a,
            "category_B": b,
            "shared_publications": n_common,
            "paired_mean_difference_A_minus_B": np.nan,
            "paired_95CI_lower": np.nan,
            "paired_95CI_upper": np.nan
        })
        continue

    differences = (
        common[a].to_numpy() - common[b].to_numpy()
    )
    boot = np.empty(B)

    for r in range(B):
        sampled = rng.integers(0, n_common, size=n_common)
        boot[r] = differences[sampled].mean()

    paired_rows.append({
        "category_A": a,
        "category_B": b,
        "shared_publications": n_common,
        "paired_mean_difference_A_minus_B": differences.mean(),
        "paired_95CI_lower": np.quantile(boot, 0.025),
        "paired_95CI_upper": np.quantile(boot, 0.975)
    })

paired = pd.DataFrame(paired_rows)

category_summary = (
    pub.groupby("resolution_proxy")
    .agg(
        publications=("provisional_source_group_id", "nunique"),
        publication_category_cells=("publication_MAE", "size"),
        macro_publication_MAE=("publication_MAE", "mean"),
        median_publication_MAE=("publication_MAE", "median"),
        publication_MAE_SD=("publication_MAE", "std")
    )
    .reset_index()
)

category_summary.to_csv(
    BASE / "62_chronology_category_summary.csv", index=False
)
overlap.to_csv(
    BASE / "62_publication_category_overlap.csv", index=False
)
comparisons.to_csv(
    BASE / "62_direct_category_bootstrap_comparisons.csv", index=False
)
paired.to_csv(
    BASE / "62_shared_publication_paired_sensitivity.csv", index=False
)

print("\nCATEGORY SUMMARY")
print(category_summary.round(4).to_string(index=False))

print("\nPUBLICATION OVERLAP BETWEEN CATEGORIES")
print(overlap.to_string(index=False))

print("\nDIRECT CLUSTER-BOOTSTRAP COMPARISONS")
print(comparisons.round(4).to_string(index=False))

print("\nSHARED-PUBLICATION PAIRED SENSITIVITY")
print(paired.round(4).to_string(index=False))

print("\nFILES SAVED")
print("06_Validation/62_chronology_category_summary.csv")
print("06_Validation/62_publication_category_overlap.csv")
print("06_Validation/62_direct_category_bootstrap_comparisons.csv")
print("06_Validation/62_shared_publication_paired_sensitivity.csv")

print("\nBootstrap unit: complete publication cluster")
print("Bootstrap replicates:", B)
print("Random seed:", SEED)
print("No raw data or previous experiment outputs modified.")
print("\nEXPERIMENT 62 COMPLETED")
