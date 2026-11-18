from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

INPUT = Path("06_Validation/61_chronology_publication_level_metrics.csv")
OUT = Path("06_Validation")

df = pd.read_csv(INPUT)

cols = [
    "resolution_proxy", "provisional_source_group_id",
    "publication_MAE", "observations", "isotope_mean", "isotope_SD"
]
df = df[cols].copy()

for c in cols[2:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna().copy()
df = df[
    (df["publication_MAE"] >= 0)
    & (df["observations"] >= 1)
    & (df["isotope_SD"] >= 0)
].copy()

# Retain publications contributing to at least two chronology categories.
category_counts = df.groupby(
    "provisional_source_group_id"
)["resolution_proxy"].nunique()

eligible = category_counts[category_counts >= 2].index
within = df[
    df["provisional_source_group_id"].isin(eligible)
].copy()

within["log_MAE"] = np.log1p(within["publication_MAE"])
within["log_isotope_SD"] = np.log1p(within["isotope_SD"])
within["absolute_isotope_mean"] = within["isotope_mean"].abs()
within["log_observations"] = np.log1p(within["observations"])

print("=" * 76)
print("EXPERIMENT 65: WITHIN-PUBLICATION FIXED-EFFECTS ANALYSIS")
print("=" * 76)

print("\nWITHIN-PUBLICATION SAMPLE")
print("Publication-category rows:", len(within))
print("Publications with multiple categories:", len(eligible))
print("Chronology categories:", within["resolution_proxy"].nunique())

overlap = (
    within.groupby("resolution_proxy")
    .agg(
        publication_category_rows=("publication_MAE", "size"),
        publications=("provisional_source_group_id", "nunique"),
        mean_MAE=("publication_MAE", "mean")
    )
    .reset_index()
)

print("\nCATEGORY COVERAGE")
print(overlap.round(4).to_string(index=False))

if len(eligible) < 10:
    raise ValueError(
        "Too few publications with multiple categories for this analysis."
    )

# Publication fixed effects absorb all time-invariant publication-level factors.
formula = (
    "log_MAE ~ C(resolution_proxy, "
    "Treatment(reference='monotonic_age_depth_candidate'))"
    " + log_isotope_SD + absolute_isotope_mean + log_observations"
    " + C(provisional_source_group_id)"
)

model = smf.ols(formula, data=within).fit(
    cov_type="cluster",
    cov_kwds={"groups": within["provisional_source_group_id"]}
)

ci = model.conf_int()
rows = []

for term in model.params.index:
    if term.startswith("C(resolution_proxy"):
        rows.append({
            "term": term,
            "coefficient": model.params[term],
            "cluster_robust_SE": model.bse[term],
            "p_value": model.pvalues[term],
            "CI_lower": ci.loc[term, 0],
            "CI_upper": ci.loc[term, 1]
        })

effects = pd.DataFrame(rows)

print("\nPUBLICATION FIXED-EFFECTS MODEL")
print(model.summary())

print("\nCHRONOLOGY CATEGORY EFFECTS")
if len(effects):
    print(effects.round(5).to_string(index=False))
else:
    print("No estimable chronology category contrasts.")

print("\nWITHIN-PUBLICATION CATEGORY PAIRS")
pair_rows = []

for doi, g in within.groupby("provisional_source_group_id"):
    values = g.set_index("resolution_proxy")["publication_MAE"]
    for a in values.index:
        for b in values.index:
            if a < b:
                pair_rows.append({
                    "publication": doi,
                    "category_A": a,
                    "category_B": b,
                    "MAE_A_minus_B": values[a] - values[b]
                })

pairs = pd.DataFrame(pair_rows)

if len(pairs):
    pair_summary = (
        pairs.groupby(["category_A", "category_B"])
        .agg(
            shared_publications=("publication", "nunique"),
            mean_within_publication_difference=("MAE_A_minus_B", "mean"),
            median_within_publication_difference=("MAE_A_minus_B", "median")
        )
        .reset_index()
    )
    print(pair_summary.round(4).to_string(index=False))
else:
    pair_summary = pd.DataFrame()

effects.to_csv(
    OUT / "65_within_publication_fixed_effects.csv", index=False
)
overlap.to_csv(
    OUT / "65_within_publication_category_coverage.csv", index=False
)
pairs.to_csv(
    OUT / "65_within_publication_pair_differences.csv", index=False
)
pair_summary.to_csv(
    OUT / "65_within_publication_pair_summary.csv", index=False
)

print("\nFILES SAVED")
print("06_Validation/65_within_publication_fixed_effects.csv")
print("06_Validation/65_within_publication_category_coverage.csv")
print("06_Validation/65_within_publication_pair_differences.csv")
print("06_Validation/65_within_publication_pair_summary.csv")

print("\nCaution: estimates apply only to publications represented in")
print("multiple chronology categories. Small overlap may limit precision.")
print("Publication fixed effects do not establish causal effects.")
print("\nEXPERIMENT 65 COMPLETED")
