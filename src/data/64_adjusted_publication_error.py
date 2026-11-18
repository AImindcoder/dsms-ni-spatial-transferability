from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

INPUT = Path("06_Validation/61_chronology_publication_level_metrics.csv")
OUT = Path("06_Validation")

print("=" * 76)
print("EXPERIMENT 64: PUBLICATION-LEVEL ADJUSTED ERROR ANALYSIS")
print("=" * 76)

df = pd.read_csv(INPUT)

required = [
    "resolution_proxy",
    "provisional_source_group_id",
    "publication_MAE",
    "observations",
    "isotope_mean",
    "isotope_SD"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required].copy()

for c in [
    "publication_MAE", "observations",
    "isotope_mean", "isotope_SD"
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=required).copy()
df = df[
    (df["publication_MAE"] >= 0)
    & (df["observations"] >= 1)
    & (df["isotope_SD"] >= 0)
].copy()

if df["provisional_source_group_id"].nunique() < 30:
    raise ValueError("Too few publication clusters for this analysis.")

reference = "monotonic_age_depth_candidate"
if reference not in df["resolution_proxy"].unique():
    raise ValueError("Expected reference chronology category is absent.")

df["log_MAE"] = np.log1p(df["publication_MAE"])
df["log_isotope_SD"] = np.log1p(df["isotope_SD"])
df["absolute_isotope_mean"] = df["isotope_mean"].abs()
df["log_observations"] = np.log1p(df["observations"])

# Model 1: chronology category only.
formula_unadjusted = (
    "log_MAE ~ C(resolution_proxy, "
    "Treatment(reference='monotonic_age_depth_candidate'))"
)

# Model 2: adjust for isotope spread, mean magnitude and sample count.
formula_adjusted = (
    "log_MAE ~ C(resolution_proxy, "
    "Treatment(reference='monotonic_age_depth_candidate'))"
    " + log_isotope_SD"
    " + absolute_isotope_mean"
    " + log_observations"
)

groups = df["provisional_source_group_id"]

model1 = smf.ols(formula_unadjusted, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups}
)

model2 = smf.ols(formula_adjusted, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": groups}
)

def extract(model, label):
    ci = model.conf_int()
    rows = []

    for term in model.params.index:
        rows.append({
            "model": label,
            "term": term,
            "coefficient_log_scale": model.params[term],
            "cluster_robust_SE": model.bse[term],
            "p_value": model.pvalues[term],
            "CI_lower": ci.loc[term, 0],
            "CI_upper": ci.loc[term, 1]
        })

    return rows

results = pd.DataFrame(
    extract(model1, "Unadjusted") +
    extract(model2, "Adjusted")
)

category_stats = (
    df.groupby("resolution_proxy")
    .agg(
        publication_category_rows=("publication_MAE", "size"),
        publications=("provisional_source_group_id", "nunique"),
        observations=("observations", "sum"),
        mean_publication_MAE=("publication_MAE", "mean"),
        median_isotope_SD=("isotope_SD", "median"),
        median_isotope_mean=("isotope_mean", "median"),
        median_observations=("observations", "median")
    )
    .reset_index()
)

print("\nANALYTIC SAMPLE")
print("Publication-category rows:", len(df))
print("Unique publication clusters:",
      df["provisional_source_group_id"].nunique())
print("Reference category:", reference)

print("\nCATEGORY-LEVEL COVARIATE SUMMARY")
print(category_stats.round(4).to_string(index=False))

print("\nUNADJUSTED MODEL")
print(model1.summary())

print("\nADJUSTED MODEL")
print(model2.summary())

print("\nCOEFFICIENT TABLE")
print(results.round(5).to_string(index=False))

results.to_csv(
    OUT / "64_adjusted_publication_error_model_coefficients.csv",
    index=False
)
category_stats.to_csv(
    OUT / "64_adjusted_publication_error_category_summary.csv",
    index=False
)

print("\nFILES SAVED")
print("06_Validation/64_adjusted_publication_error_model_coefficients.csv")
print("06_Validation/64_adjusted_publication_error_category_summary.csv")

print("\nInterpretation note:")
print("Coefficients are on the log(1 + publication MAE) scale.")
print("Standard errors are clustered by publication.")
print("This is an adjusted observational association, not causal evidence.")
print("\nEXPERIMENT 64 COMPLETED")
