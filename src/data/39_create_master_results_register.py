from pathlib import Path
from datetime import date

out = Path("09_Research_Paper")
out.mkdir(parents=True, exist_ok=True)

content = r"""# Master Scientific Results Register

## Research project

**Title:** Provenance-Aware Machine Learning for Deep-Time Marine Nitrogen Isotopes: Quantifying Publication Effects, Data Heterogeneity, and Cross-Study Generalization.

**Dataset:** DSMS-NI v0.3, Yong Du et al. (2026), Earth System Science Data.

**Central question:** How do publication-level and geological-provenance differences influence machine-learning performance, uncertainty, and generalisation when modelling deep-time marine nitrogen isotope measurements?

**Register status:** Experimental results frozen pending final methodological review.

---

## 1. Dataset and provenance

- Raw dataset: 78,300 observations and 122 columns.
- Numeric bulk nitrogen isotope target: 66,060 observations.
- Target-eligible provisional source groups: 374.
- Matched geological-context modelling cohort: 24,012 observations.
- Matched cohort source groups: 239.
- Publication-held-out validation: five folds.
- Publication groups were kept entirely within individual folds.
- Source-group identities are provisional, not all independently verified.
- Raw data remain immutable. Corrections are documented in derived provenance mappings.

## 2. Publication-held-out versus random validation

Matched cohort: 24,012 observations, 239 provisional source groups.

| Validation | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Random | Full geological context | 0.8935 | 1.4440 | 0.8474 |
| Publication-held-out | Full geological context | 2.1433 | 3.1437 | 0.2768 |
| Random | Without Setting | 0.8631 | 1.4221 | 0.8520 |
| Publication-held-out | Without Setting | 2.3291 | 3.6485 | 0.0260 |

Interpretation:
Random row splitting substantially overstates performance relative to publication-held-out evaluation. The full-context MAE is approximately 2.40 times higher under publication-held-out validation.

This is evidence of a substantial cross-study generalisation gap. It does not establish that every random-split prediction is leaked.

## 3. Geological-context ablation

Coordinate-held-out validation, matched complete-case cohort: 24,012 observations.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Age + geography | 2.3870 | 3.9061 | -0.1164 |
| Full geological context | 2.0104 | 3.0014 | 0.3408 |
| Without Material | 2.0202 | 3.0098 | 0.3371 |
| Without Setting | 2.1453 | 3.2939 | 0.2061 |
| Without LithType | 2.0124 | 2.9957 | 0.3433 |
| Without Lithology | 2.0074 | 2.9796 | 0.3504 |

The full geological-context model improves pooled MAE by approximately 15.8% relative to the age-and-geography baseline in this coordinate-held-out cohort.

Removing Setting causes the largest deterioration among the tested individual feature ablations. This is a conditional model result, not proof of a causal geological effect.

## 4. Machine learning versus simple statistical baselines

Publication-held-out validation, 24,012 observations.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Full-context ML | 2.1433 | 3.1437 | 0.2768 |
| Training-mean baseline | 2.9137 | 3.7425 | -0.0249 |
| Training-median baseline | 2.9004 | 3.7438 | -0.0256 |

Full-context ML reduces MAE by:
- 26.44% relative to the training-mean baseline.
- 26.10% relative to the training-median baseline.

The ML model achieved lower MAE in all five folds.

These are simple statistical baselines, not an exhaustive comparison against advanced hierarchical or geostatistical models.

## 5. Source-group bootstrap: ML versus simple baselines

Experiment 35:
- 239 provisional source groups.
- 5,000 cluster-bootstrap repetitions.
- Resampling unit: complete source group.

| Comparison | Observed relative MAE improvement | 95% bootstrap CI |
|---|---:|---:|
| ML versus mean | 26.44% | 16.76% to 35.16% |
| ML versus median | 26.10% | 17.07% to 34.30% |

Macro-group MAE differences:
- ML versus mean: 0.7319 (95% CI 0.5449 to 0.9188).
- ML versus median: 0.6554 (95% CI 0.4773 to 0.8341).

These intervals quantify source-group sampling uncertainty conditional on the saved out-of-fold predictions and current group mapping. They do not include full model-training uncertainty.

## 6. Source-group bootstrap: contribution of Setting

Experiment 36:

| Quantity | Result |
|---|---:|
| Full-context MAE | 2.1433 |
| Without-Setting MAE | 2.3291 |
| Absolute improvement | 0.1858 |
| Relative improvement | 7.98% |
| Pooled absolute improvement 95% CI | -0.0217 to 0.4637 |
| Relative improvement 95% CI | -0.9648% to 18.6668% |
| Macro-group improvement | 0.4029 |
| Macro-group improvement 95% CI | 0.1701 to 0.6916 |

148 source groups had lower MAE with full context; 91 had lower MAE without Setting.

Interpretation:
The estimated benefit of Setting depends on the weighting scheme. The observation-weighted interval crosses zero, whereas the macro-group interval is positive. Both results must be reported.

## 7. Setting transfer across held-out publications

Experiment 37:

| Setting status | Observations | Source groups represented | Full-context MAE | Without-Setting MAE | Absolute improvement |
|---|---:|---:|---:|---:|---:|
| Known Setting | 12,772 | 136 | 1.8783 | 2.2011 | 0.3228 |
| Unseen Setting | 11,240 | 118 | 2.4445 | 2.4747 | 0.0301 |

A source group may contain both known and unseen Setting categories. The source-group counts therefore overlap and must not be added as if they were disjoint groups.

## 8. Setting transfer cluster bootstrap

Experiment 38:

**Known Setting**
- 136 source groups represented.
- Absolute improvement: 0.3228.
- Observation-weighted absolute improvement 95% CI: -0.0260 to 0.8278.
- Relative improvement 95% CI: -1.2868% to 31.4359%.
- Macro-group improvement 95% CI: 0.2232 to 0.9512.

**Unseen Setting**
- 118 source groups represented.
- Absolute improvement: 0.0301.
- Observation-weighted absolute improvement 95% CI: -0.1392 to 0.2464.
- Relative improvement 95% CI: -5.4709% to 10.4123%.
- Macro-group improvement 95% CI: -0.0497 to 0.5395.

Interpretation:
Setting has a larger estimated contribution when its category is represented in training. However, neither observation-weighted confidence interval excludes zero. The macro-group evidence is positive for known categories, but not conclusive for unseen categories.

The two subgroup analyses are not independent: 15 source groups appear in both subsets. Do not interpret them as a formal independent-populations comparison.

## 9. Target-distribution support and prediction error

Experiment 33B used the exact saved publication-held-out fold assignments.

239 source groups and 24,012 observations.

| Support diagnostic | Spearman rho with group MAE | p-value |
|---|---:|---:|
| Mean target distance from training mean | 0.1310 | 0.04299 |
| Fraction outside training 1st–99th percentiles | 0.4230 | 8.60567e-12 |
| Fraction outside training 5th–95th percentiles | 0.4982 | 2.13275e-16 |

Interpretation:
Source groups whose target distributions extend beyond training quantiles tend to have higher prediction errors. These are exploratory group-level associations. Shared fold-specific training distributions and potential dependence between source groups mean the reported p-values should not be treated as definitive independent-sample inference.

This diagnostic measures target-distribution support, not full multivariate geological support.

## 10. Scientific limitations to preserve

1. Provisional source groups are not all independently verified studies.
2. Group-aware validation is established methodology; do not claim it was invented here.
3. Random-split optimism is evidence of potential leakage and dependence, not proof of leakage for every record.
4. Setting labels are heterogeneous and not fully harmonised.
5. Known/unseen category status does not establish identical geological meaning across publications.
6. Missingness and complete-case selection may affect representativeness.
7. Categorical features may encode publication-specific terminology.
8. Feature ablation does not establish causality.
9. Bootstrap intervals are conditional on existing predictions and group assignments; they do not include retraining uncertainty.
10. Do not claim the model is universally superior to statistical, hierarchical, Bayesian, or geostatistical approaches.
11. Age proximity and geographic proximity alone do not establish joint geological support.
12. Preserve both observation-weighted and macro-group metrics.

## 11. Current conclusion

The current experiments support three main findings:

- Random row splitting gives substantially more optimistic performance than publication-held-out evaluation.
- Geological context improves predictive performance in the tested coordinate-held-out cohort, and the full-context model outperforms simple global mean and median baselines in publication-held-out evaluation.
- The contribution of Setting is heterogeneous and depends on whether its category is represented in training publications; the evidence is stronger under macro-group weighting than observation weighting.

The findings motivate provenance-aware validation and careful treatment of geological metadata when developing machine-learning models for deep-time isotope datasets.

## 12. Final manuscript status

Experimental phase: frozen provisionally.

Next work:
- Verify methods and saved artifact consistency.
- Prepare publication-quality figures and tables.
- Draft Introduction, Methods, Results, Discussion, and Limitations.
- Conduct final literature and citation verification.
- Avoid adding experiments unless a reproducibility or methodological issue is identified.
"""

path = out / "MASTER_RESULTS_REGISTER.md"
path.write_text(content, encoding="utf-8")

print("MASTER RESULTS REGISTER CREATED")
print(path.resolve())
print("Characters:", len(content))
print("Sections:", content.count("\n## "))
print("\nThis file consolidates the current experiments and their limitations.")
