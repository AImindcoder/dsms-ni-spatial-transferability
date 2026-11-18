# Historical 37-Fold ExtraTrees Experiment

## Status

Archived experimental outputs with independently verified numerical
metrics and recorded fold assignments.

The original model-training implementation has not yet been identified.
These artifacts must not be described as a complete end-to-end
reproduction of model training.

## Study design

- Source: EXP_SPATIAL_CV_CENOZOIC_COMPLETE_CASE_AUDIT.csv
- Source complete-case records: 44,790
- Historical evaluation cohort: 37,118
- Eligible spatial blocks: 37
- Sequences represented in held-out predictions: 1,137
- Publications represented in held-out predictions: 140
- Spatial grid: 15-degree latitude by 30-degree longitude
- Longitude adjacency: cyclic
- Buffer: one edge-sharing eligible neighboring block
- Publication exclusion: all publications represented in the test block
- Sequence overlap requirement: zero shared sequences

## Model configuration

- Estimator: ExtraTreesRegressor
- n_estimators: 100
- max_depth: 18
- min_samples_leaf: 3
- max_features: 1.0
- random_state: 20261111
- n_jobs: 2

## Historical results

| Metric | ExtraTrees | Median baseline |
| --- | ---: | ---: |
| N | 37118 | 37118 |
| MAE | 1.8745134694 | 1.6735314403 |
| RMSE | 2.3702212108 | 2.1973591779 |
| R2 | -0.2309038646 | -0.0579094065 |
| Sequence-balanced MAE | 1.9550324138 | 1.9067257981 |
| Publication-balanced MAE | 1.9470384391 | 1.7554532693 |
| Block-balanced MAE | 1.7803621123 | 1.6694145813 |

## Completed independent checks

1. Pooled MAE, RMSE and R2 reconstructed from OOF predictions.
2. Fold-level test counts, MAE and RMSE reconstructed.
3. OOF test row IDs matched to the manifest.
4. No train/test row-ID overlap detected within any fold.
5. Publication and sequence separation reconstructed using the
   source complete-case dataset and 1,291,312 manifest assignments.
6. Spatial buffer neighbors reconstructed from the 37 eligible blocks,
   including cyclic longitude adjacency.
7. No training assignments found inside test or buffered eligible blocks.
8. All six balanced MAE statistics reconstructed from OOF predictions.

## Verification scripts

- docs/verify_historical_extratrees.py
- docs/verify_historical_independence.py
- docs/verify_historical_spatial_buffer.py
- docs/verify_historical_balanced_mae.py

The independence and spatial-buffer verification scripts require
access to the original complete-case source dataset at the configured
local path. They are not portable without adjusting that path.

## Important limitations

- Original training code has not yet been recovered.
- The audits verify archived predictions and recorded assignments,
  not a fresh model-training run.
- Complete reconstruction of the eligible cohort and every training
  inclusion decision remains a separate task.
- Redistribution rights for the source data and derived records
  require review before public release.
- This historical experiment is distinct from the independently
  archived five-model benchmark.

## Data availability

Historical artifacts are initially maintained in a private repository
pending provenance, redistribution, and manuscript consistency review.
