## 2024-05-23 - Temporal Distortion in Climatology Indexing
**Discovery:** The legacy leap-year handling maps Feb 29th (Leap) to Day 59, grouping it with Feb 28th. Consequently, indices 60+ in the climatology represent a mix of dates depending on leap/non-leap status (e.g., Index 60 is Mar 1 in Non-Leap, but Mar 1 is 61 in Leap).
**Protocol:** We have formalized this behavior in `get_climatology_index` to prevent accidental refactoring that would change numerical results. Future versions (v2.0) should adopt a strict 1-366 mapping or a robust `dayofyear` strategy, but for now, we preserve the quirk for reproducibility.

## 2024-05-23 - Negative Forecasts from Linear Trends
**Discovery:** Identified that naive linear extrapolation of drying trends breaks physical laws (negative water). A Theil-Sen estimator on a drying trend will eventually predict negative values, which are physically impossible for precipitation/evaporation.
**Protocol:** Implemented a `clamp_min` guard (default 0.0) in `build_robust_season_trend_series` to enforce physical constraints.

## 2024-05-23 - Forecasting Performance Optimization
**Discovery:** The custom Theil-Sen estimator was $O(N^2)$, which is a scalability risk for daily data.
**Protocol:** Replaced with `scipy.stats.theilslopes` for standardized, robust implementation.

## 2025-12-08 - [Ensemble Physicality]
**Discovery:** Identified that adding bootstrapped residuals to low-value deterministic baselines (like arid region precipitation) results in negative values ("Anti-Rain").
**Discovery:** The assumption of Deterministic Evaporation in ensembles artificially narrows the uncertainty cone during droughts.
**Protocol:** All ensemble generators for physical quantities must accept a `clamp_min` argument. Distributions must be truncated, not just shifted. `run_volume_ensemble` has been refactored to support ET uncertainty via `et_residual_sets`.

## 2025-12-08 - [Math Consolidation & Metrics]
**Discovery:** Identified duplicated $O(N^2)$ Theil-Sen loops across `forecast.py`, `trends.py`, and `seasonal.py`.
**Protocol:** Consolidated trend logic into `wbm/trends.py` as the authority, using `scipy.stats.theilslopes` ($O(N \log N)$).
**Discovery:** Identified "Metric Singularity" risk where NSE/KGE crash on constant observed data (zero variance).
**Protocol:** Implemented Epsilon Guards in `wbm/analysis.py` to handle zero-variance cases gracefully (returning `-inf`).
