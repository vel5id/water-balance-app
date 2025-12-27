## 2024-05-23 - Temporal Distortion in Climatology Indexing
**Discovery:** The legacy leap-year handling maps Feb 29th (Leap) to Day 59, grouping it with Feb 28th. Consequently, indices 60+ in the climatology represent a mix of dates depending on leap/non-leap status (e.g., Index 60 is Mar 1 in Non-Leap, but Mar 1 is 61 in Leap).
**Protocol:** We have formalized this behavior in `get_climatology_index` to prevent accidental refactoring that would change numerical results. Future versions (v2.0) should adopt a strict 1-366 mapping or a robust `dayofyear` strategy, but for now, we preserve the quirk for reproducibility.

## 2024-05-23 - Negative Forecasts from Linear Trends
**Discovery:** Identified that naive linear extrapolation of drying trends breaks physical laws (negative water). A Theil-Sen estimator on a drying trend will eventually predict negative values, which are physically impossible for precipitation/evaporation.
**Protocol:** Implemented a `clamp_min` guard (default 0.0) in `build_robust_season_trend_series` to enforce physical constraints.

## 2024-05-23 - Forecasting Performance Optimization
**Discovery:** The custom Theil-Sen estimator was $O(N^2)$, which is a scalability risk for daily data.
**Protocol:** Replaced with `scipy.stats.theilslopes` for standardized, robust implementation.
