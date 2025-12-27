from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional
from .trends import calculate_slope


@dataclass
class SeasonTrendResult:
	dates: pd.DatetimeIndex
	deterministic: pd.Series  # season + trend future deterministic extension
	season_component: pd.Series
	trend_component: pd.Series
	residuals: pd.Series


def build_robust_season_trend_series(
	series: pd.Series,
	*,
	freq: Literal["doy", "month"] = "doy",
	future_days: int = 180,
	min_history: int = 90,
	clamp_min: Optional[float] = 0.0,
    seasonal_agg: Literal["median", "mean"] = "median",
) -> SeasonTrendResult:
	r"""Decompose daily series into robust seasonal + Theil–Sen trend and extend deterministically.

	Model assumes an additive decomposition: $Y_t = T_t + S_t + \epsilon_t$,
	where $T_t$ is the linear trend estimated via Theil-Sen (median slope) and
	$S_t$ is the robust seasonal component (median of indices).

	Parameters
	----------
	series : pd.Series
		Date-indexed (daily) numeric series.
	freq : {'doy','month'}
		Seasonal grouping (day-of-year or month-of-year).
	future_days : int
		Forward deterministic extension length.
	min_history : int
		Minimum number of observations required.
	clamp_min : Optional[float], default=0.0
		Lower bound for forecasted values. Essential for physical variables (P, ET)
		to prevent negative predictions ("anti-rain") from drying trends.
		Set to None to disable clamping.
	seasonal_agg : {'median', 'mean'}, default='median'
		Aggregation method for seasonal template.
		- 'median': Robust to outliers (default).
		- 'mean': Conserves mass. Preferred for zero-inflated variables like
		  Precipitation in arid regions where median might be 0.
	"""
	s = series.dropna().sort_index()
	if len(s) < min_history:
		raise ValueError("Insufficient history for season+trend model")

	idx = s.index
	if freq == "doy":
		key = idx.dayofyear
	else:
		key = idx.month

	if seasonal_agg == "mean":
		seasonal_map = s.groupby(key).mean()
	else:
		seasonal_map = s.groupby(key).median()

	season_component = pd.Series(index=idx, dtype=float)
	for k, val in seasonal_map.items():
		season_component.loc[key == k] = val

	detrended = s - season_component
	x = (idx - idx[0]).days.to_numpy()
	slope = calculate_slope(detrended.to_numpy(), x)

	# Axiom: If using mean seasonality, we likely want mass conservation,
	# so we should use the Mean for the intercept as well, rather than Median.
	if seasonal_agg == "mean":
		intercept = float(np.mean(detrended.to_numpy() - slope * x))
	else:
		intercept = float(np.median(detrended.to_numpy() - slope * x))

	trend_component = pd.Series(intercept + slope * x, index=idx)
	deterministic_hist = season_component + trend_component
	residuals = s - deterministic_hist

	future_index = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
	if freq == "doy":
		future_key = future_index.dayofyear
	else:
		future_key = future_index.month

	# Robust Seasonality Lookup (handling missing keys like Feb 29)
	def get_seasonal_val(k):
		if k in seasonal_map:
			return seasonal_map[k]

		# Fallback strategy:
		if freq == "doy" and k == 60:
			val_59 = seasonal_map.get(59)
			val_61 = seasonal_map.get(61)
			if val_59 is not None and val_61 is not None:
				return (val_59 + val_61) / 2.0
			if val_59 is not None: return val_59
			if val_61 is not None: return val_61

		# Fallback to global central tendency
		if seasonal_agg == "mean":
			return seasonal_map.mean()
		return seasonal_map.median()

	future_season = pd.Series([get_seasonal_val(k) for k in future_key], index=future_index)
	x_future = (future_index - idx[0]).days.to_numpy()
	future_trend = pd.Series(intercept + slope * x_future, index=future_index)
	deterministic_future = future_season + future_trend

	# Phase 1: The "Physics Guard"
	if clamp_min is not None:
		deterministic_future = deterministic_future.clip(lower=clamp_min)

	return SeasonTrendResult(
		dates=future_index,
		deterministic=deterministic_future,
		season_component=season_component,
		trend_component=trend_component,
		residuals=residuals,
	)


__all__ = [
	"SeasonTrendResult",
	"build_robust_season_trend_series",
]
