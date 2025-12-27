from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional
try:
	from scipy import stats
	HAS_SCIPY = True
except ImportError:
	HAS_SCIPY = False


def _theil_sen_slope(y: np.ndarray, x: np.ndarray) -> float:
	"""Return Theil–Sen median slope for y vs x.

	Uses scipy.stats.theilslopes if available for standardized implementation.
	Falls back to a custom implementation if scipy is missing.
	"""
	n = len(y)
	if n < 2:
		return 0.0

	if HAS_SCIPY:
		# scipy.stats.theilslopes returns (slope, intercept, low_slope, high_slope)
		# We only need the slope (median slope).
		# alpha=0.95 is default, impact is on confidence intervals which we ignore.
		res = stats.theilslopes(y, x, alpha=0.95)
		return float(res[0])

	# Fallback O(N^2) implementation
	slopes = []
	for i in range(n - 1):
		dy = y[i + 1 :] - y[i]
		dx = x[i + 1 :] - x[i]
		valid = dx != 0
		s = dy[valid] / dx[valid]
		if s.size:
			slopes.append(s)
	if not slopes:
		return 0.0
	all_slopes = np.concatenate(slopes)
	return float(np.median(all_slopes))


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
	"""
	s = series.dropna().sort_index()
	if len(s) < min_history:
		raise ValueError("Insufficient history for season+trend model")

	idx = s.index
	if freq == "doy":
		key = idx.dayofyear
	else:
		key = idx.month

	seasonal_map = s.groupby(key).median()
	season_component = pd.Series(index=idx, dtype=float)
	for k, val in seasonal_map.items():
		season_component.loc[key == k] = val

	detrended = s - season_component
	x = (idx - idx[0]).days.to_numpy()
	slope = _theil_sen_slope(detrended.to_numpy(), x)
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
		# If Feb 29 (DOY 60 in leap year context, but typically handled by dayofyear) is missing:
		# We check for neighbors.
		# Note: In standard pandas dayofyear, Feb 29 is 60.
		# If '60' is missing in the map (e.g. history had no leap years),
		# we interpolate between 59 and 61 if they exist.
		if freq == "doy" and k == 60:
			val_59 = seasonal_map.get(59)
			val_61 = seasonal_map.get(61)
			if val_59 is not None and val_61 is not None:
				return (val_59 + val_61) / 2.0
			if val_59 is not None: return val_59
			if val_61 is not None: return val_61

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
