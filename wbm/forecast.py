from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


def _theil_sen_slope(y: np.ndarray, x: np.ndarray) -> float:
	"""Return Theil–Sen median slope for y vs x."""
	n = len(y)
	if n < 2:
		return 0.0
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
) -> SeasonTrendResult:
	"""Decompose daily series into robust seasonal + Theil–Sen trend and extend deterministically.

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
	future_season = pd.Series([seasonal_map.get(k, seasonal_map.median()) for k in future_key], index=future_index)
	x_future = (future_index - idx[0]).days.to_numpy()
	future_trend = pd.Series(intercept + slope * x_future, index=future_index)
	deterministic_future = future_season + future_trend

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
