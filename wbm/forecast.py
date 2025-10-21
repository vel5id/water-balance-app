from __future__ import annotations

import warnings
# Suppress sklearn warnings globally before any imports
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
# Suppress statsmodels convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal
try:
	import pmdarima as pm  # heavy; may fail if ABI mismatch
except Exception:
	pm = None  # defer error until SARIMA path is actually used
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from datetime import timedelta


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


def build_sarima_model(
	series: pd.Series,
	*,
	future_days: int = 180,
	min_history: int = 90,
) -> pd.Series:
	"""Build and forecast using auto_arima and return date-indexed Series.

	The function is robust to short histories and returns a pd.Series indexed by
	future calendar days immediately following the last observation in `series`.
	"""
	result, _ = build_sarima_model_with_params(series, future_days=future_days, min_history=min_history)
	return result


def build_sarima_model_with_params(
	series: pd.Series,
	*,
	future_days: int = 180,
	min_history: int = 90,
	max_history: int = 730,  # NEW: limit history to prevent OOM
	fixed_order: tuple = None,
	fixed_seasonal_order: tuple = None,
) -> tuple[pd.Series, dict]:
	"""Build and forecast using auto_arima, returning predictions and model parameters.

	Parameters
	----------
	series : pd.Series
		Time series data
	future_days : int
		Number of days to forecast
	min_history : int
		Minimum history length required
	max_history : int
		Maximum history to use (prevents memory issues on large data)
	fixed_order : tuple, optional
		If provided, skip auto_arima and use this (p,d,q)
	fixed_seasonal_order : tuple, optional
		If provided, use this (P,D,Q,m) for seasonal component

	Returns
	-------
	predictions : pd.Series
		Date-indexed forecast series
	model_info : dict
		Dictionary with 'order', 'seasonal_order', 'aic', 'bic'
	"""
	s = series.dropna().sort_index()
	# ensure daily frequency; if duplicates, take last
	s = s[~s.index.duplicated(keep="last")]  # de-dup
	
	# Explicitly set frequency to 'D' to avoid warnings
	if not isinstance(s.index, pd.DatetimeIndex):
		s.index = pd.DatetimeIndex(s.index)
	s = s.asfreq("D")
	# Ensure frequency is explicitly set
	if s.index.freq is None:
		s.index.freq = pd.infer_freq(s.index) or 'D'
	
	# FIX 1: Fill any remaining NaN with forward fill then backward fill
	if s.isna().any():
		s = s.fillna(method='ffill').fillna(method='bfill')
		# If still NaN (empty series), raise error
		if s.isna().any():
			raise ValueError("Unable to fill NaN values in series")
	
	if len(s) < min_history:
		raise ValueError("Insufficient history for SARIMA model")
	
	# FIX 2: MEMORY OPTIMIZATION - Limit history to prevent OOM on large datasets
	if len(s) > max_history:
		s = s.iloc[-max_history:]

	# Determine a sensible seasonal period `m`. For daily data with annual seasonality
	# m=365 is heavy; use m=365 if we have > 3 years of data, else fallback to m=7 or 30.
	days = len(s)
	if days >= 3 * 365:
		m = 365
	elif days >= 180:
		m = 30
	elif days >= 60:
		m = 7
	else:
		m = 1  # No seasonality for very short series

	# If fixed parameters provided, use them directly
	if fixed_order and fixed_seasonal_order:
		from statsmodels.tsa.statespace.sarimax import SARIMAX
		try:
			# Suppress convergence warnings explicitly
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore')
				model = SARIMAX(s, order=fixed_order, seasonal_order=fixed_seasonal_order, 
				                enforce_stationarity=False, enforce_invertibility=False)
				fitted = model.fit(disp=False, maxiter=50, method='lbfgs')
			pred = fitted.forecast(steps=future_days)
			model_info = {
				"order": fixed_order,
				"seasonal_order": fixed_seasonal_order,
				"aic": float(fitted.aic),
				"bic": float(fitted.bic),
			}
		except Exception as e:
			raise ValueError(f"SARIMAX with fixed params failed: {str(e)}")
	else:
		# Auto search with fallback
		try:
			if pm is None:
				raise ImportError("pmdarima is not available in this environment")
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore')
				model: SARIMAXResults = pm.auto_arima(
					s,
					start_p=0, start_q=0,
					test='adf',
					max_p=2, max_q=2,  # Reduced from 3 for speed
					m=m,
					d=None,
					seasonal=True if m > 1 else False,
					start_P=0,
					max_P=1, max_Q=1,  # Reduced for speed
					D=None,            # let auto_arima infer seasonal differencing
					trace=False,
					error_action='ignore',
					suppress_warnings=True,
					stepwise=True,
					maxiter=50,  # NEW: limit iterations to prevent long hangs
					method='lbfgs',  # NEW: faster than default
				)
			pred = model.predict(n_periods=future_days)
			model_info = {
				"order": model.order,
				"seasonal_order": model.seasonal_order,
				"aic": float(model.aic()),
				"bic": float(model.bic()),
			}
		except Exception as e:
			# Fallback to simpler non-seasonal model if seasonal fails
			try:
				if pm is None:
					raise ImportError("pmdarima is not available in this environment")
				model: SARIMAXResults = pm.auto_arima(
					s,
					start_p=0, start_q=0,
					max_p=2, max_q=2,
					seasonal=False,
					trace=False,
					error_action='ignore',
					suppress_warnings=True,
					stepwise=True,
					maxiter=30,
				)
				pred = model.predict(n_periods=future_days)
				model_info = {
					"order": model.order,
					"seasonal_order": (0, 0, 0, 0),
					"aic": float(model.aic()),
					"bic": float(model.bic()),
				}
			except Exception:
				raise ValueError(f"SARIMA failed: {str(e)}")

	# Forecast next future_days points and index them by date
	start_date = s.index[-1] + pd.Timedelta(days=1)
	future_index = pd.date_range(start=start_date, periods=future_days, freq="D")
	return pd.Series(pred, index=future_index), model_info


def _build_robust_season_trend_series(
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


def build_robust_season_trend_series(
	series: pd.Series,
	*,
	freq: Literal["doy", "month"] = "doy",
	future_days: int = 180,
	min_history: int = 90,
) -> SeasonTrendResult:
	"""Public wrapper for backward compatibility. See _build_robust_season_trend_series."""
	return _build_robust_season_trend_series(
		series,
		freq=freq,
		future_days=future_days,
		min_history=min_history,
	)


# ============================================================================
# Prophet Integration
# ============================================================================

def build_prophet_forecast(
	series: pd.Series,
	*,
	future_days: int = 180,
	min_history: int = 90,
	yearly_seasonality: bool = True,
	weekly_seasonality: bool = True,
	daily_seasonality: bool = False,
	interval_width: float = 0.95,
) -> tuple[pd.Series, dict]:
	"""Build and forecast using Facebook Prophet.
	
	Parameters
	----------
	series : pd.Series
		Time series data (datetime-indexed, daily frequency preferred)
	future_days : int
		Number of days to forecast
	min_history : int
		Minimum history length required
	yearly_seasonality : bool
		Enable yearly seasonality
	weekly_seasonality : bool
		Enable weekly seasonality
	daily_seasonality : bool
		Enable daily seasonality
	interval_width : float
		Prediction interval width (0-1)
	
	Returns
	-------
	forecast : pd.Series
		Date-indexed forecast series (yhat values)
	model_info : dict
		Dictionary with 'method', 'n_obs', 'n_forecast', 'mape'
	"""
	try:
		from prophet import Prophet
	except ImportError:
		raise ImportError(
			"Prophet not installed. Install with: pip install prophet pystan==2.19.1.1"
		)
	
	s = series.dropna().sort_index()
	if len(s) < min_history:
		raise ValueError(f"Insufficient history for Prophet ({len(s)} < {min_history})")
	
	# Prepare data for Prophet: must have 'ds' (datetime) and 'y' (values)
	df = pd.DataFrame({
		'ds': s.index,
		'y': s.values
	})
	
	# Build model
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		model = Prophet(
			yearly_seasonality=yearly_seasonality,
			weekly_seasonality=weekly_seasonality,
			daily_seasonality=daily_seasonality,
			interval_width=interval_width,
			stan_backend='cmdstanpy' if 'cmdstanpy' in __import__('sys').modules else None
		)
		model.fit(df)
	
	# Generate future dates
	future = model.make_future_dataframe(periods=future_days)
	
	# Forecast
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		forecast_df = model.predict(future)
	
	# Extract yhat for future dates only
	future_forecast = forecast_df[forecast_df['ds'] > s.index[-1]].copy()
	forecast_series = pd.Series(
		future_forecast['yhat'].values,
		index=pd.date_range(s.index[-1] + timedelta(days=1), periods=future_days, freq='D'),
		name='yhat'
	)
	
	model_info = {
		'method': 'Prophet',
		'n_obs': len(s),
		'n_forecast': future_days,
		'yearly_seasonality': yearly_seasonality,
		'weekly_seasonality': weekly_seasonality,
		'daily_seasonality': daily_seasonality,
	}
	
	return forecast_series, model_info


# ============================================================================
# SARIMAX with Exogenous Regressors
# ============================================================================

def build_sarimax_forecast(
	series: pd.Series,
	*,
	exog_data: pd.DataFrame | None = None,
	future_days: int = 180,
	min_history: int = 90,
	order: tuple = (1, 1, 1),
	seasonal_order: tuple = (1, 1, 1, 12),
) -> tuple[pd.Series, dict]:
	"""Build and forecast using SARIMAX (Seasonal ARIMA with eXogenous regressors).
	
	SARIMAX extends SARIMA to include external features (e.g., temperature, precipitation)
	that may improve forecast accuracy. This is particularly useful for water balance 
	predictions where climate variables drive hydrological processes.
	
	Parameters
	----------
	series : pd.Series
		Time series data (datetime-indexed, daily frequency preferred)
	exog_data : pd.DataFrame, optional
		External regressors aligned with series (e.g., temperature, precipitation).
		Index must match series.index. Only columns that overlap with series will be used.
	future_days : int
		Number of days to forecast
	min_history : int
		Minimum history length required
	order : tuple
		ARIMA order (p, d, q). Default: (1, 1, 1)
	seasonal_order : tuple
		Seasonal order (P, D, Q, m). Default: (1, 1, 1, 12) for monthly seasonality
	
	Returns
	-------
	forecast : pd.Series
		Date-indexed forecast series
	model_info : dict
		Dictionary with 'method', 'n_obs', 'n_forecast', 'order', 'seasonal_order', 'has_exog'
	
	Raises
	------
	ImportError
		If statsmodels not installed
	ValueError
		If insufficient history
	
	Notes
	-----
	- Exogenous variables should be stationary or differenced
	- For future forecasts without exog data, uses last known exog values (persistence)
	- SARIMAX is more computationally expensive than SARIMA but can be more accurate
	
	References
	----------
	- Statsmodels SARIMAX: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
	"""
	try:
		from statsmodels.tsa.statespace.sarimax import SARIMAX
	except ImportError:
		raise ImportError("statsmodels not installed")
	
	s = series.dropna().sort_index()
	if len(s) < min_history:
		raise ValueError(f"Insufficient history for SARIMAX ({len(s)} < {min_history})")
	
	# Prepare exogenous data
	exog_train = None
	exog_forecast = None
	exog_cols = []
	
	if exog_data is not None and not exog_data.empty:
		# Align external data with series
		common_idx = exog_data.index.intersection(s.index)
		
		if len(common_idx) > 0:
			exog_aligned = exog_data.loc[common_idx]
			exog_train = exog_aligned.values
			exog_cols = list(exog_aligned.columns)
			s = s.loc[common_idx]  # Trim series to match exog
			
			# For future forecast: use last exog values (persistence assumption)
			# In production: consider iterative forecasting for lag features
			if len(exog_aligned) > 0:
				last_exog = exog_aligned.iloc[-1].values
				exog_forecast = np.tile(last_exog, (future_days, 1))
	
	# Build SARIMAX model
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore')
			
			model = SARIMAX(
				s.values,
				exog=exog_train,
				order=order,
				seasonal_order=seasonal_order,
				enforce_stationarity=False,
				enforce_invertibility=False,
				disp=False
			)
			
			# Fit with limited iterations
			fitted = model.fit(disp=False, maxiter=100)
			
			# Forecast
			forecast_values = fitted.forecast(steps=future_days, exog=exog_forecast)
	
	except Exception as e:
		# Fallback: if SARIMAX fails, return NaN forecast with error info
		import sys
		print(f"SARIMAX fit failed: {str(e)}", file=sys.stderr, flush=True)
		forecast_values = np.full(future_days, np.nan)
	
	# Create forecast series
	forecast_series = pd.Series(
		forecast_values,
		index=pd.date_range(s.index[-1] + timedelta(days=1), periods=future_days, freq='D'),
		name='yhat'
	)
	
	model_info = {
		'method': 'SARIMAX',
		'n_obs': len(s),
		'n_forecast': future_days,
		'order': order,
		'seasonal_order': seasonal_order,
		'has_exog': exog_data is not None and not exog_data.empty,
		'exog_vars': exog_cols,
	}
	
	return forecast_series, model_info


# ============================================================================
# Improved SARIMA with enhanced error handling
# ============================================================================

def build_sarima_forecast_enhanced(
	series: pd.Series,
	*,
	future_days: int = 180,
	min_history: int = 90,
	max_history: int = 730,
	start_p: int = 0,
	start_q: int = 0,
	max_p: int = 5,
	max_q: int = 5,
	max_d: int = 2,
	seasonal: bool = True,
	trace: bool = False,
	error_action: str = 'ignore',
	suppress_warnings: bool = True,
	return_to_non_seasonal: bool = True,
) -> tuple[pd.Series, dict]:
	"""Enhanced SARIMA forecasting with robustness.
	
	Parameters
	----------
	series : pd.Series
		Time series data
	future_days : int
		Number of days to forecast
	min_history : int
		Minimum history required
	max_history : int
		Maximum history to use (prevents OOM)
	start_p, start_q : int
		Starting ARIMA order parameters
	max_p, max_q : int
		Max ARIMA parameters
	max_d : int
		Max differencing order
	seasonal : bool
		Enable seasonal component
	trace : bool
		Print search progress
	error_action : str
		How to handle errors ('ignore', 'warn', 'raise')
	suppress_warnings : bool
		Suppress convergence warnings
	return_to_non_seasonal : bool
		If seasonal fails, try non-seasonal
	
	Returns
	-------
	forecast : pd.Series
		Date-indexed forecast
	model_info : dict
		Model parameters and diagnostics
	"""
	if pm is None:
		raise ImportError(
			"pmdarima not installed. Install with: pip install pmdarima"
		)
	
	s = series.dropna().sort_index()
	if len(s) < min_history:
		raise ValueError(f"Insufficient history for SARIMA ({len(s)} < {min_history})")
	
	# Limit history to prevent OOM
	if len(s) > max_history:
		s = s.iloc[-max_history:]
	
	# Handle remaining NaN
	if s.isna().any():
		s = s.fillna(method='ffill').fillna(method='bfill')
		if s.isna().any():
			raise ValueError("Unable to fill NaN values")
	
	# Set daily frequency
	if not isinstance(s.index, pd.DatetimeIndex):
		s.index = pd.DatetimeIndex(s.index)
	s = s.asfreq('D')
	if s.index.freq is None:
		s.index.freq = pd.infer_freq(s.index) or 'D'
	
	# Adaptive seasonality
	days = len(s)
	m = 1
	if seasonal:
		if days >= 3 * 365:
			m = 365
		elif days >= 180:
			m = 30
		elif days >= 60:
			m = 7
	
	if suppress_warnings:
		warnings.filterwarnings('ignore')
	
	try:
		# Try seasonal model first
		model = pm.auto_arima(
			s,
			start_p=start_p,
			start_q=start_q,
			max_p=max_p,
			max_q=max_q,
			max_d=max_d,
			seasonal=seasonal,
			m=m,
			stepwise=True,
			trace=trace,
			error_action=error_action,
			suppress_warnings=suppress_warnings,
			maxiter=50,
			method='lbfgs',
		)
		model_type = 'SARIMA'
	except Exception as e:
		if return_to_non_seasonal and seasonal:
			warnings.warn(
				f"Seasonal ARIMA failed ({str(e)[:50]}...), falling back to non-seasonal ARIMA"
			)
			try:
				model = pm.auto_arima(
					s,
					start_p=start_p,
					start_q=start_q,
					max_p=max_p,
					max_q=max_q,
					max_d=max_d,
					seasonal=False,
					stepwise=True,
					trace=trace,
					error_action=error_action,
					suppress_warnings=suppress_warnings,
					maxiter=50,
					method='lbfgs',
				)
				model_type = 'ARIMA'
			except Exception as e2:
				raise RuntimeError(f"Both SARIMA and ARIMA failed: {str(e2)}")
		else:
			raise
	
	# Forecast
	forecast_vals, conf_int = model.get_forecast(steps=future_days, return_conf_int=True)
	
	forecast_series = pd.Series(
		forecast_vals,
		index=pd.date_range(s.index[-1] + timedelta(days=1), periods=future_days, freq='D'),
		name='yhat'
	)
	
	model_info = {
		'method': model_type,
		'order': model.order,
		'seasonal_order': model.seasonal_order if hasattr(model, 'seasonal_order') else None,
		'aic': model.aic(),
		'bic': model.bic(),
		'n_obs': len(s),
		'n_forecast': future_days,
		'max_history_used': max_history,
	}
	
	return forecast_series, model_info


__all__ = [
	"SeasonTrendResult",
	"build_sarima_model",
	"build_sarima_model_with_params",
	"build_sarima_forecast_enhanced",
	"build_robust_season_trend_series",
	"build_prophet_forecast",
	"_build_robust_season_trend_series",
]
