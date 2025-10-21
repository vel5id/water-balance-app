"""
Forecast accuracy metrics for water balance predictions.

This module provides functions to calculate and compare forecast accuracy across
multiple time horizons (1-day, 1-week, 1-month, 6-month), enabling model selection
and performance evaluation.

Functions:
    - calculate_mape: Mean Absolute Percentage Error
    - calculate_rmse: Root Mean Squared Error
    - calculate_mae: Mean Absolute Error
    - calculate_metrics_by_horizon: Compute metrics for multiple forecast horizons
    - backtest_forecast_accuracy: Walk-forward cross-validation with metrics

Author: Water Balance Model Team
License: MIT
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ==================== BASIC METRICS ====================

def calculate_mape(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Parameters
    ----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
    epsilon : float
        Small value to avoid division by zero
        
    Returns
    -------
    mape : float
        MAPE as percentage (0-100). Returns NaN if input invalid.
        
    Notes
    -----
    MAPE = (1/n) * Σ|actual - predicted| / |actual| * 100
    
    MAPE is undefined when actual = 0. We use epsilon to handle this edge case.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have same length")
    
    if len(actual) == 0:
        return float("nan")
    
    # Remove NaN and infinite values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Calculate absolute percentage errors
    denominator = np.maximum(np.abs(actual), epsilon)
    ape = np.abs(actual - predicted) / denominator
    
    mape = float(np.mean(ape) * 100.0)
    
    return mape


def calculate_rmse(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Parameters
    ----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
        
    Returns
    -------
    rmse : float
        RMSE in same units as input. Returns NaN if input invalid.
        
    Notes
    -----
    RMSE = sqrt((1/n) * Σ(actual - predicted)²)
    
    RMSE is sensitive to large errors and preferred for normally-distributed errors.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have same length")
    
    if len(actual) == 0:
        return float("nan")
    
    # Remove NaN and infinite values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    
    actual = actual[mask]
    predicted = predicted[mask]
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = float(np.sqrt(mse))
    
    return rmse


def calculate_mae(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Parameters
    ----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
        
    Returns
    -------
    mae : float
        MAE in same units as input. Returns NaN if input invalid.
        
    Notes
    -----
    MAE = (1/n) * Σ|actual - predicted|
    
    MAE is more robust to outliers than RMSE.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have same length")
    
    if len(actual) == 0:
        return float("nan")
    
    # Remove NaN and infinite values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    
    actual = actual[mask]
    predicted = predicted[mask]
    
    mae = float(np.mean(np.abs(actual - predicted)))
    
    return mae


def calculate_r_squared(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Parameters
    ----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values
        
    Returns
    -------
    r2 : float
        R² value between 0 and 1 (1 = perfect fit, 0 = baseline). Returns NaN if invalid.
        
    Notes
    -----
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(actual - predicted)²
          SS_tot = Σ(actual - mean(actual))²
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have same length")
    
    if len(actual) < 2:
        return float("nan")
    
    # Remove NaN and infinite values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not np.any(mask) or np.sum(mask) < 2:
        return float("nan")
    
    actual = actual[mask]
    predicted = predicted[mask]
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return float("nan")
    
    r2 = float(1.0 - (ss_res / ss_tot))
    
    return r2


# ==================== HORIZON-BASED METRICS ====================

def calculate_metrics_by_horizon(
    actual: pd.Series,
    predicted: pd.Series,
    horizons: Optional[List[int]] = None,
    freq: str = "D"
) -> Dict[int, Dict[str, float]]:
    """
    Calculate forecast accuracy metrics for multiple time horizons.
    
    This function enables comparison of forecast accuracy at different lead times,
    showing how accuracy degrades with forecast distance.
    
    Parameters
    ----------
    actual : pd.Series
        Actual time series (must have DatetimeIndex)
    predicted : pd.Series
        Predicted time series (must have DatetimeIndex)
    horizons : List[int], optional
        List of horizon values in days. Default: [1, 7, 30, 180] (1-day to 6-month)
    freq : str
        Frequency for horizon calculation. Default: 'D' (daily)
        
    Returns
    -------
    metrics : Dict[int, Dict[str, float]]
        Dictionary with structure:
        {
            1: {'mape': X, 'rmse': Y, 'mae': Z, 'r2': W, 'n_samples': N},
            7: {...},
            ...
        }
        
    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', periods=365, freq='D')
    >>> actual = pd.Series(np.random.randn(365), index=dates)
    >>> predicted = pd.Series(np.random.randn(365), index=dates)
    >>> metrics = calculate_metrics_by_horizon(actual, predicted)
    >>> print(metrics[1]['mape'])
    """
    if horizons is None:
        horizons = [1, 7, 30, 180]  # 1-day, 1-week, 1-month, 6-month
    
    # Ensure inputs are Series with DatetimeIndex
    if not isinstance(actual, pd.Series) or not hasattr(actual.index, 'date'):
        raise ValueError("actual must be pd.Series with DatetimeIndex")
    
    if not isinstance(predicted, pd.Series) or not hasattr(predicted.index, 'date'):
        raise ValueError("predicted must be pd.Series with DatetimeIndex")
    
    # Align indices
    common_idx = actual.index.intersection(predicted.index)
    if len(common_idx) == 0:
        raise ValueError("actual and predicted have no overlapping dates")
    
    actual_aligned = actual.loc[common_idx]
    predicted_aligned = predicted.loc[common_idx]
    
    results = {}
    
    for horizon in sorted(horizons):
        # Get reference date (horizon days before end)
        if freq.lower() == "d":
            horizon_td = pd.Timedelta(days=horizon)
        elif freq.lower() == "h":
            horizon_td = pd.Timedelta(hours=horizon)
        elif freq.lower() == "w":
            horizon_td = pd.Timedelta(weeks=horizon)
        else:
            horizon_td = pd.Timedelta(days=horizon)
        
        # Calculate from horizon distance
        min_date = actual_aligned.index.min() + horizon_td
        
        # Extract pairs where we can measure at this horizon
        actual_horizon = []
        predicted_horizon = []
        
        for date in actual_aligned.index:
            forecast_date = date + horizon_td
            
            if forecast_date in actual_aligned.index and date in predicted_aligned.index:
                actual_val = actual_aligned.loc[forecast_date]
                predicted_val = predicted_aligned.loc[date]  # Prediction made at 'date' for 'forecast_date'
                
                if np.isfinite(actual_val) and np.isfinite(predicted_val):
                    actual_horizon.append(actual_val)
                    predicted_horizon.append(predicted_val)
        
        if len(actual_horizon) == 0:
            results[horizon] = {
                'mape': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan'),
                'n_samples': 0
            }
        else:
            results[horizon] = {
                'mape': calculate_mape(actual_horizon, predicted_horizon),
                'rmse': calculate_rmse(actual_horizon, predicted_horizon),
                'mae': calculate_mae(actual_horizon, predicted_horizon),
                'r2': calculate_r_squared(actual_horizon, predicted_horizon),
                'n_samples': len(actual_horizon)
            }
    
    return results


# ==================== BACKTEST & CROSS-VALIDATION ====================

def backtest_forecast_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    test_size: int = 90,
    step_size: Optional[int] = None,
    horizons: Optional[List[int]] = None
) -> Dict[str, Union[Dict[int, float], int]]:
    """
    Walk-forward backtesting for forecast accuracy at multiple horizons.
    
    This function performs out-of-sample evaluation by iterating through the
    time series in a walk-forward manner, computing metrics at each step.
    
    Parameters
    ----------
    actual : pd.Series
        Actual time series (must have DatetimeIndex)
    predicted : pd.Series
        Predicted values from the forecast model
    test_size : int
        Size of test window in days. Default: 90
    step_size : int, optional
        Step size for walk-forward window. Default: test_size (no overlap)
    horizons : List[int], optional
        List of horizons in days. Default: [1, 7, 30, 180]
        
    Returns
    -------
    results : Dict
        {
            'horizons': {1: {...}, 7: {...}, ...},
            'n_windows': int,
            'test_size': int
        }
        
    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', periods=365, freq='D')
    >>> actual = pd.Series(np.random.randn(365), index=dates)
    >>> predicted = pd.Series(np.random.randn(365), index=dates)
    >>> results = backtest_forecast_accuracy(actual, predicted, test_size=30)
    >>> print(results['horizons'][1]['mape'])
    """
    if horizons is None:
        horizons = [1, 7, 30, 180]
    
    if step_size is None:
        step_size = test_size
    
    # Initialize accumulators
    errors_by_horizon = {h: [] for h in horizons}
    actuals_by_horizon = {h: [] for h in horizons}
    
    # Align data
    common_idx = actual.index.intersection(predicted.index)
    actual_aligned = actual.loc[common_idx].sort_index()
    predicted_aligned = predicted.loc[common_idx].sort_index()
    
    # Walk-forward
    window_count = 0
    start_idx = 0
    
    while start_idx + test_size < len(actual_aligned):
        end_idx = start_idx + test_size
        
        window_actual = actual_aligned.iloc[start_idx:end_idx]
        window_predicted = predicted_aligned.iloc[start_idx:end_idx]
        
        # Calculate metrics for this window
        try:
            window_metrics = calculate_metrics_by_horizon(
                window_actual,
                window_predicted,
                horizons=horizons
            )
            
            for h in horizons:
                if window_metrics[h]['n_samples'] > 0:
                    errors_by_horizon[h].append(window_metrics[h]['mape'])
                    actuals_by_horizon[h].append(window_metrics[h]['rmse'])
        
        except Exception as e:
            # Skip windows with insufficient data or alignment issues
            pass
        
        window_count += 1
        start_idx += step_size
    
    # Aggregate results
    results_by_horizon = {}
    for h in horizons:
        if len(errors_by_horizon[h]) > 0:
            mape_values = np.array(errors_by_horizon[h])
            rmse_values = np.array(actuals_by_horizon[h])
            
            results_by_horizon[h] = {
                'mape_mean': float(np.nanmean(mape_values)),
                'mape_std': float(np.nanstd(mape_values)),
                'rmse_mean': float(np.nanmean(rmse_values)),
                'rmse_std': float(np.nanstd(rmse_values)),
                'n_windows': len(errors_by_horizon[h])
            }
        else:
            results_by_horizon[h] = {
                'mape_mean': float('nan'),
                'mape_std': float('nan'),
                'rmse_mean': float('nan'),
                'rmse_std': float('nan'),
                'n_windows': 0
            }
    
    return {
        'horizons': results_by_horizon,
        'n_windows': window_count,
        'test_size': test_size,
        'step_size': step_size
    }


# ==================== COMPARISON & SUMMARY ====================

def format_metrics_for_display(
    metrics: Dict[int, Dict[str, float]],
    include_r2: bool = True
) -> Dict[int, Dict[str, str]]:
    """
    Format metrics dictionary for display (rounded to 2 decimals).
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, float]]
        Raw metrics from calculate_metrics_by_horizon
    include_r2 : bool
        Whether to include R² metric
        
    Returns
    -------
    formatted : Dict[int, Dict[str, str]]
        Metrics formatted as strings with appropriate precision
        
    Examples
    --------
    >>> metrics = {1: {'mape': 12.345, 'rmse': 0.567}}
    >>> formatted = format_metrics_for_display(metrics)
    >>> print(formatted[1]['mape'])  # '12.35%'
    """
    formatted = {}
    
    for horizon, vals in metrics.items():
        formatted[horizon] = {}
        
        # MAPE
        if 'mape' in vals and np.isfinite(vals['mape']):
            formatted[horizon]['MAPE'] = f"{vals['mape']:.2f}%"
        else:
            formatted[horizon]['MAPE'] = "N/A"
        
        # RMSE
        if 'rmse' in vals and np.isfinite(vals['rmse']):
            formatted[horizon]['RMSE'] = f"{vals['rmse']:.3f}"
        else:
            formatted[horizon]['RMSE'] = "N/A"
        
        # MAE
        if 'mae' in vals and np.isfinite(vals['mae']):
            formatted[horizon]['MAE'] = f"{vals['mae']:.3f}"
        else:
            formatted[horizon]['MAE'] = "N/A"
        
        # R²
        if include_r2 and 'r2' in vals and np.isfinite(vals['r2']):
            formatted[horizon]['R²'] = f"{vals['r2']:.3f}"
        elif include_r2:
            formatted[horizon]['R²'] = "N/A"
        
        # Sample count
        if 'n_samples' in vals:
            formatted[horizon]['N'] = str(vals['n_samples'])
    
    return formatted


def best_method_by_horizon(
    metrics_dict: Dict[str, Dict[int, Dict[str, float]]]
) -> Dict[int, str]:
    """
    Determine best performing method for each horizon based on MAPE.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict[int, Dict[str, float]]]
        Metrics for multiple methods:
        {
            'Theil-Sen': {1: {...}, 7: {...}, ...},
            'SARIMA': {...},
            'Prophet': {...},
            'SARIMAX': {...}
        }
        
    Returns
    -------
    best : Dict[int, str]
        Method with lowest MAPE for each horizon
        
    Examples
    --------
    >>> metrics = {
    ...     'Prophet': {1: {'mape': 10, ...}, 7: {'mape': 15, ...}},
    ...     'SARIMA': {1: {'mape': 12, ...}, 7: {'mape': 12, ...}}
    ... }
    >>> best = best_method_by_horizon(metrics)
    >>> print(best[1])  # 'Prophet'
    """
    best = {}
    
    # Get all available horizons
    horizons = set()
    for method_metrics in metrics_dict.values():
        horizons.update(method_metrics.keys())
    
    for horizon in sorted(horizons):
        best_method = None
        best_mape = float('inf')
        
        for method, method_metrics in metrics_dict.items():
            if horizon in method_metrics:
                mape = method_metrics[horizon].get('mape', float('inf'))
                
                if np.isfinite(mape) and mape < best_mape:
                    best_mape = mape
                    best_method = method
        
        if best_method is not None:
            best[horizon] = best_method
    
    return best


# ==================== UTILITY ====================

def horizon_name(days: int) -> str:
    """
    Convert horizon in days to human-readable name.
    
    Parameters
    ----------
    days : int
        Number of days
        
    Returns
    -------
    name : str
        Human-readable name (e.g., '1 day', '1 week', '1 month')
    """
    if days == 1:
        return "1 day"
    elif days == 7:
        return "1 week"
    elif days == 30:
        return "1 month"
    elif days == 180:
        return "6 months"
    else:
        return f"{days} days"


if __name__ == "__main__":
    # Quick test
    print("Metrics module loaded successfully")
    
    # Example
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    actual_data = pd.Series(100 + np.cumsum(np.random.randn(365) * 5), index=dates)
    predicted_data = pd.Series(100 + np.cumsum(np.random.randn(365) * 5.2), index=dates)
    
    metrics = calculate_metrics_by_horizon(actual_data, predicted_data)
    formatted = format_metrics_for_display(metrics)
    
    print("\nExample Metrics:")
    for horizon in sorted(metrics.keys()):
        print(f"\n{horizon_name(horizon)}:")
        for metric, value in formatted[horizon].items():
            print(f"  {metric}: {value}")
