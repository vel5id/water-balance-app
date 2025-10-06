from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Iterable

try:  # optional SciPy
    from scipy.stats import kendalltau  # type: ignore
except Exception:  # pragma: no cover
    kendalltau = None  # type: ignore


def theilsen_trend(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna().to_numpy()
    n = len(s)
    if n < 2:
        return 0.0, float("nan")
    slopes = []
    for i in range(n - 1):
        dy = s[i + 1 :] - s[i]
        dx = np.arange(i + 1, n) - i
        slopes.append(dy / dx)
    slopes_all = np.concatenate(slopes)
    slope = float(np.median(slopes_all))
    x = np.arange(n)
    intercept = float(np.median(s - slope * x))
    return slope, intercept


def theilsen_trend_ci(series: pd.Series, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """Compute Theil–Sen slope & intercept with simple rank-based CI (bootstrap fallback).

    For simplicity: bootstrap residuals after median slope; not an exact analytical CI but adequate for UI.
    Returns slope_per_year, intercept, lo, hi (slope per year units if index is a datetime).
    """
    s = series.dropna()
    if s.empty or len(s) < 3:
        return 0.0, float("nan"), 0.0, 0.0
    slope, intercept = theilsen_trend(s)
    # Convert slope to (value per year) if index is datetime
    if isinstance(s.index, pd.DatetimeIndex):
        days = (s.index[-1] - s.index[0]).days or 1
        slope_per_day = slope
        slope_per_year = slope_per_day * 365.25
    else:
        slope_per_year = slope
    # Bootstrap
    rng = np.random.default_rng(42)
    B = min(300, 50 + 5 * len(s))
    slopes = []
    arr = s.to_numpy()
    for _ in range(B):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        ss = pd.Series(sample, index=s.index)
        sl, _ = theilsen_trend(ss)
        if isinstance(s.index, pd.DatetimeIndex):
            sl = sl * 365.25
        slopes.append(sl)
    slopes = np.sort(slopes)
    lo_idx = int((alpha / 2) * (B - 1))
    hi_idx = int((1 - alpha / 2) * (B - 1))
    lo = float(slopes[lo_idx])
    hi = float(slopes[hi_idx])
    return float(slope_per_year), float(intercept), lo, hi


def kendall_significance(series: pd.Series) -> Tuple[float, float]:
    if kendalltau is None:
        s = series.rank().dropna().to_numpy()
        n = len(s)
        if n < 3:
            return 0.0, 1.0
        concord = 0
        discord = 0
        for i in range(n - 1):
            diff = s[i + 1 :] - s[i]
            concord += np.sum(diff > 0)
            discord += np.sum(diff < 0)
        tau = (concord - discord) / (0.5 * n * (n - 1))
        return float(tau), float("nan")
    arr = series.dropna()
    if len(arr) < 3:
        return 0.0, 1.0
    tau, p = kendalltau(arr.index.factorize()[0], arr.values)
    return float(tau), float(p)


def aggregate_series(df: pd.DataFrame, date_col: str, value_col: str, *, freq: str, years_back: int, end_anchor: pd.Timestamp) -> pd.Series:
    """Aggregate last N years to freq ("M" or "A") returning value-per-day mean (mm/day) series.

    - Filters rows within [end_anchor - years_back years, end_anchor].
    - Groups by freq and computes mean.
    - Returns DateTimeIndex series.
    """
    if df is None or df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)
    d = df[[date_col, value_col]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col])
    cutoff = end_anchor - pd.DateOffset(years=years_back)
    d = d[(d[date_col] >= cutoff) & (d[date_col] <= end_anchor)]
    if d.empty:
        return pd.Series(dtype=float)
    d = d.set_index(date_col).sort_index()
    agg = d[value_col].resample(freq).mean()
    return agg


def make_trend_comparison_figure(p_series: pd.Series, et_series: pd.Series, p_slope: float, p_inter: float, et_slope: float, et_inter: float):
    import plotly.graph_objects as go
    fig = go.Figure()
    if not p_series.empty:
        fig.add_trace(go.Scatter(x=p_series.index, y=p_series.values, name="P (mm/day)", mode="lines", line=dict(color="#1f77b4")))
        # Trend line for P (approx by linear fit using slope per year)
        if isinstance(p_series.index, pd.DatetimeIndex) and len(p_series) > 1:
            t0 = p_series.index[0]
            days = (p_series.index - t0).days.to_numpy()
            line = p_inter + (p_slope / 365.25) * days
            fig.add_trace(go.Scatter(x=p_series.index, y=line, name="P trend", mode="lines", line=dict(color="#1f77b4", dash="dash")))
    if not et_series.empty:
        fig.add_trace(go.Scatter(x=et_series.index, y=et_series.values, name="ET (mm/day)", mode="lines", line=dict(color="#ff7f0e")))
        if isinstance(et_series.index, pd.DatetimeIndex) and len(et_series) > 1:
            t0 = et_series.index[0]
            days = (et_series.index - t0).days.to_numpy()
            line = et_inter + (et_slope / 365.25) * days
            fig.add_trace(go.Scatter(x=et_series.index, y=line, name="ET trend", mode="lines", line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(template="plotly_white", title="P & ET aggregated with trends", xaxis_title="Date", yaxis_title="mm/day")
    return fig


__all__ = [
    "theilsen_trend",
    "theilsen_trend_ci",
    "kendall_significance",
    "aggregate_series",
    "make_trend_comparison_figure",
]
