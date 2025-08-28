from __future__ import annotations

import pandas as pd
import numpy as np


def rolling_trend(series: pd.Series, window: int = 30) -> pd.Series:
    """Return centered rolling mean as a simple trend proxy."""
    return series.rolling(window=window, center=True, min_periods=max(3, window//3)).mean()


def lagged_correlation(a: pd.Series, b: pd.Series, max_lag: int = 60) -> pd.DataFrame:
    """Compute Pearson correlation for lags in [-max_lag, max_lag].

    Positive lag means b is shifted forward (b leads a).
    Returns DataFrame with columns: lag, corr.
    """
    a = a.astype(float)
    b = b.astype(float)
    out = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = a.corr(b.shift(lag))
        elif lag < 0:
            corr = a.shift(-lag).corr(b)
        else:
            corr = a.corr(b)
        out.append({"lag": lag, "corr": corr})
    return pd.DataFrame(out)
