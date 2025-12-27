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


def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculates the Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 - ( sum((obs - sim)^2) / sum((obs - mean(obs))^2) )

    Range: (-inf, 1.0].
    NSE = 1.0 -> Perfect match.
    NSE = 0.0 -> Simulation is as accurate as the mean of observations.
    NSE < 0.0 -> Simulation is worse than the mean of observations.

    Axiom Guard: Returns -inf if observed variance is effectively zero.
    """
    # Axiom: Shape validation
    if observed.shape != simulated.shape:
        raise ValueError(f"Shape mismatch: {observed.shape} vs {simulated.shape}")

    # Axiom: Filter NaNs ensures we only compare valid overlapping data
    valid_mask = np.isfinite(observed) & np.isfinite(simulated)
    obs = observed[valid_mask]
    sim = simulated[valid_mask]

    if len(obs) < 2:
        return float('nan') # Not enough data points

    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    # AXIOM GUARD: Variance Collapse
    if denominator < 1e-9:
        return float('-inf')

    return 1.0 - (numerator / denominator)


def calculate_kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculates the Kling-Gupta Efficiency (KGE).

    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    where:
      r = correlation coefficient
      alpha = std_sim / std_obs (variability error)
      beta = mean_sim / mean_obs (bias error)

    Range: (-inf, 1.0].

    Axiom Guard: Returns -inf if dispersion or mean is zero to prevent div/0.
    """
    # Axiom: Shape & NaN handling (reused logic)
    valid_mask = np.isfinite(observed) & np.isfinite(simulated)
    obs = observed[valid_mask]
    sim = simulated[valid_mask]

    if len(obs) < 2:
        return float('nan')

    # Components
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    std_obs = np.std(obs)
    std_sim = np.std(sim)

    # AXIOM GUARD: Zero Division Risks
    if std_obs < 1e-9 or mean_obs == 0:
        return float('-inf')

    # Pearson Correlation
    # np.corrcoef returns [[1, r], [r, 1]]
    r_matrix = np.corrcoef(obs, sim)
    r = r_matrix[0, 1]

    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

__all__ = [
    "rolling_trend",
    "lagged_correlation",
    "calculate_nse",
    "calculate_kge",
]
