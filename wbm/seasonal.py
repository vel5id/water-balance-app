"""Seasonal decomposition and robust season+trend modeling utilities.

This module complements `forecast.py` by providing mathematical documentation and
helpers for interpreting and extending seasonal phenomena whose amplitude and
central tendency vary over the annual cycle.

Core concepts
-------------
We treat an observed daily (or monthly) hydrometeorological series X_t as:

    X_t = S_t + T_t + R_t

Where:
  S_t : Seasonal component (periodic with period = 365 (DOY) or 12 (MONTH)).
        Estimated here via a robust location estimator (median) over historical
        observations sharing the same seasonal key (day-of-year or month).
  T_t : Trend component, modeled robustly with Theil–Sen slope (median of pairwise slopes),
        which is resilient to outliers compared to ordinary least squares.
  R_t : Residual (noise / weather scale variability), assumed weakly dependent after
        removing S_t and T_t. Residuals can be bootstrapped (optionally with blocks)
        to generate stochastic future ensembles around a deterministic extension.

Mathematical details
--------------------
Let index dates be d_1, d_2, ..., d_n (daily). Map each date d_i to a seasonal key k_i:
  - DOY: k_i = day_of_year(d_i), with leap day (Feb 29) mapped to 59 (or smoothed blend).
  - MONTH: k_i = month(d_i).

Seasonal template S(k): median{ X_i | k_i = k }. For missing keys we impute using
the global median or linear interpolation across the key axis.

Detrended series after removing season: D_i = X_i - S(k_i).
Compute Theil–Sen slope:
  slope = median{ (D_j - D_i) / (t_j - t_i) : 1 <= i < j <= n, t_j != t_i }.
Given we align time axis to integer day offsets t_i = i - 1, we simplify dx = j - i.
Intercept chosen as median( D_i - slope * t_i ). Then trend T_i = intercept + slope * t_i.

Deterministic future extension for horizon H days:
  For future times t = n, n+1, ..., n+H-1:
     k_future = seasonal key for future date
     S_future = S(k_future) (fallback to seasonal median if unseen)
     T_future = intercept + slope * t
     X_future_det = S_future + T_future

Residuals R_i = X_i - (S(k_i) + T_i)

Ensemble generation (outline)
-----------------------------
1. Fit model: obtain deterministic future path X_future_det and residual history R.
2. Resample residuals (optionally using moving block bootstrap to preserve short-term autocorrelation).
3. Add each residual sequence to the deterministic path to obtain member trajectories.

Provided utilities
------------------
- `seasonal_keys(dates, freq)`: map DatetimeIndex to seasonal integer keys.
- `robust_seasonal_template(series, freq)`: compute median seasonal template with interpolation.
- `theil_sen_trend(values)`: pure slope estimate (pairwise median) for documentation / reuse.
- `describe_season_trend(series, freq)`: returns dict with slope per year, slope raw (per day), seasonal amplitude, residual std.
- `markdown_doc()`: return extended Markdown string documenting the approach (used in UI expander).

All functions avoid heavy dependencies, relying on numpy/pandas only.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Literal, Sequence, Tuple

SeasonFreq = Literal["doy", "month"]


def seasonal_keys(dates: pd.DatetimeIndex, freq: SeasonFreq) -> np.ndarray:
    if freq == "doy":
        k = dates.dayofyear.to_numpy()
        # Map leap day to 59 to avoid sparse key
        leap_mask = (dates.month == 2) & (dates.day == 29)
        k[leap_mask] = 59
        return k
    return dates.month.to_numpy()


def robust_seasonal_template(series: pd.Series, freq: SeasonFreq) -> pd.Series:
    s = series.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    k = seasonal_keys(s.index, freq)
    grp = s.groupby(k).median()
    # Ensure full key coverage
    if freq == "doy":
        full_index = range(1, 367)
    else:
        full_index = range(1, 13)
    g2 = grp.reindex(full_index).interpolate(limit_direction="both")
    return g2


def theil_sen_trend(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n - 1):
        dy = values[i + 1 :] - values[i]
        dx = np.arange(i + 1, n) - i
        valid = dx != 0
        if valid.any():
            slopes.append(dy[valid] / dx[valid])
    if not slopes:
        return 0.0
    all_slopes = np.concatenate(slopes)
    return float(np.median(all_slopes))


def compute_acf(values: Sequence[float], max_lag: int = 30) -> pd.DataFrame:
    """Compute unbiased autocorrelation up to max_lag.

    Returns DataFrame with columns: lag, acf.
    NaNs are dropped before calculation. If <2 points → empty frame.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return pd.DataFrame(columns=["lag", "acf"])
    x_mean = x.mean()
    var = np.sum((x - x_mean) ** 2)
    out_lags = []
    out_acf = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        num = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
        # Unbiased denominator (n - lag)
        acf_val = num / var if var > 0 else 0.0
        out_lags.append(lag)
        out_acf.append(acf_val)
    return pd.DataFrame({"lag": out_lags, "acf": out_acf})


def recommend_block_length(acf_df: pd.DataFrame, n: int) -> int:
    """Recommend block length based on first lag where |acf| < 1.96/sqrt(n).

    If never falls below threshold, return min(max lag, 14). Minimum = 1.
    """
    if acf_df is None or acf_df.empty:
        return 1
    thr = 1.96 / max(np.sqrt(n), 1.0)
    for lag, val in zip(acf_df["lag"], acf_df["acf"]):
        if abs(val) < thr:
            return int(max(1, lag))
    return int(min(14, acf_df["lag"].max()))


def theil_sen_trend_ci_boot(
    series: pd.Series,
    freq: SeasonFreq = "doy",
    n_boot: int = 300,
    block_size: int | None = None,
    random_state: int | None = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for Theil–Sen slope after removing seasonal median template.

    Returns (slope_per_year, ci_low, ci_high). Units: slope scaled to per-year ( *365.25 ).
    If insufficient data returns zeros / NaNs gracefully.
    """
    s = series.dropna().sort_index()
    n = len(s)
    if n < 10:
        return 0.0, float("nan"), float("nan")
    k = seasonal_keys(s.index, freq)
    template = robust_seasonal_template(s, freq)
    if freq == "doy":
        seas_vals = template.reindex(range(1, 367)).to_numpy()[k - 1]
    else:
        seas_vals = template.reindex(range(1, 13)).to_numpy()[k - 1]
    detr = s.to_numpy() - seas_vals
    slope_day = theil_sen_trend(detr)
    rng = np.random.default_rng(random_state)
    slopes = []
    idx_arr = np.arange(n)
    # Determine block size default if needed
    if block_size is None or block_size < 1:
        acf_df = compute_acf(detr, max_lag=min(30, n // 2))
        block_size = recommend_block_length(acf_df, n)
    for _ in range(n_boot):
        resampled = []
        pos = 0
        while len(resampled) < n:
            start = rng.integers(0, max(1, n - block_size))
            blk = detr[start : start + block_size]
            resampled.append(blk)
            pos += block_size
        boot_seq = np.concatenate(resampled)[:n]
        slopes.append(theil_sen_trend(boot_seq))
    slopes = np.sort(np.array(slopes))
    lo = slopes[int(0.025 * (len(slopes) - 1))]
    hi = slopes[int(0.975 * (len(slopes) - 1))]
    return slope_day * 365.25, lo * 365.25, hi * 365.25


def smooth_season_template(template: pd.Series, window: int = 0) -> pd.Series:
    """Optionally apply centered rolling median smoothing to a seasonal template.

    window=0 disables smoothing.
    """
    if window is None or window <= 1:
        return template
    return template.rolling(window=window, center=True, min_periods=max(2, window // 2)).median().interpolate(limit_direction="both")


def describe_season_trend(series: pd.Series, freq: SeasonFreq = "doy") -> dict:
    s = series.dropna().sort_index()
    if len(s) < 10:
        return {"n": len(s)}
    k = seasonal_keys(s.index, freq)
    template = robust_seasonal_template(s, freq)
    # Map template back onto dates
    if freq == "doy":
        seas_vals = template.reindex(range(1, 367)).to_numpy()[k - 1]
    else:
        seas_vals = template.reindex(range(1, 13)).to_numpy()[k - 1]
    detr = s.to_numpy() - seas_vals
    slope_per_day = theil_sen_trend(detr)
    slope_per_year = slope_per_day * 365.25
    residuals = detr - (detr * 0 + np.median(detr))  # center residuals (already removed slope for simplicity)
    amp = float(np.nanpercentile(seas_vals, 95) - np.nanpercentile(seas_vals, 5))
    return {
        "n": len(s),
        "freq": freq,
        "slope_per_day": slope_per_day,
        "slope_per_year": slope_per_year,
        "seasonal_amplitude_p90_p10": amp,
        "residual_std": float(np.nanstd(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
    }


def seasonal_spread(series: pd.Series, freq: SeasonFreq = "doy") -> pd.DataFrame:
    """Return per-key quantile spread of the seasonal distribution.

    Output columns: key, p10, p25, median, p75, p90. Requires >=3 distinct years (heuristic)
    for interpretability; otherwise returns empty DataFrame.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["key","p10","p25","median","p75","p90"])
    years = s.index.year.unique()
    if len(years) < 3:
        return pd.DataFrame(columns=["key","p10","p25","median","p75","p90"])
    k = seasonal_keys(s.index, freq)
    df = pd.DataFrame({"value": s.values, "key": k})
    g = df.groupby("key")["value"]
    out = g.quantile([0.10,0.25,0.5,0.75,0.90]).unstack()
    out = out.rename(columns={0.10:"p10",0.25:"p25",0.5:"median",0.75:"p75",0.90:"p90"})
    out.reset_index(inplace=True)
    return out

def markdown_doc() -> str:
    return (
        "## Season + Trend Modeling (Robust)\n"
        "Модель: X_t = S_t + T_t + R_t, где S_t — сезонная медиана по ключу (DOY/MONTH), "
        "T_t — Theil–Sen тренд, R_t — остаток. Остатки можно бутстрэпить для ансамблей.\n\n"
        "Шаги:\n"
        "1. Сортировка и нормализация дат\n"
        "2. Группировка и медиана по ключу (устойчиво к выбросам)\n"
        "3. Вычитание сезонности, оценка наклона Theil–Sen\n"
        "4. Интерсепт = медиана детрендированного ряда\n"
        "5. Построение будущего: сезонный шаблон + линейный тренд\n"
        "6. Остатки = наблюдение - (сезон + тренд)\n\n"
        "Преимущества: робастность к выбросам, интерпретируемость. Ограничения: не моделирует\n"
        "нелинейные тренды, не учитывает автокорреляцию явно (кроме блок-бутстрэпа).\n"
        "\n### Smoothing (сглаживание сезонности)\n"
        "Опциональное rolling-median сглаживание уменьшает шум в шаблоне для DOY. Рекомендуемый диапазон окна: 3-11 дней.\n"
        "Слишком большое окно может исказить пики и фазы переходов.\n"
        "\n### Seasonal Spread (вариабельность)\n"
        "Лента p25–p75 или p10–p90 отражает межгодовую изменчивость сезонного профиля. Требует >=3 лет данных.\n"
        "\n### Caching\n"
        "Тяжёлые вычисления (ACF, bootstrap CI) кэшируются по хэшу входной серии и параметров. Это ускоряет интерактив.\n"
        "\n### Отложенные улучшения (Deferred)\n"
        "1. Saturation/Truncation тренда при длинном горизонте.\n"
        "2. Variance inflation diagnostics (spread ratio).\n"
        "3. Box-Cox вместо лог-преобразования.\n"
        "4. Adaptive block bootstrap (оптимизация по минимизации ошибки прогноза).\n"
        "5. Мультипликативный режим (лог1p) для гетероскедастичных рядов.\n"
    )

__all__ = [
    "seasonal_keys",
    "robust_seasonal_template",
    "theil_sen_trend",
    "compute_acf",
    "recommend_block_length",
    "theil_sen_trend_ci_boot",
    "smooth_season_template",
    "seasonal_spread",
    "describe_season_trend",
    "markdown_doc",
]
