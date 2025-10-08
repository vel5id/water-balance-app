from __future__ import annotations
import pandas as pd
from typing import Tuple
from .state import LoadedData, Controls, ScenarioContext
from wbm.simulate import simulate_forward
from wbm.forecast import build_robust_season_trend_series, SeasonTrendResult

__all__ = ["prepare_drivers", "run_scenario"]

def _prepare_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    if df is None or df.empty or value_col not in df.columns or "date" not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", value_col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.set_index("date")[value_col].asfreq("D")

def prepare_drivers(ld: LoadedData, ctr: Controls):
    p_daily = et_daily = None
    if ctr.forecast_mode == "Monthly mean (all years)":
        # Simple monthly mean expansion
        p_daily = _monthly_mean_series(ld.era5_df, "precip_mm", ctr.start_date, ctr.end_date)
        et_daily = _monthly_mean_series(ld.era5_df, "evap_mm", ctr.start_date, ctr.end_date)
    elif ctr.forecast_mode == "Seasonal + trend":
        base_p = _prepare_series(ld.era5_df, "precip_mm")
        base_et = _prepare_series(ld.era5_df, "evap_mm")
        freq = "doy" if ctr.seas_basis == "DOY" else "month"
        future_days = int((ctr.end_date - ctr.start_date).days)
        min_hist = ctr.hist_window_days if ctr.hist_window_days > 0 else 90
        try:
            res_p: SeasonTrendResult = build_robust_season_trend_series(base_p, freq=freq, future_days=future_days, min_history=min_hist)
            p_daily = res_p.deterministic
        except Exception:
            p_daily = None
        try:
            res_et: SeasonTrendResult = build_robust_season_trend_series(base_et, freq=freq, future_days=future_days, min_history=min_hist)
            et_daily = res_et.deterministic
        except Exception:
            et_daily = None
    return p_daily, et_daily

def _monthly_mean_series(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.set_index("date").sort_index()
    monthly = d[col].groupby([d.index.month]).mean()
    rng = pd.date_range(start, end, freq="D")
    vals = []
    for ts in rng:
        m = ts.month
        vals.append(monthly.get(m, float("nan")))
    return pd.Series(vals, index=rng)

def select_initial_volume(balance_df: pd.DataFrame, start_date: pd.Timestamp, vols) -> Tuple[float, str]:
    if balance_df is None or balance_df.empty:
        if hasattr(vols, '__len__') and len(vols):
            return float(vols[len(vols)//2]), "fallback: curve midpoint"
        return 0.0, "fallback: 0"
    try:
        last_obs = pd.to_datetime(balance_df["date"]).max()
        if pd.isna(last_obs):
            raise ValueError
        # prefer prior or interpolate logic simplified: just take last observed <= start
        prior = balance_df[pd.to_datetime(balance_df["date"]) <= start_date]
        if not prior.empty:
            return float(prior.tail(1)["volume_mcm"].iloc[0]), "last prior observation"
        else:
            return float(balance_df.tail(1)["volume_mcm"].iloc[0]), "last observation (after start)"
    except Exception:
        return float(balance_df["volume_mcm"].median()), "median fallback"

def run_scenario(ld: LoadedData, ctr: Controls, p_daily, et_daily, init_volume: float) -> ScenarioContext:
    scen = simulate_forward(
        start_date=ctr.start_date,
        end_date=ctr.end_date,
        init_volume_mcm=init_volume,
        p_clim=ld.p_clim,
        et_clim=ld.et_clim,
        vol_to_area=ld.vol_to_area,
        p_scale=ctr.p_scale,
        et_scale=ctr.et_scale,
        q_in_mcm_per_day=ctr.q_in,
        q_out_mcm_per_day=ctr.q_out,
        p_daily=p_daily,
        et_daily=et_daily,
    )
    return ScenarioContext(scenario_df=scen, init_volume=init_volume, init_note="")
