from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Callable


def simulate_forward(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    init_volume_mcm: float,
    p_clim: pd.Series,  # DOY-indexed precipitation (mm/day)
    et_clim: pd.Series,  # DOY-indexed evaporation (mm/day)
    vol_to_area: Callable[[float], float],
    p_scale: float = 1.0,
    et_scale: float = 1.0,
    q_in_mcm_per_day: float = 0.0,
    q_out_mcm_per_day: float = 0.0,
    p_daily: Optional[pd.Series] = None,  # Optional date-indexed precipitation (mm/day)
    et_daily: Optional[pd.Series] = None,  # Optional date-indexed evaporation (mm/day)
) -> pd.DataFrame:
    """Daily forward simulation with scaled P/ET and constant inflow/outflow.

    State update:
      area_km2(t) = vol_to_area(volume_mcm(t))
      P_mcm = p_scale * P_mm(doy) * area_km2 / 1000
      ET_mcm = et_scale * ET_mm(doy) * area_km2 / 1000
      volume(t+1) = volume(t) + P_mcm - ET_mcm + q_in - q_out
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)

    vol = np.empty(n, dtype=float)
    area = np.empty(n, dtype=float)
    p_mcm = np.empty(n, dtype=float)
    et_mcm = np.empty(n, dtype=float)

    vol[0] = init_volume_mcm
    for i, dt in enumerate(dates):

        # Area from current volume
        area[i] = max(0.0, vol_to_area(vol[i]))

        # P/ET for this date
        if p_daily is not None and not p_daily.empty:
            p_mm = float(p_daily.get(dt, 0.0)) * p_scale
        else:
            doy = dt.dayofyear
            if dt.month == 2 and dt.day == 29:
                doy = 59
            p_mm = float(p_clim.get(doy, 0.0)) * p_scale
        if et_daily is not None and not et_daily.empty:
            et_mm = float(et_daily.get(dt, 0.0)) * et_scale
        else:
            doy = dt.dayofyear
            if dt.month == 2 and dt.day == 29:
                doy = 59
            et_mm = float(et_clim.get(doy, 0.0)) * et_scale

        p_mcm[i] = p_mm * area[i] / 1000.0
        et_mcm[i] = et_mm * area[i] / 1000.0

        if i < n - 1:
            vol[i+1] = vol[i] + p_mcm[i] - et_mcm[i] + q_in_mcm_per_day - q_out_mcm_per_day

    delta = np.empty(n, dtype=float)
    delta[0] = np.nan
    delta[1:] = vol[1:] - vol[:-1]

    df = pd.DataFrame({
        "date": dates,
        "volume_mcm": vol,
        "area_km2": area,
        "precipitation_volume_mcm": p_mcm,
        "evaporation_volume_mcm": et_mcm,
        "delta_volume_mcm": delta,
    })
    df["residual_mcm"] = df["delta_volume_mcm"] - df["precipitation_volume_mcm"] + df["evaporation_volume_mcm"]
    return df
