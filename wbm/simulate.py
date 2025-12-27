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

    # Pre-calculate P/ET arrays to avoid repeated lookups inside the loop
    def _prepare_flux_array(daily_series: Optional[pd.Series], clim_series: pd.Series, scale: float) -> np.ndarray:
        if daily_series is not None and not daily_series.empty:
            # Reindex to full date range, filling missing dates with 0.0
            # Assuming daily_series is date-indexed
            s = daily_series.reindex(dates, fill_value=0.0)
            return s.to_numpy(dtype=float) * scale
        else:
            # Use climatology based on DOY
            # Map DOY 1..366 to values.
            # Create lookup array for 1..366 (size 367)
            clim_lookup = np.zeros(367, dtype=float)
            # Only use indices present in clim_series that are within range
            valid_idx = clim_series.index[clim_series.index <= 366]
            clim_lookup[valid_idx] = clim_series[valid_idx].to_numpy(dtype=float)

            # Map dates to DOY
            doys = dates.dayofyear.to_numpy()

            # Handle leap year Feb 29 -> DOY 60 in pandas.
            # Original logic: if dt.month == 2 and dt.day == 29: doy = 59
            # Feb 29 is month 2, day 29.
            # In pandas dayofyear:
            # Jan 1 = 1
            # ...
            # Feb 28 = 59
            # Feb 29 = 60
            # Mar 1 = 61 (leap) or 60 (non-leap)

            # We want Feb 29 to map to 59 (Feb 28 climatology).
            is_leap_feb29 = (dates.month == 2) & (dates.day == 29)
            if is_leap_feb29.any():
                doys[is_leap_feb29] = 59

            return clim_lookup[doys] * scale

    p_mm_arr = _prepare_flux_array(p_daily, p_clim, p_scale)
    et_mm_arr = _prepare_flux_array(et_daily, et_clim, et_scale)

    vol = np.empty(n, dtype=float)
    area = np.empty(n, dtype=float)
    p_mcm = np.empty(n, dtype=float)
    et_mcm = np.empty(n, dtype=float)

    vol[0] = init_volume_mcm

    # Loop for sequential volume update
    for i in range(n):
        # Area from current volume
        area[i] = max(0.0, vol_to_area(vol[i]))

        # Calculate fluxes in MCM
        p_mcm[i] = p_mm_arr[i] * area[i] / 1000.0
        et_mcm[i] = et_mm_arr[i] * area[i] / 1000.0

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


def simulate_forward_era5(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    init_volume_mcm: float,
    vol_to_area: Callable[[float], float],
    daily_precip_mm: pd.Series,
    daily_evap_mm: pd.Series,
    *,
    daily_runoff_mm: Optional[pd.Series] = None,
    precip_scale: float = 1.0,
    evap_scale: float = 1.0,
    runoff_scale: float = 1.0,
    q_in_mcm_per_day: float = 0.0,
    q_out_mcm_per_day: float = 0.0,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """Forward simulation driven directly by ERA5 daily series (no climatology).

    Parameters
    ----------
    start_date, end_date : pd.Timestamp
        Simulation inclusive date range.
    init_volume_mcm : float
        Initial storage (million cubic meters).
    vol_to_area : Callable[[float], float]
        Function mapping volume_mcm -> area_km2.
    daily_precip_mm, daily_evap_mm : pd.Series
        Date-indexed (normalized) daily mm values. Index must be DatetimeIndex.
    daily_runoff_mm : Optional[pd.Series]
        Optional date-indexed runoff (mm/day) over the active area (if representing net runoff to reservoir surface). If this
        represents a contributing basin larger than reservoir area, pre-scale externally to mm-equivalent on reservoir area
        or convert to an explicit volumetric inflow and supply via q_in_mcm_per_day.
    precip_scale, evap_scale, runoff_scale : float
        Multiplicative scalars for scenario experimentation.
    q_in_mcm_per_day, q_out_mcm_per_day : float
        Constant external inflow/outflow (mcm/day) added after surface fluxes each step.
    fill_method : str
        Method to fill internal gaps in daily series ("ffill", "bfill", or "none").

    Returns
    -------
    DataFrame with columns:
        date, volume_mcm, area_km2,
        precipitation_volume_mcm, evaporation_volume_mcm, (optional) runoff_volume_mcm,
        delta_volume_mcm, residual_mcm

    Notes
    -----
    * Precipitation / evaporation volumes computed as mm * area_km2 / 1000.
    * Runoff term (if provided) is treated analogously and added to volume.
    * residual_mcm = delta_volume_mcm - P + ET - (runoff?) after accounting for q_in/q_out.
    """
    dates = pd.date_range(start_date, end_date, freq="D")
    # Normalize input series to full date range
    def _prep(s: pd.Series) -> pd.Series:
        if s is None or s.empty:
            return pd.Series(index=dates, data=0.0, dtype=float)
        ss = s.copy()
        if not isinstance(ss.index, pd.DatetimeIndex):
            ss.index = pd.to_datetime(ss.index)
        ss = ss.sort_index().reindex(dates)
        if fill_method in ("ffill", "bfill"):
            ss = getattr(ss, fill_method)()  # type: ignore[attr-defined]
        ss = ss.fillna(0.0)
        return ss.astype(float)

    p_mm = _prep(daily_precip_mm) * float(precip_scale)
    et_mm = _prep(daily_evap_mm) * float(evap_scale)
    ro_mm = _prep(daily_runoff_mm) * float(runoff_scale) if daily_runoff_mm is not None else None

    # Optimize by converting to numpy arrays for faster indexing in loop
    p_mm_arr = p_mm.to_numpy(dtype=float)
    et_mm_arr = et_mm.to_numpy(dtype=float)
    ro_mm_arr = ro_mm.to_numpy(dtype=float) if ro_mm is not None else None

    n = len(dates)
    vol = np.empty(n, dtype=float)
    area = np.empty(n, dtype=float)
    p_mcm = np.empty(n, dtype=float)
    et_mcm = np.empty(n, dtype=float)
    ro_mcm = np.empty(n, dtype=float) if ro_mm_arr is not None else None

    vol[0] = init_volume_mcm
    for i in range(n):
        area[i] = max(0.0, vol_to_area(vol[i]))
        p_mcm[i] = p_mm_arr[i] * area[i] / 1000.0
        et_mcm[i] = et_mm_arr[i] * area[i] / 1000.0
        if ro_mm_arr is not None:
            ro_mcm[i] = ro_mm_arr[i] * area[i] / 1000.0

        if i < n - 1:
            delta = p_mcm[i] - et_mcm[i]
            if ro_mcm is not None:
                delta += ro_mcm[i]
            vol[i+1] = vol[i] + delta + q_in_mcm_per_day - q_out_mcm_per_day

    delta_vol = np.empty(n, dtype=float)
    delta_vol[0] = np.nan
    delta_vol[1:] = vol[1:] - vol[:-1]

    data = {
        "date": dates,
        "volume_mcm": vol,
        "area_km2": area,
        "precipitation_volume_mcm": p_mcm,
        "evaporation_volume_mcm": et_mcm,
        "delta_volume_mcm": delta_vol,
    }
    if ro_mcm is not None:
        data["runoff_volume_mcm"] = ro_mcm

    df = pd.DataFrame(data)
    # residual: delta - P + ET - (runoff if present)
    if ro_mcm is not None:
        df["residual_mcm"] = df["delta_volume_mcm"] - df["precipitation_volume_mcm"] + df["evaporation_volume_mcm"] - df["runoff_volume_mcm"]
    else:
        df["residual_mcm"] = df["delta_volume_mcm"] - df["precipitation_volume_mcm"] + df["evaporation_volume_mcm"]
    return df

__all__ = [
    'simulate_forward',
    'simulate_forward_era5'
]
