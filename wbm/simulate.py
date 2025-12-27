from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from typing import Optional, Callable

# Setup logger
logger = logging.getLogger("wbm.simulate")

# Constants
MM_KM2_TO_MCM_FACTOR: float = 1000.0
"""Divisor to convert (mm * km2) volume to Million Cubic Meters (MCM).
Dimensional analysis:
1 mm * 1 km2 = (10^-3 m) * (10^6 m2) = 10^3 m3
1 MCM = 10^6 m3
Therefore, 1 (mm * km2) = 10^-3 MCM
To convert (mm * km2) to MCM, divide by 1000.
"""

def get_climatology_index(date: pd.Timestamp) -> int:
    """
    Calculates the climatology index (1-366) for a given date, preserving
    legacy leap-year handling.

    CRITICAL: This function maps Feb 29th (Leap) to Day 59, grouping it with
    Feb 28th. Consequently, indices 60+ in the climatology represent a mix
    of dates depending on leap/non-leap status.

    Refactoring this logic will alter model hydrology.

    Args:
        date (pd.Timestamp): The simulation step date.

    Returns:
        int: The 1-based index for p_clim/et_clim lookup.
    """
    doy = date.dayofyear
    if date.month == 2 and date.day == 29:
        doy = 59
    return doy


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
      P_mcm = p_scale * P_mm(doy) * area_km2 / MM_KM2_TO_MCM_FACTOR
      ET_mcm = et_scale * ET_mm(doy) * area_km2 / MM_KM2_TO_MCM_FACTOR
      volume(t+1) = volume(t) + P_mcm - ET_mcm + q_in - q_out
    """
    # Axiom Guards: Input Validation
    if init_volume_mcm < 0:
        raise ValueError(f"Initial volume cannot be negative: {init_volume_mcm}")
    # While scales can technically be negative for sensitivity analysis (inverse),
    # NaN checks are critical.
    if np.isnan(p_scale) or np.isnan(et_scale):
        raise ValueError("Simulation scales (p_scale, et_scale) cannot be NaN.")

    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)

    vol = np.empty(n, dtype=float)
    area = np.empty(n, dtype=float)
    p_mcm = np.empty(n, dtype=float)
    et_mcm = np.empty(n, dtype=float)

    vol[0] = init_volume_mcm
    for i, dt in enumerate(dates):

        # Area from current volume
        current_area = vol_to_area(vol[i])
        if np.isnan(current_area):
            raise ValueError(f"vol_to_area returned NaN for volume {vol[i]} at step {i} ({dt.date()})")

        # Max(0.0) acts as a physical safeguard for empty reservoir conditions
        area[i] = max(0.0, current_area)

        # P/ET for this date
        if p_daily is not None and not p_daily.empty:
            p_mm = float(p_daily.get(dt, 0.0)) * p_scale
        else:
            doy = get_climatology_index(dt)
            p_mm = float(p_clim.get(doy, 0.0)) * p_scale
        if et_daily is not None and not et_daily.empty:
            et_mm = float(et_daily.get(dt, 0.0)) * et_scale
        else:
            doy = get_climatology_index(dt)
            et_mm = float(et_clim.get(doy, 0.0)) * et_scale

        p_mcm[i] = p_mm * area[i] / MM_KM2_TO_MCM_FACTOR
        et_mcm[i] = et_mm * area[i] / MM_KM2_TO_MCM_FACTOR

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
    catchment_area_km2: Optional[float] = None,
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
        Optional date-indexed runoff (mm/day).
        WARNING: If catchment_area_km2 is NOT provided, this is interpreted as runoff depth
        over the dynamic reservoir surface (Inflow = mm * Area(t)), which is physically unusual
        for basin runoff (leads to positive feedback: smaller lake -> less inflow).
    catchment_area_km2 : Optional[float]
        Static catchment area. If provided, Inflow = daily_runoff_mm * catchment_area_km2.
        This decouples inflow from the shrinking reservoir surface.
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
    * Precipitation / evaporation volumes computed as mm * area_km2 / MM_KM2_TO_MCM_FACTOR.
    * Runoff term (if provided) is treated as mm * catchment_area / MM_KM2_TO_MCM_FACTOR (if area provided)
      or mm * reservoir_area (legacy behavior).
    * residual_mcm = delta_volume_mcm - P + ET - (runoff?) after accounting for q_in/q_out.
    """
    # Axiom Guards: Input Validation
    if init_volume_mcm < 0:
        raise ValueError(f"Initial volume cannot be negative: {init_volume_mcm}")
    if np.isnan(precip_scale) or np.isnan(evap_scale) or np.isnan(runoff_scale):
        raise ValueError("Simulation scales cannot be NaN.")

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

    n = len(dates)
    vol = np.empty(n, dtype=float)
    area = np.empty(n, dtype=float)
    p_mcm = np.empty(n, dtype=float)
    et_mcm = np.empty(n, dtype=float)
    ro_mcm = np.empty(n, dtype=float) if ro_mm is not None else None

    vol[0] = init_volume_mcm

    # Check for legacy runoff behavior warning
    if ro_mm is not None and catchment_area_km2 is None:
        logger.warning(
            "⚠️ PHYSICS VIOLATION: calculating Inflow based on dynamic Reservoir Area. "
            "This creates a positive feedback loop (Shrinking Basin Paradox). "
            "Provide `catchment_area_km2` to fix."
        )

    for i, dt in enumerate(dates):
        current_area = vol_to_area(vol[i])
        if np.isnan(current_area):
            raise ValueError(f"vol_to_area returned NaN for volume {vol[i]} at step {i} ({dt.date()})")

        area[i] = max(0.0, current_area)

        p_mcm[i] = p_mm.iloc[i] * area[i] / MM_KM2_TO_MCM_FACTOR
        et_mcm[i] = et_mm.iloc[i] * area[i] / MM_KM2_TO_MCM_FACTOR

        if ro_mm is not None and ro_mcm is not None:
            if catchment_area_km2 is not None:
                # AXIOM: Physically correct scaling
                ro_mcm[i] = ro_mm.iloc[i] * catchment_area_km2 / MM_KM2_TO_MCM_FACTOR
            else:
                # LEGACY: Scaled by Lake Area
                ro_mcm[i] = ro_mm.iloc[i] * area[i] / MM_KM2_TO_MCM_FACTOR

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
