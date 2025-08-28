from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Tuple


def load_baseline(
    output_dir: str,
    gleam_path: str,
    imerg_path: str,
    area_volume_curve_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load baseline artifacts and external inputs.

    Returns (balance_df, gleam_df, imerg_df, curve_df).
    """
    balance_csv = os.path.join(output_dir, "water_balance_final.csv")
    balance_df = pd.read_csv(balance_csv) if os.path.exists(balance_csv) else pd.DataFrame()
    if not balance_df.empty:
        balance_df["date"] = pd.to_datetime(balance_df["date"])  # type: ignore

    gleam_df = pd.read_csv(gleam_path) if os.path.exists(gleam_path) else pd.DataFrame()
    if not gleam_df.empty:
        gleam_df["date"] = pd.to_datetime(gleam_df["date"])  # type: ignore
        # Normalize evaporation column name to evaporation_mm
        # Try common GLEAM columns (E = total evap in mm/day)
        if "evaporation_mm" not in gleam_df.columns:
            if "E" in gleam_df.columns:
                gleam_df = gleam_df.rename(columns={"E": "evaporation_mm"})
            elif "evaporation" in gleam_df.columns:
                gleam_df = gleam_df.rename(columns={"evaporation": "evaporation_mm"})

    imerg_df = pd.read_csv(imerg_path) if os.path.exists(imerg_path) else pd.DataFrame()
    if not imerg_df.empty:
        imerg_df["date"] = pd.to_datetime(imerg_df["date"])  # type: ignore
        # Normalize precipitation column
        if "precipitation_mm" not in imerg_df.columns:
            if "mean_precip_mm_per_h" in imerg_df.columns:
                # Convert hourly mean (mm/h) to daily mm by *24
                imerg_df = imerg_df.rename(columns={"mean_precip_mm_per_h": "precipitation_mm"})
                imerg_df["precipitation_mm"] = imerg_df["precipitation_mm"] * 24.0
            elif "precipitation" in imerg_df.columns:
                imerg_df = imerg_df.rename(columns={"precipitation": "precipitation_mm"})

    curve_df = pd.read_csv(area_volume_curve_path) if os.path.exists(area_volume_curve_path) else pd.DataFrame()

    return balance_df, gleam_df, imerg_df, curve_df


def build_daily_climatology(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Return Series indexed by day-of-year (1..366) with mean value per DOY.

    Works even if leap days present; uses DOY 60 for Feb 29 by mapping to 59.
    """
    if df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)

    d = df[[date_col, value_col]].dropna()
    d = d.copy()
    d["doy"] = d[date_col].dt.dayofyear
    # Normalize leap day (Feb 29) to DOY 59 to avoid sparse index
    d.loc[(d[date_col].dt.month == 2) & (d[date_col].dt.day == 29), "doy"] = 59
    clim = d.groupby("doy")[value_col].mean()
    # Ensure 1..366 index present
    full = pd.Series(index=range(1, 367), dtype=float)
    full.loc[clim.index] = clim.values
    # Fill small gaps by interpolation
    full = full.interpolate(limit_direction="both")
    return full
