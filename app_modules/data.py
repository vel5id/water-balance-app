from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Callable
from .state import LoadedData
from wbm.data import load_baseline, build_daily_climatology, load_era5_daily, load_era5_from_raw_nc_dbs
from wbm.curve import build_area_to_volume, build_volume_to_area

try:
    from scipy.interpolate import interp1d  # type: ignore
except Exception:  # pragma: no cover
    interp1d = None  # type: ignore


def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if not df.empty and col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df[col] = pd.to_datetime(df[col]).dt.normalize()
        except Exception:
            pass
    return df


def load_all_data(data_root: str) -> LoadedData:
    OUTPUT_DIR = os.path.join(data_root, "water_balance_output")
    ERA5_DAILY_DB_PATH = os.path.join(data_root, "processing_output", "era5_daily.sqlite")
    ERA5_DAILY_CSV_PATH = os.path.join(data_root, "processing_output", "era5_daily_summary.csv")
    AREA_VOLUME_CURVE_PATH = os.path.join(data_root, "processing_output", "area_volume_curve.csv")

    balance_df, _gleam_unused, _imerg_unused, curve_df = load_baseline(
        OUTPUT_DIR, "", "", AREA_VOLUME_CURVE_PATH
    )

    era5_df = pd.DataFrame()
    if os.path.exists(ERA5_DAILY_DB_PATH):
        era5_df = load_era5_daily(ERA5_DAILY_DB_PATH)
    elif os.path.exists(ERA5_DAILY_CSV_PATH):
        era5_df = load_era5_daily(ERA5_DAILY_CSV_PATH)

    if era5_df.empty:
        raw_nc_root = os.path.join(data_root, "raw_nc")
        if os.path.exists(raw_nc_root):
            era5_df = load_era5_from_raw_nc_dbs(raw_nc_root)

    if not era5_df.empty and "date" in era5_df.columns:
        era5_df = _ensure_datetime(era5_df, "date")

    if curve_df.empty:
        raise RuntimeError("Area-volume curve not found. Run dem_processor first.")

    area_to_vol, areas, vols = build_area_to_volume(curve_df)
    vol_to_area, _, _ = build_volume_to_area(curve_df)

    # Elevation mapping
    if interp1d is not None and "elevation_m" in curve_df.columns:
        try:
            c = curve_df.dropna(subset=["volume_mcm", "elevation_m"]).sort_values("volume_mcm")
            v_vals = c["volume_mcm"].to_numpy()
            z_vals = c["elevation_m"].to_numpy()
            vol_to_elev: Callable = interp1d(
                v_vals, z_vals, kind="linear", bounds_error=False, fill_value=(float(z_vals[0]), float(z_vals[-1]))
            )
        except Exception:  # pragma: no cover
            vol_to_elev = None  # type: ignore
    else:
        vol_to_elev = None  # type: ignore

    p_clim = build_daily_climatology(era5_df, "date", "precip_mm") if not era5_df.empty else pd.Series(dtype=float)
    et_clim = build_daily_climatology(era5_df, "date", "evap_mm") if not era5_df.empty else pd.Series(dtype=float)

    return LoadedData(
        era5_df=era5_df,
        balance_df=balance_df,
        curve_df=curve_df,
        p_clim=p_clim,
        et_clim=et_clim,
        area_to_vol=area_to_vol,
        vol_to_area=vol_to_area,
        vol_to_elev=vol_to_elev,
    )
