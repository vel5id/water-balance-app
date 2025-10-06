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


def load_era5_daily(path: str) -> pd.DataFrame:
    """Load preprocessed ERA5 daily data from CSV or SQLite DB.

    CSV mode (legacy): expects columns date, precip_mm, evap_mm, etc.
    SQLite mode: tables named era5_t2m_c, era5_runoff_mm, era5_snow_depth_m, era5_evap_mm, era5_precip_mm
      each with (date TEXT PRIMARY KEY, value REAL).
    Returns merged wide DataFrame with date + available variable columns.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.sort_values("date").reset_index(drop=True)
        return df
    if ext in (".db", ".sqlite", ".sqlite3"):
        import sqlite3
        tables = [
            ("era5_t2m_c", "t2m_c"),
            ("era5_runoff_mm", "runoff_mm"),
            ("era5_snow_depth_m", "snow_depth_m"),
            ("era5_evap_mm", "evap_mm"),
            ("era5_precip_mm", "precip_mm"),
        ]
        frames = []
        with sqlite3.connect(path) as conn:
            cur = conn.cursor()
            existing = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for tbl, col in tables:
                if tbl in existing:
                    try:
                        tdf = pd.read_sql_query(f"SELECT date, value as {col} FROM {tbl}", conn)
                        tdf["date"] = pd.to_datetime(tdf["date"]).dt.normalize()
                        frames.append(tdf)
                    except Exception:
                        pass
        if not frames:
            return pd.DataFrame()
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="date", how="outer")
        out = out.sort_values("date").reset_index(drop=True)
        return out
    # Unknown format
    return pd.DataFrame()


def load_era5_from_raw_nc_dbs(raw_nc_root: str) -> pd.DataFrame:
    """Assemble ERA5 daily data from per-variable SQLite files inside raw_nc subfolders.

    Expected layout (each optional):
      raw_nc/temperature/era5_t2m_c.sqlite (table era5_t2m_c)
      raw_nc/total_precipitation/era5_precip_mm.sqlite (table era5_precip_mm)
      raw_nc/total_evaporation/era5_evap_mm.sqlite (table era5_evap_mm)
      raw_nc/runoff/era5_runoff_mm.sqlite (table era5_runoff_mm)
      raw_nc/snow/era5_snow_depth_m.sqlite (table era5_snow_depth_m)

    Each table: (date TEXT PRIMARY KEY, value REAL)

    Returns wide DataFrame with columns:
      date, t2m_c, precip_mm, evap_mm, runoff_mm, snow_depth_m (subset if some missing)
    """
    if not os.path.exists(raw_nc_root):
        return pd.DataFrame()
    import sqlite3

    specs = [
        ("temperature", "era5_t2m_c.sqlite", "era5_t2m_c", "t2m_c"),
        ("total_precipitation", "era5_precip_mm.sqlite", "era5_precip_mm", "precip_mm"),
        ("total_evaporation", "era5_evap_mm.sqlite", "era5_evap_mm", "evap_mm"),
        ("runoff", "era5_runoff_mm.sqlite", "era5_runoff_mm", "runoff_mm"),
        ("snow", "era5_snow_depth_m.sqlite", "era5_snow_depth_m", "snow_depth_m"),
    ]
    frames: list[pd.DataFrame] = []
    for sub, db_name, table, col in specs:
        db_path = os.path.join(raw_nc_root, sub, db_name)
        if not os.path.exists(db_path):
            continue
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(f"SELECT date, value as {col} FROM {table}", conn)
            if df.empty or "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            frames.append(df)
        except Exception:
            # Skip corrupt / unexpected files quietly
            continue
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out


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


# ---- Minimal SQLite partition helpers (reconstructed) ----

from pathlib import Path
from typing import Dict, Tuple as _Tuple


def partition_for_table(table_name: str) -> str:
    """Return partition key for a table.

    Heuristic mapping mirroring the original intent:
      - Tables starting with 'era5_' -> 'era5'
      - Tables starting with 'out_' or 'proc_' -> 'outputs'
      - Otherwise -> 'core'
    """
    lower = table_name.lower()
    if lower.startswith("era5_"):
        return "era5"
    if lower.startswith("out_") or lower.startswith("proc_"):
        return "outputs"
    return "core"


def resolve_sqlite_partitions(data_root: str, *, fallback_to_core: bool = True) -> _Tuple[Dict[str, str], Dict[str, str]]:
    """Return (existing, defaults) mapping of partition names to file paths.

    existing: partitions whose files currently exist on disk.
    defaults: suggested file paths regardless of existence.
    """
    root = Path(data_root)
    storage_dir = root / "wbm" / "storage"
    defaults = {
        "core": str(storage_dir / "water_balance_core.db"),
        "era5": str(storage_dir / "water_balance_era5.db"),
        "outputs": str(storage_dir / "water_balance_outputs.db"),
    }
    existing = {name: path for name, path in defaults.items() if Path(path).exists()}
    if not existing and fallback_to_core:
        # If nothing exists, assume a single legacy db name possibility
        legacy = storage_dir / "water_balance.db"
        if legacy.exists():
            existing["core"] = str(legacy)
    return existing, defaults


def resolve_sqlite_settings(data_root: str) -> _Tuple[bool, Dict[str, str]]:
    """Determine whether to use SQLite as a source and map partition->path.

    Use when at least one partition file exists.
    """
    existing, defaults = resolve_sqlite_partitions(data_root)
    if existing:
        return True, existing
    return False, defaults


__all__ = [
    "load_baseline",
    "build_daily_climatology",
    "partition_for_table",
    "resolve_sqlite_partitions",
    "resolve_sqlite_settings",
    "load_era5_daily",
    "load_era5_from_raw_nc_dbs",
]
