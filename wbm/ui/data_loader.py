"""Minimal data loader stub used by wbm.db when original implementation is absent.

Provides:
 - _read_csv_safe(path): read CSV if exists else empty DataFrame
 - load_all_data(...): returns tuple of DataFrames in expected order

Order expected by db._load_core_tables:
 balance_df, gleam_df, imerg_df, curve_df,
 era5_tp_df, era5_e_df, era5_swe_df, era5_ro_df, era5_t2m_df
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _read_csv_safe(path: str) -> pd.DataFrame:  # pragma: no cover - trivial guard
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def load_all_data(
    balance_dir: str,
    gleam_path: str,
    imerg_path: str,
    area_volume_path: str,
    era5_tp_path: str,
    era5_e_path: str,
    era5_swe_path: str,
    era5_ro_path: str,
    era5_t2m_path: str,
    *,
    use_sqlite: bool = False,  # unused placeholder
    sqlite_paths: dict[str, str] | None = None,  # unused placeholder
):
    balance_file = Path(balance_dir) / "water_balance_final.csv"
    balance_df = _read_csv_safe(str(balance_file))
    gleam_df = _read_csv_safe(gleam_path)
    imerg_df = _read_csv_safe(imerg_path)
    curve_df = _read_csv_safe(area_volume_path)
    era5_tp_df = _read_csv_safe(era5_tp_path)
    era5_e_df = _read_csv_safe(era5_e_path)
    era5_swe_df = _read_csv_safe(era5_swe_path)
    era5_ro_df = _read_csv_safe(era5_ro_path)
    era5_t2m_df = _read_csv_safe(era5_t2m_path)
    return (
        balance_df,
        gleam_df,
        imerg_df,
        curve_df,
        era5_tp_df,
        era5_e_df,
        era5_swe_df,
        era5_ro_df,
        era5_t2m_df,
    )


__all__ = [
    "_read_csv_safe",
    "load_all_data",
]
