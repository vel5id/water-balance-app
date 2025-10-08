from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import pandas as pd

__all__ = [
    "LoadedData",
    "Controls",
    "ScenarioContext",
]

@dataclass
class LoadedData:
    era5_df: pd.DataFrame
    balance_df: pd.DataFrame
    curve_df: pd.DataFrame
    p_clim: pd.Series
    et_clim: pd.Series
    area_to_vol: Callable | None
    vol_to_area: Callable | None
    vol_to_elev: Callable | None

@dataclass
class Controls:
    p_scale: float
    et_scale: float
    q_in: float
    q_out: float
    min_area_km2: float
    filter_baseline: bool
    hide_scenario_below_min: bool
    view_mode: str
    smooth_win: int
    forecast_mode: str
    hist_window_days: int
    seas_basis: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp

@dataclass
class ScenarioContext:
    scenario_df: pd.DataFrame
    init_volume: float
    init_note: str
