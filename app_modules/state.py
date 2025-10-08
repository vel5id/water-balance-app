from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import pandas as pd

@dataclass
class LoadedData:
    era5_df: pd.DataFrame
    balance_df: pd.DataFrame
    curve_df: pd.DataFrame
    p_clim: pd.Series
    et_clim: pd.Series
    area_to_vol: Callable
    vol_to_area: Callable
    vol_to_elev: Optional[Callable]

@dataclass
class Controls:
    # Sidebar controls
    p_scale: float
    et_scale: float
    q_in: float
    q_out: float
    min_area_km2: float
    filter_baseline: bool
    hide_scenario_below_min: bool
    start_date: pd.Timestamp
    horizon_days: int
    view_mode: str
    smooth_win: int
    forecast_mode: str
    hist_window_days: int
    seas_basis: str

@dataclass
class ScenarioResult:
    scenario_df: pd.DataFrame
    init_volume: float
    init_note: str

@dataclass
class AppContext:
    data: LoadedData
    controls: Controls
    scenario: ScenarioResult | None = None
    extra: dict = field(default_factory=dict)
