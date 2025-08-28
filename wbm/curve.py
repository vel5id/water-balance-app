from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Tuple

try:
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None  # Will error on use


def build_area_to_volume(curve_df: pd.DataFrame) -> Tuple[Callable[[float], float], np.ndarray, np.ndarray]:
    """Create clamped linear interpolator: area_km2 -> volume_mcm.

    Returns (fn, areas_sorted, volumes_sorted).
    """
    if interp1d is None:
        raise ImportError("scipy is required for interpolation. Install scipy.")

    if curve_df.empty:
        raise ValueError("Empty area-volume curve.")

    c = curve_df.dropna(subset=["area_km2", "volume_mcm"]).sort_values("area_km2")
    areas = c["area_km2"].to_numpy()
    volumes = c["volume_mcm"].to_numpy()

    # Ensure monotonic areas (deduplicate)
    uniq_idx = np.unique(areas, return_index=True)[1]
    areas = areas[uniq_idx]
    volumes = volumes[uniq_idx]

    f = interp1d(areas, volumes, kind="linear", bounds_error=False, fill_value=(volumes[0], volumes[-1]))

    def fn(area: float) -> float:
        return float(f(area))

    return fn, areas, volumes


def build_volume_to_area(curve_df: pd.DataFrame) -> Tuple[Callable[[float], float], np.ndarray, np.ndarray]:
    """Create clamped linear interpolator: volume_mcm -> area_km2.
    Useful to map simulated volume back to area for P/ET scaling.
    """
    if interp1d is None:
        raise ImportError("scipy is required for interpolation. Install scipy.")

    if curve_df.empty:
        raise ValueError("Empty area-volume curve.")

    c = curve_df.dropna(subset=["area_km2", "volume_mcm"]).sort_values("volume_mcm")
    vols = c["volume_mcm"].to_numpy()
    areas = c["area_km2"].to_numpy()

    uniq_idx = np.unique(vols, return_index=True)[1]
    vols = vols[uniq_idx]
    areas = areas[uniq_idx]

    f = interp1d(vols, areas, kind="linear", bounds_error=False, fill_value=(areas[0], areas[-1]))

    def fn(volume: float) -> float:
        return float(f(volume))

    return fn, vols, areas
