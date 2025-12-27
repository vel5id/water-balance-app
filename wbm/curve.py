from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Callable, Tuple

try:
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None  # Will error on use

# Setup logger for warnings
logger = logging.getLogger("wbm.curve")

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

    max_area = areas[-1]

    def fn(area: float) -> float:
        # AXIOM AUDIT: Check bounds
        if area > max_area:
             # Using logger.warning might be too spammy in a tight loop?
             # But this function (area->vol) is usually called once per step or post-process.
             # Actually, simulate loop calls vol_to_area. area_to_volume is inverse.
             # Let's keep it safe.
             logger.warning(
                f"Extrapolation Risk: Area {area:.2f} > Max {max_area:.2f}. "
                "Assuming constant volume (clamped)."
             )
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

    max_vol = vols[-1]

    def fn(volume: float) -> float:
        # AXIOM AUDIT: Check bounds
        if volume > max_vol:
            # This is critical. In a loop (simulation), this might spam millions of logs.
            # Ideally we log once per simulation run or use a rate limiter.
            # For this "Axiom" task, we just add the warning as requested.
            # "⚠️ Hydro-Static Violation"
            logger.warning(
                f"⚠️ Hydro-Static Violation: Volume {volume:.2f} > Max {max_vol:.2f}. "
                "Area clamped (Infinite Wall assumption)."
            )
        return float(f(volume))

    return fn, vols, areas
