from __future__ import annotations

import os
import numpy as np
import pandas as pd

from wbm.data import load_baseline, build_daily_climatology
from wbm.curve import build_volume_to_area
from wbm.simulate import simulate_forward


from pathlib import Path
DATA_ROOT = str(Path(__file__).resolve().parent)
OUTPUT_DIR = os.path.join(DATA_ROOT, "water_balance_output")
GLEAM_DATA_PATH = os.path.join(DATA_ROOT, "GLEAM", "processed", "gleam_summary_all_years.csv")
IMERG_DATA_PATH = os.path.join(DATA_ROOT, "precipitation_timeseries.csv")
AREA_VOLUME_CURVE_PATH = os.path.join(DATA_ROOT, "processing_output", "area_volume_curve.csv")


def build_trend_series(df: pd.DataFrame, col: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", col]].dropna().copy()
    d = d.set_index("date").asfreq("D")
    # last 4 years window
    if len(d) > 0:
        end_hist = d.index.max()
        start_hist = end_hist - pd.DateOffset(years=4)
        d = d.loc[d.index >= start_hist]
    # seasonal baseline
    doy = d.index.dayofyear
    doy = np.where((d.index.month == 2) & (d.index.day == 29), 59, doy)
    base = pd.Series(d[col].groupby(doy).mean())
    # trend
    t = np.arange(len(d), dtype=float)
    y = d[col].to_numpy()
    series = pd.Series(dtype=float)
    if len(t) >= 2 and np.isfinite(y).sum() >= 2:
        mask = np.isfinite(y)
        a, b = np.polyfit(t[mask], y[mask], 1)
        future_idx = pd.date_range(start_date, end_date, freq="D")
        tt = np.arange(len(d), len(d) + len(future_idx), dtype=float)
        trend = a * tt + b
        fdoy = future_idx.dayofyear
        fdoy = np.where((future_idx.month == 2) & (future_idx.day == 29), 59, fdoy)
        seas = base.reindex(range(1, 367)).interpolate(limit_direction="both").to_numpy()
        seas = seas[fdoy - 1]
        series = pd.Series(seas + (trend - np.nanmean(y)), index=future_idx)
    return series


def main():
    balance_df, gleam_df, imerg_df, curve_df = load_baseline(
        OUTPUT_DIR, GLEAM_DATA_PATH, IMERG_DATA_PATH, AREA_VOLUME_CURVE_PATH
    )
    if curve_df.empty:
        raise SystemExit("area_volume_curve.csv not found")

    # Determine forecast window: from last baseline date in 2025 onward to Jan 31, 2026
    last_date = pd.Timestamp("2025-01-01")
    if not balance_df.empty:
        last_date = balance_df["date"].max()
    start_date = (last_date + pd.Timedelta(days=1)).normalize()
    # If start before July 2025, align to July 1, 2025 per request
    start_date = max(start_date, pd.Timestamp("2025-07-01"))
    end_date = pd.Timestamp("2026-01-31")
    if start_date > end_date:
        print("No gap to forecast (start after end)")
        return

    # Initial volume from last known
    if not balance_df.empty:
        init_volume = float(balance_df.loc[balance_df["date"] == last_date, "volume_mcm"].tail(1).values[0])
    else:
        # fallback: mid volume of curve
        vols_sorted = curve_df.sort_values("volume_mcm")["volume_mcm"].to_numpy()
        init_volume = float(vols_sorted[len(vols_sorted)//2])

    vol_to_area, _, _ = build_volume_to_area(curve_df)

    # Drivers: seasonal + 4y trend
    p_daily = build_trend_series(imerg_df, "precipitation_mm", start_date, end_date)
    et_daily = build_trend_series(gleam_df, "evaporation_mm", start_date, end_date)
    # Climatology as fallback
    p_clim = build_daily_climatology(imerg_df, "date", "precipitation_mm")
    et_clim = build_daily_climatology(gleam_df, "date", "evaporation_mm")

    scen = simulate_forward(
        start_date=start_date,
        end_date=end_date,
        init_volume_mcm=init_volume,
        p_clim=p_clim,
        et_clim=et_clim,
        vol_to_area=vol_to_area,
        p_scale=1.0,
        et_scale=1.0,
        q_in_mcm_per_day=0.0,
        q_out_mcm_per_day=0.0,
        p_daily=p_daily,
        et_daily=et_daily,
    )

    out_path = os.path.join(OUTPUT_DIR, "forecast_2025_gap.csv")
    scen.to_csv(out_path, index=False)
    print(f"Saved forecast: {out_path}")


if __name__ == "__main__":
    main()
