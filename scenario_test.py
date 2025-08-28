import os
import pandas as pd
from wbm.data import load_baseline, build_daily_climatology
from wbm.curve import build_area_to_volume, build_volume_to_area
from wbm.simulate import simulate_forward

from pathlib import Path
DATA_ROOT = str(Path(__file__).resolve().parent)
OUTPUT_DIR = os.path.join(DATA_ROOT, "water_balance_output")
GLEAM = os.path.join(DATA_ROOT, "GLEAM", "processed", "gleam_summary_all_years.csv")
IMERG = os.path.join(DATA_ROOT, "precipitation_timeseries.csv")
CURVE = os.path.join(DATA_ROOT, "processing_output", "area_volume_curve.csv")

balance_df, gleam_df, imerg_df, curve_df = load_baseline(OUTPUT_DIR, GLEAM, IMERG, CURVE)

assert not curve_df.empty, "area_volume_curve.csv missing or empty"

_, areas, vols = build_area_to_volume(curve_df)
vol_to_area, _, _ = build_volume_to_area(curve_df)

p_clim = build_daily_climatology(imerg_df, "date", "precipitation_mm")
et_clim = build_daily_climatology(gleam_df, "date", "evaporation_mm")

start = balance_df["date"].max() if not balance_df.empty else pd.Timestamp("2024-01-01")
if not balance_df.empty:
    init_row = balance_df[balance_df["date"] == start]
    init_volume = float((init_row["volume_mcm"].iloc[-1] if not init_row.empty else balance_df["volume_mcm"].iloc[-1]))
else:
    init_volume = float(vols[len(vols)//2])

scenario = simulate_forward(
    start_date=pd.Timestamp(start), end_date=pd.Timestamp(start)+pd.Timedelta(days=180), init_volume_mcm=init_volume,
    p_clim=p_clim, et_clim=et_clim, vol_to_area=vol_to_area,
    p_scale=1.1, et_scale=0.9, q_in_mcm_per_day=0.0, q_out_mcm_per_day=0.0
)

out_path = os.path.join(OUTPUT_DIR, "scenario_test.csv")
scenario.to_csv(out_path, index=False)

summary = {
    "rows": len(scenario),
    "nan_any": scenario.isna().any().any(),
    "vol_min": float(scenario["volume_mcm"].min()),
    "vol_max": float(scenario["volume_mcm"].max()),
    "vol_end": float(scenario["volume_mcm"].iloc[-1]),
}
print("SCENARIO_OK", out_path, summary)
