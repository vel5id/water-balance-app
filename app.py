"""
Streamlit interactive water balance app.
Run with: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import rasterio

from wbm.data import load_baseline, build_daily_climatology
from wbm.curve import build_area_to_volume, build_volume_to_area
from wbm.simulate import simulate_forward
from wbm.plots import timeseries_figure, stacked_fluxes_figure
from wbm.analysis import rolling_trend, lagged_correlation


# --- Config paths (adapt if needed) ---
# Allow overriding via environment variable for server deployment.
# Default to the directory containing this file (project root).
DATA_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent))
OUTPUT_DIR = os.path.join(DATA_ROOT, "water_balance_output")
GLEAM_DATA_PATH = os.path.join(DATA_ROOT, "GLEAM", "processed", "gleam_summary_all_years.csv")
IMERG_DATA_PATH = os.path.join(DATA_ROOT, "precipitation_timeseries.csv")
AREA_VOLUME_CURVE_PATH = os.path.join(DATA_ROOT, "processing_output", "area_volume_curve.csv")
# Prefer integrated DEM if available, else fall back to reprojected bathymetry
DEM_INTEGRATED_PATH = os.path.join(DATA_ROOT, "processing_output", "integrated_bathymetry_copernicus.tif")
DEM_FALLBACK_PATH = os.path.join(DATA_ROOT, "processing_output", "bathymetry_reprojected_epsg4326.tif")
DEM_PATH = DEM_INTEGRATED_PATH if os.path.exists(DEM_INTEGRATED_PATH) else DEM_FALLBACK_PATH
NDWI_MASK_PATH = os.path.join(DATA_ROOT, "processing_output", "ndwi_mask_0275.tif")


st.set_page_config(page_title="Water Balance Interactive", layout="wide")
st.title("Water Balance Interactive Model")
st.caption("Scenario simulation with precipitation and evaporation scaling")

# --- Load data ---
with st.spinner("Loading data..."):
    balance_df, gleam_df, imerg_df, curve_df = load_baseline(
        OUTPUT_DIR, GLEAM_DATA_PATH, IMERG_DATA_PATH, AREA_VOLUME_CURVE_PATH
    )

if curve_df.empty:
    st.error("Area-volume curve not found. Please run dem_processor to generate it.")
    st.stop()

# Build interpolators
area_to_vol, areas, vols = build_area_to_volume(curve_df)
vol_to_area, _, _ = build_volume_to_area(curve_df)

# Build volume->elevation interpolator (clamped)
try:
    from scipy.interpolate import interp1d
    c = curve_df.dropna(subset=["volume_mcm", "elevation_m"]).sort_values("volume_mcm")
    v_vals = c["volume_mcm"].to_numpy()
    z_vals = c["elevation_m"].to_numpy()
    vol_to_elev = interp1d(v_vals, z_vals, kind="linear", bounds_error=False, fill_value=(float(z_vals[0]), float(z_vals[-1])))
except Exception:
    vol_to_elev = None

# Build climatology
p_clim = build_daily_climatology(imerg_df, "date", "precipitation_mm")
et_clim = build_daily_climatology(gleam_df, "date", "evaporation_mm")

# --- Helper to build trend-adjusted daily series ---
def build_trend_series(df: pd.DataFrame, col: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", col]].dropna().copy()
    d = d.set_index("date").asfreq("D")
    # Use all available history (no fixed window)
    # seasonal baseline by DOY
    doy = d.index.dayofyear
    doy = np.where((d.index.month == 2) & (d.index.day == 29), 59, doy)
    base = pd.Series(d[col].groupby(doy).mean())
    # linear trend over time
    t = np.arange(len(d), dtype=float)
    y = d[col].to_numpy()
    if len(t) >= 2 and np.isfinite(y).sum() >= 2:
        mask = np.isfinite(y)
        a, b = np.polyfit(t[mask], y[mask], 1)
        future_idx = pd.date_range(pd.Timestamp(start_date), end_date, freq="D")
        tt = np.arange(len(d), len(d) + len(future_idx), dtype=float)
        trend = a * tt + b
        fdoy = future_idx.dayofyear
        fdoy = np.where((future_idx.month == 2) & (future_idx.day == 29), 59, fdoy)
        seas = base.reindex(range(1, 367)).interpolate(limit_direction="both").to_numpy()
        seas = seas[fdoy - 1]
        series = pd.Series(seas + (trend - np.nanmean(y)), index=future_idx)
        return series
    return pd.Series(dtype=float)

def build_monthly_series(df: pd.DataFrame, col: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """Date-indexed series using monthly mean across all years (mm/day)."""
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", col]].dropna().copy()
    d["month"] = pd.to_datetime(d["date"]).dt.month
    monthly_mean = d.groupby("month")[col].mean()
    future_idx = pd.date_range(pd.Timestamp(start_date), end_date, freq="D")
    months = future_idx.month
    # Ensure 1..12 available and interpolated
    mm = monthly_mean.reindex(range(1, 13)).interpolate(limit_direction="both").to_numpy()
    vals = mm[months - 1]
    return pd.Series(vals, index=future_idx)

# --- Auto-fill baseline gaps for 2025 Jul..Jan ---
try:
    if balance_df is not None and not balance_df.empty:
        last_obs = pd.to_datetime(balance_df["date"]).max()
        start_fill = max((last_obs + pd.Timedelta(days=1)).normalize(), pd.Timestamp("2025-07-01"))
        end_fill = pd.Timestamp("2026-01-31")
        if start_fill <= end_fill:
            # Build daily P/ET with 4y trend; fallback to climatology
            p_daily_fill = build_trend_series(imerg_df, "precipitation_mm", start_fill, end_fill)
            et_daily_fill = build_trend_series(gleam_df, "evaporation_mm", start_fill, end_fill)
            init_vol = float(balance_df.loc[pd.to_datetime(balance_df["date"]) == last_obs, "volume_mcm"].tail(1).values[0])
            scen_fill = simulate_forward(
                start_date=start_fill,
                end_date=end_fill,
                init_volume_mcm=init_vol,
                p_clim=p_clim,
                et_clim=et_clim,
                vol_to_area=vol_to_area,
                p_scale=1.0,
                et_scale=1.0,
                q_in_mcm_per_day=0.0,
                q_out_mcm_per_day=0.0,
                p_daily=p_daily_fill,
                et_daily=et_daily_fill,
            )
            balance_df = balance_df.copy()
            balance_df["predicted"] = False
            scen_fill = scen_fill.copy()
            scen_fill["predicted"] = True
            balance_df = pd.concat([balance_df, scen_fill], ignore_index=True)
            balance_df["date"] = pd.to_datetime(balance_df["date"])  # ensure dtype
            # Save filled baseline
            filled_path = os.path.join(OUTPUT_DIR, "water_balance_final_filled.csv")
            try:
                balance_df.to_csv(filled_path, index=False)
            except Exception:
                pass
except Exception:
    # Do not break app if auto-fill fails
    pass

# Baseline section
with st.expander("Baseline info", expanded=False):
    if not balance_df.empty:
        st.write(balance_df.describe(include="all"))
    else:
        st.info("Baseline balance CSV not found. You can still run scenarios using climatology.")

# --- Controls ---
st.sidebar.header("Scenario Controls")
p_scale = st.sidebar.slider("Precipitation scaling", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
et_scale = st.sidebar.slider("Evaporation scaling", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
q_in = st.sidebar.number_input("Inflow (mcm/day)", value=0.0, step=0.1)
q_out = st.sidebar.number_input("Outflow (mcm/day)", value=0.0, step=0.1)

# Date range and initial volume
default_start = (balance_df["date"].max() if not balance_df.empty else pd.Timestamp("2024-01-01"))
start_date = st.sidebar.date_input("Start date", value=default_start.date())
horizon_days = st.sidebar.slider("Horizon (days)", min_value=30, max_value=730, value=365, step=30)
end_date = pd.Timestamp(start_date) + pd.Timedelta(days=horizon_days)

if not balance_df.empty:
    init_volume = float(balance_df.loc[balance_df["date"] == pd.Timestamp(start_date), "volume_mcm"].tail(1).fillna(method="ffill").values[0]) if (balance_df["date"] <= pd.Timestamp(start_date)).any() else float(balance_df["volume_mcm"].iloc[-1])
else:
    # fallback: set initial volume from curve midpoint
    init_volume = float(vols[len(vols)//2])

st.sidebar.markdown(f"Initial volume: **{init_volume:.1f} mcm**")

# View options and smoothing
st.sidebar.header("View")
view_mode = st.sidebar.radio("Display range", ["All period", "Single year"], index=0)
smooth_win = st.sidebar.slider("Smoothing window (days)", min_value=1, max_value=90, value=14, step=1)

# Forecast mode
st.sidebar.header("Forecast mode")
forecast_mode = st.sidebar.radio(
    "P/ET driver",
    ["Monthly mean (all years)", "Seasonal climatology", "Seasonal + trend"],
    index=1,
)

# --- Run simulation ---
if p_clim.empty or et_clim.empty:
    st.warning("Climatology missing (precipitation/evaporation). Load IMERG/GLEAM to enable scenarios.")
else:
    # Build optional date-indexed daily drivers
    p_daily = None
    et_daily = None
    if forecast_mode == "Monthly mean (all years)":
        p_daily = build_monthly_series(imerg_df, "precipitation_mm", pd.Timestamp(start_date), end_date)
        et_daily = build_monthly_series(gleam_df, "evaporation_mm", pd.Timestamp(start_date), end_date)
    elif forecast_mode == "Seasonal + trend":
        p_daily = build_trend_series(imerg_df, "precipitation_mm", pd.Timestamp(start_date), end_date)
        et_daily = build_trend_series(gleam_df, "evaporation_mm", pd.Timestamp(start_date), end_date)

    scenario_df = simulate_forward(
        start_date=pd.Timestamp(start_date),
        end_date=end_date,
        init_volume_mcm=init_volume,
        p_clim=p_clim,
        et_clim=et_clim,
        vol_to_area=vol_to_area,
        p_scale=p_scale,
        et_scale=et_scale,
        q_in_mcm_per_day=q_in,
        q_out_mcm_per_day=q_out,
        p_daily=p_daily,
        et_daily=et_daily,
    )

    # --- Filter by year and smoothing ---
    def filter_year(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        years = sorted(df["date"].dt.year.unique())
        if view_mode == "Single year":
            year = st.sidebar.selectbox("Year", years, index=len(years)-1)
            return df[df["date"].dt.year == year].reset_index(drop=True)
        return df

    def apply_smoothing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if df is None or df.empty or smooth_win <= 1:
            return df
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].rolling(window=smooth_win, min_periods=1, center=True).mean()
        return out

    plot_base = filter_year(balance_df.copy() if not balance_df.empty else balance_df)
    plot_scena = filter_year(scenario_df.copy())
    viz_df = plot_scena  # use this for map and bars
    # Apply smoothing to line plots (volume only)
    plot_base_s = apply_smoothing(plot_base, ["volume_mcm"]) if plot_base is not None else plot_base
    plot_scena_s = apply_smoothing(plot_scena, ["volume_mcm"]) if plot_scena is not None else plot_scena

    # --- Plots ---
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_ts = timeseries_figure(plot_base_s if plot_base_s is not None else plot_base, plot_scena_s)
        st.plotly_chart(fig_ts, use_container_width=True)
    with col2:
        st.metric("End Volume (mcm)", f"{plot_scena['volume_mcm'].iloc[-1]:.1f}")
        st.metric("Min Volume (mcm)", f"{plot_scena['volume_mcm'].min():.1f}")
        st.metric("Max Volume (mcm)", f"{plot_scena['volume_mcm'].max():.1f}")

    st.subheader("Daily P/ET volumes")
    st.plotly_chart(stacked_fluxes_figure(viz_df), use_container_width=True)
    # Hint if precipitation is zero
    if "precipitation_volume_mcm" in viz_df.columns and float(viz_df["precipitation_volume_mcm"].abs().sum()) == 0.0:
        st.info("Precipitation volumes are zero. Check IMERG coverage/units (precipitation_mm). Try 'Monthly mean (all years)' mode or provide more IMERG history.")

    # Download
    st.download_button(
        label="Download scenario CSV",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="scenario_water_balance.csv",
        mime="text/csv",
    )

    # Trends and correlations
    with st.expander("Trends and correlations", expanded=False):
        win = st.slider("Rolling window (days)", 7, 120, 30, 1)
        max_lag = st.slider("Max lag for correlation (days)", 7, 120, 60, 1)

        # Build combined DF for analysis
        base = plot_base.copy() if plot_base is not None and not plot_base.empty else plot_scena.copy()
        base = base[["date", "area_km2", "volume_mcm", "precipitation_volume_mcm", "evaporation_volume_mcm"]].copy()
        base = base.set_index("date").asfreq("D").interpolate()

        st.line_chart(
            pd.DataFrame({
                "area_trend": rolling_trend(base["area_km2"], window=win),
                "P_trend": rolling_trend(base["precipitation_volume_mcm"], window=win),
                "ET_trend": rolling_trend(base["evaporation_volume_mcm"], window=win),
            })
        )

        corr_p = lagged_correlation(base["area_km2"], base["precipitation_volume_mcm"], max_lag=max_lag)
        corr_et = lagged_correlation(base["area_km2"], base["evaporation_volume_mcm"], max_lag=max_lag)
        st.subheader("Lagged correlations with area")
        st.line_chart(corr_p.set_index("lag")["corr"], height=200)
        st.line_chart(corr_et.set_index("lag")["corr"], height=200)

    # --- Dynamic reservoir map ---
    st.subheader("Reservoir map: simulated water extent")
    if not os.path.exists(DEM_PATH):
        st.info("Bathymetry DEM not found; map overlay disabled. Expected: processing_output/bathymetry_reprojected_epsg4326.tif")
    elif vol_to_elev is None:
        st.info("Volume→Elevation mapping unavailable; ensure curve has elevation_m column.")
    else:
        # Load DEM (cached via st.session_state)
        if "_dem_cache" not in st.session_state:
            with rasterio.open(DEM_PATH) as ds:
                dem = ds.read(1).astype("float32")
                nodata = ds.nodata
                bounds = ds.bounds
                crs = str(ds.crs)
            st.session_state._dem_cache = {"dem": dem, "nodata": nodata, "bounds": bounds, "crs": crs}
        dem = st.session_state._dem_cache["dem"]
        nodata = st.session_state._dem_cache["nodata"]

        # Quick view controls
        vis_col1, vis_col2 = st.columns([2, 1])
        with vis_col2:
            alpha = st.slider("Water overlay opacity", 0.1, 0.9, 0.5, 0.05)
            idx = st.slider("Visualization day", 0, len(scenario_df) - 1, value=len(scenario_df) - 1)
            dem_view = st.selectbox("DEM view", ["Grayscale (low-dark)", "Grayscale (low-bright)", "Hillshade"], index=2)
            mask_mode = st.selectbox("Water mask mode", ["Depth & NDWI", "Simulated level"], index=0,
                                     help="Depth & NDWI: water where DEM depth<0 AND NDWI=water. Simulated level: threshold by volume→elevation (requires elevation DEM).")

        # Pick selected day volume and compute water level
        v_sel = float(scenario_df["volume_mcm"].iloc[idx])
        z_level = float(vol_to_elev(v_sel)) if vol_to_elev is not None else None

        # Build composite image
        dem_disp = dem.copy()
        if nodata is not None:
            dem_disp = np.where(dem_disp == nodata, np.nan, dem_disp)
        # Normalize DEM for display or compute hillshade
        vmin = np.nanpercentile(dem_disp, 2)
        vmax = np.nanpercentile(dem_disp, 98)
        norm = (dem_disp - vmin) / max(1e-6, (vmax - vmin))
        norm = np.clip(norm, 0.0, 1.0)

        if dem_view == "Grayscale (low-bright)":
            vis = 1.0 - norm  # invert so depressions appear bright
        elif dem_view == "Hillshade":
            z = np.nan_to_num(dem_disp, nan=float(np.nanmedian(dem_disp)))
            gx, gy = np.gradient(z)
            slope = np.pi/2 - np.arctan(np.hypot(gx, gy))
            aspect = np.arctan2(-gx, gy)
            az = np.radians(315.0)
            alt = np.radians(45.0)
            hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
            hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-6)
            vis = hs
        else:
            vis = norm

        base_rgb = (vis[..., None] * 255).astype(np.uint8)
        base_rgb = np.repeat(base_rgb, 3, axis=2)

        # Water mask options
        if mask_mode == "Depth & NDWI":
            # Load NDWI mask (1=water, 0=land)
            if "_ndwi_cache" not in st.session_state:
                if os.path.exists(NDWI_MASK_PATH):
                    with rasterio.open(NDWI_MASK_PATH) as ms:
                        ndwi = ms.read(1)
                    st.session_state._ndwi_cache = (ndwi > 0)
                else:
                    st.session_state._ndwi_cache = None
            ndwi_mask = st.session_state._ndwi_cache
            if ndwi_mask is None or ndwi_mask.shape != dem_disp.shape:
                st.warning("NDWI mask missing or shape mismatch; showing depth-only water (dem<0).")
                water_mask = (dem_disp < 0)
            else:
                water_mask = (dem_disp < 0) & ndwi_mask
            if nodata is not None:
                water_mask = water_mask & ~np.isnan(dem_disp)
        else:
            # Simulated level by volume→elevation assumes DEM is elevation; if DEM is depths (min<0, max≈0), fallback to dem<0
            if z_level is not None and (np.nanmin(dem_disp) >= 0 or np.nanmax(dem_disp) > 5):
                # Likely an elevation DEM
                water_mask = (dem_disp <= z_level)
                if nodata is not None:
                    water_mask = water_mask & ~np.isnan(dem_disp)
            else:
                st.info("Dynamic level requires elevation DEM; using depth-only water (dem<0).")
                water_mask = (dem_disp < 0)
                if nodata is not None:
                    water_mask = water_mask & ~np.isnan(dem_disp)

        water_color = np.array([30, 144, 255], dtype=np.uint8)  # DodgerBlue
        over = base_rgb.copy()
        # Alpha blend only where mask
        over_float = over.astype(np.float32)
        over_float[water_mask] = (1 - alpha) * over_float[water_mask] + alpha * water_color
        over_img = over_float.astype(np.uint8)

        with vis_col1:
            dem_src_note = "integrated_bathymetry_copernicus.tif" if os.path.basename(DEM_PATH).startswith("integrated_") else os.path.basename(DEM_PATH)
            caption = f"{scenario_df['date'].iloc[idx].date()} | Volume {v_sel:.1f} mcm | DEM: {dem_src_note} | Mask: {mask_mode}"
            if mask_mode == "Simulated level" and z_level is not None:
                caption += f" | Level {z_level:.2f} m"
            st.image(over_img, caption=caption, use_column_width=True)
            # DEM stats for verification
            dem_stats = np.array(dem, dtype="float32")
            if nodata is not None:
                dem_stats = np.where(dem_stats == nodata, np.nan, dem_stats)
            dmin = float(np.nanmin(dem_stats))
            dmax = float(np.nanmax(dem_stats))
            dmean = float(np.nanmean(dem_stats))
            st.caption(
                f"DEM stats — CRS: {st.session_state._dem_cache.get('crs','?')}, min: {dmin:.2f}, max: {dmax:.2f}, mean: {dmean:.2f}"
            )
