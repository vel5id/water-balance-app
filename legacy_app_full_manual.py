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

from wbm.data import load_baseline, build_daily_climatology, load_era5_daily, load_era5_from_raw_nc_dbs
from wbm.curve import build_area_to_volume, build_volume_to_area
from wbm.simulate import simulate_forward
from wbm.plots import timeseries_figure, stacked_fluxes_figure
import plotly.express as px  # for additional analytical scatter plots
import plotly.graph_objects as go  # for phase (loop) plots
from wbm.analysis import rolling_trend, lagged_correlation
from wbm.forecast import build_robust_season_trend_series, SeasonTrendResult
# Safe detection of statsmodels for plotly express trendlines
try:
    import statsmodels.api  # noqa: F401
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


try:
    from wbm.trends import (
        aggregate_series,
        theilsen_trend,
        kendall_significance,
        make_trend_comparison_figure,
        theilsen_trend_ci,
    )
except ImportError:
    # Fallback: force reload if module was partially loaded before code update
    import importlib, wbm.trends as _tr
    importlib.reload(_tr)
    from wbm.trends import (
        aggregate_series,
        theilsen_trend,
        kendall_significance,
        make_trend_comparison_figure,
        theilsen_trend_ci,
    )
except Exception:
    # Final fallback: define minimal local implementations
    import numpy as _np
    import pandas as _pd
    import plotly.graph_objects as _go
    def aggregate_series(df, date_col, value_col, *, freq, years_back, end_anchor):
        if df is None or df.empty or value_col not in df.columns:
            return _pd.Series(dtype=float)
        d = df[[date_col, value_col]].dropna().copy()
        d[date_col] = _pd.to_datetime(d[date_col])
        cutoff = end_anchor - _pd.DateOffset(years=years_back)
        d = d[(d[date_col] >= cutoff) & (d[date_col] <= end_anchor)]
        if d.empty: return _pd.Series(dtype=float)
        d = d.set_index(date_col).sort_index()
        return d[value_col].resample(freq).mean()
    def theilsen_trend_ci(series, alpha: float = 0.05):
        s = series.dropna()
        if s.empty or len(s) < 3:
            return 0.0, float('nan'), 0.0, 0.0
        arr = s.to_numpy(); n=len(arr)
        slopes=[]
        for i in range(n-1):
            dy = arr[i+1:]-arr[i]
            dx = _np.arange(i+1,n)-i
            slopes.append(dy/dx)
        slopes=_np.concatenate(slopes)
        slope=_np.median(slopes); x=_np.arange(n); inter=_np.median(arr - slope*x)
        if isinstance(s.index,_pd.DatetimeIndex): slope*=365.25
        # crude bootstrap
        rng=_np.random.default_rng(42); B=200; boot=[]
        for _ in range(B):
            samp = arr[rng.integers(0,n,size=n)]
            bs=_pd.Series(samp,index=s.index)
            a=bs.to_numpy(); ss=[]
            for i in range(n-1):
                dy=a[i+1:]-a[i]; dx=_np.arange(i+1,n)-i; ss.append(dy/dx)
            if ss:
                sl=_np.median(_np.concatenate(ss))
                if isinstance(s.index,_pd.DatetimeIndex): sl*=365.25
                boot.append(sl)
        boot=_np.sort(boot)
        lo=boot[int((alpha/2)*(len(boot)-1))]; hi=boot[int((1-alpha/2)*(len(boot)-1))]
        return float(slope), float(inter), float(lo), float(hi)
    def kendall_significance(series):
        return 0.0, float('nan')  # minimal placeholder
    def make_trend_comparison_figure(p_series, et_series, p_slope, p_inter, et_slope, et_inter):
        fig=_go.Figure()
        if not p_series.empty:
            fig.add_trace(_go.Scatter(x=p_series.index,y=p_series.values,name='P (mm/day)',mode='lines'))
        if not et_series.empty:
            fig.add_trace(_go.Scatter(x=et_series.index,y=et_series.values,name='ET (mm/day)',mode='lines'))
        fig.update_layout(template='plotly_white',title='P & ET aggregated with trends (fallback)')
        return fig
from wbm.ensemble import run_volume_ensemble, build_daily_ensemble


# --- Config paths (adapt if needed) ---
# Allow overriding via environment variable for server deployment.
# Default to the directory containing this file (project root).
DATA_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent))
OUTPUT_DIR = os.path.join(DATA_ROOT, "water_balance_output")
ERA5_DAILY_DB_PATH = os.path.join(DATA_ROOT, "processing_output", "era5_daily.sqlite")
ERA5_DAILY_CSV_PATH = os.path.join(DATA_ROOT, "processing_output", "era5_daily_summary.csv")
AREA_VOLUME_CURVE_PATH = os.path.join(DATA_ROOT, "processing_output", "area_volume_curve.csv")
# Prefer integrated DEM if available, else fall back to reprojected bathymetry
DEM_INTEGRATED_PATH = os.path.join(DATA_ROOT, "processing_output", "integrated_bathymetry_copernicus.tif")
DEM_FALLBACK_PATH = os.path.join(DATA_ROOT, "processing_output", "bathymetry_reprojected_epsg4326.tif")
DEM_PATH = DEM_INTEGRATED_PATH if os.path.exists(DEM_INTEGRATED_PATH) else DEM_FALLBACK_PATH
NDWI_MASK_PATH = os.path.join(DATA_ROOT, "processing_output", "ndwi_mask_0275.tif")


st.set_page_config(page_title="Water Balance Interactive", layout="wide")
st.title("Water Balance Interactive Model")
st.caption("Scenario simulation with precipitation and evaporation scaling")

# --- Load data (ERA5-only mode) ---
with st.spinner("Loading data (ERA5)..."):
    # load_baseline only to obtain balance_df + curve (other inputs ignored)
    balance_df, _gleam_unused, _imerg_unused, curve_df = load_baseline(
        OUTPUT_DIR, "", "", AREA_VOLUME_CURVE_PATH
    )
    # Primary attempt: central consolidated DB / CSV
    era5_df = pd.DataFrame()
    if os.path.exists(ERA5_DAILY_DB_PATH):
        era5_df = load_era5_daily(ERA5_DAILY_DB_PATH)
    elif os.path.exists(ERA5_DAILY_CSV_PATH):
        era5_df = load_era5_daily(ERA5_DAILY_CSV_PATH)

    # Fallback: assemble from per-variable raw_nc subfolder SQLite exports (created by specialized processors)
    if era5_df.empty:
        raw_nc_root = os.path.join(DATA_ROOT, "raw_nc")
        if os.path.exists(raw_nc_root):
            era5_df = load_era5_from_raw_nc_dbs(raw_nc_root)
    # Final normalization
    if not era5_df.empty and "date" in era5_df.columns:
        era5_df["date"] = pd.to_datetime(era5_df["date"]).dt.normalize()
    # For backward compatibility where variables imerg_df / gleam_df referenced later, define empty placeholders
    imerg_df = pd.DataFrame()
    gleam_df = pd.DataFrame()

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

# Build climatology from ERA5 (precip_mm / evap_mm)
p_clim = build_daily_climatology(era5_df, "date", "precip_mm") if not era5_df.empty else pd.Series(dtype=float)
et_clim = build_daily_climatology(era5_df, "date", "evap_mm") if not era5_df.empty else pd.Series(dtype=float)

# --- Utility: enforce datetime for 'date' columns to prevent pyarrow ArrowInvalid ---
def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df is not None and not df.empty and col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            # Leave as-is; downstream may drop or handle
            pass
    return df

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

# --- Robust initial volume selector ---
def select_initial_volume(
    balance: pd.DataFrame,
    start_date: pd.Timestamp,
    curve_vols: np.ndarray,
    prefer: str = "prior",
    max_interp_days: int = 30,
    max_future_days: int = 7,
) -> tuple[float, str]:
    """Pick initial volume at start_date with robust fallbacks.

    Priority:
      1) Exact match on date.
      2) Nearest prior value (if any).
      3) Linear interpolation between prior/after if both are close (<= max_interp_days each side).
      4) Nearest after if within max_future_days.
      5) Fallback to median of curve volumes.
    Returns (volume_mcm, note).
    """
    sdate = pd.Timestamp(start_date)
    try:
        df = balance[["date", "volume_mcm"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["date", "volume_mcm"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            v = float(np.nanmedian(curve_vols)) if len(curve_vols) else 0.0
            return v, "fallback: curve median"

        # 1) exact
        exact = df.loc[df["date"] == sdate, "volume_mcm"]
        if not exact.empty:
            return float(exact.iloc[-1]), "exact match"

        # split prior/after
        prior_df = df[df["date"] <= sdate]
        after_df = df[df["date"] >= sdate]

        # 2) prior
        if prefer == "prior" and not prior_df.empty:
            d_prev = prior_df.tail(1)
            days = int((sdate - d_prev["date"].iloc[-1]).days)
            return float(d_prev["volume_mcm"].iloc[-1]), f"nearest prior ({days} d)"

        # 3) interpolate if both sides close
        if not prior_df.empty and not after_df.empty:
            d_prev = prior_df.tail(1)
            d_next = after_df.head(1)
            dt1 = int((sdate - d_prev["date"].iloc[-1]).days)
            dt2 = int((d_next["date"].iloc[0] - sdate).days)
            if dt1 <= max_interp_days and dt2 <= max_interp_days and (dt1 + dt2) > 0:
                v1 = float(d_prev["volume_mcm"].iloc[-1])
                v2 = float(d_next["volume_mcm"].iloc[0])
                frac = dt1 / float(dt1 + dt2)
                v = v1 + (v2 - v1) * frac
                return float(v), f"interpolated between {d_prev['date'].iloc[-1].date()} and {d_next['date'].iloc[0].date()}"

        # 4) nearest after (limited)
        if prior_df.empty and not after_df.empty:
            d_next = after_df.head(1)
            dtf = int((d_next["date"].iloc[0] - sdate).days)
            if dtf <= max_future_days:
                return float(d_next["volume_mcm"].iloc[0]), f"nearest after ({dtf} d)"

        # 5) fallback
        v = float(np.nanmedian(curve_vols)) if len(curve_vols) else 0.0
        return v, "fallback: curve median"
    except Exception:
        v = float(np.nanmedian(curve_vols)) if len(curve_vols) else 0.0
        return v, "fallback: curve median"

# --- Auto-fill baseline gaps for 2025 Jul..Jan ---
try:
    if balance_df is not None and not balance_df.empty and not era5_df.empty:
        last_obs = pd.to_datetime(balance_df["date"]).max()
        start_fill = max((last_obs + pd.Timedelta(days=1)).normalize(), pd.Timestamp("2025-07-01"))
        end_fill = pd.Timestamp("2026-01-31")
        if start_fill <= end_fill:
            # Build daily P/ET with 4y trend; fallback to climatology
            p_daily_fill = build_trend_series(era5_df, "precip_mm", start_fill, end_fill)
            et_daily_fill = build_trend_series(era5_df, "evap_mm", start_fill, end_fill)
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

# Data filtering
st.sidebar.header("Data filtering")
min_area_km2 = st.sidebar.number_input("Min area to include (km²)", value=60.0, step=1.0,
                                       help="Записи с площадью ниже порога считаются недостоверными и исключаются из базовой хронологии")
filter_baseline = st.sidebar.checkbox("Filter baseline by min area", value=False)
hide_scenario_below_min = st.sidebar.checkbox("Hide scenario below min area (plots only)", value=False,
                                              help="Сценарий не изменяется — точки ниже порога скрываются только на графиках")

# Apply baseline filtering (if available and requested) before using it for initialization
if filter_baseline and not balance_df.empty and "area_km2" in balance_df.columns:
    before_n = len(balance_df)
    balance_df = balance_df[balance_df["area_km2"] >= float(min_area_km2)].reset_index(drop=True)
    removed_n = before_n - len(balance_df)
else:
    removed_n = 0

# Date range and initial volume (default to today)
today = pd.Timestamp.today().normalize()
start_date = st.sidebar.date_input("Start date", value=today.date())
horizon_days = st.sidebar.slider("Horizon (days)", min_value=30, max_value=730, value=365, step=30)
end_date = pd.Timestamp(start_date) + pd.Timedelta(days=horizon_days)

if not balance_df.empty:
    init_volume, init_note = select_initial_volume(balance_df, pd.Timestamp(start_date), vols)
else:
    init_volume, init_note = (float(vols[len(vols)//2]), "fallback: curve midpoint")

st.sidebar.markdown(f"Initial volume: **{init_volume:.1f} mcm**")
st.sidebar.caption(f"Source: {init_note}")

# Show min volume corresponding to min area
try:
    min_vol_threshold = float(area_to_vol(float(min_area_km2)))
    st.sidebar.caption(f"Min area threshold ⇒ volume ≥ {min_vol_threshold:.1f} mcm")
except Exception:
    pass

if removed_n > 0:
    st.sidebar.info(f"Baseline filtered: removed {removed_n} rows below {min_area_km2:.1f} km²")

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

from wbm import seasonal as _seasonal_doc
with st.expander("Season+Trend method (docs)", expanded=False):
    st.markdown(_seasonal_doc.markdown_doc())

# Advanced controls for Season + Trend
with st.sidebar.expander("Season+Trend options"):
    hist_window_days = st.number_input("History window (days)", min_value=0, max_value=3650, value=730, step=30,
                                       help="0 = use full history (используется как минимальная история для устойчивости тренда)")
    seas_basis = st.selectbox("Season basis", ["DOY", "MONTH"], index=0,
                              help="DOY = день года (гибче), MONTH = усреднение по месяцу (грубее, устойчивее)")

# --- Run simulation ---
if p_clim.empty or et_clim.empty:
    st.warning("Climatology missing (ERA5 precip/evap). Ensure processing scripts ran (central DB or raw_nc/*.sqlite).")
else:
    # Build optional date-indexed daily drivers
    p_daily = None
    et_daily = None
    def _prepare_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
        if df is None or df.empty or value_col not in df.columns or date_col not in df.columns:
            return pd.Series(dtype=float)
        d = df[[date_col, value_col]].dropna().copy()
        d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
        return d.set_index(date_col)[value_col].asfreq("D")

    if forecast_mode == "Monthly mean (all years)":
        p_daily = build_monthly_series(era5_df, "precip_mm", pd.Timestamp(start_date), end_date)
        et_daily = build_monthly_series(era5_df, "evap_mm", pd.Timestamp(start_date), end_date)
    elif forecast_mode == "Seasonal + trend":
        base_p = _prepare_series(era5_df, "date", "precip_mm")
        base_et = _prepare_series(era5_df, "date", "evap_mm")
        freq = "doy" if seas_basis == "DOY" else "month"
        future_days = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)
        min_hist = int(hist_window_days) if hist_window_days and hist_window_days > 0 else 90
        try:
            res_p: SeasonTrendResult = build_robust_season_trend_series(
                base_p, freq=freq, future_days=future_days, min_history=min_hist
            )
            p_daily = res_p.deterministic
        except Exception:
            p_daily = pd.Series(dtype=float)
        try:
            res_et: SeasonTrendResult = build_robust_season_trend_series(
                base_et, freq=freq, future_days=future_days, min_history=min_hist
            )
            et_daily = res_et.deterministic
        except Exception:
            et_daily = pd.Series(dtype=float)

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
    if hide_scenario_below_min and plot_scena is not None and not plot_scena.empty:
        plot_scena = plot_scena.copy()
        plot_scena.loc[plot_scena["area_km2"] < float(min_area_km2), ["volume_mcm", "area_km2"]] = np.nan
    viz_df = plot_scena  # use this for map and bars
    # Apply smoothing to line plots (volume only)
    plot_base_s = apply_smoothing(plot_base, ["volume_mcm"]) if plot_base is not None else plot_base
    plot_scena_s = apply_smoothing(plot_scena, ["volume_mcm"]) if plot_scena is not None else plot_scena

    # --- Plots ---
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_ts = timeseries_figure(plot_base_s if plot_base_s is not None else plot_base, plot_scena_s)
        st.plotly_chart(fig_ts, width="stretch")
    with col2:
        st.metric("End Volume (mcm)", f"{plot_scena['volume_mcm'].iloc[-1]:.1f}")
        st.metric("Min Volume (mcm)", f"{plot_scena['volume_mcm'].min():.1f}")
        st.metric("Max Volume (mcm)", f"{plot_scena['volume_mcm'].max():.1f}")

    st.subheader("Daily P/ET volumes")
    st.plotly_chart(stacked_fluxes_figure(viz_df), width="stretch")
    # Hint if precipitation is zero
    if "precipitation_volume_mcm" in viz_df.columns and float(viz_df["precipitation_volume_mcm"].abs().sum()) == 0.0:
        st.info("Precipitation volumes are zero. Check ERA5 processing (precip_mm). Re-run era5_process.py or switch driver mode.")

    # Download (robust to ArrowInvalid by enforcing datetime and fallback to string)
    scen_safe = _ensure_datetime(scenario_df.copy(), "date")
    try:
        csv_bytes = scen_safe.to_csv(index=False).encode("utf-8")
    except Exception:
        try:
            scen_safe["date"] = scen_safe["date"].astype(str)
        except Exception:
            pass
        csv_bytes = scen_safe.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download scenario CSV", data=csv_bytes, file_name="scenario_water_balance.csv", mime="text/csv")

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

    # --- Long-term trends P & ET ---
    with st.expander("Long-term P & ET trends", expanded=False):
        years_back = st.slider("Years back", 3, 30, 10, 1)
        # Pandas deprecates 'M' (month end) in favor of 'ME'; we keep UI labels but map internally.
        freq_label = st.selectbox(
            "Aggregation",
            ["Monthly", "Annual"],
            index=0,
            help="Monthly=month-end mean (ME), Annual=calendar year (A)"
        )
        freq = 'ME' if freq_label == 'Monthly' else 'A'
        end_anchor = pd.Timestamp.today().normalize()

        p_agg = aggregate_series(era5_df, "date", "precip_mm", freq=freq, years_back=years_back, end_anchor=end_anchor)
        et_agg = aggregate_series(era5_df, "date", "evap_mm", freq=freq, years_back=years_back, end_anchor=end_anchor)

        if not p_agg.empty and not et_agg.empty:
            p_slope_py, p_inter, p_lo, p_hi = theilsen_trend_ci(p_agg)
            et_slope_py, et_inter, et_lo, et_hi = theilsen_trend_ci(et_agg)
            p_tau, p_p = kendall_significance(p_agg)
            et_tau, et_p = kendall_significance(et_agg)

            st.plotly_chart(make_trend_comparison_figure(p_agg, et_agg, p_slope_py, p_inter, et_slope_py, et_inter), width="stretch")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("P slope (mm/day/yr)", f"{p_slope_py:.4f}")
                st.caption(f"CI [{p_lo:.4f}, {p_hi:.4f}] | Kendall tau={p_tau:.3f}, p={p_p:.3g}")
            with col_t2:
                st.metric("ET slope (mm/day/yr)", f"{et_slope_py:.4f}")
                st.caption(f"CI [{et_lo:.4f}, {et_hi:.4f}] | Kendall tau={et_tau:.3f}, p={et_p:.3g}")
        else:
            st.info("Not enough data to compute long-term trends.")

    # --- Snow vs Temperature analytical trends ---
    with st.expander("Snow vs Temperature Trends", expanded=False):
        if era5_df is None or era5_df.empty or not {"t2m_c", "snow_depth_m"}.issubset(era5_df.columns):
            st.info("ERA5 dataframe lacks required columns (t2m_c, snow_depth_m). Ensure processing scripts produced snow depth & temperature.")
        else:
            base = era5_df[["date", "t2m_c", "snow_depth_m"]].dropna().copy()
            if base.empty:
                st.info("No overlapping non-NaN snow/temperature data available.")
            else:
                base["date"] = pd.to_datetime(base["date"]).dt.normalize()
                base = base.drop_duplicates("date").set_index("date").sort_index().asfreq("D")
                # Controls
                col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                with col_cfg1:
                    win = st.slider("Rolling trend window (days)", 7, 120, 30, 1, key="snow_temp_win")
                with col_cfg2:
                    agg_mode = st.selectbox("Display aggregation", ["Daily", "Monthly"], index=0, key="snow_temp_agg")
                with col_cfg3:
                    ratio_clip = st.number_input("Clip ratio at ±", min_value=0.0, max_value=1000.0, value=50.0, step=1.0,
                                                 help="Prevents extreme spikes when temperature trend ~0")

                # Compute rolling linear trend proxy via existing rolling_trend (centered difference style)
                snow_tr = rolling_trend(base["snow_depth_m"], window=win)
                temp_tr = rolling_trend(base["t2m_c"], window=win)
                # Avoid division by near-zero: mask small temperature trends
                eps = 1e-6
                denom = temp_tr.where(temp_tr.abs() > eps)
                ratio = (snow_tr / denom).replace([np.inf, -np.inf], np.nan)
                if ratio_clip > 0:
                    ratio = ratio.clip(lower=-ratio_clip, upper=ratio_clip)

                trend_df = pd.DataFrame({
                    "snow_trend": snow_tr,
                    "temp_trend": temp_tr,
                    "ratio_snow_to_temp": ratio,
                })

                if agg_mode == "Monthly":
                    # Use 'ME' (month end) instead of deprecated 'M'
                    trend_plot_df = trend_df.resample("ME").mean()
                else:
                    trend_plot_df = trend_df

                st.markdown("**Rolling trends and their ratio** (snow depth m/day, temperature °C/day, ratio unitless).")
                st.line_chart(trend_plot_df)

                # Scatter: temperature vs snow depth (optionally monthly mean)
                if agg_mode == "Monthly":
                    scatter_df = base.resample("ME").mean().reset_index()
                else:
                    scatter_df = base.reset_index()
                scatter_df["year"] = scatter_df["date"].dt.year
                scatter_df["month"] = scatter_df["date"].dt.month

                fig_scatter = px.scatter(
                    scatter_df,
                    x="t2m_c",
                    y="snow_depth_m",
                    color="year",
                    hover_data={"date": True, "month": True},
                    trendline=("ols" if _HAS_STATSMODELS and len(scatter_df) > 10 else None),
                    labels={"t2m_c": "Temperature (°C)", "snow_depth_m": "Snow depth (m)"},
                    title="Temperature vs Snow Depth"
                )
                if not _HAS_STATSMODELS:
                    st.caption("(statsmodels не установлен: regression trendline отключён)")

                # Ratio distribution
                ratio_clean = ratio.dropna()
                if not ratio_clean.empty:
                    q = ratio_clean.quantile([0.05, 0.5, 0.95]).to_dict()
                    st.caption(f"Ratio (snow_trend / temp_trend) quantiles: 5%={q.get(0.05):.3f}, 50%={q.get(0.5):.3f}, 95%={q.get(0.95):.3f}")
                    # Year-wise quantile summary
                    ratio_yearly = ratio_clean.groupby(ratio_clean.index.year).describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                    st.dataframe(ratio_yearly[["mean","std","min","5%","25%","50%","75%","95%","max"]].rename_axis("year"), width="stretch")
                    # Boxplot across years
                    ry = ratio_clean.to_frame("ratio").reset_index()
                    ry["year"] = ry["date"].dt.year
                    fig_box = px.box(ry, x="year", y="ratio", points="suspectedoutliers", title="Year-wise distribution of snow/temp trend ratio")
                    st.plotly_chart(fig_box, width="stretch")
                st.markdown("_Примечание_: Значения отношения могут быть неустойчивыми при близком к нулю тренде температуры. Используйте клиппинг для стабилизации визуализации.")

    # --- Runoff vs Temperature analytical trends ---
    with st.expander("Runoff vs Temperature Trends", expanded=False):
        required_cols = {"t2m_c", "runoff_mm"}
        if era5_df is None or era5_df.empty or not required_cols.issubset(era5_df.columns):
            st.info("ERA5 dataframe lacks required columns (t2m_c, runoff_mm). Ensure runoff was processed.")
        else:
            rbase = era5_df[["date", "t2m_c", "runoff_mm"]].dropna().copy()
            if rbase.empty:
                st.info("No overlapping runoff/temperature data.")
            else:
                rbase["date"] = pd.to_datetime(rbase["date"]).dt.normalize()
                rbase = rbase.drop_duplicates("date").set_index("date").sort_index().asfreq("D")
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    win_r = st.slider("Rolling window (days)", 7, 180, 60, 1, key="runoff_temp_win")
                with col_r2:
                    agg_r = st.selectbox("Aggregation", ["Daily","Monthly"], index=1, key="runoff_temp_agg")
                with col_r3:
                    clip_r = st.number_input("Clip ratio at ±", 0.0, 1000.0, 50.0, 1.0, key="runoff_temp_clip")

                runoff_tr = rolling_trend(rbase["runoff_mm"], window=win_r)
                temp_tr_r = rolling_trend(rbase["t2m_c"], window=win_r)
                eps = 1e-6
                denom_r = temp_tr_r.where(temp_tr_r.abs() > eps)
                ratio_r = (runoff_tr / denom_r).replace([np.inf, -np.inf], np.nan)
                if clip_r > 0:
                    ratio_r = ratio_r.clip(-clip_r, clip_r)

                run_tr_df = pd.DataFrame({
                    "runoff_trend": runoff_tr,
                    "temp_trend": temp_tr_r,
                    "ratio_runoff_to_temp": ratio_r,
                })
                if agg_r == "Monthly":
                    plot_run_df = run_tr_df.resample("ME").mean()
                else:
                    plot_run_df = run_tr_df
                st.line_chart(plot_run_df)

                # Lagged correlations (temperature leading runoff)
                max_lag_r = st.slider("Max lag for temp→runoff corr (days)", 5, 120, 45, 5)
                # Reuse lagged_correlation on raw (not trends) for physical interpretability
                lr_corr = lagged_correlation(rbase["runoff_mm"], rbase["t2m_c"], max_lag=max_lag_r)
                st.subheader("Lagged correlation: runoff vs temperature")
                if not lr_corr.empty:
                    st.line_chart(lr_corr.set_index("lag")["corr"], height=180)
                # Scatter runoff vs temperature
                rb_scatter = (rbase.resample("ME").mean() if agg_r == "Monthly" else rbase).reset_index()
                rb_scatter["year"] = rb_scatter["date"].dt.year
                fig_run_sc = px.scatter(rb_scatter, x="t2m_c", y="runoff_mm", color="year", trendline=("ols" if _HAS_STATSMODELS else None), title="Temperature vs Runoff")
                if not _HAS_STATSMODELS:
                    st.caption("(statsmodels не установлен: regression trendline отключён)")
                # Quantiles
                rq = ratio_r.dropna().quantile([0.05,0.5,0.95]).to_dict()
                st.caption(f"Runoff/Temp trend ratio quantiles: 5%={rq.get(0.05):.3f}, 50%={rq.get(0.5):.3f}, 95%={rq.get(0.95):.3f}")

    # --- Precipitation / Evaporation diagnostics ---
    with st.expander("Precipitation & Evaporation Diagnostics", expanded=False):
        if era5_df is None or era5_df.empty or not {"precip_mm","evap_mm"}.issubset(era5_df.columns):
            st.info("ERA5 dataframe lacks precip_mm / evap_mm columns.")
        else:
            pcols = era5_df[["date","precip_mm","evap_mm"]].dropna().copy()
            if pcols.empty:
                st.info("No P/ET data available.")
            else:
                pcols["date"] = pd.to_datetime(pcols["date"])\
                    .dt.normalize()
                pcols = pcols.drop_duplicates("date").set_index("date").sort_index()
                col_pe1, col_pe2, col_pe3 = st.columns(3)
                with col_pe1:
                    win_pe = st.slider("Rolling window (days)", 7, 180, 60, 1, key="p_et_win")
                with col_pe2:
                    agg_pe = st.selectbox("Aggregation", ["Daily","Monthly"], index=1, key="p_et_agg")
                with col_pe3:
                    clip_ratio_pe = st.number_input("Clip P/ET ratio at ±", 0.0, 1000.0, 100.0, 1.0, key="p_et_clip")

                # Compute rolling P/ET ratio and drought index (P-ET)
                p_roll = pcols["precip_mm"].rolling(win_pe, min_periods=max(3,int(win_pe*0.3))).mean()
                et_roll = pcols["evap_mm"].rolling(win_pe, min_periods=max(3,int(win_pe*0.3))).mean()
                eps = 1e-6
                pet_ratio = (p_roll / et_roll.where(et_roll.abs() > eps)).replace([np.inf,-np.inf], np.nan)
                if clip_ratio_pe > 0:
                    pet_ratio = pet_ratio.clip(-clip_ratio_pe, clip_ratio_pe)
                water_balance_index = p_roll - et_roll  # positive => surplus, negative => deficit

                diag_df = pd.DataFrame({
                    "P_roll": p_roll,
                    "ET_roll": et_roll,
                    "P_over_ET": pet_ratio,
                    "P_minus_ET": water_balance_index,
                })
                if agg_pe == "Monthly":
                    diag_plot_df = diag_df.resample("ME").mean()
                else:
                    diag_plot_df = diag_df
                st.line_chart(diag_plot_df)

                # Heatmap of anomalies (monthly) for P-ET
                anom = water_balance_index.resample("ME").mean()
                anom_mean = anom.groupby(anom.index.month).transform(lambda x: x.mean())
                anom_std = anom.groupby(anom.index.month).transform(lambda x: x.std(ddof=0))
                standardized = (anom - anom_mean) / anom_std.replace(0, np.nan)
                # Pivot year x month
                df_heat = standardized.to_frame("z").reset_index()
                df_heat["year"] = df_heat["date"].dt.year
                df_heat["month"] = df_heat["date"].dt.month
                pivot = df_heat.pivot(index="year", columns="month", values="z")
                st.markdown("**Standardized anomaly (z-score) of (P-ET) by month**")
                st.dataframe(pivot.style.background_gradient(cmap="coolwarm", axis=None), width="stretch")

                # Distribution and quantiles
                quant = pet_ratio.dropna().quantile([0.05,0.25,0.5,0.75,0.95]).to_dict()
                st.caption("P/ET ratio quantiles: " + ", ".join([f"{int(k*100)}%={v:.2f}" for k,v in quant.items()]))
                quant_wb = water_balance_index.dropna().quantile([0.05,0.5,0.95]).to_dict()
                st.caption(f"(P-ET) rolling index quantiles: 5%={quant_wb.get(0.05):.2f}, 50%={quant_wb.get(0.5):.2f}, 95%={quant_wb.get(0.95):.2f}")

    # --- Seasonal Phase (Loop) Plots ---
    with st.expander("Seasonal Phase (Loop) Plots", expanded=False):
        st.markdown("Создание годовых сезонных петель (phase diagrams) для пар переменных. Каждая петля отображает траекторию внутри календарного года.")
        if era5_df is None or era5_df.empty:
            st.info("ERA5 dataframe is empty; cannot build phase plots.")
        else:
            phase_df = era5_df.copy()
            if "date" not in phase_df.columns:
                st.info("ERA5 dataframe lacks date column.")
            else:
                phase_df["date"] = pd.to_datetime(phase_df["date"])\
                    .dt.normalize()
                # Candidate numeric columns
                numeric_cols = [c for c in ["precip_mm","evap_mm","runoff_mm","t2m_c","snow_depth_m"] if c in phase_df.columns]
                if len(numeric_cols) < 2:
                    st.info("Not enough numeric ERA5 driver columns for phase plots.")
                else:
                    col_ph1, col_ph2, col_ph3 = st.columns(3)
                    with col_ph1:
                        x_var = st.selectbox("X variable", numeric_cols, index=0)
                    with col_ph2:
                        y_var = st.selectbox("Y variable", numeric_cols, index=min(1,len(numeric_cols)-1))
                    with col_ph3:
                        color_mode = st.selectbox("Color by", ["month","doy"], index=0)
                    years_available = sorted(phase_df["date"].dt.year.unique())
                    sel_years = st.multiselect("Years", years_available, default=years_available[-min(3,len(years_available)):])
                    smooth_days = st.slider("Smoothing (rolling days)", 1, 30, 5, 1, help="Rolling mean applied to both series per year")
                    show_markers = st.checkbox("Show markers", value=False)
                    if x_var == y_var:
                        st.warning("Select different variables for X and Y.")
                    else:
                        # Build figure with one trace per year (loop)
                        fig_phase = go.Figure()
                        for yr in sel_years:
                            sub = phase_df[phase_df["date"].dt.year == yr].set_index("date").sort_index()
                            if sub.empty: continue
                            x_raw = sub[x_var].astype(float)
                            y_raw = sub[y_var].astype(float)
                            if smooth_days > 1:
                                x_series = x_raw.rolling(smooth_days, min_periods=max(2,int(smooth_days*0.5))).mean()
                                y_series = y_raw.rolling(smooth_days, min_periods=max(2,int(smooth_days*0.5))).mean()
                            else:
                                x_series = x_raw
                                y_series = y_raw
                            # Color gradient within year
                            if color_mode == "month":
                                color_vals = x_series.index.month
                                cmin, cmax = 1, 12
                                colorscale = "Turbo"
                            else:
                                # day-of-year
                                doy = x_series.index.dayofyear
                                # unify leap day to 59
                                doy = np.where((x_series.index.month==2) & (x_series.index.day==29), 59, doy)
                                color_vals = doy
                                cmin, cmax = 1, 366
                                colorscale = "Viridis"
                            fig_phase.add_trace(go.Scatter(
                                x=x_series.values,
                                y=y_series.values,
                                mode="lines+markers" if show_markers else "lines",
                                line=dict(width=2),
                                marker=dict(size=5, color=color_vals, colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=False),
                                name=str(yr),
                                hovertemplate="Year=%s<Br>%s=%s<Br>%s=%s<Br>Date=%s<extra></extra>" % (
                                    yr, x_var, "%{x:.2f}", y_var, "%{y:.2f}", "%{text}"),
                                text=[d.strftime("%Y-%m-%d") for d in x_series.index]
                            ))
                        fig_phase.update_layout(
                            template="plotly_white",
                            title=f"Seasonal Loop: {x_var} vs {y_var}",
                            xaxis_title=x_var,
                            yaxis_title=y_var,
                            legend_title="Year",
                        )
                        st.plotly_chart(fig_phase, width="stretch")
                        st.caption("Петля показывает внутригодовую траекторию. Сравнение разных лет помогает выявлять смещения фенологии / фазовых переходов.")


    # --- Ensemble forecast (experimental) ---
    with st.expander("Ensemble forecast (experimental)", expanded=False):
        use_ens = st.checkbox("Enable ensemble", value=False, help="Run N stochastic members with season+trend + residual bootstrap")
        if use_ens:
            N = st.slider("Members (N)", 20, 300, 100, 10)
            hist_days = st.number_input("History window (days)", min_value=0, max_value=3650, value=5*365, step=30)
            seas_basis = st.selectbox("Season basis", ["DOY", "MONTH"], index=0)
            seas_smooth = st.slider("Seasonal smooth window (days)", 0, 31, 7, 1)
            center_mode = st.selectbox("Centering", ["median", "mean"], index=0)
            use_boot = st.checkbox("Residual block bootstrap", value=True)
            block_mode = st.selectbox("Block length mode", ["Auto","Manual"], index=0,
                                      help="Auto: оценка по затуханию ACF остатка; Manual: фиксированное значение")
            block_len_manual = st.slider("Block length (days)", 1, 14, 5, 1, disabled=block_mode=="Auto")
            # NOTE: Streamlit select_slider поддерживает одиночное значение или диапазон (tuple из 2 значений).
            # Исходная попытка выбрать три квантиля (low, median, high) напрямую приводила к ValueError (ожидалось 2 значения).
            # Решение: пользователь выбирает диапазон (low..high); медианный квантиль вычисляем автоматически (0.5 если попадает в диапазон,
            # иначе середина диапазона). Это сохраняет прежнюю семантику fan chart без ошибки распаковки.
            q_low, q_high = st.select_slider(
                "Quantile range",
                options=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                value=(0.05, 0.95),
                help="Выберите нижний и верхний квантиль для диапазона неопределенности. Центральный (median) определяется автоматически.",
            )
            if 0.5 >= q_low and 0.5 <= q_high:
                q_med = 0.5
            else:
                q_med = round((q_low + q_high) / 2.0, 4)
            # Страховка: привести к возрастанию и уникальности
            q_low, q_med, q_high = sorted({float(q_low), float(q_med), float(q_high)})[0], sorted({float(q_low), float(q_med), float(q_high)})[1], sorted({float(q_low), float(q_med), float(q_high)})[2]
            # --- Build deterministic future precipitation series using chosen season/trend method ---
            future_days = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)
            freq = "doy" if seas_basis == "DOY" else "month"
            hist_min = int(hist_days) if int(hist_days) > 0 else 90
            # Подготовка серии осадков
            base_p = _prepare_series(era5_df, "date", "precip_mm") if (era5_df is not None and not era5_df.empty) else pd.Series(dtype=float)
            try:
                future_p_res = build_robust_season_trend_series(base_p, freq=freq, future_days=future_days, min_history=hist_min)
                future_p = future_p_res.deterministic
            except Exception:
                future_p = None
            if future_p is None or future_p.empty:
                st.warning("Не удалось построить детерминированную серию осадков для ансамбля.")
            else:
                # --- Residuals history (простейший подход: фактическое - дневная климатология) ---
                try:
                    hist = era5_df[["date","precip_mm"]].dropna().copy()
                    hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
                    hist = hist.drop_duplicates("date").set_index("date").asfreq("D")
                    # build climatology by DOY (already available p_clim, но он по volume; пересчитаем для осадков)
                    doy = hist.index.dayofyear
                    doy = np.where((hist.index.month==2) & (hist.index.day==29), 59, doy)
                    clim_vals = pd.Series(hist["precip_mm"].groupby(doy).mean())
                    clim_series = pd.Series(clim_vals.reindex(range(1,367)).interpolate(limit_direction="both").to_numpy()[doy-1], index=hist.index)
                    residual_hist = (hist["precip_mm"] - clim_series)
                except Exception:
                    residual_hist = hist["precip_mm"] if 'hist' in locals() else pd.Series(dtype=float)

                # Авто выбор длины блока
                from wbm.seasonal import compute_acf, recommend_block_length, theil_sen_trend_ci_boot, seasonal_keys, robust_seasonal_template
                if use_boot and block_mode == "Auto":
                    acf_df = compute_acf(residual_hist.to_numpy(), max_lag=min(30, max(5, residual_hist.size//4)))
                    reco_block = recommend_block_length(acf_df, residual_hist.size)
                    st.caption(f"Auto block length recommendation: {reco_block}")
                    eff_block = reco_block
                else:
                    eff_block = int(block_len_manual) if use_boot else 1

                residual_sets = build_daily_ensemble(
                    deterministic_future=future_p,
                    residuals=residual_hist,
                    n_members=int(N),
                    block_size=eff_block,
                    random_state=42,
                )
                result = run_volume_ensemble(
                    start_volume_mcm=float(init_volume),
                    vol_to_area=vol_to_area,
                    p_clim=p_clim,
                    et_clim=et_clim,
                    deterministic_p=future_p * float(p_scale),  # масштабирование P
                    residual_sets=[r * float(p_scale) for r in residual_sets],
                    p_scale=1.0,  # уже учли масштаб в ряду
                    et_scale=float(et_scale),
                )
                # Построение квантилей по выбранным (q_low, q_med, q_high)
                aligned = [m.set_index("date")["volume_mcm"] for m in result.members]
                all_vols = pd.concat(aligned, axis=1)
                qdf = all_vols.quantile([q_low, q_med, q_high], axis=1).T
                qcols_map = {q_low: f"vol_q{int(q_low*100)}", q_med: f"vol_q{int(q_med*100)}", q_high: f"vol_q{int(q_high*100)}"}
                qdf.rename(columns=qcols_map, inplace=True)
                qdf.reset_index(inplace=True)
                qdf.rename(columns={"index":"date"}, inplace=True)
                ens = qdf
                if not ens.empty:
                    # Overlay fan chart by filling between quantiles
                    import plotly.graph_objects as go
                    fan = go.Figure()
                    # base scenario line for context
                    fan.add_trace(go.Scatter(x=plot_scena_s["date"], y=plot_scena_s["volume_mcm"], mode="lines",
                                             name="Scenario Volume (deterministic)", line=dict(color="#d62728")))
                    # fan fill
                    fan.add_trace(go.Scatter(x=ens["date"], y=ens[f"vol_q{int(q_high*100)}"], name=f"Q{int(q_high*100)}",
                                             line=dict(color="rgba(31,119,180,0.0)")))
                    fan.add_trace(go.Scatter(x=ens["date"], y=ens[f"vol_q{int(q_low*100)}"], name=f"Q{int(q_low*100)}",
                                             fill="tonexty", fillcolor="rgba(31,119,180,0.2)", line=dict(color="rgba(31,119,180,0.0)")))
                    fan.add_trace(go.Scatter(x=ens["date"], y=ens[f"vol_q{int(q_med*100)}"], name=f"Q{int(q_med*100)} (median)",
                                             line=dict(color="#1f77b4", width=2)))
                    fan.update_layout(title="Ensemble Volume Forecast", template="plotly_white",
                                      xaxis_title="Date", yaxis_title="Volume (million m³)")
                    st.plotly_chart(fan, width="stretch")
                    # Diagnostics expander
                    with st.expander("Ensemble diagnostics", expanded=False):
                        from wbm.seasonal import compute_acf, recommend_block_length, theil_sen_trend_ci_boot
                        st.markdown("**Residual ACF & Trend CI**")
                        acf_df = compute_acf(residual_hist.to_numpy(), max_lag=min(30, max(5, residual_hist.size//4))) if residual_hist.size else None
                        if acf_df is not None and not acf_df.empty:
                            st.dataframe(acf_df, width="stretch", height=180)
                            st.caption("ACF: автокорреляция по лагам (после удаления сезонности+тренда)")
                            rec_block = recommend_block_length(acf_df, residual_hist.size)
                            st.caption(f"Recommended block length (ACF threshold): {rec_block}")
                        else:
                            st.info("Not enough residual data for ACF.")
                        try:
                            slope_py, lo_py, hi_py = theil_sen_trend_ci_boot(base_p, freq=freq, n_boot=200)
                            st.markdown(f"**Theil–Sen slope**: {slope_py:.4f} per year (95% CI [{lo_py:.4f}, {hi_py:.4f}])")
                        except Exception:
                            st.info("Trend CI unavailable.")
                else:
                    st.info("Not enough data to run ensemble.")

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
            dem_view = st.selectbox(
                "Base background",
                [
                    "Hillshade",
                    "Grayscale (low-dark)",
                    "Grayscale (low-bright)",
                    "Terrain colors",
                    "Bathymetric",
                    "Flat",
                    "None",
                ],
                index=0,
                help="Выберите стиль подложки: Terrain/Bathymetric дают цветовую шкалу; None — прозрачная (условно) подложка без артефактов облачности.",
            )
            mask_mode = st.selectbox(
                "Water mask mode",
                ["Simulated level", "Depth & NDWI"],  # make simulated level the primary/default
                index=0,
                help=(
                    "Simulated level (default): динамическая площадь по объёму. "
                    "Depth & NDWI: статичное пересечение глубины (dem<0) и водного NDWI маска."
                ),
            )
            if mask_mode == "Simulated level":
                st.caption("По умолчанию используется динамический режим 'Simulated level'.")
        # Build composite image
        dem_disp = dem.copy()
        if nodata is not None:
            dem_disp = np.where(dem_disp == nodata, np.nan, dem_disp)
        # Normalize DEM for display or compute hillshade
        vmin = np.nanpercentile(dem_disp, 2)
        vmax = np.nanpercentile(dem_disp, 98)
        norm = (dem_disp - vmin) / max(1e-6, (vmax - vmin))
        norm = np.clip(norm, 0.0, 1.0)

        if dem_view == "Hillshade":
            z = np.nan_to_num(dem_disp, nan=float(np.nanmedian(dem_disp)))
            gx, gy = np.gradient(z)
            slope = np.pi/2 - np.arctan(np.hypot(gx, gy))
            aspect = np.arctan2(-gx, gy)
            az = np.radians(315.0)
            alt = np.radians(45.0)
            hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
            hs = (hs - hs.min()) / (hs.max() - hs.min() + 1e-6)
            vis = hs
        elif dem_view == "Grayscale (low-bright)":
            vis = 1.0 - norm  # invert so depressions appear bright
        elif dem_view == "Grayscale (low-dark)":
            vis = norm
        elif dem_view == "Terrain colors":
            # Simple terrain palette: dark green -> green -> brown -> light brown -> near white
            # Build a LUT with 256 entries
            g = norm
            # Define control points (value, RGB)
            ctrl = [
                (0.0, (20, 70, 20)),
                (0.25, (50, 120, 50)),
                (0.5, (150, 110, 60)),
                (0.75, (200, 170, 120)),
                (1.0, (245, 245, 240)),
            ]
            cp_vals = np.array([c[0] for c in ctrl])
            cp_cols = np.array([c[1] for c in ctrl], dtype=float)
            def _interp_cols(x):
                x = np.clip(x, 0, 1)
                # find indices
                idx = np.searchsorted(cp_vals, x, side='right') - 1
                idx = np.clip(idx, 0, len(cp_vals) - 2)
                left_v = cp_vals[idx]
                right_v = cp_vals[idx + 1]
                w = np.where((right_v - left_v) > 0, (x - left_v) / (right_v - left_v), 0)
                left_c = cp_cols[idx]
                right_c = cp_cols[idx + 1]
                return (left_c * (1 - w)[..., None] + right_c * w[..., None]) / 255.0
            color_vis = _interp_cols(g)
            vis = None  # flag to skip grayscale replication
        elif dem_view == "Bathymetric":
            # Separate negative (depth) and positive (land) ranges if present
            z = dem_disp.copy()
            # Normalize land part
            land = z.copy()
            # Depth palette: deep = dark blue, shallow = light cyan
            # Land palette: reuse terrain gradient simplified
            land_min = np.nanmin(land)
            land_max = np.nanmax(land)
            if land_max - land_min < 1e-6:
                land_norm = np.zeros_like(land)
            else:
                land_norm = (land - land_min) / (land_max - land_min)
            depth_mask = (z < 0) & ~np.isnan(z)
            land_mask = (z >= 0) & ~np.isnan(z)
            color_vis = np.zeros(z.shape + (3,), dtype=float)
            if depth_mask.any():
                depths = z[depth_mask]
                # more negative -> deeper
                dmin = depths.min()
                dmax = depths.max()  # close to 0
                span = max(1e-6, dmax - dmin)
                dnorm = (depths - dmin) / span
                # Map to blue gradient
                # deep: (0, 25, 90), mid: (0,90,160), shallow: (120,200,255)
                # Use quadratic easing for brightness
                def blend(c1, c2, w):
                    return c1 * (1 - w) + c2 * w
                mid_color = np.array([0, 90, 160], dtype=float)
                deep_color = np.array([0, 25, 90], dtype=float)
                shallow_color = np.array([120, 200, 255], dtype=float)
                # Two segment blend
                w_mid = np.clip(dnorm * 1.4, 0, 1)
                col_mid = blend(deep_color, mid_color, w_mid[:, None])
                w_shal = np.clip((dnorm - 0.5) * 2, 0, 1)
                col_depth = blend(col_mid, shallow_color, w_shal[:, None]) / 255.0
                color_vis[depth_mask] = col_depth
            if land_mask.any():
                ln = land_norm[land_mask]
                # Terrain simplified: green -> brown -> light
                c1 = np.array([40, 110, 40], dtype=float)
                c2 = np.array([160, 120, 60], dtype=float)
                c3 = np.array([240, 235, 225], dtype=float)
                w = ln
                mid = np.clip(w * 1.6, 0, 1)
                col_land = c1 * (1 - mid)[:, None] + c2 * mid[:, None]
                w2 = np.clip((w - 0.5) * 2, 0, 1)
                col_land = col_land * (1 - w2)[:, None] + c3 * w2[:, None]
                color_vis[land_mask] = col_land / 255.0
            vis = None
        elif dem_view == "Flat":
            color_vis = np.ones(dem_disp.shape + (3,), dtype=float) * 0.92
            vis = None
        elif dem_view == "None":
            color_vis = np.zeros(dem_disp.shape + (3,), dtype=float)
            vis = None
        else:
            vis = norm  # fallback grayscale

        if vis is not None:
            base_rgb = (vis[..., None] * 255).astype(np.uint8)
            base_rgb = np.repeat(base_rgb, 3, axis=2)
        else:
            # color_vis already RGB in 0..1
            base_rgb = (np.clip(color_vis, 0, 1) * 255).astype(np.uint8)

        # Selected day volume & corresponding level (restore after earlier refactor)
        try:
            v_sel = float(scenario_df["volume_mcm"].iloc[idx])
        except Exception:
            v_sel = float('nan')
        try:
            z_level = float(vol_to_elev(v_sel)) if vol_to_elev is not None else None
        except Exception:
            z_level = None

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
            # Simulated level mode. Support two cases:
            # 1) Elevation DEM (positive elevations) -> threshold by elevation (existing behaviour)
            # 2) Depth DEM (negative values for submerged cells, 0≈shore) -> derive dynamic extent
            dem_min = float(np.nanmin(dem_disp))
            dem_max = float(np.nanmax(dem_disp))
            is_elevation = (dem_min >= 0) or (dem_max > 5)  # heuristic

            if z_level is not None and is_elevation:
                # Elevation-based: cells with elevation <= water surface
                water_mask = (dem_disp <= z_level)
                if nodata is not None:
                    water_mask = water_mask & ~np.isnan(dem_disp)
            else:
                # Depth-based dynamic area shrink/expand.
                # We approximate target area fraction using simulated area_km2 vs max curve area.
                try:
                    target_area_km2 = float(scenario_df["area_km2"].iloc[idx])
                    max_area_curve = float(areas.max()) if 'areas' in globals() else target_area_km2
                    frac = 0.0 if max_area_curve <= 0 else float(np.clip(target_area_km2 / max_area_curve, 0.0, 1.0))
                except Exception:
                    frac = 1.0

                # Cache depth values only once (cells that can ever be wet: dem<0)
                if '_depth_values' not in st.session_state or '_depth_mask_template' not in st.session_state or st.session_state.get('_depth_shape') != dem_disp.shape:
                    depth_mask_template = (dem_disp < 0) & ~np.isnan(dem_disp)
                    depth_values = dem_disp[depth_mask_template]
                    st.session_state._depth_values = depth_values
                    st.session_state._depth_mask_template = depth_mask_template
                    st.session_state._depth_shape = dem_disp.shape
                depth_values = st.session_state._depth_values
                depth_mask_template = st.session_state._depth_mask_template

                if depth_values.size == 0:
                    # Fallback: static (no negative depths found)
                    water_mask = (dem_disp < 0)
                    if nodata is not None:
                        water_mask = water_mask & ~np.isnan(dem_disp)
                else:
                    # Depths are negative; to shrink area with lower volume, we keep only deeper cells.
                    # For fraction f, choose threshold at quantile f of sorted depths (ascending: more negative first).
                    # Ensure numerical stability for extreme fracs.
                    f = float(np.clip(frac, 0.0, 1.0))
                    if f <= 0.0:
                        water_mask = np.zeros_like(dem_disp, dtype=bool)
                    elif f >= 0.9999:
                        water_mask = depth_mask_template
                    else:
                        try:
                            thresh = float(np.quantile(depth_values, f))  # quantile f ⇒ include deepest up to that depth
                        except Exception:
                            thresh = float(depth_values.max())  # should be near 0
                        water_mask = (dem_disp <= thresh) & depth_mask_template
                    if nodata is not None:
                        water_mask = water_mask & ~np.isnan(dem_disp)

                # Optional: show a subtle caption with mode info (avoid spam each rerun)
                st.caption(f"Dynamic depth mode: area fraction={frac:.3f}")

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
            # use_column_width & use_container_width deprecated; now using width="stretch"
            st.image(over_img, caption=caption, width="stretch")
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
		# End vis_col1	
        