"""Streamlit orchestrator app (modular version).

This replaces the legacy monolithic implementation. Responsibilities are
delegated to dedicated modules under `wbm.ui`:

  wbm.ui.data.load_all           -> loading ERA5, baseline, curve & climatologies
  wbm.ui.controls.build_controls -> sidebar inputs & scenario configuration
  wbm.ui.simulation.prepare_drivers / run_scenario -> build P/ET drivers & run deterministic scenario
  wbm.ui.sections.*              -> individual analytical / visualization sections

The goal is to keep this file thin: sequencing, light session state, and high
level layout only. Heavy logic lives in packages for testability.

Run: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from wbm.ui.data import load_all, ensure_datetime
from wbm.ui.controls import build_controls
from wbm.ui.simulation import prepare_drivers, run_scenario, select_initial_volume
from wbm.ui.state import LoadedData, Controls
from wbm.plots import timeseries_figure, stacked_fluxes_figure
from wbm import seasonal as _seasonal_doc
from wbm.ui.sections.trends import render_trends_and_correlations, render_long_term_trends
from wbm.analysis import rolling_trend, lagged_correlation  # needed by some sections still inline
import plotly.express as px  # sections may rely on px
import plotly.graph_objects as go  # for fallback / map sections
from wbm.i18n import Translator, TRANSLATIONS, DEFAULT_LANG

# Unified Plotly display helper (avoids deprecated width kwarg usage in st.plotly_chart)
def show_plot(fig):
    if fig is None:
        return
    st.plotly_chart(fig, config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d","lasso2d"]}, use_container_width=True)

# Optional: statsmodels detection for regression trendlines in scatter plots
try:  # pragma: no cover - optional dependency
    import statsmodels.api  # noqa: F401
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAS_STATSMODELS = False
# (Additional sections can be imported similarly when their modular files exist)
try:
    # Optional sections (guarded imports to avoid runtime failure if not yet present)
    from wbm.ui.sections.ensemble import render_ensemble
except Exception:  # pragma: no cover - optional
    render_ensemble = None
try:
    from wbm.ui.sections.snow_temp import render_snow_temp  # type: ignore
except Exception:  # pragma: no cover
    render_snow_temp = None
try:
    from wbm.ui.sections.runoff_temp import render_runoff_temp  # type: ignore
except Exception:  # pragma: no cover
    render_runoff_temp = None
try:
    from wbm.ui.sections.p_et_diag import render_p_et_diag  # type: ignore
except Exception:  # pragma: no cover
    render_p_et_diag = None
try:
    from wbm.ui.sections.phase import render_phase_plots  # type: ignore
except Exception:  # pragma: no cover
    render_phase_plots = None
try:
    from wbm.ui.sections.map_view import render_map  # type: ignore
except Exception:  # pragma: no cover
    render_map = None


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


DATA_ROOT = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parent))
# New structured layout (2025-10): artifacts moved under processed_data/
PROC_ROOT = os.path.join(DATA_ROOT, "processed_data", "processing_output")
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed_data", "water_balance_output")
ERA5_DAILY_DB_PATH = os.path.join(PROC_ROOT, "era5_daily.sqlite")
ERA5_DAILY_CSV_PATH = os.path.join(PROC_ROOT, "era5_daily_summary.csv")
AREA_VOLUME_CURVE_PATH = os.path.join(PROC_ROOT, "area_volume_curve.csv")
DEM_INTEGRATED_PATH = os.path.join(PROC_ROOT, "integrated_bathymetry_copernicus.tif")
DEM_FALLBACK_PATH = os.path.join(PROC_ROOT, "bathymetry_reprojected_epsg4326.tif")
DEM_PATH = DEM_INTEGRATED_PATH if os.path.exists(DEM_INTEGRATED_PATH) else DEM_FALLBACK_PATH
NDWI_MASK_PATH = os.path.join(PROC_ROOT, "ndwi_mask_0275.tif")


st.set_page_config(page_title="Water Balance Interactive", layout="wide")

# --- Language handling with modern st.query_params (?lang=ru|en|kk) ---
qp = st.query_params  # Streamlit >= 1.32 style
initial_lang = qp.get("lang", DEFAULT_LANG)
if initial_lang not in TRANSLATIONS:
    initial_lang = DEFAULT_LANG

lang_codes = ["ru", "en", "kk"]
lang_display = [TRANSLATIONS[code][f"lang_{code}"] for code in lang_codes]
display_to_code = dict(zip(lang_display, lang_codes))
current_index = lang_codes.index(initial_lang)
selected_display = st.sidebar.selectbox(TRANSLATIONS[initial_lang]["language_label"], lang_display, index=current_index)
lang = display_to_code[selected_display]
if qp.get("lang", initial_lang) != lang:
    qp["lang"] = lang  # updates URL without reload
tr = Translator(lang)
st.session_state['lang'] = lang  # allow sections to reconstruct translator if not passed

st.title(tr("app_title"))
st.caption(tr("tagline"))

with st.spinner(tr("loading_data")):
    ld: LoadedData = load_all(
        DATA_ROOT,
        AREA_VOLUME_CURVE_PATH,
        ERA5_DAILY_DB_PATH,
        ERA5_DAILY_CSV_PATH,
    )
def _sanitize_dates_inplace(df: pd.DataFrame, col: str = "date"):
    if df is None or df.empty or col not in df.columns:
        return df
    ser = pd.to_datetime(df[col], errors="coerce")
    # Normalize to day precision to drop exotic ns offsets
    ser = ser.dt.floor("D")
    # Force pure datetime64[ns] (avoid object mix of Timestamp)
    try:
        df[col] = ser.astype("datetime64[ns]")
    except Exception:
        # Fallback: convert via numpy int64 days then back
        vals = ser.view("int64") // (24*3600*1_000_000_000)
        df[col] = pd.to_datetime(vals, unit="D")
    return df

def _sanitize_all_datetime_like(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in list(out.columns):
        if c.lower() == 'date' or 'date' in c.lower() or isinstance(out[c].dtype, pd.api.extensions.ExtensionDtype):
            # Broad heuristic: try datetime conversion for date-like columns
            try:
                conv = pd.to_datetime(out[c], errors='coerce')
                if conv.notna().any():
                    conv = conv.dt.floor('D')
                    out[c] = conv.astype('datetime64[ns]')
            except Exception:
                pass
        elif out[c].dtype == object:
            sample = out[c].dropna().head(5)
            if not sample.empty and any(hasattr(v,'isoformat') for v in sample):
                try:
                    conv = pd.to_datetime(out[c], errors='coerce').dt.floor('D')
                    out[c] = conv.astype('datetime64[ns]')
                except Exception:
                    pass
    return out

# Apply sanitation to loaded dataframes
for _df_name in ["curve_df","balance_df","era5_df"]:
    try:
        _df = getattr(ld, _df_name, None)
        if _df is not None and not _df.empty:
            _san = _sanitize_dates_inplace(_df.copy(), "date")
            _san = _sanitize_all_datetime_like(_san)
            setattr(ld, _df_name, _san)
    except Exception:
        pass
if ld.curve_df.empty:
    st.error(tr('curve_missing'))
    st.stop()
area_to_vol = ld.area_to_vol
vol_to_area = ld.vol_to_area
vol_to_elev = ld.vol_to_elev
curve_df = ld.curve_df
balance_df = ld.balance_df.copy()
era5_df = ld.era5_df.copy()
p_clim = ld.p_clim
et_clim = ld.et_clim
areas = curve_df.get("area_km2", pd.Series(dtype=float)).to_numpy() if not curve_df.empty else np.array([])
vols = curve_df.get("volume_mcm", pd.Series(dtype=float)).to_numpy() if not curve_df.empty else np.array([])

"""Removed legacy helper implementations in favor of modular versions (select_initial_volume, driver builders, etc.)."""

with st.expander(tr("baseline_info"), expanded=False):
    if not balance_df.empty:
        st.write(balance_df.describe(include="all"))
    else:
        st.info(tr("no_baseline"))

controls: Controls = build_controls(pd.Timestamp.today().normalize(), vols, balance_df, area_to_vol, lang=lang)

# Filter baseline if requested
removed_n = 0
if controls.filter_baseline and not balance_df.empty and "area_km2" in balance_df.columns:
    before_n = len(balance_df)
    balance_df = balance_df[balance_df["area_km2"] >= controls.min_area_km2].reset_index(drop=True)
    removed_n = before_n - len(balance_df)
if removed_n > 0:
    st.sidebar.info(tr('removed_rows_area', n=removed_n, min_area=controls.min_area_km2))

if not balance_df.empty:
    init_volume, init_note = select_initial_volume(balance_df, controls.start_date, vols)
else:
    init_volume, init_note = (float(vols[len(vols)//2]) if len(vols) else 0.0, "fallback curve midpoint")
st.sidebar.markdown(f"{tr('initial_volume')}: **{init_volume:.1f} mcm**")
st.sidebar.caption(f"{tr('source_label')}: {init_note}")
try:
    if area_to_vol is not None:
        min_vol_threshold = float(area_to_vol(float(controls.min_area_km2)))
    st.sidebar.caption(tr('min_area_to_volume', vol=min_vol_threshold))
except Exception:
    pass

with st.expander(tr("season_trend_docs"), expanded=False):
    st.markdown(_seasonal_doc.markdown_doc(lang))

# --- Run simulation ---
if p_clim.empty or et_clim.empty:
    st.warning(tr('clim_missing'))
else:
    p_daily, et_daily = prepare_drivers(ld, controls)
    scenario_ctx = run_scenario(ld, controls, p_daily, et_daily, init_volume)
    scenario_df = scenario_ctx.scenario_df
    # Sanitize scenario dates early
    if scenario_df is not None and not scenario_df.empty and "date" in scenario_df.columns:
        scenario_df = _sanitize_all_datetime_like(_sanitize_dates_inplace(scenario_df.copy(), "date"))

    def _filter_year(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if controls.view_mode == "Single year":
            years = sorted(df["date"].dt.year.unique())
            year = st.sidebar.selectbox(tr("year"), years, index=len(years)-1)
            return df[df["date"].dt.year == year].reset_index(drop=True)
        return df

    def _smooth(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if df is None or df.empty or controls.smooth_win <= 1:
            return df
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].rolling(controls.smooth_win, min_periods=1, center=True).mean()
        return out

    plot_base = _filter_year(balance_df.copy()) if not balance_df.empty else balance_df
    plot_scena = _filter_year(scenario_df.copy())
    if controls.hide_scenario_below_min and plot_scena is not None and not plot_scena.empty and "area_km2" in plot_scena.columns:
        plot_scena.loc[plot_scena["area_km2"] < controls.min_area_km2, ["volume_mcm", "area_km2"]] = np.nan
    plot_base_s = _smooth(plot_base, ["volume_mcm"]) if plot_base is not None else plot_base
    plot_scena_s = _smooth(plot_scena, ["volume_mcm"]) if plot_scena is not None else plot_scena

    col1, col2 = st.columns([2, 1])
    with col1:
        show_plot(timeseries_figure(plot_base_s, plot_scena_s, tr))
    with col2:
        if plot_scena is not None and not plot_scena.empty:
            st.metric(tr("end_volume"), f"{plot_scena['volume_mcm'].iloc[-1]:.1f}")
            st.metric(tr("min_volume"), f"{plot_scena['volume_mcm'].min():.1f}")
            st.metric(tr("max_volume"), f"{plot_scena['volume_mcm'].max():.1f}")

    st.subheader(tr("daily_pet"))
    show_plot(stacked_fluxes_figure(plot_scena, tr))

    scen_safe = ensure_datetime(scenario_df.copy(), "date")
    csv_bytes = scen_safe.to_csv(index=False).encode("utf-8")
    st.download_button(tr("download_csv"), csv_bytes, file_name="scenario_water_balance.csv", mime="text/csv")

    # Analytical sections
    render_trends_and_correlations(plot_base, plot_scena, tr)
    render_long_term_trends(era5_df, tr)
    if render_snow_temp: render_snow_temp(era5_df, tr=tr)
    if render_runoff_temp: render_runoff_temp(era5_df, tr=tr)
    if render_p_et_diag: render_p_et_diag(era5_df, tr=tr)
    if render_phase_plots: render_phase_plots(era5_df, tr=tr)
    if render_ensemble: render_ensemble(era5_df, vol_to_area, p_clim, et_clim, init_volume, controls.p_scale, controls.et_scale, controls.start_date, controls.end_date, plot_scena_s, tr=tr)
    if render_map: render_map(scenario_df, vol_to_elev, DEM_PATH, NDWI_MASK_PATH, areas, tr=tr)

st.caption(tr("footer_caption"))
