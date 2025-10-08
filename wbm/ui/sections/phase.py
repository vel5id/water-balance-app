from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Callable, Optional
try:
    from wbm.i18n import Translator, DEFAULT_LANG
except Exception:
    class Translator:  # type: ignore
        def __init__(self, lang: str='ru'): self.lang=lang
        def __call__(self, key: str, **fmt): return key if not fmt else key
    DEFAULT_LANG='ru'

__all__ = ["render_phase_plots"]

def render_phase_plots(era5_df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None):
    if tr is None:
        lang = getattr(st.session_state, 'lang', DEFAULT_LANG)
        try:
            tr = Translator(lang)
        except Exception:
            tr = lambda k, **_: k  # type: ignore
    with st.expander(tr("phase_plots"), expanded=False):
        if era5_df is None or era5_df.empty or "date" not in era5_df.columns:
            st.info("ERA5 dataframe is empty or lacks date column.")
            return
        phase_df = era5_df.copy()
        phase_df["date"] = pd.to_datetime(phase_df["date"]).dt.normalize()
        numeric_cols = [c for c in ["precip_mm","evap_mm","runoff_mm","t2m_c","snow_depth_m"] if c in phase_df.columns]
        if len(numeric_cols) < 2:
            st.info("Not enough numeric ERA5 driver columns for phase plots.")
            return
        col_ph1, col_ph2, col_ph3 = st.columns(3)
        with col_ph1:
            x_var = st.selectbox(tr("x_variable"), numeric_cols, index=0, key="phase_x")
        with col_ph2:
            y_var = st.selectbox(tr("y_variable"), numeric_cols, index=min(1,len(numeric_cols)-1), key="phase_y")
        with col_ph3:
            color_mode = st.selectbox(tr("color_by"), ["month","doy"], index=0, key="phase_color")
        years_available = sorted(phase_df["date"].dt.year.unique())
        sel_years = st.multiselect(tr("years"), years_available, default=years_available[-min(3,len(years_available)):], key="phase_years")
        smooth_days = st.slider(tr("smoothing_days"), 1, 30, 5, 1, key="phase_smooth")
        show_markers = st.checkbox(tr("show_markers"), value=False, key="phase_markers")
        if x_var == y_var:
            st.warning(tr("select_diff_vars"))
            return
        fig_phase = go.Figure()
        for yr in sel_years:
            sub = phase_df[phase_df["date"].dt.year == yr].set_index("date").sort_index()
            if sub.empty:
                continue
            x_raw = sub[x_var].astype(float)
            y_raw = sub[y_var].astype(float)
            if smooth_days > 1:
                x_series = x_raw.rolling(smooth_days, min_periods=max(2,int(smooth_days*0.5))).mean()
                y_series = y_raw.rolling(smooth_days, min_periods=max(2,int(smooth_days*0.5))).mean()
            else:
                x_series = x_raw
                y_series = y_raw
            if color_mode == "month":
                color_vals = x_series.index.month
                cmin, cmax = 1, 12
                colorscale = "Turbo"
            else:
                doy = x_series.index.dayofyear
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
        fig_phase.update_layout(template="plotly_white", title=f"{x_var} vs {y_var}", xaxis_title=x_var, yaxis_title=y_var, legend_title="Year")
        st.plotly_chart(fig_phase, use_container_width=True, config={"displaylogo": False})
