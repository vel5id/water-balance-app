from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Callable, Optional
try:
    from wbm.i18n import Translator, DEFAULT_LANG
except Exception:  # fallback minimal
    class Translator:  # type: ignore
        def __init__(self, lang: str='ru'): self.lang=lang
        def __call__(self, key: str, **fmt): return key if not fmt else key
    DEFAULT_LANG='ru'
from wbm.analysis import rolling_trend

__all__ = ["render_snow_temp"]

def render_snow_temp(era5_df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None):
    if tr is None:
        lang = getattr(st.session_state, 'lang', DEFAULT_LANG)
        try:
            tr = Translator(lang)
        except Exception:
            tr = lambda k, **_: k  # type: ignore
    with st.expander(tr("snow_temp_trends"), expanded=False):
        required = {"date","t2m_c","snow_depth_m"}
        if era5_df is None or era5_df.empty or not required.issubset(era5_df.columns):
            st.info("ERA5 dataframe lacks required columns (t2m_c, snow_depth_m)")
            return
        base = era5_df[list(required)].dropna().copy()
        if base.empty:
            st.info("No overlapping non-NaN snow/temperature data available.")
            return
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        base = base.drop_duplicates("date").set_index("date").sort_index().asfreq("D")
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            win = st.slider(tr("rolling_window_days"), 7, 120, 30, 1, key="snow_temp_win")
        with col_cfg2:
            agg_mode = st.selectbox(tr("display_agg"), [tr("daily_pet"), tr("monthly")], index=0, key="snow_temp_agg")
        with col_cfg3:
            ratio_clip = st.number_input(tr("clip_ratio"), 0.0, 1000.0, 50.0, 1.0, key="snow_temp_clip")
        snow_tr = rolling_trend(base["snow_depth_m"], window=win)
        temp_tr = rolling_trend(base["t2m_c"], window=win)
        eps = 1e-6
        denom = temp_tr.where(temp_tr.abs() > eps)
        ratio = (snow_tr / denom).replace([np.inf,-np.inf], np.nan)
        if ratio_clip > 0:
            ratio = ratio.clip(-ratio_clip, ratio_clip)
        trend_df = pd.DataFrame({
            "snow_trend": snow_tr,
            "temp_trend": temp_tr,
            "ratio_snow_to_temp": ratio,
        })
        trend_plot_df = trend_df.resample("ME").mean() if agg_mode == tr("monthly") else trend_df
        st.markdown(tr("rolling_trends_ratio"))
        st.line_chart(trend_plot_df)
        scatter_df = (base.resample("ME").mean() if agg_mode == tr("monthly") else base).reset_index()
        scatter_df["year"] = scatter_df["date"].dt.year
        scatter_df["month"] = scatter_df["date"].dt.month
        fig_scatter = px.scatter(
            scatter_df,
            x="t2m_c", y="snow_depth_m", color="year",
            hover_data={"date": True, "month": True},
            trendline=None,
            labels={"t2m_c": "T (°C)", "snow_depth_m": "Snow depth (m)"},
            title=tr("snow_temp_trends")
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displaylogo": False})
        ratio_clean = ratio.dropna()
        if not ratio_clean.empty:
            q = ratio_clean.quantile([0.05,0.5,0.95]).to_dict()
            st.caption(f"Ratio 5%={q.get(0.05):.3f}, 50%={q.get(0.5):.3f}, 95%={q.get(0.95):.3f}")
