from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Callable, Optional
try:
    from wbm.i18n import Translator, DEFAULT_LANG
except Exception:
    class Translator:  # type: ignore
        def __init__(self, lang: str='ru'): self.lang=lang
        def __call__(self, key: str, **fmt): return key if not fmt else key
    DEFAULT_LANG='ru'
from wbm.analysis import rolling_trend, lagged_correlation

__all__ = ["render_runoff_temp"]

def render_runoff_temp(era5_df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None):
    if tr is None:
        lang = getattr(st.session_state, 'lang', DEFAULT_LANG)
        try:
            tr = Translator(lang)
        except Exception:
            tr = lambda k, **_: k  # type: ignore
    with st.expander(tr("runoff_temp_trends"), expanded=False):
        required = {"date","t2m_c","runoff_mm"}
        if era5_df is None or era5_df.empty or not required.issubset(era5_df.columns):
            st.info("ERA5 dataframe lacks required columns (t2m_c, runoff_mm).")
            return
        rbase = era5_df[list(required)].dropna().copy()
        if rbase.empty:
            st.info("No overlapping runoff/temperature data.")
            return
        rbase["date"] = pd.to_datetime(rbase["date"]).dt.normalize()
        rbase = rbase.drop_duplicates("date").set_index("date").sort_index().asfreq("D")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            win_r = st.slider(tr("rolling_window_days"), 7, 180, 60, 1, key="runoff_temp_win")
        with col_r2:
            agg_r = st.selectbox(tr("aggregation"), [tr("daily_pet"), tr("monthly")], index=1, key="runoff_temp_agg")
        with col_r3:
            clip_r = st.number_input(tr("clip_ratio"), 0.0, 1000.0, 50.0, 1.0, key="runoff_temp_clip")
        runoff_tr = rolling_trend(rbase["runoff_mm"], window=win_r)
        temp_tr_r = rolling_trend(rbase["t2m_c"], window=win_r)
        eps = 1e-6
        denom_r = temp_tr_r.where(temp_tr_r.abs() > eps)
        ratio_r = (runoff_tr / denom_r).replace([np.inf,-np.inf], np.nan)
        if clip_r > 0:
            ratio_r = ratio_r.clip(-clip_r, clip_r)
        run_tr_df = pd.DataFrame({
            "runoff_trend": runoff_tr,
            "temp_trend": temp_tr_r,
            "ratio_runoff_to_temp": ratio_r,
        })
        plot_run_df = run_tr_df.resample("ME").mean() if agg_r == tr("monthly") else run_tr_df
        st.line_chart(plot_run_df)
        max_lag_r = st.slider(tr("max_lag_temp_runoff"), 5, 120, 45, 5, key="runoff_temp_lag")
        lr_corr = lagged_correlation(rbase["runoff_mm"], rbase["t2m_c"], max_lag=max_lag_r)
        st.subheader(tr("lagged_corr_runoff_temp"))
        if not lr_corr.empty:
            st.line_chart(lr_corr.set_index("lag")["corr"], height=180)
        rb_scatter = (rbase.resample("ME").mean() if agg_r == tr("monthly") else rbase).reset_index()
        rb_scatter["year"] = rb_scatter["date"].dt.year
        fig_run_sc = px.scatter(rb_scatter, x="t2m_c", y="runoff_mm", color="year", trendline=None, title=tr("runoff_temp_trends"))
        st.plotly_chart(fig_run_sc, use_container_width=True, config={"displaylogo": False})
        rq = ratio_r.dropna().quantile([0.05,0.5,0.95]).to_dict()
        st.caption(f"Runoff/Temp ratio 5%={rq.get(0.05):.3f}, 50%={rq.get(0.5):.3f}, 95%={rq.get(0.95):.3f}")
