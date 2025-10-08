from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from typing import Callable, Optional
try:
    from wbm.i18n import Translator, DEFAULT_LANG
except Exception:
    class Translator:  # type: ignore
        def __init__(self, lang: str='ru'): self.lang=lang
        def __call__(self, key: str, **fmt): return key if not fmt else key
    DEFAULT_LANG='ru'

__all__ = ["render_p_et_diag"]

def render_p_et_diag(era5_df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None):
    if tr is None:
        lang = getattr(st.session_state, 'lang', DEFAULT_LANG)
        try:
            tr = Translator(lang)
        except Exception:
            tr = lambda k, **_: k  # type: ignore
    with st.expander(tr("p_et_diag"), expanded=False):
        if era5_df is None or era5_df.empty or not {"date","precip_mm","evap_mm"}.issubset(era5_df.columns):
            st.info("ERA5 dataframe lacks precip_mm / evap_mm columns.")
            return
        pcols = era5_df[["date","precip_mm","evap_mm"]].dropna().copy()
        if pcols.empty:
            st.info("No P/ET data available.")
            return
        pcols["date"] = pd.to_datetime(pcols["date"]).dt.normalize()
        pcols = pcols.drop_duplicates("date").set_index("date").sort_index()
        col_pe1, col_pe2, col_pe3 = st.columns(3)
        with col_pe1:
            win_pe = st.slider(tr("rolling_window_days"), 7, 180, 60, 1, key="p_et_win")
        with col_pe2:
            agg_pe = st.selectbox(tr("aggregation"), [tr("daily_pet"), tr("monthly")], index=1, key="p_et_agg")
        with col_pe3:
            clip_ratio_pe = st.number_input(tr("p_et_ratio_clip"), 0.0, 1000.0, 100.0, 1.0, key="p_et_clip")
        p_roll = pcols["precip_mm"].rolling(win_pe, min_periods=max(3,int(win_pe*0.3))).mean()
        et_roll = pcols["evap_mm"].rolling(win_pe, min_periods=max(3,int(win_pe*0.3))).mean()
        eps = 1e-6
        pet_ratio = (p_roll / et_roll.where(et_roll.abs() > eps)).replace([np.inf,-np.inf], np.nan)
        if clip_ratio_pe > 0:
            pet_ratio = pet_ratio.clip(-clip_ratio_pe, clip_ratio_pe)
        water_balance_index = p_roll - et_roll
        diag_df = pd.DataFrame({
            "P_roll": p_roll,
            "ET_roll": et_roll,
            "P_over_ET": pet_ratio,
            "P_minus_ET": water_balance_index,
        })
        diag_plot_df = diag_df.resample("ME").mean() if agg_pe == tr("monthly") else diag_df
        st.line_chart(diag_plot_df)
        anom = water_balance_index.resample("ME").mean()
        anom_mean = anom.groupby(anom.index.month).transform(lambda x: x.mean())
        anom_std = anom.groupby(anom.index.month).transform(lambda x: x.std(ddof=0))
        standardized = (anom - anom_mean) / anom_std.replace(0, np.nan)
        df_heat = standardized.to_frame("z").reset_index()
        df_heat["year"] = df_heat["date"].dt.year
        df_heat["month"] = df_heat["date"].dt.month
        pivot = df_heat.pivot(index="year", columns="month", values="z")
        st.markdown("**Standardized anomaly (z-score) (P-ET)**")
        st.dataframe(
            pivot.style.background_gradient(cmap="coolwarm", axis=None),
            use_container_width=True,
        )
        quant = pet_ratio.dropna().quantile([0.05,0.25,0.5,0.75,0.95]).to_dict()
        st.caption("P/ET ratio: " + ", ".join([f"{int(k*100)}%={v:.2f}" for k,v in quant.items()]))
        quant_wb = water_balance_index.dropna().quantile([0.05,0.5,0.95]).to_dict()
        st.caption(f"(P-ET) index: 5%={quant_wb.get(0.05):.2f}, 50%={quant_wb.get(0.5):.2f}, 95%={quant_wb.get(0.95):.2f}")
