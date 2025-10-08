from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Callable
from wbm.analysis import rolling_trend, lagged_correlation

__all__ = ["render_trends_and_correlations", "render_long_term_trends"]


def render_trends_and_correlations(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None):
    tr = tr or (lambda k, **_: k)
    with st.expander(tr("trends_corr"), expanded=False):
        if scenario_df is None or scenario_df.empty:
            st.info(tr("no_scenario_data"))
            return
        cols_needed = ["date", "volume_mcm", "area_km2"]
        miss = [c for c in cols_needed if c not in scenario_df.columns]
        if miss:
            st.warning(tr("missing_columns", cols=miss))
            return
        rolling_window = st.slider(tr("rolling_window_days"), 7, 180, 30, 1)
        max_lag = st.slider(tr("max_lag_corr"), 1, 120, 30, 1)
        df = scenario_df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        if 'volume_mcm' in df.columns:
            vol_trend_series = rolling_trend(df.set_index('date')['volume_mcm'], window=rolling_window)
            st.line_chart(vol_trend_series)
        if 'area_km2' in scenario_df.columns:
            area_series = df.set_index('date')['area_km2'].astype(float)
            vol_series = df.set_index('date')['volume_mcm'].astype(float)
            lag_corr = lagged_correlation(area_series, vol_series, max_lag=max_lag)
            if not lag_corr.empty:
                st.bar_chart(lag_corr.set_index('lag')['corr'])


def render_long_term_trends(era5_df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None):
    tr = tr or (lambda k, **_: k)
    with st.expander(tr("long_term_trends"), expanded=False):
        if era5_df is None or era5_df.empty:
            st.info(tr("era5_insufficient"))
            return
        years_back = st.slider(tr("years_back"), 1, 20, 10, 1)
        agg_choice = st.selectbox(tr("aggregation"), [tr("monthly"), tr("annual")], index=0)
        era5_df = era5_df.copy()
        era5_df['date'] = pd.to_datetime(era5_df['date'])
        cutoff = era5_df['date'].max() - pd.DateOffset(years=years_back)
        era5_df = era5_df[era5_df['date'] >= cutoff]
        if agg_choice == tr("monthly"):
            # Use month-end alias 'ME' (instead of deprecated 'M')
            grp = era5_df.set_index('date').resample('ME').mean()
        else:
            # Use year-end alias 'YE' (instead of deprecated 'A')
            grp = era5_df.set_index('date').resample('YE').mean()
        if grp.empty:
            st.warning(tr("not_enough_agg"))
            return
        if 'p_mm' in grp.columns:
            st.line_chart(grp['p_mm'])
        if 'et_mm' in grp.columns:
            st.line_chart(grp['et_mm'])
        if 'p_mm' in grp.columns:
            st.metric(tr("p_slope"), f"{np.nan}")
        if 'et_mm' in grp.columns:
            st.metric(tr("et_slope"), f"{np.nan}")
