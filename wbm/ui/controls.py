from __future__ import annotations
import pandas as pd
import streamlit as st
from .state import Controls

__all__ = ["build_controls"]

def build_controls(default_start: pd.Timestamp, vols, balance_df, area_to_vol) -> Controls:
    st.sidebar.header("Scenario Controls")
    p_scale = st.sidebar.slider("Precipitation scaling", 0.0, 2.0, 1.0, 0.05)
    et_scale = st.sidebar.slider("Evaporation scaling", 0.0, 2.0, 1.0, 0.05)
    q_in = st.sidebar.number_input("Inflow (mcm/day)", value=0.0, step=0.1)
    q_out = st.sidebar.number_input("Outflow (mcm/day)", value=0.0, step=0.1)

    st.sidebar.header("Data filtering")
    min_area_km2 = st.sidebar.number_input(
        "Min area to include (km²)", value=60.0, step=1.0,
        help="Записи с площадью ниже порога исключаются из baseline"
    )
    filter_baseline = st.sidebar.checkbox("Filter baseline by min area", value=False)
    hide_scenario_below_min = st.sidebar.checkbox(
        "Hide scenario below min area (plots only)", value=False,
        help="Скрытие влияет только на визуализацию"
    )

    today = pd.Timestamp.today().normalize()
    start_date = st.sidebar.date_input("Start date", value=default_start.date())
    horizon_days = st.sidebar.slider("Horizon (days)", 30, 730, 365, 30)
    end_date = pd.Timestamp(start_date) + pd.Timedelta(days=horizon_days)

    view_mode = st.sidebar.radio("Display range", ["All period", "Single year"], index=0)
    smooth_win = st.sidebar.slider("Smoothing window (days)", 1, 90, 14, 1)

    st.sidebar.header("Forecast mode")
    forecast_mode = st.sidebar.radio(
        "P/ET driver",
        ["Monthly mean (all years)", "Seasonal climatology", "Seasonal + trend"],
        index=1,
    )

    with st.sidebar.expander("Season+Trend options"):
        hist_window_days = st.number_input(
            "History window (days)", 0, 3650, 730, 30,
            help="0 = full history"
        )
        seas_basis = st.selectbox("Season basis", ["DOY", "MONTH"], index=0)

    return Controls(
        p_scale=float(p_scale),
        et_scale=float(et_scale),
        q_in=float(q_in),
        q_out=float(q_out),
        min_area_km2=float(min_area_km2),
        filter_baseline=bool(filter_baseline),
        hide_scenario_below_min=bool(hide_scenario_below_min),
        view_mode=view_mode,
        smooth_win=int(smooth_win),
        forecast_mode=forecast_mode,
        hist_window_days=int(hist_window_days),
        seas_basis=seas_basis,
        start_date=pd.Timestamp(start_date),
        end_date=end_date,
    )
