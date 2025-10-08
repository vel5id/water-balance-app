from __future__ import annotations
import pandas as pd
import streamlit as st
from .state import Controls
from wbm.i18n import Translator, DEFAULT_LANG

__all__ = ["build_controls"]


def build_controls(default_start: pd.Timestamp, vols, balance_df, area_to_vol, lang: str = DEFAULT_LANG) -> Controls:
    tr = Translator(lang)
    st.sidebar.header(tr("scenario_controls"))
    p_scale = st.sidebar.slider("P scale", 0.0, 2.0, 1.0, 0.05)
    et_scale = st.sidebar.slider("ET scale", 0.0, 2.0, 1.0, 0.05)
    q_in = st.sidebar.number_input("Q in (mcm/day)", value=0.0, step=0.1)
    q_out = st.sidebar.number_input("Q out (mcm/day)", value=0.0, step=0.1)

    st.sidebar.header(tr("data_filtering"))
    min_area_km2 = st.sidebar.number_input(
        tr("min_area"), value=60.0, step=1.0,
        help="Records below threshold treated as unreliable and removed from baseline"
    )
    filter_baseline = st.sidebar.checkbox(tr("filter_baseline"), value=False)
    hide_scenario_below_min = st.sidebar.checkbox(
        tr("hide_below"), value=False,
        help="Hidden only on plots, scenario calculation unchanged"
    )

    start_date = st.sidebar.date_input("Start", value=default_start.date())
    horizon_days = st.sidebar.slider("Horizon (days)", 30, 730, 365, 30)
    end_date = pd.Timestamp(start_date) + pd.Timedelta(days=horizon_days)

    view_mode = st.sidebar.radio(tr("display_range"), [tr("all_period"), tr("single_year")], index=0)
    smooth_win = st.sidebar.slider(tr("smoothing_window"), 1, 90, 14, 1)

    st.sidebar.header(tr("forecast_mode"))
    forecast_mode = st.sidebar.radio(
        "P/ET",
        [tr("monthly_mean"), tr("seasonal_clim"), tr("seasonal_trend")],
        index=1,
    )

    with st.sidebar.expander(tr("season_trend_opts")):
        hist_window_days = st.number_input(
            tr("history_window"), 0, 3650, 730, 30,
            help="0 = full history"
        )
        seas_basis = st.selectbox(tr("season_basis"), [tr("basis_doy"), tr("basis_month")], index=0)

    # Map localized labels back to internal codes
    if forecast_mode == tr("monthly_mean"):
        forecast_mode_internal = "Monthly mean (all years)"
    elif forecast_mode == tr("seasonal_clim"):
        forecast_mode_internal = "Seasonal climatology"
    else:
        forecast_mode_internal = "Seasonal + trend"
    if seas_basis == tr("basis_doy"):
        seas_basis_internal = "DOY"
    else:
        seas_basis_internal = "MONTH"

    return Controls(
        p_scale=float(p_scale),
        et_scale=float(et_scale),
        q_in=float(q_in),
        q_out=float(q_out),
        min_area_km2=float(min_area_km2),
        filter_baseline=bool(filter_baseline),
        hide_scenario_below_min=bool(hide_scenario_below_min),
        view_mode=view_mode if view_mode in ("All period","Single year") else ("All period" if view_mode.startswith(tr("all_period")) else "Single year"),
        smooth_win=int(smooth_win),
        forecast_mode=forecast_mode_internal,
        hist_window_days=int(hist_window_days),
        seas_basis=seas_basis_internal,
        start_date=pd.Timestamp(start_date),
        end_date=end_date,
    )
