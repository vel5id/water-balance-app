from __future__ import annotations
import pandas as pd
import streamlit as st
from wbm.analysis import rolling_trend, lagged_correlation

__all__ = ["render_trends_and_correlations", "render_long_term_trends"]

def render_trends_and_correlations(plot_base: pd.DataFrame, plot_scena: pd.DataFrame):
    with st.expander("Trends and correlations", expanded=False):
        if plot_scena is None or plot_scena.empty:
            st.info("No scenario data.")
            return
        win = st.slider("Rolling window (days)", 7, 120, 30, 1, key="tr_win")
        max_lag = st.slider("Max lag for correlation (days)", 7, 120, 60, 1, key="tr_lag")
        base = plot_base.copy() if plot_base is not None and not plot_base.empty else plot_scena.copy()
        needed = ["date","area_km2","volume_mcm","precipitation_volume_mcm","evaporation_volume_mcm"]
        missing = [c for c in needed if c not in base.columns]
        if missing:
            st.warning(f"Missing columns for trend analysis: {missing}")
            return
        base = base[needed].copy()
        base = base.set_index("date").asfreq("D").interpolate()
        st.line_chart(pd.DataFrame({
            "area_trend": rolling_trend(base["area_km2"], window=win),
            "P_trend": rolling_trend(base["precipitation_volume_mcm"], window=win),
            "ET_trend": rolling_trend(base["evaporation_volume_mcm"], window=win),
        }))
        corr_p = lagged_correlation(base["area_km2"], base["precipitation_volume_mcm"], max_lag=max_lag)
        corr_et = lagged_correlation(base["area_km2"], base["evaporation_volume_mcm"], max_lag=max_lag)
        st.subheader("Lagged correlations with area")
        if not corr_p.empty:
            st.line_chart(corr_p.set_index("lag")["corr"], height=200)
        if not corr_et.empty:
            st.line_chart(corr_et.set_index("lag")["corr"], height=200)

def render_long_term_trends(era5_df: pd.DataFrame):
    from wbm.trends import aggregate_series, theilsen_trend_ci, kendall_significance, make_trend_comparison_figure
    with st.expander("Long-term P & ET trends", expanded=False):
        if era5_df is None or era5_df.empty or not {"date","precip_mm","evap_mm"}.issubset(era5_df.columns):
            st.info("ERA5 data insufficient for long-term trends.")
            return
        years_back = st.slider("Years back", 3, 30, 10, 1, key="lt_years")
        freq_label = st.selectbox("Aggregation", ["Monthly","Annual"], index=0, key="lt_freq")
        freq = 'ME' if freq_label == 'Monthly' else 'A'
        end_anchor = pd.Timestamp.today().normalize()
        p_agg = aggregate_series(era5_df, "date", "precip_mm", freq=freq, years_back=years_back, end_anchor=end_anchor)
        et_agg = aggregate_series(era5_df, "date", "evap_mm", freq=freq, years_back=years_back, end_anchor=end_anchor)
        if p_agg.empty or et_agg.empty:
            st.info("Not enough aggregated data.")
            return
        p_slope_py, p_inter, p_lo, p_hi = theilsen_trend_ci(p_agg)
        et_slope_py, et_inter, et_lo, et_hi = theilsen_trend_ci(et_agg)
        p_tau, p_p = kendall_significance(p_agg)
        et_tau, et_p = kendall_significance(et_agg)
        st.plotly_chart(
            make_trend_comparison_figure(p_agg, et_agg, p_slope_py, p_inter, et_slope_py, et_inter),
            use_container_width=True,
            config={"displaylogo": False},
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("P slope (mm/day/yr)", f"{p_slope_py:.4f}")
            st.caption(f"CI [{p_lo:.4f}, {p_hi:.4f}] | Kendall tau={p_tau:.3f}, p={p_p:.3g}")
        with col_t2:
            st.metric("ET slope (mm/day/yr)", f"{et_slope_py:.4f}")
            st.caption(f"CI [{et_lo:.4f}, {et_hi:.4f}] | Kendall tau={et_tau:.3f}, p={et_p:.3g}")
