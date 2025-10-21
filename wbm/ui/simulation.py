from __future__ import annotations
import pandas as pd
import streamlit as st
from typing import Tuple, Literal, Optional, Dict
from .state import LoadedData, Controls, ScenarioContext
from wbm.simulate import simulate_forward
from wbm.forecast import (
    build_robust_season_trend_series, 
    SeasonTrendResult,
    build_sarima_forecast_enhanced,
    build_prophet_forecast,
    build_sarimax_forecast,
)
from wbm.metrics import calculate_metrics_by_horizon, format_metrics_for_display

__all__ = ["prepare_drivers", "run_scenario", "display_forecast_metrics"]

def _prepare_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    if df is None or df.empty or value_col not in df.columns or "date" not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", value_col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.set_index("date")[value_col].asfreq("D")

def prepare_drivers(ld: LoadedData, ctr: Controls):
    p_daily = et_daily = None
    if ctr.forecast_mode == "Monthly mean (all years)":
        # Simple monthly mean expansion
        p_daily = _monthly_mean_series(ld.era5_df, "precip_mm", ctr.start_date, ctr.end_date)
        et_daily = _monthly_mean_series(ld.era5_df, "evap_mm", ctr.start_date, ctr.end_date)
    elif ctr.forecast_mode == "Seasonal + trend":
        # Use new forecast method selector
        forecast_method = getattr(ctr, 'forecast_method', 'Theil-Sen')
        base_p = _prepare_series(ld.era5_df, "precip_mm")
        base_et = _prepare_series(ld.era5_df, "evap_mm")
        freq = "doy" if ctr.seas_basis == "DOY" else "month"
        future_days = int((ctr.end_date - ctr.start_date).days)
        min_hist = ctr.hist_window_days if ctr.hist_window_days > 0 else 90
        
        # Get precipitation forecast
        p_daily = _get_forecast_by_method(
            base_p, 
            forecast_method, 
            future_days, 
            min_hist,
            freq=freq
        )
        
        # Get evapotranspiration forecast
        et_daily = _get_forecast_by_method(
            base_et, 
            forecast_method, 
            future_days, 
            min_hist,
            freq=freq
        )
    
    return p_daily, et_daily


def _get_forecast_by_method(
    series: pd.Series,
    method: Literal["Theil-Sen", "SARIMA", "Prophet", "SARIMAX"],
    future_days: int,
    min_history: int,
    freq: str = "doy"
) -> pd.Series:
    """Get forecast using specified method."""
    try:
        if method == "Theil-Sen":
            res: SeasonTrendResult = build_robust_season_trend_series(
                series, 
                freq=freq, 
                future_days=future_days, 
                min_history=min_history
            )
            return res.deterministic
        
        elif method == "SARIMA":
            try:
                forecast, model_info = build_sarima_forecast_enhanced(
                    series,
                    future_days=future_days,
                    min_history=min_history,
                    max_history=730,
                    seasonal=True,
                    return_to_non_seasonal=True,
                )
                st.info(f"✅ SARIMA ({model_info['method']}): AIC={model_info['aic']:.2f}")
                return forecast
            except Exception as e:
                st.warning(f"⚠️ SARIMA failed: {str(e)[:100]}. Falling back to Theil-Sen.")
                res: SeasonTrendResult = build_robust_season_trend_series(
                    series, 
                    freq=freq, 
                    future_days=future_days, 
                    min_history=min_history
                )
                return res.deterministic
        
        elif method == "Prophet":
            try:
                forecast, model_info = build_prophet_forecast(
                    series,
                    future_days=future_days,
                    min_history=min_history,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                )
                st.info(f"✅ Prophet: {model_info['n_obs']} observations used")
                return forecast
            except Exception as e:
                st.warning(f"⚠️ Prophet failed: {str(e)[:100]}. Falling back to Theil-Sen.")
                res: SeasonTrendResult = build_robust_season_trend_series(
                    series, 
                    freq=freq, 
                    future_days=future_days, 
                    min_history=min_history
                )
                return res.deterministic
        
        elif method == "SARIMAX":
            try:
                forecast, model_info = build_sarimax_forecast(
                    series,
                    exog_data=None,  # Can be extended to include climate features
                    future_days=future_days,
                    min_history=min_history,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                )
                exog_info = f" with {len(model_info['exog_vars'])} features" if model_info['has_exog'] else ""
                st.info(f"✅ SARIMAX: {model_info['n_obs']} observations{exog_info}")
                return forecast
            except Exception as e:
                st.warning(f"⚠️ SARIMAX failed: {str(e)[:100]}. Falling back to Theil-Sen.")
                res: SeasonTrendResult = build_robust_season_trend_series(
                    series, 
                    freq=freq, 
                    future_days=future_days, 
                    min_history=min_history
                )
                return res.deterministic
        
        else:
            # Default to Theil-Sen
            res: SeasonTrendResult = build_robust_season_trend_series(
                series, 
                freq=freq, 
                future_days=future_days, 
                min_history=min_history
            )
            return res.deterministic
    
    except Exception as e:
        st.error(f"❌ Forecast error: {str(e)}")
        return None

def _monthly_mean_series(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.set_index("date").sort_index()
    monthly = d[col].groupby([d.index.month]).mean()
    rng = pd.date_range(start, end, freq="D")
    vals = []
    for ts in rng:
        m = ts.month
        vals.append(monthly.get(m, float("nan")))
    return pd.Series(vals, index=rng)

def select_initial_volume(balance_df: pd.DataFrame, start_date: pd.Timestamp, vols) -> Tuple[float, str]:
    if balance_df is None or balance_df.empty:
        if hasattr(vols, '__len__') and len(vols):
            return float(vols[len(vols)//2]), "fallback: curve midpoint"
        return 0.0, "fallback: 0"
    try:
        last_obs = pd.to_datetime(balance_df["date"]).max()
        if pd.isna(last_obs):
            raise ValueError
        # prefer prior or interpolate logic simplified: just take last observed <= start
        prior = balance_df[pd.to_datetime(balance_df["date"]) <= start_date]
        if not prior.empty:
            return float(prior.tail(1)["volume_mcm"].iloc[0]), "last prior observation"
        else:
            return float(balance_df.tail(1)["volume_mcm"].iloc[0]), "last observation (after start)"
    except Exception:
        return float(balance_df["volume_mcm"].median()), "median fallback"

def run_scenario(ld: LoadedData, ctr: Controls, p_daily, et_daily, init_volume: float) -> ScenarioContext:
    scen = simulate_forward(
        start_date=ctr.start_date,
        end_date=ctr.end_date,
        init_volume_mcm=init_volume,
        p_clim=ld.p_clim,
        et_clim=ld.et_clim,
        vol_to_area=ld.vol_to_area,
        p_scale=ctr.p_scale,
        et_scale=ctr.et_scale,
        q_in_mcm_per_day=ctr.q_in,
        q_out_mcm_per_day=ctr.q_out,
        p_daily=p_daily,
        et_daily=et_daily,
    )
    return ScenarioContext(scenario_df=scen, init_volume=init_volume, init_note="")


def display_forecast_metrics(
    actual_series: pd.Series,
    forecast_series: pd.Series,
    variable_name: str = "Variable",
    horizons: Optional[list[int]] = None
) -> Optional[Dict]:
    """
    Display forecast accuracy metrics for multiple time horizons.
    
    Shows MAPE, RMSE, MAE, and R² for 1-day, 1-week, 1-month, and 6-month horizons.
    Uses Streamlit columns to display metrics in a grid format.
    
    Parameters
    ----------
    actual_series : pd.Series
        Actual observed values (must have DatetimeIndex)
    forecast_series : pd.Series
        Forecasted values (must have DatetimeIndex)
    variable_name : str
        Name of variable for display (e.g., "Precipitation", "Evapotranspiration")
    horizons : list[int], optional
        List of forecast horizons in days. Default: [1, 7, 30, 180]
        
    Returns
    -------
    metrics : Dict or None
        Calculated metrics dictionary, or None if calculation failed
    """
    if horizons is None:
        horizons = [1, 7, 30, 180]
    
    if actual_series is None or actual_series.empty or forecast_series is None or forecast_series.empty:
        return None
    
    try:
        # Calculate metrics
        metrics = calculate_metrics_by_horizon(actual_series, forecast_series, horizons=horizons)
        formatted = format_metrics_for_display(metrics, include_r2=True)
        
        # Display in Streamlit
        with st.expander(f"📊 Forecast Accuracy Metrics: {variable_name}", expanded=False):
            st.markdown("""
            **Metrics explanation:**
            - **MAPE**: Mean Absolute Percentage Error (lower is better, typical: 5-30%)
            - **RMSE**: Root Mean Squared Error in same units as data
            - **MAE**: Mean Absolute Error (more robust to outliers)
            - **R²**: Coefficient of determination (1 = perfect, 0 = baseline)
            - **N**: Number of samples used for calculation
            """)
            
            # Create columns for each horizon
            cols = st.columns(len(horizons))
            
            for idx, horizon in enumerate(sorted(horizons)):
                with cols[idx]:
                    # Determine horizon name
                    if horizon == 1:
                        horizon_label = "1-day"
                    elif horizon == 7:
                        horizon_label = "1-week"
                    elif horizon == 30:
                        horizon_label = "1-month"
                    elif horizon == 180:
                        horizon_label = "6-month"
                    else:
                        horizon_label = f"{horizon}-day"
                    
                    st.subheader(horizon_label)
                    
                    if horizon in formatted:
                        metric_data = formatted[horizon]
                        
                        # Display metrics as columns
                        for metric_name, metric_value in metric_data.items():
                            if metric_name != 'N':
                                col_m1, col_m2 = st.columns(2)
                                with col_m1:
                                    st.metric(metric_name, metric_value, delta=None)
                        
                        # Show sample count at bottom
                        if 'N' in metric_data:
                            st.caption(f"Samples: {metric_data['N']}")
                    else:
                        st.warning("N/A")
        
        return metrics
    
    except Exception as e:
        st.warning(f"⚠️ Could not calculate metrics: {str(e)[:100]}")
        return None

