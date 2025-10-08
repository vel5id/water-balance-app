from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from wbm.ensemble import run_volume_ensemble, build_daily_ensemble
from wbm.forecast import build_robust_season_trend_series
from wbm.seasonal import compute_acf, recommend_block_length, theil_sen_trend_ci_boot

__all__ = ["render_ensemble"]

def render_ensemble(era5_df: pd.DataFrame, vol_to_area, p_clim, et_clim, init_volume: float, p_scale: float, et_scale: float, start_date: pd.Timestamp, end_date: pd.Timestamp, plot_scena_s: pd.DataFrame):
    with st.expander("Ensemble forecast (experimental)", expanded=False):
        use_ens = st.checkbox("Enable ensemble", value=False, key="ens_enable", help="Run N stochastic members")
        if not use_ens:
            return
        N = st.slider("Members (N)", 20, 300, 100, 10, key="ens_N")
        hist_days = st.number_input("History window (days)", 0, 3650, 5*365, 30, key="ens_hist")
        seas_basis = st.selectbox("Season basis", ["DOY","MONTH"], index=0, key="ens_basis")
        seas_smooth = st.slider("Seasonal smooth window (days)", 0, 31, 7, 1, key="ens_smooth")
        center_mode = st.selectbox("Centering", ["median","mean"], index=0, key="ens_center")
        use_boot = st.checkbox("Residual block bootstrap", value=True, key="ens_boot")
        block_mode = st.selectbox("Block length mode", ["Auto","Manual"], index=0, key="ens_block_mode")
        block_len_manual = st.slider("Block length (days)", 1, 14, 5, 1, disabled=block_mode=="Auto", key="ens_block_len")
        q_low, q_high = st.select_slider("Quantile range", options=[0.05,0.1,0.25,0.5,0.75,0.9,0.95], value=(0.05,0.95), key="ens_qrange")
        q_med = 0.5 if (0.5 >= q_low and 0.5 <= q_high) else round((q_low + q_high)/2.0,4)
        q_low, q_med, q_high = sorted({float(q_low), float(q_med), float(q_high)})
        future_days = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)
        freq = "doy" if seas_basis == "DOY" else "month"
        hist_min = int(hist_days) if int(hist_days) > 0 else 90
        base_p = _prepare_series(era5_df, "precip_mm") if (era5_df is not None and not era5_df.empty) else pd.Series(dtype=float)
        base_len = len(base_p)
        if future_days <= 0:
            future_days = 1  # гарантируем хотя бы один день вперёд чтобы серия не была пустой
        # Эффективный минимум истории: не больше доступной длины и не менее 30 (если есть столько)
        hist_min_eff = min(hist_min, base_len) if base_len else hist_min
        hist_min_eff = max(30, hist_min_eff) if base_len >= 30 else base_len
        build_error = None
        future_p = None
        if base_len == 0:
            build_error = "Пустая серия осадков (нет данных precip_mm)."
        elif base_len < 10:
            build_error = f"Слишком мало точек ({base_len}) для построения сезонно-трендовой модели. Нужно >=10." 
        else:
            # Первая попытка с hist_min_eff
            try:
                future_p_res = build_robust_season_trend_series(
                    base_p, freq=freq, future_days=future_days, min_history=hist_min_eff
                )
                future_p = future_p_res.deterministic
            except ValueError as e:
                # Попробуем адаптивно уменьшить порог истории если пользователь задал слишком большой
                for frac in (0.75, 0.6, 0.5, 0.4):
                    alt_min = max(30, int(base_len * frac)) if base_len * frac >= 30 else int(base_len * frac)
                    if alt_min < 10:
                        continue
                    try:
                        future_p_res = build_robust_season_trend_series(
                            base_p, freq=freq, future_days=future_days, min_history=alt_min
                        )
                        future_p = future_p_res.deterministic
                        st.caption(
                            f"⚠️ Использован пониженный min_history={alt_min} (изначально {hist_min}) из-за ошибки: {e}" 
                        )
                        break
                    except Exception:
                        continue
                if future_p is None:
                    build_error = f"Не удалось после адаптивных попыток: {e} (длина серии={base_len})."
            except Exception as e:  # любые другие ошибки
                build_error = f"Неожиданная ошибка построения: {e}"
        if future_p is None or future_p.empty:
            st.warning("Не удалось построить детерминированную серию осадков.")
            st.caption(
                f"Диагностика: длина исходной серии={base_len}, выбранный min_history={hist_min}, эффективный={hist_min_eff}. {build_error or ''}".strip()
            )
            st.caption("Попробуйте: уменьшить 'History window', выбрать MONTH вместо DOY или снять Ensemble.")
            return
        try:
            hist = era5_df[["date","precip_mm"]].dropna().copy()
            hist["date"] = pd.to_datetime(hist["date"]).dt.normalize()
            hist = hist.drop_duplicates("date").set_index("date").asfreq("D")
            doy = hist.index.dayofyear
            doy = np.where((hist.index.month==2) & (hist.index.day==29), 59, doy)
            clim_vals = pd.Series(hist["precip_mm"].groupby(doy).mean())
            clim_series = pd.Series(clim_vals.reindex(range(1,367)).interpolate(limit_direction="both").to_numpy()[doy-1], index=hist.index)
            residual_hist = (hist["precip_mm"] - clim_series)
        except Exception:
            residual_hist = hist["precip_mm"] if 'hist' in locals() else pd.Series(dtype=float)
        if use_boot and block_mode == "Auto":
            acf_df = compute_acf(residual_hist.to_numpy(), max_lag=min(30, max(5,residual_hist.size//4))) if residual_hist.size else None
            reco_block = recommend_block_length(acf_df, residual_hist.size) if acf_df is not None and not acf_df.empty else 5
            st.caption(f"Auto block length recommendation: {reco_block}")
            eff_block = reco_block
        else:
            eff_block = int(block_len_manual) if use_boot else 1
        residual_sets = build_daily_ensemble(
            deterministic_future=future_p,
            residuals=residual_hist,
            n_members=int(N),
            block_size=eff_block,
            random_state=42,
        )
        result = run_volume_ensemble(
            start_volume_mcm=float(init_volume),
            vol_to_area=vol_to_area,
            p_clim=p_clim,
            et_clim=et_clim,
            deterministic_p=future_p * float(p_scale),
            residual_sets=[r * float(p_scale) for r in residual_sets],
            p_scale=1.0,
            et_scale=float(et_scale),
        )
        aligned = [m.set_index("date")["volume_mcm"] for m in result.members]
        all_vols = pd.concat(aligned, axis=1)
        qdf = all_vols.quantile([q_low,q_med,q_high], axis=1).T
        qcols_map = {q_low:f"vol_q{int(q_low*100)}", q_med:f"vol_q{int(q_med*100)}", q_high:f"vol_q{int(q_high*100)}"}
        qdf.rename(columns=qcols_map, inplace=True)
        qdf.reset_index(inplace=True); qdf.rename(columns={"index":"date"}, inplace=True)
        ens = qdf
        if ens.empty:
            st.info("Not enough data to run ensemble.")
            return
        fan = go.Figure()
        fan.add_trace(go.Scatter(x=plot_scena_s["date"], y=plot_scena_s["volume_mcm"], mode="lines", name="Scenario Volume", line=dict(color="#d62728")))
        fan.add_trace(go.Scatter(x=ens["date"], y=ens[f"vol_q{int(q_high*100)}"], name=f"Q{int(q_high*100)}", line=dict(color="rgba(31,119,180,0.0)")))
        fan.add_trace(go.Scatter(x=ens["date"], y=ens[f"vol_q{int(q_low*100)}"], name=f"Q{int(q_low*100)}", fill="tonexty", fillcolor="rgba(31,119,180,0.2)", line=dict(color="rgba(31,119,180,0.0)")))
        fan.add_trace(go.Scatter(x=ens["date"], y=ens[f"vol_q{int(q_med*100)}"], name=f"Q{int(q_med*100)} (median)", line=dict(color="#1f77b4", width=2)))
        fan.update_layout(title="Ensemble Volume Forecast", template="plotly_white", xaxis_title="Date", yaxis_title="Volume (million m³)")
        st.plotly_chart(fan, use_container_width=True, config={"displaylogo": False})
        with st.expander("Ensemble diagnostics", expanded=False):
            acf_df = compute_acf(residual_hist.to_numpy(), max_lag=min(30, max(5,residual_hist.size//4))) if residual_hist.size else None
            if acf_df is not None and not acf_df.empty:
                # width parameter must be int; switch to container-based sizing
                st.dataframe(acf_df, use_container_width=True, height=180)
                st.caption("ACF: автокорреляция по лагам")
                rec_block = recommend_block_length(acf_df, residual_hist.size)
                st.caption(f"Recommended block length (ACF threshold): {rec_block}")
            else:
                st.info("Not enough residual data for ACF.")
            try:
                slope_py, lo_py, hi_py = theil_sen_trend_ci_boot(base_p, freq=freq, n_boot=200)
                st.markdown(f"**Theil–Sen slope**: {slope_py:.4f} per year (95% CI [{lo_py:.4f}, {hi_py:.4f}])")
            except Exception:
                st.info("Trend CI unavailable.")

def _prepare_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    if df is None or df.empty or value_col not in df.columns or "date" not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", value_col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.set_index("date")[value_col].asfreq("D")
