from __future__ import annotations

# Set environment variable to suppress warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings("ignore")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, total=None, desc: str | None = None):
        total = total or (len(iterable) if hasattr(iterable, "__len__") else None)
        printed = -1
        for i, x in enumerate(iterable, 1):
            if total:
                pct = int(i * 100 / total)
                if pct // 10 > printed // 10:
                    printed = pct
                    print(f"[progress] {desc or 'backtest'}: {pct}% ({i}/{total})", flush=True)
            yield x

# Ensure local package import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==================== PROPHET MODEL ====================

def build_prophet_model(
    series: pd.Series,
    future_days: int = 180,
    use_external_regressors: bool = False,
    external_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """Build Prophet model and make predictions.
    
    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex
    future_days : int
        Number of days to forecast
    use_external_regressors : bool
        Whether to use external regressors (temperature, precipitation, etc.)
    external_data : pd.DataFrame, optional
        External regressors data
        
    Returns
    -------
    forecast : pd.DataFrame
        FULL Prophet forecast (with 'ds' column) for ENTIRE period (history + future)
    model_info : dict
        Model metadata
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet not installed. Run: pip install prophet")
    
    # Prepare data in Prophet format
    df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    # 🔧 ОПТИМИЗИРОВАННАЯ КОНФИГУРАЦИЯ для водохранилища
    # Баланс между гибкостью и регуляризацией
    model = Prophet(
        growth='linear',  # Linear для общего случая (logistic требует capacity)
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',  # 🔧 ИЗМЕНЕНО: Additive более стабилен для backtesting
        seasonality_prior_scale=10.0,  # 🔧 УВЕЛИЧЕНО: Больше гибкости для сезонности (было 5.0)
        changepoint_prior_scale=0.05,  # 🔧 ОПТИМИЗИРОВАНО: Средняя гибкость (было 0.01 - слишком жестко)
        changepoint_range=0.85,        # 🔧 УВЕЛИЧЕНО: Changepoints в 85% данных (было 0.8)
        n_changepoints=25,             # 🔧 УВЕЛИЧЕНО: Больше точек адаптации (было 15)
        interval_width=0.95,
        uncertainty_samples=0
    )
    
    # Monthly seasonality с оптимальным fourier_order
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5,  # 🔧 УВЕЛИЧЕНО: Больше гибкости (было 3)
        prior_scale=10.0
    )
    
    # Quarterly seasonality
    model.add_seasonality(
        name='quarterly',
        period=91.25,
        fourier_order=8,  # 🔧 УВЕЛИЧЕНО: Максимальная гибкость (было 5)
        prior_scale=10.0
    )
    
    # Add external regressors if provided
    if use_external_regressors and external_data is not None:
        for col in external_data.columns:
            if col not in ['ds', 'y']:
                model.add_regressor(col)
        # Merge external data
        df = df.merge(external_data, on='ds', how='left')
    
    # Fit model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model.fit(df)
    
    # Make predictions for ENTIRE period (history + future)
    future = model.make_future_dataframe(periods=future_days, freq='D')
    
    # Add external regressors to future dataframe if needed
    if use_external_regressors and external_data is not None:
        # Extend external data (простейший подход - forward fill)
        future = future.merge(external_data, on='ds', how='left')
        for col in external_data.columns:
            if col not in ['ds']:
                future[col] = future[col].fillna(method='ffill')
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        forecast = model.predict(future)
    
    # Model info
    model_info = {
        'model_type': 'Prophet',
        'growth': model.growth,
        'seasonality_mode': model.seasonality_mode,
        'yearly_seasonality': True,
        'monthly_seasonality': True,
        'quarterly_seasonality': True,
        'external_regressors': list(external_data.columns) if external_data is not None else []
    }
    
    # 🔧 ИСПРАВЛЕНО: Возвращаем ВЕСЬ forecast DataFrame с колонкой 'ds'
    return forecast[['ds', 'yhat']].set_index('ds'), model_info


# ==================== METRICS ====================

def compute_metrics(err: pd.Series, actual: pd.Series) -> Dict[str, float]:
    """Compute forecast error metrics."""
    e = err.dropna()
    n = int(e.shape[0])
    if n < 2:
        return {
            "n": n, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
            "bias": float("nan"), "mape_pct": float("nan"), "r2": float("nan"),
            "nse": float("nan"), "kge": float("nan")
        }
    
    mse = float(np.mean(np.square(e)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    
    a = actual.reindex(e.index).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.abs(e) / np.where(np.abs(a) > 1e-9, np.abs(a), np.nan)
    mape_pct = float(np.nanmean(mape) * 100.0)
    
    # Calculate R², NSE, KGE
    sim = (a + e).values
    eval = a.values
    
    with np.errstate(all="ignore"):
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(eval, sim))
        except Exception:
            r2 = float("nan")
        
        try:
            import hydroeval as he
            nse = float(he.evaluator(he.nse, sim, eval)[0])
            kge, r, alpha, beta = he.evaluator(he.kge, sim, eval)
            kge = float(kge[0])
        except Exception:
            nse = float("nan")
            kge = float("nan")
    
    return {
        "n": n, "mse": mse, "rmse": rmse, "mae": mae, "bias": bias,
        "mape_pct": mape_pct, "r2": r2, "nse": nse, "kge": kge
    }


# ==================== BACKTEST ====================

def backtest_prophet_horizons(
    df: pd.DataFrame,
    horizons: List[int],
    n_jobs: int = -1,
    chunk_size: int = 100,
    use_external_regressors: bool = False,
    external_data: Optional[pd.DataFrame] = None
) -> Dict[int, Dict[str, float]]:
    """Backtest Prophet model at different forecast horizons.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'volume_mcm' columns
    horizons : List[int]
        Forecast horizons in days (e.g., [1, 30, 365])
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    chunk_size : int
        Process in chunks for memory efficiency
    use_external_regressors : bool
        Whether to use external regressors
    external_data : pd.DataFrame, optional
        External data (temperature, precipitation, etc.)
    """
    print("\n" + "="*70)
    print("🔮 PROPHET BACKTEST STARTING")
    print("="*70)
    
    # Prepare data
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "volume_mcm"]).sort_values("date")
    vol = pd.Series(df["volume_mcm"].values, index=df["date"].values)
    
    max_h = max(horizons)
    errors_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    actual_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    
    # Sliding window
    dates = vol.index
    start_i = 180  # 🔧 УВЕЛИЧЕНО: Минимум 180 дней истории (было 90) для полугодовой сезонности
    end_i = len(vol) - max_h
    
    if end_i <= start_i:
        return {h: {"n": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                    "bias": float("nan"), "mape_pct": float("nan"), "r2": float("nan"),
                    "nse": float("nan"), "kge": float("nan")} for h in horizons}
    
    # 🔧 УВЕЛИЧЕНО: Используем 1000 origins (синхронизировано с SARIMA)
    step = max(1, (end_i - start_i) // 1000)
    total_steps = max(0, (end_i - start_i + (step - 1)) // step)
    
    print(f"📊 Total origins: {total_steps}")
    print(f"🔧 Parallelism: {n_jobs if n_jobs > 0 else 'all CPUs'}")
    print(f"📈 Horizons: {horizons} days")
    print(f"🌊 External regressors: {'Yes' if use_external_regressors else 'No'}\n")
    
    def _fit_one_origin(i: int) -> Dict[str, object]:
        """Fit Prophet for one origin."""
        # 🔧 ИСПРАВЛЕНО: Используем правильный индекс дат
        origin_date = dates[i]
        hist = vol.loc[: origin_date]  # Используем loc вместо iloc для DatetimeIndex
        
        result = {"i": i, "origin_date": str(origin_date), "preds": {}, "error": None}
        
        try:
            # Get external data for training period if needed
            ext_train = None
            if use_external_regressors and external_data is not None:
                ext_train = external_data[external_data['ds'] <= origin_date]
            
            # Build Prophet model - возвращает ВЕСЬ forecast DataFrame
            forecast_df, model_info = build_prophet_model(
                hist,
                future_days=max_h,
                use_external_regressors=use_external_regressors,
                external_data=ext_train
            )
            
            # Extract predictions for each horizon
            for h in horizons:
                target_date = origin_date + pd.Timedelta(days=h)
                
                # 🔧 ИСПРАВЛЕНО: Проверяем наличие даты в реальных данных
                if target_date not in vol.index:
                    continue
                
                # 🔧 ИСПРАВЛЕНО: Используем правильный индекс для получения прогноза
                try:
                    if target_date in forecast_df.index:
                        pred = float(forecast_df.loc[target_date, 'yhat'])
                    else:
                        # 🔧 ИСПРАВЛЕНО: Логируем отсутствие прогноза
                        result["error"] = f"Missing forecast for {target_date}"
                        continue
                except Exception as e:
                    result["error"] = f"Error extracting prediction: {str(e)}"
                    continue
                
                actual = float(vol.loc[target_date])
                
                if np.isfinite(pred) and np.isfinite(actual):
                    result["preds"][h] = {
                        "target_date": str(target_date),
                        "pred": pred,
                        "actual": actual,
                        "error": pred - actual,
                    }
                else:
                    result["error"] = f"Invalid prediction or actual: pred={pred}, actual={actual}"
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    # Process in chunks
    origins = list(range(start_i, end_i, step))
    total_chunks = (len(origins) + chunk_size - 1) // chunk_size
    
    print(f"Processing {total_chunks} chunks...")
    
    # 🔧 ДОБАВЛЕНО: Счетчики для диагностики
    total_errors = 0
    missing_forecasts = 0
    successful_predictions = {h: 0 for h in horizons}
    
    for chunk_idx in tqdm(range(total_chunks), desc="Chunks", unit="chunk"):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(origins))
        chunk_origins = origins[chunk_start:chunk_end]
        
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_one_origin)(i) for i in chunk_origins
        )
        
        # Aggregate results
        for result in results:
            if result["error"]:
                total_errors += 1
                if "Missing forecast" in result["error"]:
                    missing_forecasts += 1
                continue
            
            for h, pred_data in result["preds"].items():
                errors_by_h[h].append(pred_data["error"])
                actual_by_h[h].append(pred_data["actual"])
                successful_predictions[h] += 1
    
    # 🔧 ДОБАВЛЕНО: Вывод диагностики
    print("\n" + "="*70)
    print("📊 BACKTEST DIAGNOSTICS")
    print("="*70)
    print(f"Total origins processed: {len(origins)}")
    print(f"Total errors: {total_errors}")
    print(f"Missing forecasts: {missing_forecasts}")
    print(f"\nSuccessful predictions by horizon:")
    for h in horizons:
        print(f"  {h}-day: {successful_predictions[h]} predictions")
    print("="*70 + "\n")
    
    # Compute final metrics
    out: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        e = pd.Series(errors_by_h[h])
        a = pd.Series(actual_by_h[h])
        out[h] = compute_metrics(e, a)
    
    return out


# ==================== MAIN ====================

def main():
    root = Path(__file__).resolve().parents[1]
    csv = root / "processed_data" / "water_balance_output" / "water_balance_final.csv"
    
    if not csv.exists():
        raise FileNotFoundError(f"Data file not found: {csv}")
    
    df = pd.read_csv(csv)
    
    # 🔧 РАСШИРЕННЫЙ НАБОР ГОРИЗОНТОВ для детального анализа
    horizons = [
        1,      # 1 день - краткосрочный прогноз
        2,      # 2 дня
        3,      # 3 дня
        7,      # 1 неделя
        30,     # 1 месяц (30 дней)
        90,     # 3 месяца (квартал)
        180,    # 6 месяцев (полугодие)
        365     # 1 год (365 дней)
    ]
    
    print("\n" + "="*70)
    print("🔮 PROPHET BACKTEST - Water Volume Forecasting")
    print("="*70)
    print(f"📈 Forecast horizons: {horizons} days")
    print(f"   - Ultra-short: 1-3 days (SARIMA expected to win)")
    print(f"   - Short-term: 7-30 days (Comparison zone)")
    print(f"   - Mid-term: 90-180 days (Prophet potential zone)")
    print(f"   - Long-term: 365+ days (Prophet expected to win)")
    print("="*70)
    
    # Backtest Prophet
    metrics = backtest_prophet_horizons(
        df,
        horizons,
        n_jobs=-1,
        chunk_size=100,
        use_external_regressors=False  # Set to True if you have external data
    )
    
    # Save results
    out_dir = root / "processed_data" / "water_balance_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "forecast_backtest_prophet_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "="*70)
    print("✅ PROPHET BACKTEST COMPLETE")
    print("="*70)
    
    # 🔧 Группируем результаты по категориям прогнозов
    ultra_short = [h for h in horizons if h <= 3]
    short_term = [h for h in horizons if 4 <= h <= 30]
    mid_term = [h for h in horizons if 31 <= h <= 180]
    long_term = [h for h in horizons if h > 180]
    
    def print_horizon_group(group_name: str, group_horizons: List[int]):
        """Print metrics for a group of horizons."""
        if not group_horizons:
            return
        
        print(f"\n{'='*70}")
        print(f"📊 {group_name}")
        print(f"{'='*70}")
        
        for h in group_horizons:
            m = metrics[h]
            print(f"\n🔹 {h}-day forecast:")
            print(f"  R²:    {m['r2']:7.4f}")
            print(f"  NSE:   {m['nse']:7.4f}")
            print(f"  KGE:   {m['kge']:7.4f}")
            print(f"  RMSE:  {m['rmse']:7.2f} млн.м³")
            print(f"  MAE:   {m['mae']:7.2f} млн.м³")
            print(f"  MAPE:  {m['mape_pct']:7.2f}%")
            print(f"  Bias:  {m['bias']:7.2f} млн.м³")
            print(f"  n:     {m['n']}")
    
    print_horizon_group("ULTRA-SHORT TERM (1-3 days)", ultra_short)
    print_horizon_group("SHORT TERM (7-30 days)", short_term)
    print_horizon_group("MID TERM (3-6 months)", mid_term)
    print_horizon_group("LONG TERM (12+ months)", long_term)
    
    print("\n" + "="*70)
    print("💾 Results saved to:", out_path.name)
    print("="*70)
    
    # Compare with SARIMA
    sarima_path = out_dir / "forecast_backtest_metrics.json"
    if sarima_path.exists():
        with open(sarima_path) as f:
            sarima_metrics = json.load(f)
        
        print("\n" + "="*70)
        print("⚔️ PROPHET vs SARIMA - HEAD-TO-HEAD COMPARISON")
        print("="*70)
        
        # Подсчёт побед
        prophet_wins = 0
        sarima_wins = 0
        
        # Сравниваем только общие горизонты
        common_horizons = [h for h in horizons if str(h) in sarima_metrics]
        
        if not common_horizons:
            print("\n⚠️ No common horizons found in SARIMA results")
            print(f"   SARIMA horizons: {list(sarima_metrics.keys())}")
            print(f"   Prophet horizons: {horizons}")
        else:
            for h in common_horizons:
                print(f"\n{'='*70}")
                print(f"🔹 Horizon: {h} days")
                print(f"{'='*70}")
                
                # Используем int ключи для Prophet, str для SARIMA
                prophet_r2 = metrics[h]['r2']
                sarima_r2 = sarima_metrics[str(h)]['r2']
                prophet_rmse = metrics[h]['rmse']
                sarima_rmse = sarima_metrics[str(h)]['rmse']
                prophet_nse = metrics[h]['nse']
                sarima_nse = sarima_metrics[str(h)]['nse']
                
                # R² comparison
                r2_winner = "Prophet" if prophet_r2 > sarima_r2 else "SARIMA"
                r2_improvement = ((prophet_r2 - sarima_r2) / abs(sarima_r2) * 100) if sarima_r2 != 0 else float('inf')
                
                print(f"\n  📈 R² Score:")
                print(f"     Prophet: {prophet_r2:8.4f}")
                print(f"     SARIMA:  {sarima_r2:8.4f}")
                print(f"     Winner:  🏆 {r2_winner} ({abs(r2_improvement):.1f}% {'better' if r2_winner == 'Prophet' else 'worse'})")
                
                # RMSE comparison
                rmse_winner = "Prophet" if prophet_rmse < sarima_rmse else "SARIMA"
                rmse_improvement = ((sarima_rmse - prophet_rmse) / sarima_rmse * 100) if sarima_rmse != 0 else 0
                
                print(f"\n  📉 RMSE (lower is better):")
                print(f"     Prophet: {prophet_rmse:8.2f} млн.м³")
                print(f"     SARIMA:  {sarima_rmse:8.2f} млн.м³")
                print(f"     Winner:  🏆 {rmse_winner} ({abs(rmse_improvement):.1f}% {'better' if rmse_winner == 'Prophet' else 'worse'})")
                
                # NSE comparison
                nse_winner = "Prophet" if prophet_nse > sarima_nse else "SARIMA"
                
                print(f"\n  � NSE Score:")
                print(f"     Prophet: {prophet_nse:8.4f}")
                print(f"     SARIMA:  {sarima_nse:8.4f}")
                print(f"     Winner:  �🏆 {nse_winner}")
                
                # Overall winner (based on R² and RMSE)
                if r2_winner == rmse_winner:
                    overall_winner = r2_winner
                    if overall_winner == "Prophet":
                        prophet_wins += 1
                    else:
                        sarima_wins += 1
                    print(f"\n  🏅 Overall winner: 🏆 {overall_winner}")
                else:
                    print(f"\n  🏅 Overall winner: 🤝 Mixed (R²: {r2_winner}, RMSE: {rmse_winner})")
            
            # Final summary
            print("\n" + "="*70)
            print("🏆 FINAL SCORE")
            print("="*70)
            print(f"  Prophet wins: {prophet_wins}/{len(common_horizons)}")
            print(f"  SARIMA wins:  {sarima_wins}/{len(common_horizons)}")
            
            if prophet_wins > sarima_wins:
                print(f"\n  🎉 Prophet is the overall winner!")
            elif sarima_wins > prophet_wins:
                print(f"\n  🎉 SARIMA is the overall winner!")
            else:
                print(f"\n  🤝 It's a tie!")
            
            print("\n  💡 RECOMMENDATION:")
            print(f"     - Use SARIMA for horizons: {[h for h in common_horizons if (metrics[h]['r2'] < sarima_metrics[str(h)]['r2'])]}")
            print(f"     - Use Prophet for horizons: {[h for h in common_horizons if (metrics[h]['r2'] >= sarima_metrics[str(h)]['r2'])]}")
            print("="*70)


if __name__ == "__main__":
    main()
