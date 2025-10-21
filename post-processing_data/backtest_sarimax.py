from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
import os

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


# ==================== FEATURE ENGINEERING ====================

def prepare_exog_features(df: pd.DataFrame, exog_climate: pd.DataFrame) -> pd.DataFrame:
    """Prepare external regressors with climate context.
    
    Strategy:
    - Single regressor: Temperature (current value)
    - SARIMAX uses 365+ days of history to learn annual patterns
    - AR component handles temporal dependencies internally
    
    Parameters
    ----------
    df : pd.DataFrame
        Water volume data with 'date' and 'volume_mcm'
    exog_climate : pd.DataFrame
        Climate data with 'date' and climate variables
        
    Returns
    -------
    exog_df : pd.DataFrame
        Feature dataframe with date index (only temperature)
    """
    # Prepare volume data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Prepare climate data
    exog_climate = exog_climate.copy()
    exog_climate['date'] = pd.to_datetime(exog_climate['date'])
    exog_climate = exog_climate.set_index('date').sort_index()
    
    # Combine
    combined = pd.concat([df[['volume_mcm']], exog_climate], axis=1)
    
    # === CLIMATE CONTEXT ===
    # Temperature - important for evaporation and snowmelt
    combined['temperature_c'] = exog_climate['temperature_c']
    
    # Drop volume_mcm (it's the target, not a regressor)
    combined = combined.drop(columns=['volume_mcm'])
    
    # Drop raw climate columns (keep only processed features)
    raw_climate_cols = ['runoff_mm', 'precipitation_mm', 'evaporation_mm']
    combined = combined.drop(columns=raw_climate_cols, errors='ignore')
    
    # Drop rows with NaN (due to rolling windows)
    combined = combined.dropna()
    
    return combined


# ==================== SARIMAX MODEL ====================

def build_sarimax_model(
    series: pd.Series,
    exog_data: Optional[pd.DataFrame] = None,
    future_days: int = 365,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12)
) -> tuple:
    """Build SARIMAX model with external regressors.
    
    Parameters
    ----------
    series : pd.Series
        Historical time series
    exog_data : pd.DataFrame, optional
        External regressors (lag features + climate context)
    future_days : int
        Forecast horizon
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        Seasonal order (P, D, Q, m)
        
    Returns
    -------
    forecast : pd.Series
        Forecasted values
    model_info : dict
        Model metadata
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels not installed")
    
    # Align external data with series if provided
    exog_train = None
    exog_forecast = None
    
    if exog_data is not None:
        # Match indices - only use data that overlaps with series
        common_idx = exog_data.index.intersection(series.index)
        exog_train = exog_data.loc[common_idx].values
        series_aligned = series.loc[common_idx]
        
        # Generate future indices for forecast
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
        
        # For future external regressors:
        # LAG FEATURES: Use iterative forecasting (forecast becomes lag for next step)
        # CLIMATE FEATURES: Use last known values (persistence)
        # TEMPORAL FEATURES: Calculate directly from date
        
        # For simplicity in backtest: use last row values (persistence)
        # In production: implement iterative forecasting for lag features
        last_exog_values = exog_data.iloc[-1].values
        exog_forecast = np.tile(last_exog_values, (future_days, 1))
        
        # Update temporal features for future dates
        if 'month' in exog_data.columns or 'quarter' in exog_data.columns:
            future_df = pd.DataFrame(index=future_dates)
            
            # Get column indices for temporal features
            col_names = list(exog_data.columns)
            
            if 'month' in col_names:
                month_idx = col_names.index('month')
                exog_forecast[:, month_idx] = future_dates.month
            
            if 'quarter' in col_names:
                quarter_idx = col_names.index('quarter')
                exog_forecast[:, quarter_idx] = future_dates.quarter
    else:
        series_aligned = series
    
    # Build SARIMAX model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        model = SARIMAX(
            series_aligned.values,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False, maxiter=100)
        
        # Forecast
        forecast_values = fitted.forecast(steps=future_days, exog=exog_forecast)
    
    # Create forecast series
    last_date = series_aligned.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
    forecast = pd.Series(forecast_values, index=forecast_dates)
    
    model_info = {
        'model_type': 'SARIMAX',
        'order': order,
        'seasonal_order': seasonal_order,
        'has_exog': exog_data is not None,
        'exog_vars': list(exog_data.columns) if exog_data is not None else []
    }
    
    return forecast, model_info


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

def backtest_sarimax_horizons(
    df: pd.DataFrame,
    horizons: List[int],
    exog_df: Optional[pd.DataFrame] = None,
    n_jobs: int = None,
    chunk_size: int = 100,
    lookback_days: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """Backtest SARIMAX model at different forecast horizons.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'volume_mcm' columns
    horizons : List[int]
        Forecast horizons in days
    exog_df : pd.DataFrame, optional
        Feature dataframe with lag and climate features (already prepared)
    n_jobs : int
        Number of parallel jobs (None = cpu_count - 1)
    chunk_size : int
        Chunk size for processing
    """
    # Determine optimal number of cores (leave one free for system)
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)
    
    print("\n" + "="*70)
    print("🔮 SARIMAX BACKTEST STARTING")
    print("="*70)
    
    # Prepare data
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "volume_mcm"]).sort_values("date")
    vol = pd.Series(df["volume_mcm"].values, index=df["date"].values)
    
    # Check if external features are provided
    exog_prepared = None
    if exog_df is not None:
        print(f"📊 External features: {list(exog_df.columns)}")
        print(f"📊 Feature count: {len(exog_df.columns)}")
        exog_prepared = exog_df
    else:
        print("📊 External features: None")
    
    max_h = max(horizons)
    errors_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    actual_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    
    # Sliding window
    dates = vol.index
    # Determine minimal history needed
    min_history = 365
    if lookback_days is not None:
        min_history = max(min_history, lookback_days)
    start_i = min_history  # require at least this much history before first origin
    end_i = len(vol) - max_h
    
    if end_i <= start_i:
        return {h: {"n": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                    "bias": float("nan"), "mape_pct": float("nan"), "r2": float("nan"),
                    "nse": float("nan"), "kge": float("nan")} for h in horizons}
    
    # Sample origins (~1000)
    step = max(1, (end_i - start_i) // 1000)
    total_steps = max(0, (end_i - start_i + (step - 1)) // step)
    
    print(f"📊 Total origins: {total_steps}")
    print(f"🔧 Parallelism: {n_jobs} cores (CPU count - 1)")
    if lookback_days is not None:
        print(f"🪟 Lookback window: last {lookback_days} days per origin")
    else:
        print(f"🪟 Lookback window: full history up to origin")
    print(f"📈 Horizons: {horizons} days\n")
    
    def _fit_one_origin(i: int) -> Dict[str, object]:
        """Fit SARIMAX for one origin."""
        origin_date = dates[i]
        # rolling window: restrict to last lookback_days if provided
        if lookback_days is not None:
            start_date = origin_date - pd.Timedelta(days=lookback_days-1)
            hist = vol.loc[max(start_date, dates[0]) : origin_date]
        else:
            hist = vol.loc[: origin_date]
        
        # Get external data for training period
        exog_train = None
        if exog_prepared is not None:
            if lookback_days is not None:
                start_date = origin_date - pd.Timedelta(days=lookback_days-1)
                exog_train = exog_prepared.loc[max(start_date, exog_prepared.index.min()) : origin_date]
            else:
                exog_train = exog_prepared.loc[: origin_date]
        
        result = {"i": i, "origin_date": str(origin_date), "preds": {}, "error": None}
        
        try:
            # Build SARIMAX model
            forecast, model_info = build_sarimax_model(
                hist,
                exog_data=exog_train,
                future_days=max_h,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12)
            )
            
            # Extract predictions for each horizon
            for h in horizons:
                target_date = origin_date + pd.Timedelta(days=h)
                
                if target_date not in vol.index:
                    continue
                
                try:
                    if target_date in forecast.index:
                        pred = float(forecast.loc[target_date])
                    else:
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
    
    total_errors = 0
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
                continue
            
            for h, pred_data in result["preds"].items():
                errors_by_h[h].append(pred_data["error"])
                actual_by_h[h].append(pred_data["actual"])
                successful_predictions[h] += 1
    
    # Diagnostics
    print("\n" + "="*70)
    print("📊 BACKTEST DIAGNOSTICS")
    print("="*70)
    print(f"Total origins processed: {len(origins)}")
    print(f"Total errors: {total_errors}")
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
    horizons = [1, 2, 3, 7, 30, 90, 180, 365]
    
    # Load external regressors from ERA5 data
    print("\n📊 Loading climate data for feature engineering...")
    
    raw_nc = root / "raw_data" / "raw_nc"
    
    # Load all 5 climate variables
    runoff_df = pd.read_csv(raw_nc / "runoff" / "era5_runoff_mm.csv")
    runoff_df.columns = ['date', 'runoff_mm']
    
    snow_df = pd.read_csv(raw_nc / "snow" / "era5_snow_depth_m.csv")
    snow_df.columns = ['date', 'snow_depth_m']
    
    temp_df = pd.read_csv(raw_nc / "temperature" / "era5_t2m_c.csv")
    temp_df.columns = ['date', 'temperature_c']
    
    evap_df = pd.read_csv(raw_nc / "total_evaporation" / "era5_evap_mm.csv")
    evap_df.columns = ['date', 'evaporation_mm']
    
    precip_df = pd.read_csv(raw_nc / "total_precipitation" / "era5_precip_mm.csv")
    precip_df.columns = ['date', 'precipitation_mm']
    
    # Merge all climate variables
    climate_df = runoff_df
    for regr_df in [snow_df, temp_df, evap_df, precip_df]:
        climate_df = pd.merge(climate_df, regr_df, on='date', how='outer')
    
    # Sort by date
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df = climate_df.sort_values('date')
    
    # Forward fill missing values
    climate_df = climate_df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Loaded {len(climate_df)} rows of climate data:")
    print(f"   - runoff_mm")
    print(f"   - snow_depth_m")
    print(f"   - temperature_c")
    print(f"   - evaporation_mm")
    print(f"   - precipitation_mm")
    print(f"   Date range: {climate_df['date'].min()} to {climate_df['date'].max()}")
    
    # === FEATURE ENGINEERING ===
    print("\n🔧 Engineering features (lags + climate context + temporal)...")
    exog_df = prepare_exog_features(df, climate_df)
    
    print(f"✅ Created {len(exog_df.columns)} features:")
    print(f"   Total rows: {len(exog_df)}")
    print(f"   Date range: {exog_df.index.min()} to {exog_df.index.max()}")
    print(f"\n📋 Features:")
    print(f"   - Climate: temperature_c (current value)")
    print(f"   - History: 365+ days for annual pattern learning")
    
    print("\n" + "="*70)
    print("🔮 SARIMAX BACKTEST - Water Volume Forecasting")
    print("="*70)
    print(f"📈 Forecast horizons: {horizons} days")
    print(f"📊 Total features: {len(exog_df.columns)}")
    print(f"💡 Strategy: Temperature only + 730-day lookback window")
    print(f"💡 SARIMAX AR(1,1,1)x(1,1,1,12) handles seasonality")
    print("="*70)
    
    # Backtest SARIMAX
    metrics = backtest_sarimax_horizons(
        df,
        horizons,
        exog_df=exog_df,
        n_jobs=None,  # Will use CPU count - 1
        chunk_size=100,
        lookback_days=730
    )
    
    # Save results
    out_dir = root / "processed_data" / "water_balance_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "forecast_backtest_sarimax_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "="*70)
    print("✅ SARIMAX BACKTEST COMPLETE")
    print("="*70)
    
    # Print results by category
    ultra_short = [h for h in horizons if h <= 3]
    short_term = [h for h in horizons if 4 <= h <= 30]
    mid_term = [h for h in horizons if 31 <= h <= 180]
    long_term = [h for h in horizons if h > 180]
    
    def print_horizon_group(group_name: str, group_horizons: List[int]):
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
    
    # Compare with SARIMA and Prophet
    sarima_path = out_dir / "forecast_backtest_metrics.json"
    prophet_path = out_dir / "forecast_backtest_prophet_metrics.json"
    
    if sarima_path.exists() and prophet_path.exists():
        print("\n" + "="*70)
        print("⚔️ THREE-WAY COMPARISON: SARIMAX vs SARIMA vs Prophet")
        print("="*70)
        
        with open(sarima_path) as f:
            sarima_metrics = json.load(f)
        with open(prophet_path) as f:
            prophet_metrics = json.load(f)
        
        for h in horizons:
            if str(h) in sarima_metrics and h in prophet_metrics:
                print(f"\n{'='*70}")
                print(f"🔹 Horizon: {h} days")
                print(f"{'='*70}")
                
                sarimax_r2 = metrics[h]['r2']
                sarima_r2 = sarima_metrics[str(h)]['r2']
                prophet_r2 = prophet_metrics[h]['r2']
                
                print(f"\n  📈 R² Score:")
                print(f"     SARIMAX: {sarimax_r2:8.4f}")
                print(f"     SARIMA:  {sarima_r2:8.4f}")
                print(f"     Prophet: {prophet_r2:8.4f}")
                
                # Find winner
                best_r2 = max(sarimax_r2, sarima_r2, prophet_r2)
                if best_r2 == sarimax_r2:
                    print(f"     Winner:  🏆 SARIMAX")
                elif best_r2 == sarima_r2:
                    print(f"     Winner:  🏆 SARIMA")
                else:
                    print(f"     Winner:  🏆 Prophet")


if __name__ == "__main__":
    main()
