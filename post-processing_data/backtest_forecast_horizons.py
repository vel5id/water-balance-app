from __future__ import annotations

# Set environment variable to suppress ALL warnings before any imports
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings("ignore")
# Specifically suppress convergence warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import pickle
import hashlib

import hydroeval as he
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error

# Optional progress bar (tqdm). Provide a safe fallback if not installed.
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable, total=None, desc: str | None = None):  # pragma: no cover
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
from wbm.forecast import build_sarima_model, build_sarima_model_with_params


# ==================== DATA VALIDATION ====================

def validate_input_data(df: pd.DataFrame, required_min_points: int = 90) -> Tuple[bool, List[str]]:
    """Validate input DataFrame for water balance forecasting.
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required columns
    if "date" not in df.columns:
        issues.append("❌ Missing required column: 'date'")
    if "volume_mcm" not in df.columns:
        issues.append("❌ Missing required column: 'volume_mcm'")
    
    if issues:
        return False, issues
    
    # Check data types
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        issues.append(f"❌ Cannot convert 'date' to datetime: {e}")
        return False, issues
    
    # Check for missing values
    null_dates = df["date"].isna().sum()
    null_volumes = df["volume_mcm"].isna().sum()
    
    if null_dates > 0:
        issues.append(f"⚠️  Found {null_dates} null dates ({null_dates/len(df)*100:.1f}%)")
    if null_volumes > 0:
        issues.append(f"⚠️  Found {null_volumes} null volumes ({null_volumes/len(df)*100:.1f}%)")
    
    # Check for duplicates
    dup_dates = df["date"].duplicated().sum()
    if dup_dates > 0:
        issues.append(f"⚠️  Found {dup_dates} duplicate dates")
    
    # Check volume range
    valid_volumes = df["volume_mcm"].dropna()
    if len(valid_volumes) > 0:
        min_vol, max_vol = valid_volumes.min(), valid_volumes.max()
        mean_vol = valid_volumes.mean()
        
        if min_vol < 0:
            issues.append(f"❌ Negative volumes found: min={min_vol:.2f}")
        
        # Check for outliers (beyond 5 std)
        std_vol = valid_volumes.std()
        outliers = ((valid_volumes - mean_vol).abs() > 5 * std_vol).sum()
        if outliers > 0:
            issues.append(f"⚠️  Found {outliers} potential outliers (>5σ from mean)")
        
        # Data range info
        issues.append(f"✅ Volume range: {min_vol:.1f} - {max_vol:.1f} млн.м³ (mean={mean_vol:.1f})")
    
    # Check temporal coverage
    clean_df = df.dropna(subset=["date"])
    if len(clean_df) > 0:
        date_range = (clean_df["date"].max() - clean_df["date"].min()).days
        issues.append(f"✅ Date range: {date_range} days ({clean_df['date'].min().date()} to {clean_df['date'].max().date()})")
        
        # Check for gaps
        clean_df = clean_df.sort_values("date")
        date_diffs = clean_df["date"].diff()
        large_gaps = (date_diffs > pd.Timedelta(days=7)).sum()
        if large_gaps > 0:
            issues.append(f"⚠️  Found {large_gaps} gaps >7 days in time series")
    
    # Minimum data requirement
    if len(df.dropna(subset=["date", "volume_mcm"])) < required_min_points:
        issues.append(
            f"❌ Insufficient data: need ≥{required_min_points} points, have {len(df.dropna(subset=['date', 'volume_mcm']))}"
        )
        return False, issues
    
    # All critical checks passed
    is_valid = not any(issue.startswith("❌") for issue in issues)
    return is_valid, issues


# ==================== MODEL CACHING ====================

class ModelCache:
    """Cache for SARIMA models to speed up backtesting."""
    
    def __init__(self, cache_dir: Path, max_cache_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        self.hits = 0
        self.misses = 0
        self.params_cache: Dict[str, Tuple] = {}  # In-memory cache for params
    
    def _get_cache_key(self, data: pd.Series) -> str:
        """Generate cache key from data hash."""
        data_str = f"{len(data)}_{data.index[0]}_{data.index[-1]}_{data.mean():.6f}_{data.std():.6f}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"sarima_{key}.pkl"
    
    def get_params(self, data: pd.Series) -> Optional[Tuple]:
        """Get cached SARIMA parameters for similar data."""
        key = self._get_cache_key(data)
        
        # Check in-memory cache first
        if key in self.params_cache:
            self.hits += 1
            return self.params_cache[key]
        
        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    params = pickle.load(f)
                self.params_cache[key] = params
                self.hits += 1
                return params
            except Exception as e:
                print(f"  ⚠️  Cache read error: {e}", flush=True)
        
        self.misses += 1
        return None
    
    def save_params(self, data: pd.Series, order: Tuple, seasonal_order: Tuple):
        """Save SARIMA parameters to cache."""
        key = self._get_cache_key(data)
        params = (order, seasonal_order)
        
        # Save to in-memory cache
        self.params_cache[key] = params
        
        # Save to disk cache
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
                pickle.dump(params, f)
            
            # Clean up old cache files if size limit exceeded
            self._cleanup_cache()
        except Exception as e:
            print(f"  ⚠️  Cache write error: {e}", flush=True)
    
    def _cleanup_cache(self):
        """Remove oldest cache files if total size exceeds limit."""
        try:
            cache_files = list(self.cache_dir.glob("sarima_*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            if total_size > self.max_cache_size_bytes:
                # Sort by modification time, oldest first
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                
                # Remove oldest files until under limit
                for f in cache_files:
                    if total_size <= self.max_cache_size_bytes * 0.8:  # Leave 20% buffer
                        break
                    size = f.stat().st_size
                    f.unlink()
                    total_size -= size
        except Exception as e:
            print(f"  ⚠️  Cache cleanup error: {e}", flush=True)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        # Calculate cache size - but do it more efficiently (cached value)
        try:
            cache_size_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("sarima_*.pkl"))
            cache_size_mb = cache_size_bytes / 1024**2
        except Exception:
            cache_size_mb = 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate_pct": round(hit_rate, 1),
            "cache_size_mb": round(cache_size_mb, 2)
        }


@dataclass
class HorizonMetrics:
    horizon_days: int
    n: int
    mse: float
    rmse: float
    mae: float
    bias: float
    mape_pct: float


def compute_metrics(err: pd.Series, actual: pd.Series) -> Dict[str, float]:
    e = err.dropna()
    n = int(e.shape[0])
    if n < 2:  # Need at least 2 points for correlation/variance
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

    # Calculate predictions from errors and actuals
    sim = (a + e).values
    eval = a.values

    # Calculate R², NSE, KGE
    # Suppress errors for cases with no variance etc.
    with np.errstate(all="ignore"):
        try:
            kge, r, alpha, beta = he.evaluator(he.kge, sim, eval)
            kge = float(kge[0])
            r2 = float(r[0] ** 2)
        except Exception:
            kge, r2 = float("nan"), float("nan")

        try:
            nse = float(he.evaluator(he.nse, sim, eval)[0])
        except Exception:
            nse = float("nan")

    return {
        "n": n, "mse": mse, "rmse": rmse, "mae": mae, "bias": bias,
        "mape_pct": mape_pct, "r2": r2, "nse": nse, "kge": kge
    }


def backtest_volume_horizons(
    df: pd.DataFrame, 
    horizons: List[int], 
    n_jobs: int = -1, 
    chunk_size: int = 500,  # Увеличено с 100 до 500 для 80GB RAM
    target_r2: float = 0.7,
    target_nse: float = 0.7,
    target_rmse_pct: float = 10.0,
    early_stop_after: int = 50,
    use_cache: bool = True,
    cache_size_gb: float = 5.0,
    min_history: int = 90,
    max_origins: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Backtest volume forecasts at given horizons using SARIMA with multiprocessing and early stopping.

    For each day t in history (excluding last max(h) days), we fit SARIMA on history up to t
    and read deterministic forecast at t+h. Compare to actual volume at t+h.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'volume_mcm' columns
    horizons : List[int]
        Forecast horizons in days (e.g., [1, 30, 365])
    n_jobs : int, optional
        Number of parallel jobs (-1 = all cores, 1 = sequential), by default -1
    chunk_size : int, optional
        Process backtest in chunks of this size to save memory, by default 500 (optimized for 80GB RAM)
    target_r2 : float, optional
        Target R² to achieve (early stop if exceeded), by default 0.7
    target_nse : float, optional
        Target NSE to achieve (early stop if exceeded), by default 0.7
    target_rmse_pct : float, optional
        Target RMSE as % of mean volume (early stop if below), by default 10.0
    early_stop_after : int, optional
        Check early stopping criteria after this many origins, by default 50
    use_cache : bool, optional
        Enable model parameter caching for speedup, by default True
    cache_size_gb : float, optional
        Maximum cache size in GB, by default 5.0
    """
    # ==================== VALIDATION ====================
    print("\n" + "="*70)
    print("🔍 VALIDATING INPUT DATA")
    print("="*70)
    
    is_valid, issues = validate_input_data(df, required_min_points=min_history)
    for issue in issues:
        print(f"  {issue}")
    
    if not is_valid:
        print("\n❌ Validation failed. Cannot proceed with backtest.\n")
        return {
            h: {
                "n": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan"), "mape_pct": float("nan"), "r2": float("nan"),
                "nse": float("nan"), "kge": float("nan")
            } for h in horizons
        }
    
    print("\n✅ Validation passed!\n")
    
    # ==================== SETUP ====================
    
    # ==================== SETUP ====================
    out: Dict[int, Dict[str, float]] = {}
    if df.empty or "date" not in df.columns or "volume_mcm" not in df.columns:
        return {
            h: {
                "n": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan"), "mape_pct": float("nan"), "r2": float("nan"),
                "nse": float("nan"), "kge": float("nan")
            } for h in horizons
        }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "volume_mcm"]).sort_values("date")
    # Build a proper datetime-indexed series and upsample to daily to handle sparse observations (e.g., Sentinel)
    vol = pd.Series(df["volume_mcm"].astype(float).values, index=pd.DatetimeIndex(df["date"]))
    # Ensure strictly increasing index
    vol = vol.sort_index()
    # Reindex to daily frequency so forecast targets (t+h) exist; then fill gaps conservatively
    original_points = len(vol)
    vol = vol.asfreq("D")
    # Forward-fill then back-fill to cover leading NaNs
    vol = vol.ffill().bfill()
    print(f"🗓️  Resampled to daily frequency: {original_points} → {len(vol)} points; date span {vol.index.min().date()}..{vol.index.max().date()}")

    # Initialize cache
    cache = None
    if use_cache:
        cache_dir = ROOT / "processed_data" / "water_balance_output" / "model_cache"
        cache = ModelCache(cache_dir, max_cache_size_gb=cache_size_gb)
        print(f"💾 Model caching: ENABLED (max {cache_size_gb}GB)")
    else:
        print(f"💾 Model caching: DISABLED")

    max_h = max(horizons)
    errors_by_h: Dict[int, List[float]] = {h: [] for h in horizons}
    actual_by_h: Dict[int, List[float]] = {h: [] for h in horizons}

    # Sliding origin: for each t up to len - max_h - 1
    dates = vol.index
    # Performance guard: subsample origins to keep runtime reasonable
    start_i = min_history  # require at least min_history points for stability
    end_i = len(vol) - max_h
    if end_i <= start_i:
        return {
            h: {
                "n": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "bias": float("nan"), "mape_pct": float("nan"), "r2": float("nan"),
                "nse": float("nan"), "kge": float("nan")
            } for h in horizons
        }
    # 🔧 УВЕЛИЧЕНО: Aim for ~1000 origins (было 800) для лучшей статистической значимости
    step = max(1, (end_i - start_i) // 1000)
    # progress logging
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    progress_dir = ROOT / "processed_data" / "water_balance_output"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / "forecast_backtest_progress.jsonl"
    samples_path = progress_dir / "forecast_backtest_samples.csv"
    best_params_path = progress_dir / "best_sarima_params.json"
    samples_rows: List[Dict[str, object]] = []
    best_sarima_params: Dict[str, object] = {}
    early_stopped = False
    
    # Calculate mean volume for RMSE normalization
    mean_volume = float(vol.mean()) if len(vol) > 0 else 1.0

    def log_progress(step_idx: int):
        try:
            summary = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "run_id": run_id,
                "step": step_idx,
                "total": total_steps,
                "pct": round(100.0 * (step_idx + 1) / max(1, total_steps), 2),
                "counts": {str(h): len(errors_by_h[h]) for h in horizons},
            }
            # lightweight running metrics
            metrics_so_far: Dict[str, Dict[str, float]] = {}
            for h in horizons:
                e = pd.Series(errors_by_h[h])
                a = pd.Series(actual_by_h[h])
                if len(e) >= 2 and len(a) >= 2:
                    metrics_so_far[str(h)] = compute_metrics(e, a)
            if metrics_so_far:
                summary["metrics"] = metrics_so_far
            
            # Add cache stats if available
            if cache:
                summary["cache"] = cache.get_stats()
            
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except Exception:
            # do not fail the run due to logging issues
            pass

    def _fit_one_origin(i: int) -> Dict[str, object]:
        """Fit model for one origin, return predictions for all horizons."""
        hist = vol.iloc[: i + 1]
        
        # Ensure frequency is explicitly set to avoid warnings
        if hist.index.freq is None:
            hist.index.freq = pd.infer_freq(hist.index) or 'D'
        
        result = {"i": i, "preds": {}, "error": None, "sarima_order": None, "sarima_seasonal_order": None, "cache_hit": False}
        try:
            # Try to get cached parameters
            cached_params = None
            if cache:
                cached_params = cache.get_params(hist)
            
            if cached_params:
                # Use cached parameters (FAST PATH)
                order, seasonal_order = cached_params
                result["cache_hit"] = True
                result["sarima_order"] = order
                result["sarima_seasonal_order"] = seasonal_order
                
                # Build model with known params - much faster than auto_arima
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    # Suppress convergence warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        model = SARIMAX(hist, order=order, seasonal_order=seasonal_order, 
                                       enforce_stationarity=False, enforce_invertibility=False)
                        fitted = model.fit(disp=False, maxiter=50, method='lbfgs')
                    forecast = fitted.forecast(steps=max_h)
                    
                    # Build date index for forecast
                    last_date = hist.index[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                                   periods=max_h, freq='D')
                    res_deterministic = pd.Series(forecast.values, index=forecast_dates)
                except Exception as e:
                    # Fallback to full model if cached params fail
                    print(f"  ⚠️  Cached params failed, using auto_arima: {e}", flush=True)
                    res_deterministic, model_info = build_sarima_model_with_params(
                        hist, future_days=max_h, min_history=min_history
                    )
                    result["sarima_order"] = model_info.get("order")
                    result["sarima_seasonal_order"] = model_info.get("seasonal_order")
            else:
                # Find optimal parameters (SLOW PATH)
                res_deterministic, model_info = build_sarima_model_with_params(
                    hist, future_days=max_h, min_history=min_history
                )
                order = model_info.get("order")
                seasonal_order = model_info.get("seasonal_order")
                result["sarima_order"] = order
                result["sarima_seasonal_order"] = seasonal_order
                
                # Cache the parameters for future use
                if cache and order and seasonal_order:
                    cache.save_params(hist, order, seasonal_order)
            
            for h in horizons:
                target_date = dates[i] + pd.Timedelta(days=h)
                if target_date not in vol.index:
                    continue
                try:
                    pred = float(res_deterministic.loc[target_date]) if target_date in res_deterministic.index else np.nan
                except Exception:
                    pred = np.nan
                actual = float(vol.loc[target_date])
                if np.isfinite(pred) and np.isfinite(actual):
                    result["preds"][h] = {
                        "target_date": target_date,
                        "pred": pred,
                        "actual": actual,
                        "error": pred - actual,
                    }
        except Exception as e:
            result["error"] = str(e)
        return result

    # Process in chunks to manage memory
    origins = list(range(start_i, end_i, step))
    if max_origins and max_origins > 0:
        origins = origins[:max_origins]
        print(f"🔎 Limiting origins to first {len(origins)} as requested (max-origins)")
    total_steps = len(origins)
    total_chunks = (total_steps + chunk_size - 1) // chunk_size
    
    # write initial progress snapshot so the file exists immediately
    log_progress(-1)
    
    # Calculate absolute RMSE threshold from percentage
    mean_vol = vol.mean()
    rmse_threshold = mean_vol * target_rmse_pct / 100.0
    
    print("="*70)
    print("🚀 STARTING BACKTEST")
    print("="*70)
    print(f"📊 Total origins: {total_steps} in {total_chunks} chunks")
    print(f"🔧 Parallelism: {n_jobs if n_jobs > 0 else 'all CPUs'} | Chunk size: {chunk_size}")
    print(f"💾 RAM optimized for: 80GB (chunk_size={chunk_size})")
    print(f"🎯 Early stopping: R²>{target_r2:.2f}, NSE>{target_nse:.2f}, RMSE<{rmse_threshold:.1f} млн.м³")
    print(f"� Best params will be saved and reused for speed\n")
    
    chunk_ranges = []
    for chunk_start in range(0, len(origins), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(origins))
        chunk_ranges.append((chunk_start, chunk_end))
    
    # Process chunks with progress bar
    for chunk_idx, (chunk_start, chunk_end) in enumerate(tqdm(chunk_ranges, desc="Chunks", unit="chunk", position=0)):
        chunk_origins = origins[chunk_start:chunk_end]
        
        # Parallel processing of this chunk with its own progress
        print(f"\n  📦 Chunk {chunk_idx+1}/{total_chunks}: processing {len(chunk_origins)} origins...", flush=True)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_fit_one_origin)(i) for i in tqdm(chunk_origins, desc=f"  Origins", leave=False, position=1)
        )
        
        # Aggregate results from this chunk
        for step_idx_in_chunk, result in enumerate(results):
            step_idx = chunk_start + step_idx_in_chunk
            i = result["i"]
            if result["error"]:
                print(f"  ⚠️  Skipping origin {step_idx+1}/{total_steps} due to error: {result['error']}", flush=True)
                log_progress(step_idx)
                continue
            
            for h, pred_data in result["preds"].items():
                errors_by_h[h].append(pred_data["error"])
                actual_by_h[h].append(pred_data["actual"])
                # collect lightweight sample rows roughly every 10th origin to limit size
                if (step_idx % 10) == 0:
                    samples_rows.append({
                        "run_id": run_id,
                        "origin_date": dates[i].strftime('%Y-%m-%d'),
                        "horizon_days": h,
                        "target_date": pred_data["target_date"].strftime('%Y-%m-%d'),
                        "pred": pred_data["pred"],
                        "actual": pred_data["actual"],
                        "error": pred_data["error"],
                    })
                    
            # Track SARIMA parameters if available
            if result.get("sarima_order") is not None and result.get("sarima_seasonal_order") is not None:
                origin_key = f"origin_{i}"
                best_sarima_params[origin_key] = {
                    "order": result["sarima_order"],
                    "seasonal_order": result["sarima_seasonal_order"],
                }
            
            # log progress every step for visibility
            log_progress(step_idx)
        
        # Print chunk summary
        total_collected = sum(len(errors_by_h[h]) for h in horizons)
        cache_hits = sum(1 for r in results if r.get("cache_hit", False))
        print(f"  ✅ Chunk {chunk_idx+1} done: {total_collected} total predictions | Cache hits: {cache_hits}/{len(results)}", flush=True)
        
        # Clear chunk results to free memory
        del results
        
        # === Check early stopping criteria ===
        total_processed = chunk_end
        if total_processed >= early_stop_after and not early_stopped:
            # Compute current metrics across all horizons
            all_errors = []
            all_actuals = []
            for h in horizons:
                all_errors.extend(errors_by_h[h])
                all_actuals.extend(actual_by_h[h])
            
            if len(all_errors) >= 50:  # Need at least 50 points
                errors_arr = np.array(all_errors)
                actuals_arr = np.array(all_actuals)
                preds_arr = actuals_arr + errors_arr
                
                try:
                    # Compute metrics
                    r2_current = r2_score(actuals_arr, preds_arr)
                    nse_current = he.evaluator(he.nse, preds_arr, actuals_arr)[0]
                    rmse_current = np.sqrt(mean_squared_error(actuals_arr, preds_arr))
                    rmse_pct_current = (rmse_current / mean_volume) * 100
                    
                    print(f"\n📊 Checking early stop criteria after {total_processed} origins:", flush=True)
                    print(f"   R²={r2_current:.4f} (target≥{target_r2:.2f})", flush=True)
                    print(f"   NSE={nse_current:.4f} (target≥{target_nse:.2f})", flush=True)
                    print(f"   RMSE%={rmse_pct_current:.2f}% (target≤{target_rmse_pct:.1f}%)", flush=True)
                    
                    # Check if all targets met
                    if (r2_current >= target_r2 and 
                        nse_current >= target_nse and 
                        rmse_pct_current <= target_rmse_pct):
                        print(f"\n✅ 🎉 TARGET METRICS ACHIEVED! Stopping early at origin {total_processed}.", flush=True)
                        early_stopped = True
                        
                        # Save best parameters
                        if best_sarima_params:
                            with open(best_params_path, 'w') as f:
                                json.dump(best_sarima_params, f, indent=2)
                            print(f"💾 Saved {len(best_sarima_params)} SARIMA parameter sets to {best_params_path}", flush=True)
                            
                            # Show summary of most common parameters
                            orders = [v["order"] for v in best_sarima_params.values()]
                            seasonal_orders = [v["seasonal_order"] for v in best_sarima_params.values()]
                            from collections import Counter
                            most_common_order = Counter(tuple(o) if isinstance(o, list) else o for o in orders).most_common(1)
                            most_common_seasonal = Counter(tuple(s) if isinstance(s, list) else s for s in seasonal_orders).most_common(1)
                            if most_common_order and most_common_seasonal:
                                print(f"📈 Most common SARIMA parameters:", flush=True)
                                print(f"   order (p,d,q): {most_common_order[0][0]} ({most_common_order[0][1]} times)", flush=True)
                                print(f"   seasonal (P,D,Q,m): {most_common_seasonal[0][0]} ({most_common_seasonal[0][1]} times)", flush=True)
                        
                        break  # Exit chunk loop
                except Exception as e:
                    print(f"  ⚠️  Error computing early stop metrics: {e}", flush=True)

    for h in horizons:
        e = pd.Series(errors_by_h[h])
        a = pd.Series(actual_by_h[h])
        out[h] = compute_metrics(e, a)
    
    # Print final cache statistics
    if cache:
        stats = cache.get_stats()
        print("\n" + "="*70)
        print("💾 CACHE STATISTICS")
        print("="*70)
        print(f"  Hits: {stats['hits']} | Misses: {stats['misses']} | Total: {stats['total']}")
        print(f"  Hit rate: {stats['hit_rate_pct']:.1f}%")
        print(f"  Cache size: {stats['cache_size_mb']:.1f} MB")
        print("="*70 + "\n")
    
    # Write samples if any
    try:
        if samples_rows:
            pd.DataFrame(samples_rows).to_csv(samples_path, index=False)
    except Exception:
        pass
    return out


def main():
    import argparse
    root = Path(__file__).resolve().parents[1]

    # Defaults: prefer Sentinel-derived CSV if present, else fall back to water_balance_final
    sentinel_csv = root / "processed_data" / "water_balance_output" / "sentinel_area_volume.csv"
    default_csv = sentinel_csv if sentinel_csv.exists() else (root / "processed_data" / "water_balance_output" / "water_balance_final.csv")

    parser = argparse.ArgumentParser(description="Backtest SARIMA on volume time series")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="Input CSV with columns: date, volume_mcm [and optional area_km2, source]")
    parser.add_argument("--n-jobs", type=int, default=-2, help="Parallel jobs (-1 all cores; -2 all but one)")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size for origins to control memory")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Enable SARIMA params cache")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable SARIMA params cache")
    parser.add_argument("--cache-size-gb", type=float, default=2.0, help="Max cache size (GB)")
    parser.add_argument("--min-history", type=int, default=90, help="Minimum history length for model fit")
    parser.add_argument("--early-stop-after", type=int, default=50, help="Check early stop after N origins")
    parser.add_argument("--max-origins", type=int, default=0, help="Optional cap on number of origins to process (0 = no cap)")
    parser.add_argument("--horizons", type=str, default="1,2,3,7,30,90,180,365", help="Comma-separated horizons in days")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    print("\n" + "="*70)
    print("📈 SARIMA BACKTEST - Extended Horizons")
    print("="*70)
    print(f"Input: {csv_path}")
    print(f"Horizons: {horizons} days")
    print("="*70 + "\n")

    # Wrap the backtest to optionally limit origins via environment hook
    # We pass through args for runtime controls
    metrics = backtest_volume_horizons(
        df,
        horizons,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        early_stop_after=args.early_stop_after,
        use_cache=args.use_cache,
        cache_size_gb=args.cache_size_gb,
        min_history=args.min_history,
    )

    out_path = root / "processed_data" / "water_balance_output" / "forecast_backtest_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "="*70)
    print("✅ BACKTEST COMPLETE")
    print("="*70)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("="*70)


if __name__ == "__main__":
    main()
