from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    # Parse numeric columns
    res = pd.to_numeric(df.get("residual_mcm"), errors="coerce").dropna()
    if res.empty:
        raise ValueError("No residual_mcm data to compute stats.")
    n = int(res.shape[0])
    bias = float(res.mean())
    # Sample std (ddof=1)
    std = float(res.std(ddof=1)) if n > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(res))))
    mae = float(np.mean(np.abs(res)))
    medae = float(np.median(np.abs(res)))
    q90_abs = float(np.quantile(np.abs(res), 0.90))
    # Residual Standard Error (sigma_hat) ~ sqrt(SSE/(n-1))
    sse = float(np.sum(np.square(res - bias)))
    rse_sigma = float(np.sqrt(sse / (n - 1))) if n > 1 else float("nan")
    # Normalizations by volume scale
    vol = pd.to_numeric(df.get("volume_mcm"), errors="coerce").dropna()
    nrmse_mean_pct = float("nan")
    nrmse_range_pct = float("nan")
    if not vol.empty:
        vmean = float(np.mean(vol))
        vrange = float(np.max(vol) - np.min(vol))
        if vmean:
            nrmse_mean_pct = float(rmse / vmean * 100.0)
        if vrange:
            nrmse_range_pct = float(rmse / vrange * 100.0)
    return {
        "n": n,
        "bias_mean_mcm": bias,
        "std_mcm": std,
        "rmse_mcm": rmse,
        "mae_mcm": mae,
        "medae_mcm": medae,
        "q90_abs_residual_mcm": q90_abs,
        "RSE_sigma_mcm": rse_sigma,
        "NRMSE_mean_volume_pct": nrmse_mean_pct,
        "NRMSE_range_pct": nrmse_range_pct,
        "source": str(csv_path),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "processed_data" / "water_balance_output"
    csv = out_dir / "water_balance_final.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing CSV: {csv}")
    metrics = compute_stats(csv)
    out_json = out_dir / "model_error_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
