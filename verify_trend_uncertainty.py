import numpy as np
import pandas as pd
from wbm.forecast import build_robust_season_trend_series
from wbm.ensemble import build_daily_ensemble

print("--- Trend Uncertainty Verification ---")

# Synthetic data: Linear trend + Noise
dates = pd.date_range("2000-01-01", "2010-12-31", freq="D")
x = np.arange(len(dates))
rng = np.random.default_rng(42)
y = 0.01 * x + rng.normal(0, 5, len(dates)) # Slope 0.01
series = pd.Series(y, index=dates)

# Forecast 10 years out
future_days = 365 * 10
res = build_robust_season_trend_series(series, future_days=future_days, transformation="none")

# Build Ensemble (Linear)
members = build_daily_ensemble(res.deterministic, res.residuals, n_members=100, random_state=42)

# Analyze Spread at Start vs End
# Ensemble dataframe
ens_df = pd.DataFrame(members).T # (days, members)
spread = ens_df.quantile(0.95, axis=1) - ens_df.quantile(0.05, axis=1)

start_spread = spread.iloc[0]
end_spread = spread.iloc[-1]

print(f"Spread Start: {start_spread:.4f}")
print(f"Spread End:   {end_spread:.4f}")

ratio = end_spread / start_spread
print(f"Ratio End/Start: {ratio:.4f}")

if ratio < 1.1:
    print("FAIL: Spread is constant. Trend uncertainty is IGNORED.")
else:
    print("PASS: Spread grows. Trend uncertainty is captured.")
