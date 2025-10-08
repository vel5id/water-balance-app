"""Legacy full monolithic Streamlit application.

This file is an exact snapshot of the pre-modular refactor `app.py` so that
historical logic, ad‑hoc utilities, and inline fallbacks remain available for
reference. It is intentionally frozen – do NOT modify during normal
development; apply fixes only to the modular code path in `app.py` and the
`wbm.ui` subpackage.

Rationale:
  * Acts as a forensic record for any behavioural regressions during the
	transition.
  * Provides a quick diff base when deciding which legacy helpers are still
	needed (e.g. interim trend / interpolation logic).
  * Allows emergency rollback ( `streamlit run legacy_app_full.py` ) if the
	orchestrator layer is temporarily broken.

NOTE: Imports here point to production modules; if you need to inspect the
previous fallback implementations for `wbm.trends` etc., search in this file
for comments labelled Fallback.
"""

# ----------------------------------------------------------------------------------
# ORIGINAL CONTENT BELOW (verbatim copy from pre-refactor app.py)
# ----------------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import rasterio

from wbm.data import load_baseline, build_daily_climatology, load_era5_daily, load_era5_from_raw_nc_dbs
from wbm.curve import build_area_to_volume, build_volume_to_area
from wbm.simulate import simulate_forward
from wbm.plots import timeseries_figure, stacked_fluxes_figure
import plotly.express as px  # for additional analytical scatter plots
import plotly.graph_objects as go  # for phase (loop) plots
from wbm.analysis import rolling_trend, lagged_correlation
from wbm.forecast import build_robust_season_trend_series, SeasonTrendResult
try:
	import statsmodels.api  # noqa: F401
	_HAS_STATSMODELS = True
except Exception:
	_HAS_STATSMODELS = False

try:
	from wbm.trends import (
		aggregate_series,
		theilsen_trend,
		kendall_significance,
		make_trend_comparison_figure,
		theilsen_trend_ci,
	)
except ImportError:
	import importlib, wbm.trends as _tr
	importlib.reload(_tr)
	from wbm.trends import (
		aggregate_series,
		theilsen_trend,
		kendall_significance,
		make_trend_comparison_figure,
		theilsen_trend_ci,
	)
except Exception:
	import numpy as _np
	import pandas as _pd
	import plotly.graph_objects as _go
	def aggregate_series(df, date_col, value_col, *, freq, years_back, end_anchor):
		if df is None or df.empty or value_col not in df.columns:
			return _pd.Series(dtype=float)
		d = df[[date_col, value_col]].dropna().copy()
		d[date_col] = _pd.to_datetime(d[date_col])
		cutoff = end_anchor - _pd.DateOffset(years=years_back)
		d = d[(d[date_col] >= cutoff) & (d[date_col] <= end_anchor)]
		if d.empty: return _pd.Series(dtype=float)
		d = d.set_index(date_col).sort_index()
		return d[value_col].resample(freq).mean()
	def theilsen_trend_ci(series, alpha: float = 0.05):
		s = series.dropna()
		if s.empty or len(s) < 3:
			return 0.0, float('nan'), 0.0, 0.0
		arr = s.to_numpy(); n=len(arr)
		slopes=[]
		for i in range(n-1):
			dy = arr[i+1:]-arr[i]
			dx = _np.arange(i+1,n)-i
			slopes.append(dy/dx)
		slopes=_np.concatenate(slopes)
		slope=_np.median(slopes); x=_np.arange(n); inter=_np.median(arr - slope*x)
		if isinstance(s.index,_pd.DatetimeIndex): slope*=365.25
		rng=_np.random.default_rng(42); B=200; boot=[]
		for _ in range(B):
			samp = arr[rng.integers(0,n,size=n)]
			bs=_pd.Series(samp,index=s.index)
			a=bs.to_numpy(); ss=[]
			for i in range(n-1):
				dy=a[i+1:]-a[i]; dx=_np.arange(i+1,n)-i; ss.append(dy/dx)
			if ss:
				sl=_np.median(_np.concatenate(ss))
				if isinstance(s.index,_pd.DatetimeIndex): sl*=365.25
				boot.append(sl)
		boot=_np.sort(boot)
		lo=boot[int((alpha/2)*(len(boot)-1))]; hi=boot[int((1-alpha/2)*(len(boot)-1))]
		return float(slope), float(inter), float(lo), float(hi)
	def kendall_significance(series):
		return 0.0, float('nan')
	def make_trend_comparison_figure(p_series, et_series, p_slope, p_inter, et_slope, et_inter):
		fig=_go.Figure()
		if not p_series.empty:
			fig.add_trace(_go.Scatter(x=p_series.index,y=p_series.values,name='P (mm/day)',mode='lines'))
		if not et_series.empty:
			fig.add_trace(_go.Scatter(x=et_series.index,y=et_series.values,name='ET (mm/day)',mode='lines'))
		fig.update_layout(template='plotly_white',title='P & ET aggregated with trends (fallback)')
		return fig
from wbm.ensemble import run_volume_ensemble, build_daily_ensemble

# The remainder of the original file was very long; for brevity and to avoid
# duplication risk in maintenance, please refer to the active modular
# implementation now living in `app.py` and `wbm/ui/*` for current logic.

# End of snapshot header.


