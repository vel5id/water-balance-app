"""Water Balance Model (wbm) utility package.

Includes original + reconstructed analytical components.

Modules:
  curve: area<->volume interpolators
  data: loading baseline and climatology
  simulate: forward scenario simulation
  plots: interactive Plotly figures
  analysis: rolling trends & lag correlations
  forecast: robust seasonal + Theil–Sen trend decomposition
  trends: Theil–Sen + Kendall significance helpers
  ensemble: residual bootstrap ensemble volume projections
"""

from . import curve, data, simulate, plots, analysis  # noqa: F401

# Lazy optional imports for reconstructed pieces; wrap to avoid hard failure if missing
try:  # forecast / trends / ensemble were in reconstructed package
	from . import forecast, trends, ensemble  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
	forecast = None  # type: ignore
	trends = None  # type: ignore
	ensemble = None  # type: ignore

__all__ = [
	"curve",
	"data",
	"simulate",
	"plots",
	"analysis",
	"forecast",
	"trends",
	"ensemble",
]
