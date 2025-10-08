from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from typing import Callable, Optional


def timeseries_figure(baseline: pd.DataFrame | None, scenario: pd.DataFrame, tr: Optional[Callable[[str], str]] = None) -> go.Figure:
    tr = tr or (lambda k, **_: k)
    fig = go.Figure()
    if baseline is not None and not baseline.empty:
        fig.add_trace(go.Scatter(x=baseline["date"], y=baseline["volume_mcm"], name=tr("baseline_volume_name"),
                                 mode="lines", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=scenario["date"], y=scenario["volume_mcm"], name=tr("scenario_volume_name"),
                             mode="lines", line=dict(color="#d62728", width=2)))
    fig.update_layout(title=tr("reservoir_timeseries_title"), xaxis_title=tr("date_axis"), yaxis_title=tr("volume_axis"),
                      template="plotly_white")
    return fig


def stacked_fluxes_figure(df: pd.DataFrame, tr: Optional[Callable[[str], str]] = None) -> go.Figure:
    tr = tr or (lambda k, **_: k)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["precipitation_volume_mcm"], name=tr("p_bar"), marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=df["date"], y=-df["evaporation_volume_mcm"], name=tr("et_bar"), marker_color="#ff7f0e"))
    fig.update_layout(barmode="relative", title=tr("daily_p_et_title"), template="plotly_white")
    return fig
