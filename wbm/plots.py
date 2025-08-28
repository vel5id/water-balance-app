from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


def timeseries_figure(baseline: pd.DataFrame | None, scenario: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if baseline is not None and not baseline.empty:
        fig.add_trace(go.Scatter(x=baseline["date"], y=baseline["volume_mcm"], name="Baseline Volume (mcm)",
                                 mode="lines", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=scenario["date"], y=scenario["volume_mcm"], name="Scenario Volume (mcm)",
                             mode="lines", line=dict(color="#d62728", width=2)))
    fig.update_layout(title="Reservoir Volume Over Time", xaxis_title="Date", yaxis_title="Volume (million m³)",
                      template="plotly_white")
    return fig


def stacked_fluxes_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["precipitation_volume_mcm"], name="P (mcm/day)", marker_color="#1f77b4"))
    fig.add_trace(go.Bar(x=df["date"], y=-df["evaporation_volume_mcm"], name="ET (mcm/day)", marker_color="#ff7f0e"))
    fig.update_layout(barmode="relative", title="Daily P and ET Volumes (mcm/day)", template="plotly_white")
    return fig
