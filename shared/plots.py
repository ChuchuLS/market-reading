"""
Shared plotting helpers used across all dashboard pages.

The Regime Timeline used to be implemented per-page with horizontal go.Bar
traces (base + width_ms on a date axis). That pattern was fragile and
repeatedly rendered blank. This module centralizes a single, robust
implementation built on go.Scatter thick horizontal lines, which size
reliably on a date axis.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_regime_timeline(
    runs,
    color_func,
    x_min,
    x_max,
    height=120,
    dark_layout=None,
    text_dim="#9ca3af",
):
    """
    Plot regime runs as continuous horizontal colored time bands.

    Parameters
    ----------
    runs : DataFrame
        Output of regime_runs(regimes); columns Start, End, Regime, Duration.
    color_func : callable
        Maps a regime label (str) to a color (str).
    x_min, x_max : date-like
        Bounds of the x-axis (typically regimes.index.min()/max()).
    height : int
        Chart height in px.
    dark_layout : dict | None
        Base layout dict (e.g. theming.DARK_LAYOUT) merged into the figure.
    text_dim : str
        Tick label color.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure (caller is responsible for st.plotly_chart).

    Notes
    -----
    Each run is drawn as a single thick horizontal line from its Start to
    End + 1 day (so a one-day run still has visible width). Using Scatter
    lines instead of Bars avoids the Timedelta/base/barmode pitfalls that
    left the old timelines blank.
    """
    fig = go.Figure()

    if runs is None or len(runs) == 0:
        # Empty but valid figure; keeps callers simple.
        layout = dict(dark_layout or {})
        fig.update_layout(
            **layout,
            height=height,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=35),
            yaxis=dict(visible=False, range=[-1, 1], fixedrange=True),
            xaxis=dict(
                type="date", showgrid=False, tickfont=dict(size=10, color=text_dim)
            ),
        )
        return fig

    for _, run in runs.iterrows():
        start = pd.to_datetime(run["Start"])
        end = pd.to_datetime(run["End"])
        end_exclusive = end + pd.Timedelta(days=1)

        line = dict(color=color_func(run["Regime"]), width=26)
        fig.add_trace(
            go.Scatter(
                x=[start, end_exclusive],
                y=[0, 0],
                mode="lines",
                line=line,
                hovertemplate=(
                    f"{run['Regime']}<br>"
                    f"{start.strftime('%Y-%m-%d')} \u2192 "
                    f"{end.strftime('%Y-%m-%d')}<br>"
                    f"{int(run['Duration'])} day(s)"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    layout = dict(dark_layout or {})
    fig.update_layout(
        **layout,
        height=height,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=35),
        yaxis=dict(visible=False, range=[-1, 1], fixedrange=True),
        xaxis=dict(
            type="date",
            range=[
                pd.to_datetime(x_min),
                pd.to_datetime(x_max) + pd.Timedelta(days=1),
            ],
            showgrid=False,
            tickfont=dict(size=10, color=text_dim),
        ),
    )
    return fig
