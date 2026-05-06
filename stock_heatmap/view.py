"""
Stock Heatmap tab — TradingView-style treemap.

Features:
  - Index selector: S&P 500 / Nasdaq 100 / Dow Jones / Russell 2000 (proxy)
  - Size by: Market Cap / Volume
  - Color by: 1D / 1W / 1M / 3M / YTD return
  - Group by: Sector / Industry / None
  - Filters: Sector (multi-select), Min market cap, Search by ticker/name

Render uses Plotly's go.Treemap with a custom sector → ticker hierarchy.
Color is a diverging green/red scale centered on 0% with caps at ±5/10/30%
depending on the chosen timeframe (so 1D doesn't get washed out by YTD outliers).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_heatmap.data_yf import get_heatmap_data


# Theming constants — matched to the rest of the app
BG = "#0a0a0a"
TEXT_DIM = "#888"

# Per-timeframe color scale caps (percent) — 1D is naturally smaller than YTD
COLOR_CAPS = {
    "1D": 3.0,
    "1W": 7.0,
    "1M": 15.0,
    "3M": 25.0,
    "YTD": 35.0,
}


def _format_marketcap(v: float) -> str:
    if pd.isna(v) or v == 0:
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"


def render_stock_heatmap():
    """Render the Stock Heatmap section. Public entry point called from app.py."""
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">STOCK HEATMAP</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Treemap of index constituents · sized and colored by selectable metrics · data: Yahoo Finance
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Top control row — TradingView-style toolbar
    # -------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    with c1:
        index_name = st.selectbox(
            "Index",
            ["S&P 500", "Nasdaq 100", "Dow Jones 30", "Russell 2000 (Top 50 proxy)"],
            index=1,
            help="Universe of stocks to display. S&P 500 is fetched live from Wikipedia; "
                 "others use a snapshot list. Russell 2000 uses a 50-name proxy "
                 "(full 2000 would overwhelm yfinance).",
        )

    with c2:
        size_by = st.selectbox(
            "Size by",
            ["MarketCap", "AvgVolume", "Equal"],
            index=0,
            format_func=lambda x: {
                "MarketCap": "Market Cap",
                "AvgVolume": "Avg Volume",
                "Equal": "Equal Size",
            }[x],
        )

    with c3:
        color_by = st.selectbox(
            "Color by",
            ["1D", "1W", "1M", "3M", "YTD"],
            index=0,
            help="Performance window used for color intensity.",
        )

    with c4:
        group_by = st.selectbox(
            "Group by",
            ["Sector", "Industry", "None"],
            index=0,
        )

    # ----------------------------------------------------------------
    # Load / refresh button
    # Public app — protect yfinance from abuse via two layers:
    #   1. The @st.cache_data on fetch_prices has a 1-hour TTL shared
    #      across ALL users of this deployment. So the first click in
    #      an hour fetches; everyone else gets cached results instantly.
    #   2. Per-user 60-second cooldown between clicks (anti-spam).
    # ----------------------------------------------------------------
    import time

    cache_key = f"heatmap_data_{index_name}"
    last_load_key = f"heatmap_last_load_{index_name}"
    cooldown_seconds = 60

    needs_load = cache_key not in st.session_state or st.session_state[cache_key] is None
    last_load = st.session_state.get(last_load_key, 0)
    seconds_since_last = time.time() - last_load
    in_cooldown = (not needs_load) and seconds_since_last < cooldown_seconds

    cl, cr = st.columns([2, 5])
    with cl:
        if in_cooldown:
            wait = int(cooldown_seconds - seconds_since_last)
            st.button(
                f"⟳ Refresh available in {wait}s",
                use_container_width=True, disabled=True,
            )
            clicked = False
        else:
            clicked = st.button(
                "⟳ Load data" if needs_load else "⟳ Refresh data",
                use_container_width=True,
                type="primary" if needs_load else "secondary",
            )

    with cr:
        if not needs_load and not in_cooldown:
            st.caption(
                "💡 Data is shared across all users via 1-hour cache · "
                "click Refresh to force a re-fetch from Yahoo Finance."
            )

    if clicked:
        result = get_heatmap_data(index_name)
        st.session_state[cache_key] = result
        st.session_state[last_load_key] = time.time()
        if result is None or result.empty:
            return

    if needs_load:
        st.info(
            f"Click **Load data** to fetch {index_name} from Yahoo Finance. "
            "First load per hour takes 30-90s; subsequent visits within the hour "
            "use cached data and are instant."
        )
        return

    df = st.session_state[cache_key]
    if df is None or df.empty:
        st.error("No data returned. Try refreshing or choosing a different index.")
        return

    # -------------------------------------------------------------------
    # Filters row
    # -------------------------------------------------------------------
    fc1, fc2, fc3 = st.columns([3, 2, 2])

    with fc1:
        all_sectors = sorted([s for s in df["Sector"].unique() if s and s != "Unknown"])
        sectors_filter = st.multiselect(
            "Filter sectors",
            all_sectors,
            default=[],
            placeholder="All sectors",
        )

    with fc2:
        min_cap_b = st.number_input(
            "Min market cap ($B)",
            min_value=0.0, max_value=5000.0, value=0.0, step=1.0,
            help="Hide stocks with market cap below this threshold.",
        )

    with fc3:
        search = st.text_input(
            "Search ticker or name",
            placeholder="e.g. NVDA, semi…",
        )

    # Apply filters
    view = df.copy()
    if sectors_filter:
        view = view[view["Sector"].isin(sectors_filter)]
    if min_cap_b > 0:
        view = view[view["MarketCap"] >= min_cap_b * 1e9]
    if search.strip():
        q = search.strip().lower()
        view = view[
            view["Ticker"].str.lower().str.contains(q, na=False)
            | view["Name"].str.lower().str.contains(q, na=False)
        ]

    # Drop rows with missing color metric
    view = view.dropna(subset=[color_by]).copy()

    # Drop rows with zero/missing size (treemap can't render them)
    if size_by == "Equal":
        view["_size"] = 1.0
    else:
        view["_size"] = view[size_by].fillna(0)
        view = view[view["_size"] > 0]

    if view.empty:
        st.warning("No stocks match the current filters.")
        return

    # -------------------------------------------------------------------
    # Build the treemap
    # -------------------------------------------------------------------
    cap = COLOR_CAPS[color_by]

    # Hierarchy: optional grouping → ticker
    if group_by == "Sector":
        labels = ["All"] + view["Sector"].unique().tolist() + view["Ticker"].tolist()
        parents = [""] + ["All"] * len(view["Sector"].unique()) + view["Sector"].tolist()
        # Sector aggregate sizes (sum of children)
        sector_sizes = view.groupby("Sector")["_size"].sum().to_dict()
        sector_sizes_list = [sector_sizes[s] for s in view["Sector"].unique()]
        values = [sum(sector_sizes_list)] + sector_sizes_list + view["_size"].tolist()
        # Sector aggregate colors (cap-weighted mean of children's color metric)
        sector_colors = (
            view.groupby("Sector")
            .apply(lambda g: (g[color_by] * g["_size"]).sum() / g["_size"].sum())
            .to_dict()
        )
        sector_colors_list = [sector_colors[s] for s in view["Sector"].unique()]
        # Top "All" gets the overall weighted average
        overall_color = (view[color_by] * view["_size"]).sum() / view["_size"].sum()
        colors = [overall_color] + sector_colors_list + view[color_by].tolist()

        # Hover and label text per node
        ticker_text = [
            f"<b>{r['Ticker']}</b><br>{r[color_by]:+.2f}%"
            for _, r in view.iterrows()
        ]
        text = ["All"] + list(view["Sector"].unique()) + ticker_text

        ticker_hover = [
            f"<b>{r['Ticker']}</b> — {r['Name']}<br>"
            f"Sector: {r['Sector']}<br>"
            f"Market Cap: {_format_marketcap(r['MarketCap'])}<br>"
            f"{color_by}: {r[color_by]:+.2f}%"
            f"<extra></extra>"
            for _, r in view.iterrows()
        ]
        sector_hover = [
            f"<b>{s}</b><br>"
            f"{(view['Sector']==s).sum()} stocks<br>"
            f"Avg {color_by}: {sector_colors[s]:+.2f}%"
            f"<extra></extra>"
            for s in view["Sector"].unique()
        ]
        hovers = [f"All<extra></extra>"] + sector_hover + ticker_hover

    elif group_by == "Industry":
        labels = ["All"] + view["Industry"].unique().tolist() + view["Ticker"].tolist()
        parents = [""] + ["All"] * len(view["Industry"].unique()) + view["Industry"].tolist()
        ind_sizes = view.groupby("Industry")["_size"].sum().to_dict()
        ind_sizes_list = [ind_sizes[s] for s in view["Industry"].unique()]
        values = [sum(ind_sizes_list)] + ind_sizes_list + view["_size"].tolist()
        ind_colors = (
            view.groupby("Industry")
            .apply(lambda g: (g[color_by] * g["_size"]).sum() / g["_size"].sum())
            .to_dict()
        )
        ind_colors_list = [ind_colors[s] for s in view["Industry"].unique()]
        overall_color = (view[color_by] * view["_size"]).sum() / view["_size"].sum()
        colors = [overall_color] + ind_colors_list + view[color_by].tolist()
        ticker_text = [
            f"<b>{r['Ticker']}</b><br>{r[color_by]:+.2f}%" for _, r in view.iterrows()
        ]
        text = ["All"] + list(view["Industry"].unique()) + ticker_text
        ticker_hover = [
            f"<b>{r['Ticker']}</b> — {r['Name']}<br>"
            f"Industry: {r['Industry']}<br>"
            f"Market Cap: {_format_marketcap(r['MarketCap'])}<br>"
            f"{color_by}: {r[color_by]:+.2f}%<extra></extra>"
            for _, r in view.iterrows()
        ]
        ind_hover = [
            f"<b>{s}</b><br>"
            f"{(view['Industry']==s).sum()} stocks<br>"
            f"Avg {color_by}: {ind_colors[s]:+.2f}%<extra></extra>"
            for s in view["Industry"].unique()
        ]
        hovers = [f"All<extra></extra>"] + ind_hover + ticker_hover

    else:  # No grouping — flat treemap of tickers
        labels = ["All"] + view["Ticker"].tolist()
        parents = [""] + ["All"] * len(view)
        values = [view["_size"].sum()] + view["_size"].tolist()
        overall_color = (view[color_by] * view["_size"]).sum() / view["_size"].sum()
        colors = [overall_color] + view[color_by].tolist()
        text = ["All"] + [
            f"<b>{r['Ticker']}</b><br>{r[color_by]:+.2f}%"
            for _, r in view.iterrows()
        ]
        ticker_hover = [
            f"<b>{r['Ticker']}</b> — {r['Name']}<br>"
            f"Market Cap: {_format_marketcap(r['MarketCap'])}<br>"
            f"{color_by}: {r[color_by]:+.2f}%<extra></extra>"
            for _, r in view.iterrows()
        ]
        hovers = ["All<extra></extra>"] + ticker_hover

    # Diverging green/red colorscale
    colorscale = [
        [0.0,  "#7f1d1d"],   # deep red
        [0.25, "#dc2626"],
        [0.45, "#fca5a5"],
        [0.5,  "#262626"],   # neutral / zero
        [0.55, "#86efac"],
        [0.75, "#16a34a"],
        [1.0,  "#14532d"],   # deep green
    ]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=text,
        textinfo="text",
        textfont=dict(family="Inter, sans-serif", size=11, color="#ffffff"),
        marker=dict(
            colors=colors,
            colorscale=colorscale,
            cmid=0,
            cmin=-cap,
            cmax=cap,
            line=dict(color="#0a0a0a", width=1),
            colorbar=dict(
                title=dict(text=f"<i>{color_by} %</i>",
                           font=dict(size=10, color="#aaa")),
                tickfont=dict(size=9, color=TEXT_DIM),
                outlinewidth=0,
                thickness=10,
                len=0.5,
                ticksuffix="%",
            ),
        ),
        hovertemplate=hovers,
        branchvalues="total",
        pathbar=dict(visible=True, thickness=18,
                     textfont=dict(family="Inter", size=10, color="#aaa")),
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color="#e0e0e0", size=11),
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # -------------------------------------------------------------------
    # Footer stats
    # -------------------------------------------------------------------
    n_shown = len(view)
    n_total = len(df)
    pos = (view[color_by] > 0).sum()
    neg = (view[color_by] < 0).sum()
    breadth = pos / n_shown * 100 if n_shown else 0
    avg = view[color_by].mean()

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.metric("Stocks shown", f"{n_shown} / {n_total}")
    with fc2:
        st.metric(f"{color_by} Breadth", f"{pos} / {neg}", f"{breadth:.0f}% positive",
                  delta_color="off")
    with fc3:
        leader_idx = view[color_by].idxmax()
        leader = view.loc[leader_idx]
        st.metric(f"{color_by} Leader", leader["Ticker"], f"+{leader[color_by]:.2f}%")
    with fc4:
        laggard_idx = view[color_by].idxmin()
        laggard = view.loc[laggard_idx]
        st.metric(f"{color_by} Laggard", laggard["Ticker"], f"{laggard[color_by]:.2f}%",
                  delta_color="inverse")

    st.caption(
        f"Color scale capped at ±{cap:.0f}% for visual contrast. "
        f"Stocks beyond this range still show the extreme color. "
        f"Data via Yahoo Finance — may be 15min delayed."
    )
