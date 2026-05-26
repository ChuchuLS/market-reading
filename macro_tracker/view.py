"""
Macro Tracker section — BQL-driven cross-asset performance dashboard.

Reads data/MARKET_DATA.xlsx (sheet: macro_tracker). Renders 4 internal sub-tabs:
  - Heatmap (sortable color-coded table)
  - Leaders & Laggards (top/bottom horizontal bars)
  - Category Rotation (grouped bars + category averages)
  - Cross-Timeframe (scatter rotation map)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theming import (
    BG,
    GRID,
    TEXT,
    TEXT_DIM,
    GREEN,
    RED,
    AMBER,
    BLUE,
    CATEGORY_COLORS,
    DARK_LAYOUT,
    COLS,
)

# ---------------------------------------------------------------------------
# Data file location — relative to this module
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "data" / "MARKET_DATA.xlsx"
SHEET_NAME = "macro_tracker"


# ---------------------------------------------------------------------------
# Ticker → (Name, Category) registry
# ---------------------------------------------------------------------------
# Categories are grouped into ETF buckets (above the line) and commodity
# buckets (below). The CATEGORY_ORDER list below controls display ordering.
TICKER_MAP = {
    # --- Tech (8) ---
    "SMH US Equity": ("Semiconductors", "Tech"),
    "IGV US Equity": ("Software", "Tech"),
    "SKYY US Equity": ("Cloud Computing", "Tech"),
    "HACK US Equity": ("Cybersecurity", "Tech"),
    "BOTZ US Equity": ("AI", "Tech"),
    "ROBO US Equity": ("Robotics", "Tech"),
    "QTUM US Equity": ("Quantum", "Tech"),
    "SOCL US Equity": ("Social Media", "Tech"),
    # --- Materials & Mining (7) ---
    "SIL US Equity": ("Silver Miners", "Materials & Mining"),
    "GDX US Equity": ("Gold Miners", "Materials & Mining"),
    "COPX US Equity": ("Copper Miners", "Materials & Mining"),
    "REMX US Equity": ("Rare Earth Metals", "Materials & Mining"),
    "LIT US Equity": ("Lithium / Battery", "Materials & Mining"),
    "SLX US Equity": ("Steel", "Materials & Mining"),
    "XLB US Equity": ("Materials", "Materials & Mining"),
    # --- Energy & Power (3) ---
    "XLE US Equity": ("Energy ETF", "Energy & Power"),
    "URA US Equity": ("Uranium / Nuclear", "Energy & Power"),
    "XLU US Equity": ("Utilities", "Energy & Power"),
    # --- Defensive (4) ---
    "XLP US Equity": ("Consumer Staples", "Defensive"),
    "XLV US Equity": ("Health Care", "Defensive"),
    "IHI US Equity": ("Medical Devices", "Defensive"),
    "IYZ US Equity": ("Telecom", "Defensive"),
    # --- Cyclical (8) ---
    "XLY US Equity": ("Consumer Discretionary", "Cyclical"),
    "XRT US Equity": ("Retail", "Cyclical"),
    "XHB US Equity": ("Homebuilders", "Cyclical"),
    "XLI US Equity": ("Industrials", "Cyclical"),
    "IYT US Equity": ("Transports", "Cyclical"),
    "JETS US Equity": ("Airlines", "Cyclical"),
    "DJUSCA Index": ("Casinos", "Cyclical"),
    "ITA US Equity": ("Aerospace & Defense", "Cyclical"),
    # --- Growth & Bio (3) ---
    "VUG US Equity": ("Growth Stocks", "Growth & Bio"),
    "ARKG US Equity": ("Genomics", "Growth & Bio"),
    "XBI US Equity": ("Biotech", "Growth & Bio"),
    # --- Speculative (2) ---
    "WGMI US Equity": ("Bitcoin Miners", "Speculative"),
    "ARKX US Equity": ("Space", "Speculative"),
    # --- International (2) ---
    "EEM US Equity": ("Emerging Markets", "International"),
    "KWEB US Equity": ("China Internet", "International"),
    # --- Real Estate (1) ---
    "XLRE US Equity": ("Real Estate", "Real Estate"),
    # --- Financials (1) ---
    "KRE US Equity": ("Regional Banks", "Financials"),
    # --- Bonds (1) ---
    "TLT US Equity": ("Long Term Treasuries", "Bonds"),
    # --- Commodities ---
    "GC1 Comdty": ("Gold", "Precious Metals"),
    "SI1 Comdty": ("Silver", "Precious Metals"),
    "PL1 Comdty": ("Platinum", "Precious Metals"),
    "PA1 Comdty": ("Palladium", "Precious Metals"),
    "HG1 Comdty": ("Copper", "Industrial Metals"),
    "CL1 Comdty": ("WTI Crude Oil", "Energy"),
    "CO1 Comdty": ("Brent Crude", "Energy"),
    "NG1 Comdty": ("Natural Gas", "Energy"),
    "HO1 Comdty": ("Heating Oil", "Energy"),
    "XB1 Comdty": ("Gasoline (RBOB)", "Energy"),
    "COAL US Equity": ("Coal", "Energy"),
    "KC1 Comdty": ("Coffee", "Softs"),
    "SB1 Comdty": ("Sugar", "Softs"),
    "CC1 Comdty": ("Cocoa", "Softs"),
    "CT1 Comdty": ("Cotton", "Softs"),
    "WOOD US Equity": ("Lumber", "Softs"),
    "LH1 Comdty": ("Lean Hogs", "Livestock"),
}

# Display order — controls how categories appear in dropdowns and table sorts
CATEGORY_ORDER_ETF = [
    "Tech",
    "Materials & Mining",
    "Energy & Power",
    "Defensive",
    "Cyclical",
    "Growth & Bio",
    "Speculative",
    "International",
    "Real Estate",
    "Financials",
    "Bonds",
]
CATEGORY_ORDER_COMDTY = [
    "Precious Metals",
    "Industrial Metals",
    "Energy",
    "Softs",
    "Livestock",
]
CATEGORY_ORDER = CATEGORY_ORDER_ETF + CATEGORY_ORDER_COMDTY


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path, _mtime: float) -> pd.DataFrame:
    """Read MARKET_DATA.xlsx (sheet: macro_tracker) and return tidy DataFrame keyed by ticker.

    Tolerates BQL output that includes an auto-added DATES column between
    the ticker column and the metric columns.
    """
    raw = pd.read_excel(path, sheet_name=SHEET_NAME)

    # BQL =BQL(range, "metrics") output adds a DATES column between the
    # ticker column and the metric columns. Detect and drop it so the
    # loader works whether the user pasted output with or without it.
    if len(raw.columns) >= 2:
        second = raw.iloc[:, 1]
        # Datetime dtype OR header literally named "DATES" → it's the BQL date column
        is_dates_col = (
            pd.api.types.is_datetime64_any_dtype(second)
            or str(raw.columns[1]).strip().upper() == "DATES"
        )
        if is_dates_col:
            raw = raw.drop(raw.columns[1], axis=1)

    raw = raw.iloc[:, :6].copy()
    raw.columns = ["Ticker", "1D", "1W", "1M", "3M", "YTD"]
    raw = raw.dropna(subset=["Ticker"]).reset_index(drop=True)

    rows = []
    for _, r in raw.iterrows():
        ticker = str(r["Ticker"]).strip()
        if ticker in TICKER_MAP:
            name, cat = TICKER_MAP[ticker]
            # Classify instrument type from the Bloomberg suffix.
            if ticker.endswith("Comdty"):
                itype = "Comdty"
            else:
                itype = "ETF"

            rows.append(
                {
                    "Category": cat,
                    "Name": name,
                    "Ticker": ticker,
                    "InstrumentType": itype,
                    "1D": float(r["1D"]),
                    "1W": float(r["1W"]),
                    "1M": float(r["1M"]),
                    "3M": float(r["3M"]),
                    "YTD": float(r["YTD"]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def cell_color(v: float, cap: float = 30.0) -> str:
    if pd.isna(v) or v == 0:
        return "rgba(60,60,65,0.4)"
    t = max(-1, min(1, v / cap))
    intensity = abs(t)
    if t >= 0:
        return f"hsl(140, {35 + intensity * 45}%, {88 - intensity * 50}%)"
    else:
        return f"hsl(355, {40 + intensity * 50}%, {88 - intensity * 50}%)"


def text_color_for(v: float, cap: float = 30.0) -> str:
    if pd.isna(v) or v == 0:
        return "#999"
    intensity = min(1, abs(v / cap))
    return "#fff" if intensity > 0.45 else ("#0a3d18" if v > 0 else "#4a0d12")


# ===========================================================================
# Public render entry point
# ===========================================================================
def render_macro_tracker():
    """Render the entire Macro Tracker section (header + 4 internal tabs)."""

    if not DATA_PATH.exists():
        st.error(
            f"MARKET_DATA.xlsx not found at {DATA_PATH} (sheet: macro_tracker). "
            "Place your BQL output there and click the sidebar Refresh."
        )
        return

    mtime = DATA_PATH.stat().st_mtime
    df = load_data(DATA_PATH, mtime)
    last_updated = datetime.fromtimestamp(mtime).strftime("%b %d, %Y · %H:%M")

    # -------------------------------------------------------------------
    # Sidebar controls — only shown while Macro Tracker is the active tab
    # -------------------------------------------------------------------
    with st.sidebar:
        st.markdown("**MACRO TRACKER FILTERS**")
        all_cats = ["All"] + sorted(df["Category"].unique().tolist())
        cat_filter = st.selectbox(
            "Category", all_cats, index=0, label_visibility="collapsed", key="mt_cat"
        )
        search = st.text_input(
            "Search",
            placeholder="Filter name or ticker…",
            label_visibility="collapsed",
            key="mt_search",
        )
        sort_col = "YTD"
        sort_dir = "Descending"

        st.markdown("---")
        st.markdown("**DATA**")
        st.caption(f"📁 {DATA_PATH.name}")
        st.caption(f"🕐 Updated: {last_updated}")
        st.caption(f"🔢 {len(df)} instruments")
        if st.button("↻ Refresh BQL data", use_container_width=True, key="mt_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Apply filters
    view = df.copy()
    if cat_filter != "All":
        view = view[view["Category"] == cat_filter]
    if search.strip():
        q = search.strip().lower()
        view = view[
            view["Name"].str.lower().str.contains(q, na=False)
            | view["Ticker"].str.lower().str.contains(q, na=False)
            | view["Category"].str.lower().str.contains(q, na=False)
        ]
    view = view.sort_values(sort_col, ascending=(sort_dir == "Ascending")).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------
    # Header banner
    # -------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="background:#0a0a0a;padding:1rem 1.25rem;margin:0 0 1rem 0;
                    border-bottom:1px solid #1a1a1a;color:#fff;">
          <div style="font-size:11px;color:#4ade80;letter-spacing:0.2em;
                      text-transform:uppercase;font-weight:600;margin-bottom:0.25rem;">
            ⬢ Cross-Asset Performance · BQL Snapshot
          </div>
          <div style="font-size:36px;font-weight:800;letter-spacing:-0.01em;
                      font-family:'Inter',sans-serif;line-height:1;">
            MACRO TRACKER
          </div>
          <div style="font-size:11px;color:#888;letter-spacing:0.05em;margin-top:0.5rem;
                      text-transform:uppercase;">
            Latest: {last_updated} · {len(df)} instruments · {len(df['Category'].unique())} categories
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Top stats row
    # -------------------------------------------------------------------
    ytd_vals = df["YTD"].dropna()
    positive = (ytd_vals > 0).sum()
    negative = (ytd_vals < 0).sum()
    breadth_pct = positive / len(ytd_vals) * 100 if len(ytd_vals) else 0
    leader_row = df.loc[df["YTD"].idxmax()]
    laggard_row = df.loc[df["YTD"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Instruments", f"{len(df)}")
    with c2:
        st.metric(
            "YTD Breadth",
            f"{positive} / {negative}",
            f"{breadth_pct:.0f}% positive",
            delta_color="off",
        )
    with c3:
        st.metric("YTD Leader", leader_row["Name"], f"+{leader_row['YTD']:.2f}%")
    with c4:
        st.metric(
            "YTD Laggard",
            laggard_row["Name"],
            f"{laggard_row['YTD']:.2f}%",
            delta_color="inverse",
        )

    # 1D=0 issue is now handled silently in load_data (ETFs with 1D=0 become NaN).
    st.markdown("---")

    # -------------------------------------------------------------------
    # Internal sub-tabs
    # -------------------------------------------------------------------
    sub1, sub2 = st.tabs(["Heatmap", "Cross-Timeframe"])

    with sub1:
        _render_heatmap(view, df)
    with sub2:
        _render_cross_timeframe(df)


# ===========================================================================
# Sub-tab renderers
# ===========================================================================
def _render_table(
    rows_df: pd.DataFrame, title: str, subtitle: str, sort_key_prefix: str
):
    """Render a single heatmap table for a subset of instruments.

    Adds a row of clickable column header buttons that sort the table.
    Sort state is stored in st.session_state under the given prefix.
    """
    st.markdown(
        f"""
        <div style="font-size:14px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-top:1.5rem;margin-bottom:0.25rem;">
          {title}
        </div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:0.5rem;">
          {subtitle}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(rows_df) == 0:
        st.caption("No instruments in this group.")
        return

    # ---- Sort state ----
    sort_col_state = f"{sort_key_prefix}_sort_col"
    sort_dir_state = f"{sort_key_prefix}_sort_dir"
    if sort_col_state not in st.session_state:
        st.session_state[sort_col_state] = "YTD"
        st.session_state[sort_dir_state] = "desc"

    def _toggle_sort(col: str):
        if st.session_state[sort_col_state] == col:
            st.session_state[sort_dir_state] = (
                "asc" if st.session_state[sort_dir_state] == "desc" else "desc"
            )
        else:
            st.session_state[sort_col_state] = col
            # Default direction depends on column type
            st.session_state[sort_dir_state] = (
                "asc" if col in ("Category", "Name", "Ticker") else "desc"
            )

    # ---- Header row: Streamlit-button column headers ----
    # 8 columns: Category, Name, Ticker, 1D, 1W, 1M, 3M, YTD
    # Width ratios chosen to roughly match the table cell widths below.
    HEADER_RATIOS = [2.5, 2.8, 1.8, 1.2, 1.2, 1.2, 1.2, 1.2]
    cols = st.columns(HEADER_RATIOS)
    HEADER_LABELS = ["Category", "Name", "Ticker"] + COLS

    cur_col = st.session_state[sort_col_state]
    cur_dir = st.session_state[sort_dir_state]

    for col_idx, (col_widget, label) in enumerate(zip(cols, HEADER_LABELS)):
        # Sort indicator
        indicator = ""
        if label == cur_col:
            indicator = "  ↓" if cur_dir == "desc" else "  ↑"

        align = "left" if col_idx < 3 else "right"
        with col_widget:
            # Use a tiny button with a key per (table, label)
            st.button(
                f"{label}{indicator}",
                key=f"{sort_key_prefix}_btn_{label}",
                on_click=_toggle_sort,
                args=(label,),
                use_container_width=True,
            )

    # ---- Apply sort ----
    sort_col = st.session_state[sort_col_state]
    asc = st.session_state[sort_dir_state] == "asc"
    if sort_col in ("Category", "Name", "Ticker"):
        rows_df = rows_df.sort_values(sort_col, ascending=asc, kind="stable")
    else:
        # Numeric: NaN goes last regardless of direction
        rows_df = rows_df.sort_values(
            sort_col, ascending=asc, na_position="last", kind="stable"
        )

    # ---- Build rows HTML ----
    rows_html = []
    for _, r in rows_df.iterrows():
        cat_dot = CATEGORY_COLORS.get(r["Category"], "#888")
        cells = []
        for c in COLS:
            v = r[c]
            bg = cell_color(v)
            fg = text_color_for(v)
            txt = "—" if pd.isna(v) else f"{v:.2f}%"
            cells.append(
                f'<td style="background:{bg};color:{fg};text-align:right;'
                f"padding:6px 12px;font-family:JetBrains Mono,monospace;"
                f'font-size:11px;font-weight:500;">{txt}</td>'
            )
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<td style="padding:6px 12px;color:#aaa;font-size:11px;">'
            f'<span style="display:inline-block;width:6px;height:6px;'
            f'background:{cat_dot};margin-right:8px;vertical-align:middle;"></span>'
            f'{r["Category"]}</td>'
            f'<td style="padding:6px 12px;color:#fff;font-size:12px;font-weight:500;">{r["Name"]}</td>'
            f'<td style="padding:6px 12px;color:#666;font-size:10px;'
            f'font-family:JetBrains Mono,monospace;">{r["Ticker"]}</td>'
            + "".join(cells)
            + "</tr>"
        )

    table_html = (
        '<div style="border:1px solid #1a1a1a;background:#0a0a0a;overflow-x:auto;'
        'margin-top:-0.5rem;">'
        '<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        + "".join(rows_html)
        + "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _render_heatmap(view: pd.DataFrame, df: pd.DataFrame):
    """Render the heatmap split into two parts: ETFs and Commodities."""
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">PERFORMANCE HEATMAP</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Multi-timeframe scoreboard \u00b7 click column headers to sort \u00b7 color scale capped at \u00b130%
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(view) == 0:
        st.info("No instruments match the current filters.")
        return

    etfs = view[view["InstrumentType"] == "ETF"]
    comdty = view[view["InstrumentType"] == "Comdty"]

    _render_table(
        etfs,
        "EQUITIES & ETFS",
        f"{len(etfs)} instruments \u00b7 grouped into {etfs['Category'].nunique()} sectors",
        sort_key_prefix="etf_table",
    )

    _render_table(
        comdty,
        "COMMODITY FUTURES",
        f"{len(comdty)} instruments \u00b7 front-month continuous contracts",
        sort_key_prefix="comdty_table",
    )

    st.caption(
        f"{len(view)} of {len(df)} instruments shown \u00b7 "
        f"ETF 1D values are 0% when the BQL pull ran outside US market hours "
        f"(today\u2019s close == prior close). Refresh BQL during US trading hours for live 1D."
    )


def _render_cross_timeframe(df: pd.DataFrame):
    """Cross-timeframe rotation map, split into two scatters (ETFs vs Comdty),
    with circle size proportional to abs(Y-axis return)."""
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">CROSS-TIMEFRAME ROTATION MAP</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Short-term momentum vs longer-term trend \u00b7 circle size = magnitude of recent move \u00b7 top-right = sustained moves
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X-axis (longer trend)", COLS, index=4, key="ct_x")
    with col_y:
        y_axis = st.selectbox("Y-axis (recent move)", COLS, index=1, key="ct_y")

    valid = df.dropna(subset=[x_axis, y_axis]).copy()
    # Marker size based on abs(Y-axis return). Map range to readable bubble sizes.
    abs_y = valid[y_axis].abs()
    if abs_y.max() > 0:
        # Min/max marker size in pixels (sqrt-scaled by Plotly when sizemode=area)
        valid["_size"] = abs_y
    else:
        valid["_size"] = 1.0
    # sizeref normalizes the largest bubble to ~40px. (Plotly convention.)
    sizeref_val = max(2.0 * valid["_size"].max() / (40**2), 1e-6)

    etfs = valid[valid["InstrumentType"] == "ETF"]
    comdty = valid[valid["InstrumentType"] == "Comdty"]

    def _build_fig(scatter_df: pd.DataFrame, header: str, n_total: int) -> go.Figure:
        fig = go.Figure()
        if len(scatter_df) == 0:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                text="No instruments in this group",
                showarrow=False,
                font=dict(size=12, color="#666"),
            )
            fig.update_layout(
                **DARK_LAYOUT,
                height=300,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            return fig

        for cat in sorted(scatter_df["Category"].unique()):
            sub = scatter_df[scatter_df["Category"] == cat]
            color = CATEGORY_COLORS.get(cat, "#888")
            fig.add_trace(
                go.Scatter(
                    x=sub[x_axis],
                    y=sub[y_axis],
                    mode="markers+text",
                    name=cat,
                    text=sub["Name"],
                    textposition="top center",
                    textfont=dict(size=8, color="#aaa"),
                    marker=dict(
                        size=sub["_size"],
                        sizemode="area",
                        sizeref=sizeref_val,
                        sizemin=4,
                        color=color,
                        line=dict(color="rgba(255,255,255,0.3)", width=0.5),
                        opacity=0.75,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + f"{x_axis}: %{{x:.2f}}%<br>"
                        + f"{y_axis}: %{{y:.2f}}%<br>"
                        + f"|{y_axis}|: %{{marker.size:.2f}}%"
                        + "<extra></extra>"
                    ),
                )
            )

        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
        fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

        x_range = (scatter_df[x_axis].min(), scatter_df[x_axis].max())
        y_range = (scatter_df[y_axis].min(), scatter_df[y_axis].max())
        # Pad ranges so labels & quadrant boxes don't get clipped
        x_pad = max((x_range[1] - x_range[0]) * 0.10, 1.0)
        y_pad = max((y_range[1] - y_range[0]) * 0.15, 1.0)

        fig.add_annotation(
            x=x_range[1],
            y=y_range[1] + y_pad * 0.3,
            text="<b>STRONG &amp; ACCELERATING</b>",
            showarrow=False,
            font=dict(size=10, color="#4ade80"),
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(74,222,128,0.08)",
            bordercolor="rgba(74,222,128,0.3)",
            borderwidth=1,
            borderpad=4,
        )
        fig.add_annotation(
            x=x_range[0],
            y=y_range[0] - y_pad * 0.3,
            text="<b>WEAK &amp; DETERIORATING</b>",
            showarrow=False,
            font=dict(size=10, color="#f87171"),
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(248,113,113,0.08)",
            bordercolor="rgba(248,113,113,0.3)",
            borderwidth=1,
            borderpad=4,
        )

        fig.update_layout(
            **DARK_LAYOUT,
            height=520,
            showlegend=True,
            title=dict(
                text=f"<span style='font-size:12px;letter-spacing:0.1em;color:#aaa;'>{header} \u00b7 {len(scatter_df)} of {n_total}</span>",
                x=0,
                xanchor="left",
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10, color="#ccc"),
            ),
            xaxis=dict(
                title=dict(
                    text=f"<i>{x_axis} return (%)</i>", font=dict(size=11, color="#aaa")
                ),
                range=[x_range[0] - x_pad, x_range[1] + x_pad],
                showgrid=True,
                gridcolor=GRID,
                zeroline=False,
                tickfont=dict(size=9, color=TEXT_DIM),
                ticksuffix="%",
            ),
            yaxis=dict(
                title=dict(
                    text=f"<i>{y_axis} return (%)</i>", font=dict(size=11, color="#aaa")
                ),
                range=[y_range[0] - y_pad, y_range[1] + y_pad],
                showgrid=True,
                gridcolor=GRID,
                zeroline=False,
                tickfont=dict(size=9, color=TEXT_DIM),
                ticksuffix="%",
            ),
            margin=dict(l=60, r=180, t=40, b=50),
        )
        return fig

    fig_etf = _build_fig(etfs, "EQUITIES & ETFS", (df["InstrumentType"] == "ETF").sum())
    st.plotly_chart(fig_etf, use_container_width=True, config={"displayModeBar": False})

    fig_com = _build_fig(
        comdty, "COMMODITY FUTURES", (df["InstrumentType"] == "Comdty").sum()
    )
    st.plotly_chart(fig_com, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Larger circles = bigger move on the recent timeframe. "
        "Top-right quadrant = positive on both axes (sustained leadership). "
        "Bottom-left = negative on both (sustained weakness). "
        "Top-left or bottom-right = inflection / reversal candidates."
    )
