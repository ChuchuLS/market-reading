"""
Macro Tracker — Streamlit dashboard, OFR/Bloomberg-style dark theme.

Reads BQL output from data/DATA.xlsx (5 timeframes × 57 instruments) and renders:
  - Header with breadth and leader/laggard stats
  - Category YTD performance bar
  - Multi-timeframe heatmap (sortable)
  - Top/bottom 10 leaders & laggards
  - Cross-timeframe scatter (rotation view)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path(__file__).parent / "data" / "DATA.xlsx"

# ---------------------------------------------------------------------------
# Color tokens — match the rates monitor aesthetic
# ---------------------------------------------------------------------------
BG = "#0a0a0a"
PANEL = "#0f0f0f"
GRID = "rgba(255,255,255,0.05)"
TEXT = "#e0e0e0"
TEXT_DIM = "#888"
LINE = "#ffffff"

GREEN = "#4ade80"
RED = "#f87171"
AMBER = "#fbbf24"
BLUE = "#60a5fa"

CATEGORY_COLORS = {
    "Theme": "#4ade80",
    "Precious Metals": "#d4af37",
    "Industrial Metals": "#b87333",
    "Energy": "#ff6b35",
    "Softs": "#a0522d",
    "Livestock": "#c97064",
}

DARK_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=11),
)

# ---------------------------------------------------------------------------
# Global dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #1a1a1a;
    }
    section[data-testid="stSidebar"] * { color: #ccc !important; }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        letter-spacing: 0.04em;
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #888 !important;
        letter-spacing: 0.04em;
    }
    /* Hide Streamlit's hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tighter padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        background: #0a0a0a;
    }

    /* Dividers */
    hr { border-color: #1a1a1a !important; margin: 1.5rem 0 !important; }

    /* Selectbox / inputs */
    [data-baseweb="select"] > div {
        background-color: #0f0f0f !important;
        border-color: #2a2a2a !important;
    }
    .stTextInput > div > div > input {
        background-color: #0f0f0f !important;
        color: #e0e0e0 !important;
        border-color: #2a2a2a !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 10px !important;
        color: #888 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #1a1a1a;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #888;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.75rem 1.25rem;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #4ade80 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Ticker → (Name, Category) registry
# ---------------------------------------------------------------------------
TICKER_MAP = {
    'SIL US Equity':   ('Silver Miners',         'Theme'),
    'REMX US Equity':  ('Rare Earth Metals',     'Theme'),
    'URA US Equity':   ('Uranium / Nuclear',     'Theme'),
    'GDX US Equity':   ('Gold Miners',           'Theme'),
    'COPX US Equity':  ('Copper Miners',         'Theme'),
    'ARKX US Equity':  ('Space',                 'Theme'),
    'WGMI US Equity':  ('Bitcoin Miners',        'Theme'),
    'LIT US Equity':   ('Lithium / Battery',     'Theme'),
    'XHB US Equity':   ('Homebuilders',          'Theme'),
    'ARKG US Equity':  ('Genomics',              'Theme'),
    'SLX US Equity':   ('Steel',                 'Theme'),
    'ITA US Equity':   ('Aerospace & Defense',   'Theme'),
    'XLE US Equity':   ('Energy ETF',            'Theme'),
    'SMH US Equity':   ('Semiconductors',        'Theme'),
    'KRE US Equity':   ('Regional Banks',        'Theme'),
    'XLB US Equity':   ('Materials',             'Theme'),
    'XLP US Equity':   ('Consumer Staples',      'Theme'),
    'XRT US Equity':   ('Retail',                'Theme'),
    'XLI US Equity':   ('Industrials',           'Theme'),
    'XBI US Equity':   ('Biotech',               'Theme'),
    'QTUM US Equity':  ('Quantum',               'Theme'),
    'EEM US Equity':   ('Emerging Markets',      'Theme'),
    'IYT US Equity':   ('Transports',            'Theme'),
    'ROBO US Equity':  ('Robotics',              'Theme'),
    'XLRE US Equity':  ('Real Estate',           'Theme'),
    'BOTZ US Equity':  ('AI',                    'Theme'),
    'JETS US Equity':  ('Airlines',              'Theme'),
    'XLY US Equity':   ('Consumer Discretionary','Theme'),
    'XLV US Equity':   ('Health Care',           'Theme'),
    'TLT US Equity':   ('Long Term Treasuries',  'Theme'),
    'IHI US Equity':   ('Medical Devices',       'Theme'),
    'KWEB US Equity':  ('China Internet',        'Theme'),
    'XLU US Equity':   ('Utilities',             'Theme'),
    'SOCL US Equity':  ('Social Media',          'Theme'),
    'IYZ US Equity':   ('Telecom',               'Theme'),
    'HACK US Equity':  ('Cybersecurity',         'Theme'),
    'VUG US Equity':   ('Growth Stocks',         'Theme'),
    'DJUSCA Index':    ('Casinos',               'Theme'),
    'SKYY US Equity':  ('Cloud Computing',       'Theme'),
    'IGV US Equity':   ('Software',              'Theme'),
    'GC1 Comdty':      ('Gold',                  'Precious Metals'),
    'SI1 Comdty':      ('Silver',                'Precious Metals'),
    'PL1 Comdty':      ('Platinum',              'Precious Metals'),
    'PA1 Comdty':      ('Palladium',             'Precious Metals'),
    'HG1 Comdty':      ('Copper',                'Industrial Metals'),
    'CL1 Comdty':      ('WTI Crude Oil',         'Energy'),
    'CO1 Comdty':      ('Brent Crude',           'Energy'),
    'NG1 Comdty':      ('Natural Gas',           'Energy'),
    'HO1 Comdty':      ('Heating Oil',           'Energy'),
    'XB1 Comdty':      ('Gasoline (RBOB)',       'Energy'),
    'COAL US Equity':  ('Coal',                  'Energy'),
    'KC1 Comdty':      ('Coffee',                'Softs'),
    'SB1 Comdty':      ('Sugar',                 'Softs'),
    'CC1 Comdty':      ('Cocoa',                 'Softs'),
    'CT1 Comdty':      ('Cotton',                'Softs'),
    'WOOD US Equity':  ('Lumber',                'Softs'),
    'LH1 Comdty':      ('Lean Hogs',             'Livestock'),
}

COLS = ["1D", "1W", "1M", "3M", "YTD"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path, _mtime: float) -> pd.DataFrame:
    """Read DATA.xlsx and return tidy DataFrame keyed by ticker."""
    raw = pd.read_excel(path)
    raw = raw.iloc[:, :6].copy()
    raw.columns = ["Ticker", "1D", "1W", "1M", "3M", "YTD"]
    raw = raw.dropna(subset=["Ticker"]).reset_index(drop=True)

    rows = []
    for _, r in raw.iterrows():
        ticker = str(r["Ticker"]).strip()
        if ticker in TICKER_MAP:
            name, cat = TICKER_MAP[ticker]
            rows.append({
                "Category": cat,
                "Name": name,
                "Ticker": ticker,
                "1D": float(r["1D"]),
                "1W": float(r["1W"]),
                "1M": float(r["1M"]),
                "3M": float(r["3M"]),
                "YTD": float(r["YTD"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Color helper for cells
# ---------------------------------------------------------------------------
def cell_color(v: float, cap: float = 30.0) -> str:
    """Diverging green/red HSL color for a percent value."""
    if pd.isna(v) or v == 0:
        return "rgba(60,60,65,0.4)"
    t = max(-1, min(1, v / cap))
    intensity = abs(t)
    if t >= 0:
        lightness = 88 - intensity * 50
        sat = 35 + intensity * 45
        return f"hsl(140, {sat}%, {lightness}%)"
    else:
        lightness = 88 - intensity * 50
        sat = 40 + intensity * 50
        return f"hsl(355, {sat}%, {lightness}%)"


def text_color_for(v: float, cap: float = 30.0) -> str:
    if pd.isna(v) or v == 0:
        return "#999"
    intensity = min(1, abs(v / cap))
    return "#fff" if intensity > 0.45 else ("#0a3d18" if v > 0 else "#4a0d12")


# ---------------------------------------------------------------------------
# Try to load
# ---------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"DATA.xlsx not found at {DATA_PATH}. Place your BQL output there and refresh.")
    st.stop()

mtime = DATA_PATH.stat().st_mtime
df = load_data(DATA_PATH, mtime)
last_updated = datetime.fromtimestamp(mtime).strftime("%b %d, %Y · %H:%M")


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:18px;font-weight:800;letter-spacing:0.06em;color:#fff;
                    margin-bottom:0.25rem;">
          MACRO TRACKER
        </div>
        <div style="font-size:10px;letter-spacing:0.1em;color:#888;text-transform:uppercase;
                    margin-bottom:1.5rem;">
          BQL Cross-Asset Snapshot
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**FILTERS**")
    all_cats = ["All"] + sorted(df["Category"].unique().tolist())
    cat_filter = st.selectbox("Category", all_cats, index=0, label_visibility="collapsed")

    search = st.text_input("Search", placeholder="Filter name or ticker…",
                           label_visibility="collapsed")

    sort_col = st.selectbox("Sort by", COLS, index=4, label_visibility="visible")
    sort_dir = st.radio("Direction", ["Descending", "Ascending"],
                        index=0, horizontal=True, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**DATA**")
    st.caption(f"📁 {DATA_PATH.name}")
    st.caption(f"🕐 Updated: {last_updated}")
    st.caption(f"🔢 {len(df)} instruments")

    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(
        "Replace `data/DATA.xlsx` with a fresh BQL export "
        "and click Refresh to update the dashboard."
    )


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
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
view = view.sort_values(sort_col, ascending=(sort_dir == "Ascending")).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Top stats row — breadth, leader, laggard, best timeframe
# ---------------------------------------------------------------------------
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
    st.metric(
        "YTD Leader",
        leader_row["Name"],
        f"+{leader_row['YTD']:.2f}%",
    )
with c4:
    st.metric(
        "YTD Laggard",
        laggard_row["Name"],
        f"{laggard_row['YTD']:.2f}%",
        delta_color="inverse",
    )

# Warning banner if all theme 1D = 0
themes_only = df[df["Category"] == "Theme"]
if len(themes_only) and (themes_only["1D"] == 0).all():
    st.markdown(
        """
        <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                    padding:0.6rem 1rem;margin:0.5rem 0;border-radius:2px;
                    font-size:11px;color:#fcd34d;">
          <b>⚠ Note:</b> All theme ETF 1-day returns are 0%. This is a BQL artifact —
          query ran when US markets had not moved since prior close (likely outside US
          trading hours from Asia/Europe). Commodity futures 1D values are valid because
          futures trade nearly 24h. Refresh the BQL sheet during US market hours for live values.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs for the main views
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Heatmap", "Leaders & Laggards", "Category Rotation", "Cross-Timeframe"
])


# ===========================================================================
# TAB 1 — Multi-timeframe heatmap
# ===========================================================================
with tab1:
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">PERFORMANCE HEATMAP</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Multi-timeframe scoreboard · color scale capped at ±30%
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(view) == 0:
        st.info("No instruments match the current filters.")
    else:
        # Build a Plotly table-like heatmap using a heatmap with custom annotations
        # Better: build a styled HTML table directly for full control
        rows_html = []
        for _, r in view.iterrows():
            cat_dot = CATEGORY_COLORS.get(r["Category"], "#888")
            cells = []
            for c in COLS:
                v = r[c]
                bg = cell_color(v)
                fg = text_color_for(v)
                txt = "—" if pd.isna(v) else f"{v:.2f}"
                cells.append(
                    f'<td style="background:{bg};color:{fg};text-align:right;'
                    f'padding:6px 12px;font-family:JetBrains Mono,monospace;'
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

        header_html = (
            '<tr style="border-bottom:1px solid rgba(255,255,255,0.1);'
            'background:rgba(255,255,255,0.02);">'
            + ''.join(
                f'<th style="text-align:left;padding:10px 12px;font-size:10px;'
                f'text-transform:uppercase;letter-spacing:0.1em;color:#888;'
                f'font-weight:500;">{h}</th>'
                for h in ["Category", "Name", "Ticker"]
            )
            + ''.join(
                f'<th style="text-align:right;padding:10px 12px;font-size:10px;'
                f'text-transform:uppercase;letter-spacing:0.1em;color:#888;'
                f'font-weight:500;">{c}</th>'
                for c in COLS
            )
            + '</tr>'
        )

        table_html = (
            '<div style="border:1px solid #1a1a1a;background:#0a0a0a;overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
            + header_html
            + ''.join(rows_html)
            + '</table></div>'
        )

        st.markdown(table_html, unsafe_allow_html=True)
        st.caption(f"{len(view)} of {len(df)} instruments shown")


# ===========================================================================
# TAB 2 — Leaders & Laggards horizontal bars
# ===========================================================================
with tab2:
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">LEADERS & LAGGARDS</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Top and bottom performers across timeframes
        </div>
        """,
        unsafe_allow_html=True,
    )

    timeframe = st.radio(
        "Timeframe",
        COLS,
        index=4,
        horizontal=True,
        label_visibility="collapsed",
    )

    n_show = st.slider("Show top/bottom N", 5, 20, 10, label_visibility="collapsed")

    sub = df.dropna(subset=[timeframe]).sort_values(timeframe, ascending=False)
    top_n = sub.head(n_show)
    bot_n = sub.tail(n_show).iloc[::-1]

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_n[timeframe],
            y=top_n["Name"],
            orientation="h",
            marker=dict(
                color=top_n[timeframe],
                colorscale=[[0, "#0a3d18"], [1, "#4ade80"]],
                cmin=0, cmax=top_n[timeframe].max(),
                line=dict(width=0),
            ),
            text=[f"+{v:.2f}%" for v in top_n[timeframe]],
            textposition="outside",
            textfont=dict(size=10, color="#4ade80", family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>" + timeframe + ": %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            **DARK_LAYOUT,
            height=max(320, n_show * 28),
            title=dict(
                text=f"<span style='font-size:12px;letter-spacing:0.1em;'>TOP {n_show} · {timeframe}</span>",
                x=0, xanchor="left", font=dict(color="#4ade80"),
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10, color="#ccc"),
                       gridcolor=GRID, showgrid=False),
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                       tickfont=dict(size=9, color=TEXT_DIM), ticksuffix="%"),
            showlegend=False,
            margin=dict(l=10, r=60, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bot_n[timeframe],
            y=bot_n["Name"],
            orientation="h",
            marker=dict(
                color=bot_n[timeframe],
                colorscale=[[0, "#f87171"], [1, "#4a0d12"]],
                cmin=bot_n[timeframe].min(), cmax=0,
                line=dict(width=0),
            ),
            text=[f"{v:.2f}%" for v in bot_n[timeframe]],
            textposition="outside",
            textfont=dict(size=10, color="#f87171", family="JetBrains Mono"),
            hovertemplate="<b>%{y}</b><br>" + timeframe + ": %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            **DARK_LAYOUT,
            height=max(320, n_show * 28),
            title=dict(
                text=f"<span style='font-size:12px;letter-spacing:0.1em;'>BOTTOM {n_show} · {timeframe}</span>",
                x=0, xanchor="left", font=dict(color="#f87171"),
            ),
            yaxis=dict(tickfont=dict(size=10, color="#ccc"),
                       gridcolor=GRID, showgrid=False),
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                       tickfont=dict(size=9, color=TEXT_DIM), ticksuffix="%"),
            showlegend=False,
            margin=dict(l=10, r=60, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# TAB 3 — Category rotation
# ===========================================================================
with tab3:
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">CATEGORY ROTATION</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Average performance per category across timeframes · spot momentum shifts
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_avg = df.groupby("Category")[COLS].mean().reset_index()
    cat_avg = cat_avg.sort_values("YTD", ascending=False).reset_index(drop=True)

    # Bar chart: one bar per category per timeframe
    fig = go.Figure()
    timeframe_colors = {
        "1D": "#60a5fa",
        "1W": "#a78bfa",
        "1M": "#fbbf24",
        "3M": "#fb923c",
        "YTD": "#4ade80",
    }
    for col in COLS:
        fig.add_trace(go.Bar(
            x=cat_avg["Category"],
            y=cat_avg[col],
            name=col,
            marker=dict(color=timeframe_colors[col], line=dict(width=0)),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=420,
        barmode="group",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#ccc"),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#ccc")),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.2)",
                   tickfont=dict(size=10, color=TEXT_DIM), ticksuffix="%"),
        margin=dict(l=40, r=40, t=40, b=60),
    )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Category summary table
    st.markdown(
        """
        <div style="font-size:11px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin:1.5rem 0 0.5rem 0;">
          Category averages · sorted by YTD
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_rows_html = []
    for _, r in cat_avg.iterrows():
        cat_dot = CATEGORY_COLORS.get(r["Category"], "#888")
        n_in_cat = (df["Category"] == r["Category"]).sum()
        cells = []
        for c in COLS:
            v = r[c]
            bg = cell_color(v)
            fg = text_color_for(v)
            cells.append(
                f'<td style="background:{bg};color:{fg};text-align:right;'
                f'padding:8px 12px;font-family:JetBrains Mono,monospace;'
                f'font-size:11px;font-weight:600;">{v:.2f}</td>'
            )
        cat_rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<td style="padding:8px 12px;color:#fff;font-size:13px;font-weight:500;">'
            f'<span style="display:inline-block;width:8px;height:8px;background:{cat_dot};'
            f'margin-right:10px;vertical-align:middle;"></span>'
            f'{r["Category"]}</td>'
            f'<td style="padding:8px 12px;color:#666;font-size:10px;text-align:right;'
            f'font-family:JetBrains Mono;">{n_in_cat}</td>'
            + "".join(cells)
            + "</tr>"
        )

    cat_table_html = (
        '<div style="border:1px solid #1a1a1a;background:#0a0a0a;">'
        '<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
        '<tr style="border-bottom:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.02);">'
        '<th style="text-align:left;padding:10px 12px;font-size:10px;'
        'text-transform:uppercase;letter-spacing:0.1em;color:#888;font-weight:500;">Category</th>'
        '<th style="text-align:right;padding:10px 12px;font-size:10px;'
        'text-transform:uppercase;letter-spacing:0.1em;color:#888;font-weight:500;">N</th>'
        + ''.join(
            f'<th style="text-align:right;padding:10px 12px;font-size:10px;'
            f'text-transform:uppercase;letter-spacing:0.1em;color:#888;'
            f'font-weight:500;">{c}</th>'
            for c in COLS
        )
        + '</tr>'
        + ''.join(cat_rows_html)
        + '</table></div>'
    )
    st.markdown(cat_table_html, unsafe_allow_html=True)


# ===========================================================================
# TAB 4 — Cross-timeframe scatter (rotation map)
# ===========================================================================
with tab4:
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;letter-spacing:0.04em;color:#fff;
                    margin-bottom:0.25rem;">CROSS-TIMEFRAME ROTATION MAP</div>
        <div style="font-size:10px;color:#888;letter-spacing:0.08em;text-transform:uppercase;
                    margin-bottom:1rem;">
          Short-term momentum vs longer-term trend · top-right = strong sustained moves
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X-axis (longer trend)", COLS, index=4)
    with col_y:
        y_axis = st.selectbox("Y-axis (recent move)", COLS, index=1)

    valid = df.dropna(subset=[x_axis, y_axis])

    fig = go.Figure()
    for cat in sorted(valid["Category"].unique()):
        sub = valid[valid["Category"] == cat]
        color = CATEGORY_COLORS.get(cat, "#888")
        fig.add_trace(go.Scatter(
            x=sub[x_axis],
            y=sub[y_axis],
            mode="markers+text",
            name=cat,
            text=sub["Name"],
            textposition="top center",
            textfont=dict(size=8, color="#aaa"),
            marker=dict(
                size=10,
                color=color,
                line=dict(color="rgba(255,255,255,0.3)", width=0.5),
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                + f"{x_axis}: %{{x:.2f}}%<br>"
                + f"{y_axis}: %{{y:.2f}}%"
                + "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    # Quadrant labels
    x_range = (valid[x_axis].min(), valid[x_axis].max())
    y_range = (valid[y_axis].min(), valid[y_axis].max())
    fig.add_annotation(
        x=x_range[1] * 0.95, y=y_range[1] * 0.95,
        text="<b>STRONG &amp; ACCELERATING</b>", showarrow=False,
        font=dict(size=10, color="#4ade80"), xanchor="right", yanchor="top",
        bgcolor="rgba(74,222,128,0.08)", bordercolor="rgba(74,222,128,0.3)",
        borderwidth=1, borderpad=4,
    )
    fig.add_annotation(
        x=x_range[0] * 0.95 if x_range[0] < 0 else x_range[0] * 1.05,
        y=y_range[0] * 0.95 if y_range[0] < 0 else y_range[0] * 1.05,
        text="<b>WEAK &amp; DETERIORATING</b>", showarrow=False,
        font=dict(size=10, color="#f87171"), xanchor="left", yanchor="bottom",
        bgcolor="rgba(248,113,113,0.08)", bordercolor="rgba(248,113,113,0.3)",
        borderwidth=1, borderpad=4,
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc"),
        ),
        xaxis=dict(
            title=dict(text=f"<i>{x_axis} return (%)</i>",
                       font=dict(size=11, color="#aaa")),
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM), ticksuffix="%",
        ),
        yaxis=dict(
            title=dict(text=f"<i>{y_axis} return (%)</i>",
                       font=dict(size=11, color="#aaa")),
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM), ticksuffix="%",
        ),
        margin=dict(l=60, r=180, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        f"Top-right quadrant = positive on both timeframes (sustained leadership). "
        f"Bottom-left = negative on both (sustained weakness). "
        f"Top-left or bottom-right = inflection / reversal candidates."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"📊 {len(df)} instruments across {len(df['Category'].unique())} categories · "
    f"Source: Bloomberg BQL via DATA.xlsx · "
    f"Last refreshed: {last_updated}"
)
