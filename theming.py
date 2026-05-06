"""
Shared theme tokens and global CSS injection for the unified dashboard.

Imported once by app.py at startup; every view module imports the color
constants from here so the look stays consistent across tabs.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Color tokens
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

COLS = ["1D", "1W", "1M", "3M", "YTD"]


# ---------------------------------------------------------------------------
# Global CSS — injected once
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stDataFrame"] { background: #0a0a0a; }

hr { border-color: #1a1a1a !important; margin: 1.5rem 0 !important; }

[data-baseweb="select"] > div {
    background-color: #0f0f0f !important;
    border-color: #2a2a2a !important;
}
.stTextInput > div > div > input {
    background-color: #0f0f0f !important;
    color: #e0e0e0 !important;
    border-color: #2a2a2a !important;
}

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

/* Top-level tabs (the section switcher) */
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
"""


def apply_theme():
    """Inject the global CSS. Call once from app.py."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
