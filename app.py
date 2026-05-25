"""
Market Reading — unified dashboard.

Sidebar navigation: only the selected page is rendered, so Streamlit no
longer computes every dashboard page on each run (the old nested-st.tabs
structure evaluated all tab bodies eagerly).
"""

from __future__ import annotations

import streamlit as st

from theming import apply_theme

from macro_tracker.view import render_macro_tracker
from stock_heatmap.view import render_stock_heatmap
from cross_asset.view import render_cross_asset
from cross_asset_ficc.view import render_cross_asset_ficc
from rates_complex.view import render_rates_complex
from credit_complex.view import render_credit_complex
from fx_complex.view import render_fx_complex
from equity_complex.view import render_equity_complex
from comdty_complex.view import render_commodities_complex
from sector_complex.view import render_sector_complex

st.set_page_config(
    page_title="Market Reading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

with st.sidebar:
    st.markdown(
        """
<div style="font-size:18px;font-weight:800;letter-spacing:0.06em;color:#fff;
margin-bottom:0.25rem;">
MARKET READING
</div>
<div style="font-size:10px;letter-spacing:0.1em;color:#888;text-transform:uppercase;
margin-bottom:1.5rem;">
Unified Cross-Asset Dashboard
</div>
""",
        unsafe_allow_html=True,
    )

PAGES = [
    "Macro Tracker",
    "Stock Heatmap",
    "3-Asset Classic",
    "FICC",
    "Rates",
    "Credit",
    "FX",
    "Equity",
    "Commodities",
    "Sectors",
]

page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")

if page == "Macro Tracker":
    render_macro_tracker()
elif page == "Stock Heatmap":
    render_stock_heatmap()
elif page == "3-Asset Classic":
    render_cross_asset()
elif page == "FICC":
    render_cross_asset_ficc()
elif page == "Rates":
    render_rates_complex()
elif page == "Credit":
    render_credit_complex()
elif page == "FX":
    render_fx_complex()
elif page == "Equity":
    render_equity_complex()
elif page == "Commodities":
    render_commodities_complex()
elif page == "Sectors":
    render_sector_complex()
