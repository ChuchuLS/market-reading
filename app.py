"""
Market Reading — unified dashboard.
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

top_tab_macro, top_tab_stocks, top_tab_xasset = st.tabs([
    "📊 Macro Tracker",
    "🌐 Stock Heatmap",
    "🔗 Cross-Asset",
])

with top_tab_macro:
    render_macro_tracker()

with top_tab_stocks:
    render_stock_heatmap()

with top_tab_xasset:
    sub_classic, sub_ficc, sub_rates, sub_credit, sub_fx, sub_equity, sub_comdty, sub_sector = st.tabs([
        "3-Asset Classic",
        "FICC (5-asset)",
        "Rates",
        "Credit",
        "FX",
        "Equity",
        "Commodities",
        "Sectors",
    ])

    with sub_classic:
        render_cross_asset()

    with sub_ficc:
        render_cross_asset_ficc()

    with sub_rates:
        render_rates_complex()

    with sub_credit:
        render_credit_complex()

    with sub_fx:
        render_fx_complex()

    with sub_equity:
        render_equity_complex()

    with sub_comdty:
        render_commodities_complex()

    with sub_sector:
        render_sector_complex()
