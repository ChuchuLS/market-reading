"""
Market Reading — unified dashboard.

Top-level tabs:
  - Macro Tracker: BQL cross-asset performance (themes + commodities)
  - Stock Heatmap: TradingView-style treemap of US index constituents
  - Cross-Asset: unified group with sub-tabs:
      · 3-Asset Classic (SPX / UST 10Y / DXY)
      · FICC (5-asset)
      · Rates (10Y / 2s10s / 10Y BE / 10Y Real / MOVE)
      · Credit (US HY / US IG / EU iTraxx)
      · FX (TBD)
      · Equity (TBD)
      · Commodities (TBD)
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Market Reading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ---------------------------------------------------------------------------
# Sidebar branding
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Top-level tabs
# ---------------------------------------------------------------------------
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
    sub_classic, sub_ficc, sub_rates, sub_credit = st.tabs([
        "3-Asset Classic",
        "FICC (5-asset)",
        "Rates",
        "Credit",
    ])

    with sub_classic:
        render_cross_asset()

    with sub_ficc:
        render_cross_asset_ficc()

    with sub_rates:
        render_rates_complex()

    with sub_credit:
        render_credit_complex()
