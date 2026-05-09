"""
Market Reading — unified dashboard.

Top-level tabs:
  - Macro Tracker: BQL cross-asset performance (themes + commodities)
  - Stock Heatmap: TradingView-style treemap of US index constituents
  - Cross-Asset: 3-asset (SPX / UST 10Y / DXY) PCA + 8-bucket regime cube
  - FICC: 5-asset (SPX / UST 10Y / DXY / BCOM / HY OAS) PCA + continuous regime

Each section lives in its own subpackage. This file just wires them together.
"""

from __future__ import annotations

import streamlit as st

from theming import apply_theme

from macro_tracker.view import render_macro_tracker
from stock_heatmap.view import render_stock_heatmap
from cross_asset.view import render_cross_asset
from cross_asset_ficc.view import render_cross_asset_ficc

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
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
# Top-level horizontal tabs
# ---------------------------------------------------------------------------
top_tab_macro, top_tab_stocks, top_tab_xasset, top_tab_ficc = st.tabs([
    "📊 Macro Tracker",
    "🌐 Stock Heatmap",
    "🔗 Cross-Asset",
    "⬢ FICC",
])

with top_tab_macro:
    render_macro_tracker()

with top_tab_stocks:
    render_stock_heatmap()

with top_tab_xasset:
    render_cross_asset()

with top_tab_ficc:
    render_cross_asset_ficc()
