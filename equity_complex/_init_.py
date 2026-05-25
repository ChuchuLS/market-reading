"""
Equity complex — within-complex PCA for the equity sub-components.

Sub-components: SPX (cap-weighted), SPW (equal-weight), VIX (implied vol).
Anchored on SPX positive (SPX up = risk-on direction).

Reads the same MARKET_DATA.xlsx (sheet: ficc) as cross_asset_ficc/.
"""
