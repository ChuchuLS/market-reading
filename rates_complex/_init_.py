"""
Rates complex — within-complex PCA for the rates sub-components.

Sub-components: 10Y yield, 2s10s spread, 10Y breakeven, 10Y real yield, MOVE.
Anchored on USGG10YR positive (rates-up = positive direction).

Reads the same MARKET_DATA.xlsx (sheet: ficc) as cross_asset_ficc/. No separate data file.
"""
