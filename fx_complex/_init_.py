"""
FX complex — within-complex PCA for the FX sub-components.

Sub-components: DXY (broad dollar), FXJPEMCS (EM FX index), JYBSS12M (xccy basis).
Anchored on DXY positive (USD-strong direction).

JYBSS12M (12M USDJPY basis) is a funding-stress signal, not a directional FX
signal. Its sign relative to DXY tells you what KIND of dollar regime is active:
  same-sign as DXY  → USD strength is funding-driven (dollar shortage)
  opposite to DXY   → USD strength is growth/rate-divergence driven
  basis dominates with DXY weak → funding event without directional move

Reads the same FICCREADING.xlsx as cross_asset_ficc/.
"""
