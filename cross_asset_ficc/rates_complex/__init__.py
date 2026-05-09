"""
Cross-Asset FICC — 5-asset extension of cross_asset/.

Assets: SPX, UST 10Y, DXY, BCOM, HY OAS
The classic 3-asset cross_asset/ module remains unchanged. This module
applies the same rolling-PCA machinery to a 5-asset FICC basket and replaces
the 8-bucket sign-cube classification with a continuous regime characterization
(leader + concentration + sign pattern), since 32 sign-buckets stop being
legible.
"""
