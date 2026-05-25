"""
Shared data utilities used across dashboard pages.
"""

from __future__ import annotations

import pandas as pd


def drop_all_zero_return_rows(returns: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop only rows where EVERY asset has a zero return (a fully stale day,
    e.g. a holiday the data source forward-filled across all series).

    This is deliberately softer than the old per-asset filter
    ``(returns != 0).all(axis=1)``, which deleted an entire trading day if
    *any single* asset was unchanged. A zero daily change is legitimate for
    spreads, curves, OAS, breakevens, vol indices, and illiquid series, so
    dropping the whole row on one zero discarded valid data.

    Parameters
    ----------
    returns : DataFrame
        Asset returns (one column per asset).

    Returns
    -------
    (filtered_returns, n_dropped)
        The frame with all-zero rows removed, and the count removed.
    """
    if returns.empty:
        return returns, 0
    stale_mask = (returns == 0).all(axis=1)
    dropped = int(stale_mask.sum())
    return returns.loc[~stale_mask], dropped
