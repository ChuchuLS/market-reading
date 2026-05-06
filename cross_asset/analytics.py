"""
Cross-asset analytics for SPX / UST 10Y / DXY.

Pure math, no Streamlit. Functions take DataFrames in and return DataFrames out.

Conventions
-----------
- `prices` is a DataFrame with columns SPX, USGG10YR, DXY indexed by Date.
- For SPX and DXY we compute log returns.
- For USGG10YR (yield) we compute first differences (in pct points).
  This is correct for a yield series: a 10bp move is informative as
  +0.10 in the yield series; a "log return" on a yield is meaningless.
- All rolling stats are right-aligned (today's value uses the last N days
  including today).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw levels to a returns DataFrame:
      SPX, DXY  -> log returns
      USGG10YR  -> first differences (pct points)
    """
    out = pd.DataFrame(index=prices.index)
    out["SPX"] = np.log(prices["SPX"]).diff()
    out["USGG10YR"] = prices["USGG10YR"].diff()
    out["DXY"] = np.log(prices["DXY"]).diff()
    return out.dropna()


# ---------------------------------------------------------------------------
# Pairwise rolling correlations
# ---------------------------------------------------------------------------
def rolling_pairwise_corrs(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling pairwise Pearson correlations.
    Returns a DataFrame with 3 columns:
      'SPX_vs_USGG10YR', 'SPX_vs_DXY', 'USGG10YR_vs_DXY'
    NaN rows trimmed.
    """
    a, b, c = "SPX", "USGG10YR", "DXY"
    out = pd.DataFrame(index=returns.index)
    out[f"{a}_vs_{b}"] = returns[a].rolling(window).corr(returns[b])
    out[f"{a}_vs_{c}"] = returns[a].rolling(window).corr(returns[c])
    out[f"{b}_vs_{c}"] = returns[b].rolling(window).corr(returns[c])
    return out.dropna()


def latest_pairwise_corrs(returns: pd.DataFrame, window: int = 60) -> dict:
    """Return today's pairwise correlations as a dict of floats."""
    rolled = rolling_pairwise_corrs(returns, window)
    last = rolled.iloc[-1]
    return {col: float(last[col]) for col in rolled.columns}


# ---------------------------------------------------------------------------
# PCA / Dominant theme
# ---------------------------------------------------------------------------
def pca_dominant_theme(returns: pd.DataFrame, window: int = 60) -> dict:
    """
    Run PCA on standardized returns over the LAST `window` days.
    Returns dict with:
      - explained_variance: fraction explained by PC1
      - loadings: dict {asset: loading on PC1, signed so SPX is positive}
      - n_obs: number of observations used
    """
    sub = returns.tail(window).dropna()
    if len(sub) < window // 2:
        return {"explained_variance": np.nan, "loadings": {}, "n_obs": len(sub)}

    # Standardize each column (z-score)
    z = (sub - sub.mean()) / sub.std(ddof=1)
    # Covariance of standardized data == correlation matrix
    cov = z.cov().values

    eig_vals, eig_vecs = np.linalg.eigh(cov)
    # eigh returns ascending — take the largest
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    pc1 = eig_vecs[:, 0]
    # Sign convention: make SPX loading positive so the theme is interpretable
    spx_idx = list(z.columns).index("SPX")
    if pc1[spx_idx] < 0:
        pc1 = -pc1

    explained = float(eig_vals[0] / eig_vals.sum())
    loadings = {col: float(pc1[i]) for i, col in enumerate(z.columns)}

    return {
        "explained_variance": explained,
        "loadings": loadings,
        "n_obs": len(sub),
    }


def rolling_pca_loadings(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling PC1 loadings over time. For each day t, compute PCA on the
    preceding `window` days. Returns DataFrame indexed by Date with columns:
      SPX_load, USGG10YR_load, DXY_load, ExplainedVar, EigGap

    [v2 — sign-stabilized + spike-resistant]

    Sign convention:
      - Within the time series, signs are aligned to the previous day's PC1
        so the curves don't randomly flip when a loading is near zero.
      - The entire series is then globally flipped (if needed) so that the
        MOST RECENT window has positive SPX loading, matching
        pca_dominant_theme()'s convention.
      - As a final guard, any single-day "spike" (a day where loadings differ
        wildly from BOTH neighbors) is forced to align with its neighbors.
        This catches edge cases where the sign-flip detector tied at
        dot-product == 0.

    EigGap = (lambda1 - lambda2) / lambda1.  When small (<~0.15), PC1 and
    PC2 are nearly equal in importance; the loadings are unreliable and
    callers should fade or filter those rows.
    """
    cols = list(returns.columns)
    spx_idx = cols.index("SPX")
    out_records = []
    prev_pc1 = None

    for end_idx in range(window, len(returns) + 1):
        sub = returns.iloc[end_idx - window:end_idx]
        if sub.isna().any().any():
            continue
        z = (sub - sub.mean()) / sub.std(ddof=1)
        cov = z.cov().values
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        pc1 = eig_vecs[:, 0]

        # Within-time stability: align to previous day's PC1.
        # If it's the very first window OR the dot product is exactly zero
        # (orthogonal), fall back to anchoring SPX positive.
        if prev_pc1 is None:
            if pc1[spx_idx] < 0:
                pc1 = -pc1
        else:
            dot = np.dot(pc1, prev_pc1)
            if dot < 0:
                pc1 = -pc1
            elif abs(dot) < 1e-6:
                # Degenerate — fall back to SPX-positive anchor
                if pc1[spx_idx] < 0:
                    pc1 = -pc1
        prev_pc1 = pc1.copy()

        eig_gap = float((eig_vals[0] - eig_vals[1]) / eig_vals[0])
        explained = float(eig_vals[0] / eig_vals.sum())

        out_records.append({
            "Date": sub.index[-1],
            "SPX_load": float(pc1[cols.index("SPX")]),
            "USGG10YR_load": float(pc1[cols.index("USGG10YR")]),
            "DXY_load": float(pc1[cols.index("DXY")]),
            "ExplainedVar": explained,
            "EigGap": eig_gap,
        })

    df = pd.DataFrame(out_records).set_index("Date")
    if df.empty:
        return df

    # Global sign flip so the most recent reading has SPX positive
    if df["SPX_load"].iloc[-1] < 0:
        df[["SPX_load", "USGG10YR_load", "DXY_load"]] *= -1

    # Final spike-killer: any day where ALL three loadings differ by >0.8
    # from BOTH the previous and next day is a sign artifact — flip it.
    # Run this as a single pass.
    load_cols = ["SPX_load", "USGG10YR_load", "DXY_load"]
    arr = df[load_cols].to_numpy()
    n = len(arr)
    if n >= 3:
        for i in range(1, n - 1):
            prev_diff = np.abs(arr[i] - arr[i - 1]).max()
            next_diff = np.abs(arr[i] - arr[i + 1]).max()
            flipped_diff_prev = np.abs(-arr[i] - arr[i - 1]).max()
            flipped_diff_next = np.abs(-arr[i] - arr[i + 1]).max()
            # If flipping would dramatically reduce both gaps, do it
            if (prev_diff > 0.8 and next_diff > 0.8 and
                flipped_diff_prev < prev_diff * 0.5 and
                flipped_diff_next < next_diff * 0.5):
                arr[i] = -arr[i]
        df[load_cols] = arr

    return df


# Module version marker — bump when math changes so we can verify deployment
__ANALYTICS_VERSION__ = "v2.1-2026-05-06"


# ---------------------------------------------------------------------------
# Helpers for interpreting correlations / loadings
# ---------------------------------------------------------------------------
def correlation_label(rho: float) -> str:
    """Human-readable strength label for a correlation."""
    a = abs(rho)
    if a < 0.2:
        return "weak"
    elif a < 0.4:
        return "moderate"
    elif a < 0.6:
        return "strong"
    else:
        return "very strong"


def correlation_story(pair: str, rho: float) -> str:
    """One-line interpretation of a correlation between two macro assets."""
    a = abs(rho)
    sign = "positive" if rho > 0 else "negative"
    if pair == "SPX_vs_USGG10YR":
        if rho < -0.3:
            return "Equities up when yields fall (flight-to-quality / central bank put)"
        elif rho > 0.3:
            return "Equities and yields rising together (growth-on regime)"
        else:
            return "Decoupled — neither growth nor risk dominating"
    if pair == "SPX_vs_DXY":
        if rho < -0.3:
            return "Equities and dollar moving opposite (typical risk-on/risk-off)"
        elif rho > 0.3:
            return "Equities and dollar both strong (US exceptionalism)"
        else:
            return "Decoupled — currency and equities driven by separate factors"
    if pair == "USGG10YR_vs_DXY":
        if rho > 0.3:
            return "Yields and dollar rising together (rate-driven dollar strength)"
        elif rho < -0.3:
            return "Yields and dollar diverging (unusual; often EM stress or carry unwind)"
        else:
            return "Rate-currency link weakening"
    return f"{sign} ({correlation_label(rho)})"


def loading_label(load: float) -> str:
    """Describe how strongly an asset loads on the dominant theme."""
    a = abs(load)
    if a < 0.3:
        return "barely"
    elif a < 0.5:
        return "moderate"
    else:
        return "heavy"
