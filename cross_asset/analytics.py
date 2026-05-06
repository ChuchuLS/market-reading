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
    preceding `window` days. Returns DataFrame indexed by Date with
    columns: SPX_load, USGG10YR_load, DXY_load, ExplainedVar.
    Sign of loadings is anchored so SPX_load >= 0.
    """
    cols = list(returns.columns)
    spx_idx = cols.index("SPX")
    out_records = []

    for end_idx in range(window, len(returns) + 1):
        sub = returns.iloc[end_idx - window:end_idx]
        if sub.isna().any().any():
            continue
        z = (sub - sub.mean()) / sub.std(ddof=1)
        cov = z.cov().values
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eig_vals)[::-1]
        pc1 = eig_vecs[:, idx[0]]
        if pc1[spx_idx] < 0:
            pc1 = -pc1
        explained = eig_vals[idx[0]] / eig_vals.sum()
        out_records.append({
            "Date": sub.index[-1],
            "SPX_load": float(pc1[cols.index("SPX")]),
            "USGG10YR_load": float(pc1[cols.index("USGG10YR")]),
            "DXY_load": float(pc1[cols.index("DXY")]),
            "ExplainedVar": float(explained),
        })

    return pd.DataFrame(out_records).set_index("Date")


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
