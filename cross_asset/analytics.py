"""
Cross-asset analytics for SPX / UST 10Y / DXY.

Pure math, no Streamlit. Functions take DataFrames in and return DataFrames out.

Methodology variants
--------------------
The PCA/correlation results depend on two preprocessing choices:

1. Return scaling — how do we put SPX (~1% daily moves), UST 10Y yield
   (~5bp daily moves), and DXY (~0.4% daily moves) on a common scale?
     - "zscore"   : subtract window mean, divide by window std (default,
                    equivalent to PCA on the correlation matrix)
     - "volscale" : divide each return by its long-history rolling vol
                    BEFORE the window starts; preserves cross-window
                    comparability of magnitudes
     - "raw"      : no scaling — SPX dominates because of its variance.
                    Mathematically valid but rarely useful.

2. Window weighting — within each rolling window, do recent days weight
   more than old days?
     - "equal"    : every day weighted the same (default)
     - "ewm"      : exponentially-weighted, half-life = window/3 days

Conventions
-----------
- `prices` is a DataFrame with columns SPX, USGG10YR, DXY indexed by Date.
- For SPX and DXY we compute log returns.
- For USGG10YR (yield) we compute first differences (in pct points / 100bp).
- All rolling stats are right-aligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame, vol_scale: bool = False,
                    vol_window: int = 60) -> pd.DataFrame:
    """
    Convert raw levels to a returns DataFrame:
      SPX, DXY  -> log returns
      USGG10YR  -> first differences (pct points)

    If vol_scale=True, each return is divided by its trailing realized
    volatility (sqrt of trailing variance over `vol_window` days). This is
    the macro standard for putting heterogeneous assets on a common scale
    before correlation/PCA. Note: this introduces a vol_window-day burn-in
    where returns are NaN.
    """
    out = pd.DataFrame(index=prices.index)
    out["SPX"] = np.log(prices["SPX"]).diff()
    out["USGG10YR"] = prices["USGG10YR"].diff()
    out["DXY"] = np.log(prices["DXY"]).diff()
    out = out.dropna()

    if vol_scale:
        # Trailing realized vol per asset, then divide each day's return by it
        rv = out.rolling(window=vol_window, min_periods=max(20, vol_window // 2)).std()
        # Avoid division by zero on early NaNs
        scaled = out / rv
        out = scaled.dropna()

    return out


# ---------------------------------------------------------------------------
# Level-scaled state (for macro-regime PCA)
# ---------------------------------------------------------------------------
def compute_level_scaled(
    prices: pd.DataFrame,
    lookback: int = 500,
    smooth_halflife: int = 20,
) -> pd.DataFrame:
    """
    Convert raw prices to a "level-scaled" macro state series:
      SPX        -> log(SPX), then rolling z-score
      USGG10YR   -> raw yield level, then rolling z-score
      DXY        -> log(DXY), then rolling z-score
    Then optionally smooth the z-scored result with an EWMA halflife.

    This is the input for level-based PCA (vs the daily-returns-based PCA).
    The output captures slow macro regime evolution: where each asset sits
    relative to its trailing 2-year mean, in standard deviations.

    Parameters
    ----------
    lookback : int
        Rolling window for mean/std normalization (default 500 ≈ 2 years).
    smooth_halflife : int
        EWMA halflife applied AFTER z-scoring. 0 disables smoothing.
        Default 20 produces trending macro regime curves.
    """
    x = pd.DataFrame(index=prices.index)
    x["SPX"] = np.log(prices["SPX"])
    x["USGG10YR"] = prices["USGG10YR"]
    x["DXY"] = np.log(prices["DXY"])

    min_periods = max(60, lookback // 4)
    mu = x.rolling(lookback, min_periods=min_periods).mean()
    sig = x.rolling(lookback, min_periods=min_periods).std(ddof=1)

    z = ((x - mu) / sig).dropna()

    if smooth_halflife and smooth_halflife > 0:
        z = z.ewm(halflife=smooth_halflife, min_periods=1).mean()

    return z.dropna()


# ---------------------------------------------------------------------------
# Pairwise rolling correlations
# ---------------------------------------------------------------------------
def rolling_pairwise_corrs(returns: pd.DataFrame, window: int = 60,
                           weighting: str = "equal") -> pd.DataFrame:
    """
    Compute rolling pairwise Pearson correlations.

    weighting:
      "equal" — standard rolling correlation, all days in the window weighted same
      "ewm"   — exponentially-weighted, halflife = window/3 days

    Returns a DataFrame with 3 columns:
      'SPX_vs_USGG10YR', 'SPX_vs_DXY', 'USGG10YR_vs_DXY'
    """
    a, b, c = "SPX", "USGG10YR", "DXY"
    out = pd.DataFrame(index=returns.index)

    if weighting == "ewm":
        halflife = max(window / 3, 5)
        ewm = returns.ewm(halflife=halflife, min_periods=window // 2)
        # pandas ewm.corr returns a multi-index; pick the pairs we want
        # We compute three EWM correlations manually for clarity
        ewm_a = returns[a].ewm(halflife=halflife, min_periods=window // 2)
        out[f"{a}_vs_{b}"] = ewm_a.corr(returns[b])
        out[f"{a}_vs_{c}"] = ewm_a.corr(returns[c])
        ewm_b = returns[b].ewm(halflife=halflife, min_periods=window // 2)
        out[f"{b}_vs_{c}"] = ewm_b.corr(returns[c])
    else:
        # Equal-weighted rolling
        out[f"{a}_vs_{b}"] = returns[a].rolling(window).corr(returns[b])
        out[f"{a}_vs_{c}"] = returns[a].rolling(window).corr(returns[c])
        out[f"{b}_vs_{c}"] = returns[b].rolling(window).corr(returns[c])

    return out.dropna()


def latest_pairwise_corrs(returns: pd.DataFrame, window: int = 60,
                          weighting: str = "equal") -> dict:
    """Return today's pairwise correlations as a dict of floats."""
    rolled = rolling_pairwise_corrs(returns, window, weighting)
    if rolled.empty:
        return {}
    last = rolled.iloc[-1]
    return {col: float(last[col]) for col in rolled.columns}


# ---------------------------------------------------------------------------
# PCA / Dominant theme
# ---------------------------------------------------------------------------
def _make_weights(n: int, weighting: str) -> np.ndarray:
    """Construct a length-n weight vector that sums to n.
    'equal' = all 1s. 'ewm' = exponential decay w/ halflife=n/3."""
    if weighting == "ewm":
        halflife = max(n / 3, 5)
        decay = 0.5 ** (1.0 / halflife)
        # weights from oldest (i=0) to newest (i=n-1)
        idx = np.arange(n)
        w = decay ** (n - 1 - idx)
        w = w * (n / w.sum())  # normalize so weights sum to n
        return w
    return np.ones(n)


def _weighted_corr_matrix(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted correlation matrix on already-standardized data z (shape n x p).
    Weights sum to n. Returns p x p correlation matrix."""
    # weighted mean, std
    wsum = w.sum()
    mean = (w[:, None] * z).sum(axis=0) / wsum
    centered = z - mean
    cov = (w[:, None] * centered).T @ centered / wsum
    # standardize back to correlation
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    corr = cov / np.outer(std, std)
    return corr


def _weighted_cov_matrix(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted covariance matrix on data x (shape n x p). Weights sum to n.
    Returns p x p covariance matrix WITHOUT unit-variance standardization.

    Use this when you want PCA to preserve relative variance information
    between assets. Caller is responsible for ensuring `x` is on a
    comparable scale across columns (e.g., vol-scaled returns) — otherwise
    units will dominate the eigendecomposition.
    """
    wsum = w.sum()
    mean = (w[:, None] * x).sum(axis=0) / wsum
    centered = x - mean
    cov = (w[:, None] * centered).T @ centered / wsum
    return cov


def pca_dominant_theme(returns: pd.DataFrame, window: int = 60,
                       weighting: str = "equal") -> dict:
    """
    Run PCA on standardized returns over the LAST `window` days.
    Returns dict with:
      - explained_variance: fraction explained by PC1
      - loadings: dict {asset: loading on PC1, signed so SPX is positive}
      - n_obs: number of observations used
      - eig_gap: (lambda1 - lambda2) / lambda1
    """
    sub = returns.tail(window).dropna()
    if len(sub) < window // 2:
        return {"explained_variance": np.nan, "loadings": {}, "n_obs": len(sub),
                "eig_gap": np.nan}

    # z-score within window so PCA operates on correlation matrix
    z = ((sub - sub.mean()) / sub.std(ddof=1)).values
    w = _make_weights(len(sub), weighting)
    corr = _weighted_corr_matrix(z, w)

    eig_vals, eig_vecs = np.linalg.eigh(corr)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    pc1 = eig_vecs[:, 0]
    cols = list(sub.columns)
    spx_idx = cols.index("SPX")
    if pc1[spx_idx] < 0:
        pc1 = -pc1

    explained = float(eig_vals[0] / eig_vals.sum())
    loadings = {col: float(pc1[i]) for i, col in enumerate(cols)}
    eig_gap = float((eig_vals[0] - eig_vals[1]) / eig_vals[0])

    return {
        "explained_variance": explained,
        "loadings": loadings,
        "n_obs": len(sub),
        "eig_gap": eig_gap,
    }


def rolling_pca_loadings(returns: pd.DataFrame, window: int = 60,
                         weighting: str = "equal",
                         pca_method: str = "standard",
                         presmooth_halflife: int = 0) -> pd.DataFrame:
    """
    Rolling PC1 loadings over time. For each day t, compute weighted PCA on
    the preceding `window` days. Returns DataFrame indexed by Date with cols:
      SPX_load, USGG10YR_load, DXY_load, ExplainedVar, EigGap

    Parameters
    ----------
    window : int
        Rolling window length in days.
    weighting : "equal" | "ewm"
        Within-window weighting (passed to _make_weights).
    pca_method : "standard" | "procrustes"
        Sign convention only:
          - "standard"   : Anchor SPX positive independently per day.
                           Honest about regime changes; can show sign
                           jitter when SPX_load crosses zero.
          - "procrustes" : Rotate today's PC1 to be closest to yesterday's
                           PC1 (sign continuity). Smoother curves; may
                           obscure real regime changes.
    presmooth_halflife : int
        If > 0, pre-smooth daily returns with EWMA filter of this halflife
        BEFORE running rolling PCA. Common values: 5 (light), 10 (moderate),
        15-20 (heavy). 0 = no pre-smoothing.

    Pre-smoothing and pca_method are independent — combining
    `presmooth_halflife=15` with `pca_method="procrustes"` produces the
    smoothest curves the model can generate.
    """
    cols = list(returns.columns)
    spx_idx = cols.index("SPX")

    # ---- Pre-smoothing: applied first, independent of PCA method ----------
    if presmooth_halflife and presmooth_halflife > 0:
        halflife = max(int(presmooth_halflife), 1)
        returns_used = returns.ewm(halflife=halflife, min_periods=1).mean()
    else:
        returns_used = returns

    out_records = []
    prev_pc1 = None

    for end_idx in range(window, len(returns_used) + 1):
        sub = returns_used.iloc[end_idx - window:end_idx]
        if sub.isna().any().any():
            continue
        # Avoid degenerate std (e.g., if pre-smoothing made a column flat)
        std = sub.std(ddof=1)
        if (std == 0).any():
            continue
        z = ((sub - sub.mean()) / std).values
        w = _make_weights(len(sub), weighting)
        corr = _weighted_corr_matrix(z, w)

        eig_vals, eig_vecs = np.linalg.eigh(corr)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        pc1 = eig_vecs[:, 0]

        # Sign convention
        if pca_method == "procrustes":
            if prev_pc1 is None:
                if pc1[spx_idx] < 0:
                    pc1 = -pc1
            else:
                if np.dot(pc1, prev_pc1) < 0:
                    pc1 = -pc1
            prev_pc1 = pc1.copy()
        else:
            # "standard": anchor SPX positive each day
            if pc1[spx_idx] < 0:
                pc1 = -pc1

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

    # For procrustes only: ensure the LATEST window has SPX positive.
    if pca_method == "procrustes" and df["SPX_load"].iloc[-1] < 0:
        df[["SPX_load", "USGG10YR_load", "DXY_load"]] *= -1

    return df


def rolling_pca_loadings_level_scaled(
    level_scaled: pd.DataFrame,
    window: int = 120,
    weighting: str = "equal",
    pca_method: str = "procrustes",
) -> pd.DataFrame:
    """
    Rolling COVARIANCE PCA on the level-scaled macro state (output of
    `compute_level_scaled`).

    Captures slow regime evolution: which macro state variable contributes
    most to the dominant factor's direction over the rolling window.

    Loadings should be read as: "in level-scaled units (z-scores of
    log-prices and yield levels), how much does each asset contribute to
    the dominant macro factor."

    Parameters
    ----------
    level_scaled : DataFrame
        Output of compute_level_scaled(). Columns: SPX, USGG10YR, DXY.
    window : int
        Rolling window length (default 120 ≈ 6 months — appropriate for
        the slow level-scaled time series).
    weighting : "equal" | "ewm"
        Within-window weighting.
    pca_method : "standard" | "procrustes"
        Sign convention. Default procrustes for smooth regime tracking.

    Returns
    -------
    DataFrame with columns SPX_load, USGG10YR_load, DXY_load,
    ExplainedVar, EigGap. Indexed by Date.
    """
    cols = list(level_scaled.columns)
    spx_idx = cols.index("SPX")

    out_records = []
    prev_pc1 = None

    for end_idx in range(window, len(level_scaled) + 1):
        sub = level_scaled.iloc[end_idx - window:end_idx].dropna()
        if len(sub) < window // 2:
            continue

        x = sub.values
        w = _make_weights(len(sub), weighting)
        cov = _weighted_cov_matrix(x, w)

        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        pc1 = eig_vecs[:, 0]

        # Sign convention. For procrustes, anchor SPX positive on the very
        # first iteration (no previous reference) so the chain has a stable
        # starting orientation.
        if pca_method == "procrustes":
            if prev_pc1 is not None:
                if np.dot(pc1, prev_pc1) < 0:
                    pc1 = -pc1
            else:
                if pc1[spx_idx] < 0:
                    pc1 = -pc1
        else:
            if pc1[spx_idx] < 0:
                pc1 = -pc1
        prev_pc1 = pc1.copy()

        total_var = eig_vals.sum()
        explained = float(eig_vals[0] / total_var) if total_var > 0 else np.nan
        eig_gap = float((eig_vals[0] - eig_vals[1]) / eig_vals[0]) \
                  if eig_vals[0] > 0 else np.nan

        out_records.append({
            "Date": sub.index[-1],
            "SPX_load": float(pc1[cols.index("SPX")]),
            "USGG10YR_load": float(pc1[cols.index("USGG10YR")]),
            "DXY_load": float(pc1[cols.index("DXY")]),
            "ExplainedVar": explained,
            "EigGap": eig_gap,
        })

    df = pd.DataFrame(out_records)
    if df.empty:
        return df
    df = df.set_index("Date")

    if pca_method == "procrustes" and df["SPX_load"].iloc[-1] < 0:
        df[["SPX_load", "USGG10YR_load", "DXY_load"]] *= -1

    return df


# Module version marker — bump when math changes so we can verify deployment
__ANALYTICS_VERSION__ = "v7.2-2026-05-08"


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
