"""
Cross-asset FICC analytics for SPX / UST 10Y / DXY / BCOM / HY OAS.

Pure math, no Streamlit. Functions take DataFrames in and return DataFrames out.

This is the 5-asset extension of cross_asset/analytics.py. The PCA machinery
is dimension-agnostic — it just works with 5 columns instead of 3. The new
addition is the leadership decomposition (leader + concentration), which makes
sense at any dimensionality but becomes essential at 5D where the eyeball
"who's leading?" question is harder to answer from raw loadings.

Asset conventions
-----------------
- SPX, DXY, BCOM       -> log returns
- USGG10YR (yield)     -> first differences (pct points)
- LF98OAS (HY spread)  -> first differences (bp / 100, ie pct points)

The two diff-based series are on the same scale (pct points) and the three
log-return series are on the same scale (~%/day), so within-window z-scoring
puts everything on a common footing. Vol-scaling is also supported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Asset list — single source of truth for column ordering
# ---------------------------------------------------------------------------
ASSETS = ["SPX", "USGG10YR", "DXY", "BCOM", "LF98OAS"]

# Display labels for UI (LF98OAS shown as "HY OAS")
ASSET_LABELS = {
    "SPX": "SPX",
    "USGG10YR": "UST 10Y",
    "DXY": "DXY",
    "BCOM": "BCOM",
    "LF98OAS": "HY OAS",
}

# Loading-column names in DataFrames
LOAD_COLS = [f"{a}_load" for a in ASSETS]


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def compute_returns(
    prices: pd.DataFrame, vol_scale: bool = False, vol_window: int = 60
) -> pd.DataFrame:
    """
    Convert raw levels to a returns DataFrame:
        SPX, DXY, BCOM  -> log returns
        USGG10YR        -> first differences (pct points)
        LF98OAS         -> first differences (pct points)

    If vol_scale=True, each return is divided by its trailing realized vol
    (sqrt of trailing variance over `vol_window` days). Introduces a
    vol_window-day burn-in.
    """
    out = pd.DataFrame(index=prices.index)
    out["SPX"] = np.log(prices["SPX"]).diff()
    out["USGG10YR"] = prices["USGG10YR"].diff()
    out["DXY"] = np.log(prices["DXY"]).diff()
    out["BCOM"] = np.log(prices["BCOM"]).diff()
    out["LF98OAS"] = prices["LF98OAS"].diff()
    out = out.dropna()

    if vol_scale:
        rv = out.rolling(window=vol_window, min_periods=max(20, vol_window // 2)).std()
        out = (out / rv).dropna()

    return out


# ---------------------------------------------------------------------------
# Pairwise rolling correlations
# ---------------------------------------------------------------------------
def _pair_key(a: str, b: str) -> str:
    return f"{a}_vs_{b}"


def all_pair_keys() -> list[str]:
    """Return the 10 pair keys for the 5-asset universe, in canonical order."""
    keys = []
    for i in range(len(ASSETS)):
        for j in range(i + 1, len(ASSETS)):
            keys.append(_pair_key(ASSETS[i], ASSETS[j]))
    return keys


def rolling_pairwise_corrs(
    returns: pd.DataFrame, window: int = 60, weighting: str = "equal"
) -> pd.DataFrame:
    """
    Compute rolling pairwise Pearson correlations for all 10 unique pairs
    in the 5-asset basket.

    weighting:
        "equal" — standard rolling correlation
        "ewm"   — exponentially-weighted, halflife = window/3 days
    """
    out = pd.DataFrame(index=returns.index)

    if weighting == "ewm":
        halflife = max(window / 3, 5)
        for i in range(len(ASSETS)):
            for j in range(i + 1, len(ASSETS)):
                a, b = ASSETS[i], ASSETS[j]
                ewm_a = returns[a].ewm(halflife=halflife, min_periods=window // 2)
                out[_pair_key(a, b)] = ewm_a.corr(returns[b])
    else:
        for i in range(len(ASSETS)):
            for j in range(i + 1, len(ASSETS)):
                a, b = ASSETS[i], ASSETS[j]
                out[_pair_key(a, b)] = returns[a].rolling(window).corr(returns[b])

    return out.dropna()


def latest_pairwise_corrs(
    returns: pd.DataFrame, window: int = 60, weighting: str = "equal"
) -> dict:
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
    """Length-n weight vector that sums to n. 'equal' = all 1s.
    'ewm' = exponential decay w/ halflife=n/3."""
    if weighting == "ewm":
        halflife = max(n / 3, 5)
        decay = 0.5 ** (1.0 / halflife)
        idx = np.arange(n)
        w = decay ** (n - 1 - idx)
        w = w * (n / w.sum())
        return w
    return np.ones(n)


def _weighted_corr_matrix(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted correlation matrix on already-standardized data z (n x p).
    Weights sum to n. Returns p x p correlation matrix."""
    wsum = w.sum()
    mean = (w[:, None] * z).sum(axis=0) / wsum
    centered = z - mean
    cov = (w[:, None] * centered).T @ centered / wsum
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    corr = cov / np.outer(std, std)
    return corr


def pca_dominant_theme(
    returns: pd.DataFrame, window: int = 60, weighting: str = "equal"
) -> dict:
    """
    Run PCA on standardized returns over the LAST `window` days (5 assets).

    Returns dict with:
        - explained_variance: fraction explained by PC1
        - loadings: dict {asset: loading on PC1, signed so SPX is positive}
        - n_obs: number of observations used
        - eig_gap: (lambda1 - lambda2) / lambda1
    """
    sub = returns.tail(window).dropna()
    if len(sub) < window // 2:
        return {
            "explained_variance": np.nan,
            "loadings": {},
            "n_obs": len(sub),
            "eig_gap": np.nan,
        }

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


def rolling_pca_loadings(
    returns: pd.DataFrame,
    window: int = 60,
    weighting: str = "equal",
    pca_method: str = "standard",
    presmooth_halflife: int = 0,
) -> pd.DataFrame:
    """
    Rolling PC1 loadings over time for the 5-asset basket. For each day t,
    compute weighted PCA on the preceding `window` days. Returns a DataFrame
    indexed by Date with columns:

        SPX_load, USGG10YR_load, DXY_load, BCOM_load, LF98OAS_load,
        ExplainedVar, EigGap

    Parameters mirror cross_asset/analytics.rolling_pca_loadings:
      pca_method : "standard" anchors SPX positive each day
                   "procrustes" rotates today's PC1 to be closest to yesterday's
      presmooth_halflife : EWMA pre-smoothing of returns (0 = off)
    """
    cols = list(returns.columns)
    if cols != ASSETS:
        # Reorder to canonical so the loading column ordering is stable
        returns = returns[ASSETS]
        cols = ASSETS
    spx_idx = cols.index("SPX")

    # Pre-smoothing applied first
    if presmooth_halflife and presmooth_halflife > 0:
        halflife = max(int(presmooth_halflife), 1)
        returns_used = returns.ewm(halflife=halflife, min_periods=1).mean()
    else:
        returns_used = returns

    out_records = []
    prev_pc1 = None

    for end_idx in range(window, len(returns_used) + 1):
        sub = returns_used.iloc[end_idx - window : end_idx]
        if sub.isna().any().any():
            continue
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

        if pca_method == "procrustes":
            if prev_pc1 is None:
                if pc1[spx_idx] < 0:
                    pc1 = -pc1
            else:
                if np.dot(pc1, prev_pc1) < 0:
                    pc1 = -pc1
            prev_pc1 = pc1.copy()
        else:
            if pc1[spx_idx] < 0:
                pc1 = -pc1

        eig_gap = float((eig_vals[0] - eig_vals[1]) / eig_vals[0])
        explained = float(eig_vals[0] / eig_vals.sum())

        record = {"Date": sub.index[-1]}
        for i, col in enumerate(cols):
            record[f"{col}_load"] = float(pc1[i])
        record["ExplainedVar"] = explained
        record["EigGap"] = eig_gap
        out_records.append(record)

    df = pd.DataFrame(out_records).set_index("Date")
    if df.empty:
        return df

    # For procrustes only: ensure the LATEST window has SPX positive
    if pca_method == "procrustes" and df["SPX_load"].iloc[-1] < 0:
        for col in LOAD_COLS:
            df[col] *= -1

    return df


# ---------------------------------------------------------------------------
# Leadership decomposition — what's missing in the 3-asset module
# ---------------------------------------------------------------------------
def leadership_stats(loadings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-day leadership characterization of the dominant theme.

    Returns DataFrame with columns:
        Leader        : asset key with largest |loading| (e.g. "SPX")
        LeaderLabel   : display label (e.g. "SPX")
        LeaderLoad    : signed loading of the leader
        Concentration : max(load^2) / sum(load^2). Range 1/n (perfectly
                        diffuse) to 1.0 (single-asset move). For 5 assets,
                        floor is 0.20.
        SignPattern   : 5-char string of signs in canonical asset order,
                        e.g. "+−+−+" (SPX, UST10Y, DXY, BCOM, HY OAS)

    Concentration is the right "is this regime concentrated?" measure because
    it's invariant to total loading magnitude — it only cares about the shape
    of the loading vector across assets.
    """
    if loadings_df.empty:
        return pd.DataFrame(
            index=loadings_df.index,
            columns=[
                "Leader",
                "LeaderLabel",
                "LeaderLoad",
                "Concentration",
                "SignPattern",
            ],
        )

    L = loadings_df[LOAD_COLS].values  # (T, 5)
    abs_L = np.abs(L)
    sq = L**2
    sq_sum = sq.sum(axis=1)
    sq_sum_safe = np.where(sq_sum > 0, sq_sum, 1.0)

    leader_idx = abs_L.argmax(axis=1)
    leader_keys = [ASSETS[i] for i in leader_idx]
    leader_labels = [ASSET_LABELS[k] for k in leader_keys]
    leader_loads = L[np.arange(len(L)), leader_idx]
    concentration = sq.max(axis=1) / sq_sum_safe

    # Sign pattern uses Unicode minus for clarity in display
    def _sign_str(row: np.ndarray) -> str:
        return "".join("+" if v >= 0 else "−" for v in row)

    sign_patterns = [_sign_str(row) for row in L]

    return pd.DataFrame(
        {
            "Leader": leader_keys,
            "LeaderLabel": leader_labels,
            "LeaderLoad": leader_loads,
            "Concentration": concentration,
            "SignPattern": sign_patterns,
        },
        index=loadings_df.index,
    )


# ---------------------------------------------------------------------------
# Helpers for interpretation (less story-driven than 3-asset version since
# 5-asset sign patterns don't have canonical economic narratives)
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


def correlation_summary(pair: str, rho: float) -> str:
    """One-line factual summary of a correlation between two assets."""
    a_lab, b_lab = pair.split("_vs_")
    a_lab = ASSET_LABELS.get(a_lab, a_lab)
    b_lab = ASSET_LABELS.get(b_lab, b_lab)
    sign_word = "co-move" if rho > 0 else "diverge"
    return f"{a_lab} and {b_lab} {sign_word} ({correlation_label(rho)})"


def loading_label(load: float) -> str:
    """Describe how strongly an asset loads on the dominant theme."""
    a = abs(load)
    if a < 0.25:
        return "barely"
    elif a < 0.45:
        return "moderate"
    else:
        return "heavy"


def concentration_label(c: float) -> str:
    """Describe how concentrated leadership is. Floor for 5 assets is 0.20."""
    if c < 0.30:
        return "diffuse"  # close to 1/5 = 0.20
    elif c < 0.40:
        return "mild lead"
    elif c < 0.55:
        return "clear lead"
    else:
        return "dominant"


# ---------------------------------------------------------------------------
__ANALYTICS_VERSION__ = "ficc-v1.0-2026-05-09"
