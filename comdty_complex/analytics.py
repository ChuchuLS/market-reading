"""
Commodities within-complex analytics.

Sub-components:
- BCOM   : Bloomberg Commodity Index (broad)
- CL1    : WTI Crude (energy)
- HG1    : Copper (industrial metals — growth proxy)
- GC1    : Gold (precious — fear / real-rates proxy)
- S 1    : Soybeans (ags)

Asset conventions
-----------------
All five are log returns — clean homogeneous units.

Sign anchor
-----------
PC1 anchored on BCOM positive. So "+" = broad commodity move up.
The interesting sign patterns are:
- All five same-sign         : pure broad commodity regime
- Energy and metals split    : growth-vs-inflation tension
- Gold opposite              : flight-to-quality vs growth
- Ags opposite               : weather/idiosyncratic vs macro
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ASSETS = ["BCOM", "CL1", "HG1", "GC1", "S1"]

ASSET_LABELS = {
    "BCOM": "BCOM",
    "CL1":  "WTI Crude",
    "HG1":  "Copper",
    "GC1":  "Gold",
    "S1":   "Soybeans",
}

ANCHOR = "BCOM"

LOAD_COLS = [f"{a}_load" for a in ASSETS]


def compute_returns(prices: pd.DataFrame, vol_scale: bool = False,
                    vol_window: int = 60) -> pd.DataFrame:
    """All 5 sub-components are log returns."""
    out = pd.DataFrame(index=prices.index)
    for a in ASSETS:
        out[a] = np.log(prices[a]).diff()
    out = out.dropna()

    if vol_scale:
        rv = out.rolling(window=vol_window,
                         min_periods=max(20, vol_window // 2)).std()
        out = (out / rv).dropna()

    return out


def _pair_key(a: str, b: str) -> str:
    return f"{a}_vs_{b}"


def all_pair_keys() -> list[str]:
    keys = []
    for i in range(len(ASSETS)):
        for j in range(i + 1, len(ASSETS)):
            keys.append(_pair_key(ASSETS[i], ASSETS[j]))
    return keys


def rolling_pairwise_corrs(returns: pd.DataFrame, window: int = 60,
                           weighting: str = "equal") -> pd.DataFrame:
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


def latest_pairwise_corrs(returns: pd.DataFrame, window: int = 60,
                          weighting: str = "equal") -> dict:
    rolled = rolling_pairwise_corrs(returns, window, weighting)
    if rolled.empty:
        return {}
    last = rolled.iloc[-1]
    return {col: float(last[col]) for col in rolled.columns}


def _make_weights(n: int, weighting: str) -> np.ndarray:
    if weighting == "ewm":
        halflife = max(n / 3, 5)
        decay = 0.5 ** (1.0 / halflife)
        idx = np.arange(n)
        w = decay ** (n - 1 - idx)
        w = w * (n / w.sum())
        return w
    return np.ones(n)


def _weighted_corr_matrix(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    wsum = w.sum()
    mean = (w[:, None] * z).sum(axis=0) / wsum
    centered = z - mean
    cov = (w[:, None] * centered).T @ centered / wsum
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    return cov / np.outer(std, std)


def pca_dominant_theme(returns: pd.DataFrame, window: int = 60,
                       weighting: str = "equal") -> dict:
    sub = returns.tail(window).dropna()
    if len(sub) < window // 2:
        return {"explained_variance": np.nan, "loadings": {},
                "n_obs": len(sub), "eig_gap": np.nan}

    z = ((sub - sub.mean()) / sub.std(ddof=1)).values
    w = _make_weights(len(sub), weighting)
    corr = _weighted_corr_matrix(z, w)
    eig_vals, eig_vecs = np.linalg.eigh(corr)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    pc1 = eig_vecs[:, 0]

    cols = list(sub.columns)
    anchor_idx = cols.index(ANCHOR)
    if pc1[anchor_idx] < 0:
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
    cols = list(returns.columns)
    if cols != ASSETS:
        returns = returns[ASSETS]
        cols = ASSETS
    anchor_idx = cols.index(ANCHOR)

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
                if pc1[anchor_idx] < 0:
                    pc1 = -pc1
            else:
                if np.dot(pc1, prev_pc1) < 0:
                    pc1 = -pc1
            prev_pc1 = pc1.copy()
        else:
            if pc1[anchor_idx] < 0:
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

    if pca_method == "procrustes" and df[f"{ANCHOR}_load"].iloc[-1] < 0:
        for col in LOAD_COLS:
            df[col] *= -1

    return df


def leadership_stats(loadings_df: pd.DataFrame) -> pd.DataFrame:
    if loadings_df.empty:
        return pd.DataFrame(
            index=loadings_df.index,
            columns=["Leader", "LeaderLabel", "LeaderLoad",
                     "Concentration", "SignPattern"],
        )

    L = loadings_df[LOAD_COLS].values
    abs_L = np.abs(L)
    sq = L ** 2
    sq_sum = sq.sum(axis=1)
    sq_sum_safe = np.where(sq_sum > 0, sq_sum, 1.0)

    leader_idx = abs_L.argmax(axis=1)
    leader_keys = [ASSETS[i] for i in leader_idx]
    leader_labels = [ASSET_LABELS[k] for k in leader_keys]
    leader_loads = L[np.arange(len(L)), leader_idx]
    concentration = sq.max(axis=1) / sq_sum_safe

    def _sign_str(row: np.ndarray) -> str:
        return "".join("+" if v >= 0 else "−" for v in row)
    sign_patterns = [_sign_str(row) for row in L]

    return pd.DataFrame({
        "Leader":        leader_keys,
        "LeaderLabel":   leader_labels,
        "LeaderLoad":    leader_loads,
        "Concentration": concentration,
        "SignPattern":   sign_patterns,
    }, index=loadings_df.index)


def headline_vs_breadth(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(columns=["HeadlineMove", "ComplexMedian", "Divergence"])
    rv = returns.rolling(window=60, min_periods=20).std()
    scaled = (returns / rv).abs()
    cum = scaled.rolling(window=window, min_periods=window).sum()
    headline = cum[ANCHOR]
    complex_median = cum[ASSETS].median(axis=1)
    out = pd.DataFrame({
        "HeadlineMove":  headline,
        "ComplexMedian": complex_median,
        "Divergence":    headline - complex_median,
    }, index=returns.index)
    return out.dropna()


def gold_copper_ratio(prices: pd.DataFrame) -> dict:
    """
    Gold/Copper ratio — classic 'fear vs growth' commodity signal.
    Rising ratio = gold outperforming copper = risk-off / slowdown bias.
    Falling ratio = copper outperforming gold = growth-on / cyclical bias.
    """
    if "GC1" not in prices.columns or "HG1" not in prices.columns:
        return {}
    ratio = prices["GC1"] / prices["HG1"]
    last = float(ratio.iloc[-1])
    pct_chg_1m = float(ratio.iloc[-1] / ratio.iloc[-21] - 1) * 100 if len(ratio) >= 21 else 0.0
    pct_chg_3m = float(ratio.iloc[-1] / ratio.iloc[-63] - 1) * 100 if len(ratio) >= 63 else 0.0

    if pct_chg_1m > 5:
        regime_label = "FEAR-ON"
    elif pct_chg_1m > 1:
        regime_label = "DEFENSIVE TILT"
    elif pct_chg_1m < -5:
        regime_label = "GROWTH-ON"
    elif pct_chg_1m < -1:
        regime_label = "CYCLICAL TILT"
    else:
        regime_label = "BALANCED"

    return {
        "ratio":        ratio,
        "last":         last,
        "pct_chg_1m":   pct_chg_1m,
        "pct_chg_3m":   pct_chg_3m,
        "regime_label": regime_label,
    }


def oil_summary(prices: pd.DataFrame) -> dict:
    """Oil level + recent moves."""
    if "CL1" not in prices.columns:
        return {}
    s = prices["CL1"].dropna()
    if s.empty:
        return {}
    last = float(s.iloc[-1])
    pct_chg_1m = float(s.iloc[-1] / s.iloc[-21] - 1) * 100 if len(s) >= 21 else 0.0
    pct_chg_3m = float(s.iloc[-1] / s.iloc[-63] - 1) * 100 if len(s) >= 63 else 0.0
    return {
        "last":       last,
        "pct_chg_1m": pct_chg_1m,
        "pct_chg_3m": pct_chg_3m,
    }


def correlation_label(rho: float) -> str:
    a = abs(rho)
    if a < 0.2:   return "weak"
    elif a < 0.4: return "moderate"
    elif a < 0.6: return "strong"
    else:         return "very strong"


def correlation_summary(pair: str, rho: float) -> str:
    a_lab, b_lab = pair.split("_vs_")
    a_lab = ASSET_LABELS.get(a_lab, a_lab)
    b_lab = ASSET_LABELS.get(b_lab, b_lab)
    sign_word = "co-move" if rho > 0 else "diverge"
    return f"{a_lab} and {b_lab} {sign_word} ({correlation_label(rho)})"


def loading_label(load: float) -> str:
    a = abs(load)
    if a < 0.25:  return "barely"
    elif a < 0.45: return "moderate"
    else:          return "heavy"


def concentration_label(c: float) -> str:
    """For 5 assets, floor is 0.20."""
    if c < 0.30:   return "diffuse"
    elif c < 0.40: return "mild lead"
    elif c < 0.55: return "clear lead"
    else:          return "dominant"


__ANALYTICS_VERSION__ = "comdty-v1.0-2026-05-09"
