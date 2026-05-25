"""
Sector within-complex analytics.

Universe: 11 US GICS sector SPDR ETFs.
  XLK Technology · XLF Financials · XLE Energy · XLV Health Care
  XLI Industrials · XLY Cons. Discretionary · XLP Cons. Staples
  XLU Utilities · XLB Materials · XLRE Real Estate · XLC Communications

Three lenses combined in this complex:
  1. Inter-sector correlation regime — average pairwise correlation across
     the 11 sectors. High = macro-driven market (top-down bets work),
     low = stock-picker's market (security selection works). Shown as a
     percentile vs trailing history because avg correlation drifts
     structurally over time.
  2. Sector PCA — PC1 is typically broad market beta; PC2/PC3 capture style
     rotation (growth/value, cyclical/defensive). Reuses the leader+sign
     regime-classification framework shared across all complexes.
  3. Relative strength leadership — each sector's trailing return vs SPY,
     ranked. The actionable "which sectors lead/lag right now" layer.

Asset conventions
-----------------
All 11 ETFs: price level → log returns.
SPY (benchmark) is loaded alongside for relative-strength calcs but is NOT
part of the PCA universe.

Sign anchor
-----------
PC1 anchored so XLK_load is positive each day (tech is the largest sector
weight and a natural risk-on proxy). So "+" = risk-on direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

ASSET_LABELS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Cons Disc",
    "XLP": "Cons Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Comm Svcs",
}

# Cyclical vs defensive grouping (used for the risk-appetite read)
CYCLICAL = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLB", "XLC"]
DEFENSIVE = ["XLV", "XLP", "XLU", "XLRE"]

ANCHOR = "XLK"
BENCHMARK = "SPY"

LOAD_COLS = [f"{a}_load" for a in ASSETS]


def compute_returns(
    prices: pd.DataFrame, vol_scale: bool = False, vol_window: int = 60
) -> pd.DataFrame:
    """All sector ETFs: log returns. Benchmark excluded from this frame."""
    out = pd.DataFrame(index=prices.index)
    for a in ASSETS:
        if a in prices.columns:
            out[a] = np.log(prices[a]).diff()
    out = out.dropna()

    if vol_scale:
        rv = out.rolling(window=vol_window, min_periods=max(20, vol_window // 2)).std()
        out = (out / rv).dropna()

    return out


def benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """Log returns of the SPY benchmark (for relative-strength calcs)."""
    if BENCHMARK not in prices.columns:
        return pd.Series(dtype=float)
    return np.log(prices[BENCHMARK]).diff().dropna()


# ---------------------------------------------------------------------------
# Pairwise correlations (full matrix support for many assets)
# ---------------------------------------------------------------------------
def _pair_key(a: str, b: str) -> str:
    return f"{a}_vs_{b}"


def all_pair_keys() -> list[str]:
    keys = []
    for i in range(len(ASSETS)):
        for j in range(i + 1, len(ASSETS)):
            keys.append(_pair_key(ASSETS[i], ASSETS[j]))
    return keys


def _make_weights(n: int, weighting: str) -> np.ndarray:
    if weighting == "ewm":
        halflife = max(n / 3.0, 1.0)
        lam = np.log(2) / halflife
        w = np.exp(-lam * (n - 1 - np.arange(n)))
    else:
        w = np.ones(n)
    return w * n / w.sum()


def _weighted_corr_matrix(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    wsum = w.sum()
    mean = (w[:, None] * z).sum(axis=0) / wsum
    centered = z - mean
    cov = (w[:, None] * centered).T @ centered / wsum
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0
    return cov / np.outer(std, std)


# ---------------------------------------------------------------------------
# 1. Inter-sector correlation regime
# ---------------------------------------------------------------------------
def avg_pairwise_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling average of the off-diagonal entries of the sector correlation
    matrix. High = sectors moving together (macro-driven market);
    low = sectors diverging (stock-picker's market).
    """
    cols = [a for a in ASSETS if a in returns.columns]
    sub_all = returns[cols]
    n_assets = len(cols)
    if n_assets < 2:
        return pd.DataFrame(columns=["AvgCorr"])

    out = []
    for end in range(window, len(sub_all) + 1):
        sub = sub_all.iloc[end - window : end]
        if sub.isna().any().any():
            continue
        C = sub.corr().values
        # mean of off-diagonal entries
        off_mean = (C.sum() - n_assets) / (n_assets * (n_assets - 1))
        out.append({"Date": sub.index[-1], "AvgCorr": float(off_mean)})

    if not out:
        return pd.DataFrame(columns=["AvgCorr"])
    return pd.DataFrame(out).set_index("Date")


def correlation_regime_summary(
    corr_series: pd.DataFrame, pctile_lookback: int = 504
) -> dict:
    """
    Summarize the latest avg-correlation reading as a percentile vs trailing
    history (default ~2 years of trading days). Returns level, percentile,
    and a label.
    """
    if corr_series.empty:
        return {}

    s = corr_series["AvgCorr"].dropna()
    if s.empty:
        return {}

    latest = float(s.iloc[-1])
    trailing = s.tail(pctile_lookback)
    pctile = float((trailing < latest).mean() * 100)

    if pctile >= 80:
        label = "MACRO MARKET"
        desc = "sectors moving together — top-down/factor bets favored"
    elif pctile >= 60:
        label = "LEANING MACRO"
        desc = "moderately correlated — mixed regime"
    elif pctile >= 40:
        label = "NEUTRAL"
        desc = "average dispersion"
    elif pctile >= 20:
        label = "LEANING MICRO"
        desc = "below-average correlation — selection starting to matter"
    else:
        label = "STOCK-PICKER'S MARKET"
        desc = "sectors diverging — security selection favored"

    return {
        "latest": latest,
        "percentile": pctile,
        "label": label,
        "desc": desc,
        "median": float(trailing.median()),
    }


# ---------------------------------------------------------------------------
# 2. Sector PCA
# ---------------------------------------------------------------------------
def pca_dominant_theme(
    returns: pd.DataFrame, window: int = 60, weighting: str = "equal"
) -> dict:
    cols = [a for a in ASSETS if a in returns.columns]
    sub = returns[cols].tail(window).dropna()
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

    anchor_idx = cols.index(ANCHOR) if ANCHOR in cols else 0
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


def rolling_pca_loadings(
    returns: pd.DataFrame,
    window: int = 60,
    weighting: str = "equal",
    pca_method: str = "standard",
    presmooth_halflife: int = 0,
) -> pd.DataFrame:
    cols = [a for a in ASSETS if a in returns.columns]
    returns = returns[cols]
    anchor_idx = cols.index(ANCHOR) if ANCHOR in cols else 0

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

    df = pd.DataFrame(out_records)
    if df.empty:
        return df
    df = df.set_index("Date")

    if pca_method == "procrustes" and df[f"{ANCHOR}_load"].iloc[-1] < 0:
        for col in LOAD_COLS:
            if col in df.columns:
                df[col] *= -1

    return df


def leadership_stats(loadings_df: pd.DataFrame) -> pd.DataFrame:
    """Per-day leader (largest |loading|), concentration, and sign pattern."""
    present_cols = [c for c in LOAD_COLS if c in loadings_df.columns]
    present_assets = [c.replace("_load", "") for c in present_cols]

    if loadings_df.empty or not present_cols:
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

    L = loadings_df[present_cols].values
    abs_L = np.abs(L)
    sq = L**2
    sq_sum = sq.sum(axis=1)
    sq_sum_safe = np.where(sq_sum > 0, sq_sum, 1.0)

    leader_idx = abs_L.argmax(axis=1)
    leader_keys = [present_assets[i] for i in leader_idx]
    leader_labels = [ASSET_LABELS[k] for k in leader_keys]
    leader_loads = L[np.arange(len(L)), leader_idx]
    concentration = sq.max(axis=1) / sq_sum_safe

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
# 3. Relative strength leadership
# ---------------------------------------------------------------------------
def relative_strength(
    returns: pd.DataFrame, bench_ret: pd.Series, window: int = 60
) -> pd.DataFrame:
    """
    Trailing cumulative return of each sector minus the benchmark's
    cumulative return over the same window. Sorted strongest to weakest.
    """
    cols = [a for a in ASSETS if a in returns.columns]
    sec = returns[cols].tail(window)
    if len(sec) < 2 or bench_ret.empty:
        return pd.DataFrame(columns=["Sector", "Label", "RelStrength", "AbsReturn"])

    # Align benchmark to the same window
    bench = bench_ret.reindex(sec.index).fillna(0.0)

    cum_sec = np.exp(sec.sum()) - 1.0  # log-returns → simple cum return
    cum_bench = float(np.exp(bench.sum()) - 1.0)

    rs = (cum_sec - cum_bench).sort_values(ascending=False)

    return pd.DataFrame(
        {
            "Sector": rs.index,
            "Label": [ASSET_LABELS[s] for s in rs.index],
            "RelStrength": rs.values,
            "AbsReturn": [float(cum_sec[s]) for s in rs.index],
        }
    ).reset_index(drop=True)


def cyclical_defensive_spread(returns: pd.DataFrame, window: int = 60) -> dict:
    """
    Cyclical basket return minus defensive basket return over the window.
    Positive = market favoring growth/risk; negative = market favoring safety.
    Equal-weighted within each basket.
    """
    cyc = [a for a in CYCLICAL if a in returns.columns]
    dfn = [a for a in DEFENSIVE if a in returns.columns]
    if not cyc or not dfn:
        return {}

    sub = returns.tail(window)
    cyc_ret = float(np.exp(sub[cyc].mean(axis=1).sum()) - 1.0)
    dfn_ret = float(np.exp(sub[dfn].mean(axis=1).sum()) - 1.0)
    spread = cyc_ret - dfn_ret

    if spread > 0.02:
        label = "RISK-ON (cyclicals leading)"
    elif spread > -0.02:
        label = "NEUTRAL"
    else:
        label = "RISK-OFF (defensives leading)"

    return {
        "cyclical_return": cyc_ret,
        "defensive_return": dfn_ret,
        "spread": spread,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Two-axis sector regime classification
#   Axis 1 (risk):      cyclical − defensive spread  → Risk-On / Risk-Off
#   Axis 2 (structure): inter-sector correlation     → Macro / Micro
# This replaces the single-leader PCA labeling, which is meaningless across
# 11 diffuse sectors. The two axes give four interpretable quadrant regimes.
# ---------------------------------------------------------------------------
RISK_DEADBAND = 0.01  # |spread| below this → risk-neutral
CORR_MACRO_PCTILE = 60.0  # corr percentile ≥ this → Macro
CORR_MICRO_PCTILE = 40.0  # corr percentile ≤ this → Micro


def rolling_cyc_def_spread(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling cyclical−defensive cumulative-return spread (one value per day)."""
    cyc = [a for a in CYCLICAL if a in returns.columns]
    dfn = [a for a in DEFENSIVE if a in returns.columns]
    if not cyc or not dfn:
        return pd.Series(dtype=float, name="Spread")

    cyc_mean = returns[cyc].mean(axis=1)
    dfn_mean = returns[dfn].mean(axis=1)
    out = []
    idx = []
    for end in range(window, len(returns) + 1):
        c = float(np.exp(cyc_mean.iloc[end - window : end].sum()) - 1.0)
        d = float(np.exp(dfn_mean.iloc[end - window : end].sum()) - 1.0)
        out.append(c - d)
        idx.append(returns.index[end - 1])
    return pd.Series(out, index=idx, name="Spread")


def classify_sector_regime(
    spread_series: pd.Series,
    corr_series: pd.DataFrame,
    pctile_lookback: int = 504,
) -> pd.Series:
    """
    Classify each day into one of the two-axis regimes.

    Risk axis from the cyclical-defensive spread (deadband → "Neutral").
    Structure axis from the rolling-correlation percentile vs trailing
    history (between thresholds → "Bal").

    Labels: "Risk-On / Macro", "Risk-On / Micro", "Risk-Off / Macro",
    "Risk-Off / Micro", and balanced/neutral variants. The label is built
    as "<risk> / <structure>".
    """
    if spread_series.empty or corr_series.empty:
        return pd.Series(dtype="object", name="Regime")

    corr = corr_series["AvgCorr"]
    # align both series to common dates
    common = spread_series.index.intersection(corr.index)
    spread = spread_series.reindex(common)
    corr = corr.reindex(common)

    # rolling percentile of correlation vs trailing lookback
    def _pctile_at(i: int) -> float:
        lo = max(0, i - pctile_lookback + 1)
        window_vals = corr.iloc[lo : i + 1]
        cur = corr.iloc[i]
        if window_vals.empty or pd.isna(cur):
            return np.nan
        return float((window_vals < cur).mean() * 100)

    labels = []
    for i, date in enumerate(common):
        sp = spread.iloc[i]
        pct = _pctile_at(i)

        # risk axis
        if pd.isna(sp):
            risk = "Neutral"
        elif sp > RISK_DEADBAND:
            risk = "Risk-On"
        elif sp < -RISK_DEADBAND:
            risk = "Risk-Off"
        else:
            risk = "Neutral"

        # structure axis
        if pd.isna(pct):
            struct = "Bal"
        elif pct >= CORR_MACRO_PCTILE:
            struct = "Macro"
        elif pct <= CORR_MICRO_PCTILE:
            struct = "Micro"
        else:
            struct = "Bal"

        labels.append(f"{risk} / {struct}")

    return pd.Series(labels, index=common, name="Regime")


def smooth_regime(regime_series: pd.Series, min_days: int = 3) -> pd.Series:
    """
    Suppress regime flips shorter than `min_days` by carrying forward the
    previous confirmed regime. Reduces threshold-boundary chatter so the
    timeline shows durable regimes rather than 1-2 day blips.

    A new regime is only "confirmed" once it has held for min_days
    consecutive observations; until then the prior confirmed regime persists.
    """
    if regime_series.empty:
        return regime_series

    vals = regime_series.tolist()
    idx = regime_series.index
    out = vals[:]  # copy

    confirmed = vals[0]
    out[0] = confirmed
    i = 1
    while i < len(vals):
        if vals[i] == confirmed:
            out[i] = confirmed
            i += 1
            continue
        # candidate new regime — look ahead to see if it holds min_days
        cand = vals[i]
        run_len = 1
        j = i + 1
        while j < len(vals) and vals[j] == cand:
            run_len += 1
            j += 1
        if run_len >= min_days:
            for k in range(i, j):
                out[k] = cand
            confirmed = cand
            i = j
        else:
            for k in range(i, j):
                out[k] = confirmed
            i = j

    return pd.Series(out, index=idx, name="Regime")


def sector_regime_info(
    regime_series: pd.Series,
    spread_series: pd.Series,
    corr_summary: dict,
    cd_detail: dict,
) -> dict:
    """Current-state summary for the regime card."""
    if regime_series.empty:
        return {}
    current = regime_series.iloc[-1]
    last_date = regime_series.index[-1]
    start_date = last_date
    for d in regime_series.index[::-1]:
        if regime_series.loc[d] == current:
            start_date = d
        else:
            break
    days_in = int((regime_series.index >= start_date).sum())

    latest_spread = float(spread_series.iloc[-1]) if not spread_series.empty else np.nan

    return {
        "regime": current,
        "since": start_date,
        "days_in": days_in,
        "spread": latest_spread,
        "corr_percentile": corr_summary.get("percentile"),
        "corr_level": corr_summary.get("latest"),
        "cyclical_return": cd_detail.get("cyclical_return"),
        "defensive_return": cd_detail.get("defensive_return"),
    }


SECTOR_REGIME_COLOR = {
    "Risk-On / Macro": "#22c55e",  # green — broad rally
    "Risk-On / Micro": "#14b8a6",  # teal — narrow/selective risk
    "Risk-On / Bal": "#4ade80",
    "Risk-Off / Macro": "#ef4444",  # red — correlated selloff
    "Risk-Off / Micro": "#f97316",  # orange — dispersed defensive
    "Risk-Off / Bal": "#fb923c",
    "Neutral / Macro": "#6b7280",
    "Neutral / Micro": "#9ca3af",
    "Neutral / Bal": "#525252",
}


def sector_regime_color(label: str) -> str:
    return SECTOR_REGIME_COLOR.get(label, "#fbbf24")


# ---------------------------------------------------------------------------
# Display helpers (mirror other complexes)
# ---------------------------------------------------------------------------
def loading_label(load: float) -> str:
    a = abs(load)
    if a >= 0.45:
        strength = "dominant"
    elif a >= 0.30:
        strength = "strong"
    elif a >= 0.15:
        strength = "moderate"
    else:
        strength = "weak"
    direction = "positive" if load >= 0 else "negative"
    return f"{strength} {direction}"


def concentration_label(c: float) -> str:
    if c >= 0.50:
        return "concentrated"
    elif c >= 0.35:
        return "mild lead"
    else:
        return "diffuse"


__ANALYTICS_VERSION__ = "sector-v1.1-2026-05-22"
