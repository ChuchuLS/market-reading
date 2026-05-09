"""
Continuous regime characterization for the credit within-complex basket.

Same machinery as the other complexes — leader + sign pattern.
Anchored on LF98OAS positive ("+" in HY position = HY wider, risk-off).

Note: with 3 sub-components, EXP_VAR_THRESHOLD is set higher (0.70) since
3-asset PCs naturally explain more variance than 5-asset PCs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_complex.analytics import (
    ASSETS, ASSET_LABELS, LOAD_COLS, leadership_stats,
)


# ---- Classification thresholds -------------------------------------------
LOADING_MAGNITUDE_THRESHOLD = 0.40   # 3 assets: floor for "leader" is higher
EXP_VAR_THRESHOLD = 0.70             # 3 assets: PC1 typically explains 60-85%
PERSISTENCE_THRESHOLD = 0.85


# ---- Leader colors (colorblind-safe) ------------------------------------
LEADER_COLOR = {
    "LF98OAS": "#3b82f6",   # blue (HY — primary risk asset)
    "LUACOAS": "#06b6d4",   # cyan (IG — quality)
    "ITRXEBE": "#f97316",   # orange (EU — region split)
}

SPECIAL_COLOR = {
    "Mixed":         "#525252",
    "Transitioning": "#e5e7eb",
}


def _format_label(leader: str, sign_pattern: str) -> str:
    return f"{ASSET_LABELS[leader]} {sign_pattern}"


def classify_loadings_series(
    loadings_df: pd.DataFrame,
    mag_threshold: float = LOADING_MAGNITUDE_THRESHOLD,
    var_threshold: float = EXP_VAR_THRESHOLD,
) -> pd.Series:
    if loadings_df.empty:
        return pd.Series(dtype="object", name="Regime")

    lead = leadership_stats(loadings_df)
    expvar = loadings_df["ExplainedVar"]

    labels = []
    for date in loadings_df.index:
        ev = expvar.loc[date]
        leader_load = lead.loc[date, "LeaderLoad"]
        if pd.isna(ev) or ev < var_threshold:
            labels.append("Mixed")
            continue
        if pd.isna(leader_load) or abs(leader_load) < mag_threshold:
            labels.append("Mixed")
            continue
        labels.append(_format_label(
            lead.loc[date, "Leader"],
            lead.loc[date, "SignPattern"],
        ))
    return pd.Series(labels, index=loadings_df.index, name="Regime")


def cosine_persistence(loadings_df: pd.DataFrame) -> pd.Series:
    if loadings_df.empty:
        return pd.Series(dtype=float, name="Persistence")
    V = loadings_df[LOAD_COLS].values
    if V.shape[0] < 2:
        return pd.Series([np.nan] * V.shape[0],
                         index=loadings_df.index, name="Persistence")
    dots = np.einsum("ij,ij->i", V[1:], V[:-1])
    norms = np.linalg.norm(V[1:], axis=1) * np.linalg.norm(V[:-1], axis=1)
    cos = np.where(norms > 0, dots / norms, np.nan)
    out = np.concatenate([[np.nan], cos])
    return pd.Series(out, index=loadings_df.index, name="Persistence")


def apply_persistence_filter(
    regime_series: pd.Series,
    persistence_series: pd.Series,
    threshold: float = PERSISTENCE_THRESHOLD,
) -> pd.Series:
    if regime_series.empty:
        return regime_series
    out = regime_series.copy()
    aligned = persistence_series.reindex(out.index)
    mask = aligned.notna() & (aligned < threshold)
    out[mask] = "Transitioning"
    return out


def regime_runs(regime_series: pd.Series) -> pd.DataFrame:
    if regime_series.empty:
        return pd.DataFrame(columns=["Regime", "Start", "End", "Duration"])
    s = regime_series.reset_index()
    s.columns = ["Date", "Regime"]
    s["RunId"] = (s["Regime"] != s["Regime"].shift()).cumsum()
    return (
        s.groupby("RunId")
         .agg(Regime=("Regime", "first"),
              Start=("Date", "first"),
              End=("Date", "last"),
              Duration=("Date", "count"))
         .reset_index(drop=True)
    )


def regime_stats(regime_series: pd.Series) -> pd.DataFrame:
    if regime_series.empty:
        return pd.DataFrame(columns=[
            "Regime", "Days", "Pct", "Runs",
            "AvgRun", "MedRun", "MaxRun", "LastEntry", "Active",
        ])
    runs = regime_runs(regime_series)
    total = len(regime_series)
    current = regime_series.iloc[-1]
    rows = []
    for bucket, group in runs.groupby("Regime"):
        days = group["Duration"].sum()
        rows.append({
            "Regime":    bucket,
            "Days":      int(days),
            "Pct":       float(days / total * 100),
            "Runs":      int(len(group)),
            "AvgRun":    float(group["Duration"].mean()),
            "MedRun":    float(group["Duration"].median()),
            "MaxRun":    int(group["Duration"].max()),
            "LastEntry": group["Start"].max(),
            "Active":    bucket == current,
        })
    return (pd.DataFrame(rows)
              .sort_values("Days", ascending=False)
              .reset_index(drop=True))


def current_regime_info(
    regime_series: pd.Series,
    loadings_df: pd.DataFrame,
    persistence_series: pd.Series,
) -> dict:
    if regime_series.empty or loadings_df.empty:
        return {}
    current = regime_series.iloc[-1]
    last_date = regime_series.index[-1]
    start_date = last_date
    for d in regime_series.index[::-1]:
        if regime_series.loc[d] == current:
            start_date = d
        else:
            break
    days_in = int((regime_series.index <= last_date).sum() -
                  (regime_series.index < start_date).sum())
    last_loadings = loadings_df.iloc[-1]
    lead_today = leadership_stats(loadings_df.iloc[[-1]]).iloc[0]
    last_persistence = (
        persistence_series.iloc[-1]
        if not persistence_series.empty
        and not pd.isna(persistence_series.iloc[-1])
        else None
    )
    if current in SPECIAL_COLOR:
        color = SPECIAL_COLOR[current]
    else:
        color = LEADER_COLOR.get(lead_today["Leader"], "#fbbf24")
    return {
        "regime":        current,
        "color":         color,
        "since":         start_date,
        "days_in":       days_in,
        "leader":        lead_today["Leader"],
        "leader_label":  lead_today["LeaderLabel"],
        "leader_load":   float(lead_today["LeaderLoad"]),
        "concentration": float(lead_today["Concentration"]),
        "sign_pattern":  lead_today["SignPattern"],
        "loadings":      {a: float(last_loadings[f"{a}_load"]) for a in ASSETS},
        "expvar":        float(last_loadings["ExplainedVar"]),
        "persistence":   last_persistence,
    }


def transitions_log(
    regime_series: pd.Series,
    persistence_series: pd.Series,
    last_n: int = 20,
) -> pd.DataFrame:
    if regime_series.empty or len(regime_series) < 2:
        return pd.DataFrame(columns=[
            "Date", "From", "To", "Persistence", "DurationFrom",
        ])
    runs = regime_runs(regime_series)
    if len(runs) < 2:
        return pd.DataFrame(columns=[
            "Date", "From", "To", "Persistence", "DurationFrom",
        ])
    aligned_pers = persistence_series.reindex(regime_series.index)
    transitions = []
    for i in range(1, len(runs)):
        prev_run = runs.iloc[i - 1]
        curr_run = runs.iloc[i]
        transition_date = curr_run["Start"]
        end_idx = aligned_pers.index.get_loc(transition_date)
        start_idx = max(0, end_idx - 4)
        window_pers = aligned_pers.iloc[start_idx:end_idx + 1].dropna()
        pers = float(window_pers.min()) if len(window_pers) > 0 else None
        transitions.append({
            "Date":         transition_date,
            "From":         prev_run["Regime"],
            "To":           curr_run["Regime"],
            "Persistence":  pers,
            "DurationFrom": int(prev_run["Duration"]),
        })
    df = pd.DataFrame(transitions).sort_values("Date", ascending=False).reset_index(drop=True)
    if last_n:
        df = df.head(last_n)
    return df


def regime_color(regime_label: str) -> str:
    if regime_label in SPECIAL_COLOR:
        return SPECIAL_COLOR[regime_label]
    leader_label = regime_label.split(" ", 1)[0]
    for asset_key, lbl in ASSET_LABELS.items():
        if lbl == leader_label:
            return LEADER_COLOR.get(asset_key, "#fbbf24")
    # Multi-word labels like "US HY OAS" — take first 2 tokens
    leader_label2 = " ".join(regime_label.split(" ")[:3])
    for asset_key, lbl in ASSET_LABELS.items():
        if lbl == leader_label2 or regime_label.startswith(lbl):
            return LEADER_COLOR.get(asset_key, "#fbbf24")
    return "#fbbf24"


__REGIME_VERSION__ = "credit-v1.0-2026-05-09"
