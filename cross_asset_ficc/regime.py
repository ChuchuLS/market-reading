"""
Continuous regime characterization for the 5-asset FICC basket.

Unlike the 3-asset cross_asset/regime.py which assigns each day to one of
8 sign-buckets (+ Mixed/Transitioning), this module describes regimes
continuously: leader, leadership concentration, sign pattern, plus the
same day-over-day persistence and run-aggregation utilities.

A regime is identified not by a discrete bucket label but by the (Leader,
SignPattern) tuple — so two consecutive days with the same leader AND the
same 5-bit sign pattern are considered the same regime. This produces
~2-3x more distinct labels than the 3-asset cube but each label retains
clear meaning, and regime *runs* still aggregate naturally.

Mixed/Transitioning rules (parallel to cross_asset/regime.py):
- "Mixed"         when ExplainedVar < EXP_VAR_THRESHOLD or LeaderLoad below
                  LOADING_MAGNITUDE_THRESHOLD (theme weak / leader unconvincing)
- "Transitioning" when day-over-day cosine persistence < PERSISTENCE_THRESHOLD
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cross_asset_ficc.analytics import (
    ASSETS,
    ASSET_LABELS,
    LOAD_COLS,
    leadership_stats,
)

# ---- Classification thresholds (parallel to cross_asset/regime.py) ----------
LOADING_MAGNITUDE_THRESHOLD = 0.30  # leader's |loading|
EXP_VAR_THRESHOLD = 0.45  # 5 assets: lower than 3-asset 0.60 is fine
PERSISTENCE_THRESHOLD = 0.85  # day-over-day cosine for "Transitioning"

# Leader color palette — colorblind-safe (no red/green pairings).
# Distinguishable under deuteranopia, protanopia, and tritanopia by varying
# both hue and lightness.
LEADER_COLOR = {
    "SPX": "#3b82f6",  # blue
    "USGG10YR": "#06b6d4",  # cyan (distinguishable from blue by saturation)
    "DXY": "#f97316",  # orange
    "BCOM": "#a855f7",  # purple
    "LF98OAS": "#facc15",  # yellow (distinct from orange by lightness)
}

# Special-state colors — also colorblind-safe.
# Mixed is gray (achromatic), Transitioning is white (high-contrast against
# the dark theme; distinct from all leader hues).
SPECIAL_COLOR = {
    "Mixed": "#525252",  # gray (achromatic — universally readable)
    "Transitioning": "#e5e7eb",  # near-white (distinct from all leader colors)
}


# ---------------------------------------------------------------------------
# Per-day regime label
# ---------------------------------------------------------------------------
def _format_label(leader: str, sign_pattern: str) -> str:
    """e.g. 'SPX +−+−+' — display label."""
    return f"{ASSET_LABELS[leader]} {sign_pattern}"


def classify_loadings_series(
    loadings_df: pd.DataFrame,
    mag_threshold: float = LOADING_MAGNITUDE_THRESHOLD,
    var_threshold: float = EXP_VAR_THRESHOLD,
) -> pd.Series:
    """
    Per-day regime label as "Leader Sign-Pattern" (e.g. "SPX +−+−+").
    Days that fail thresholds get "Mixed".

    Expected columns: SPX_load, USGG10YR_load, DXY_load, BCOM_load,
                      LF98OAS_load, ExplainedVar.
    """
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
        labels.append(
            _format_label(
                lead.loc[date, "Leader"],
                lead.loc[date, "SignPattern"],
            )
        )
    return pd.Series(labels, index=loadings_df.index, name="Regime")


# ---------------------------------------------------------------------------
# Persistence (cosine similarity day over day)
# ---------------------------------------------------------------------------
def cosine_persistence(loadings_df: pd.DataFrame) -> pd.Series:
    """
    Day-over-day cosine similarity of the 5-D loading vector.

    Range: -1.0 (full sign flip) to +1.0 (perfectly stable).
    First row is NaN.
    """
    if loadings_df.empty:
        return pd.Series(dtype=float, name="Persistence")

    V = loadings_df[LOAD_COLS].values
    if V.shape[0] < 2:
        return pd.Series(
            [np.nan] * V.shape[0], index=loadings_df.index, name="Persistence"
        )

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
    """
    Relabel days where day-over-day persistence < threshold as 'Transitioning'.
    """
    if regime_series.empty:
        return regime_series
    out = regime_series.copy()
    aligned = persistence_series.reindex(out.index)
    mask = aligned.notna() & (aligned < threshold)
    out[mask] = "Transitioning"
    return out


# ---------------------------------------------------------------------------
# Run aggregation
# ---------------------------------------------------------------------------
def regime_runs(regime_series: pd.Series) -> pd.DataFrame:
    """
    Convert a per-day regime series into contiguous "runs".

    Returns DataFrame with columns: Regime, Start, End, Duration.
    """
    if regime_series.empty:
        return pd.DataFrame(columns=["Regime", "Start", "End", "Duration"])

    s = regime_series.reset_index()
    s.columns = ["Date", "Regime"]
    s["RunId"] = (s["Regime"] != s["Regime"].shift()).cumsum()
    runs = (
        s.groupby("RunId")
        .agg(
            Regime=("Regime", "first"),
            Start=("Date", "first"),
            End=("Date", "last"),
            Duration=("Date", "count"),
        )
        .reset_index(drop=True)
    )
    return runs


def regime_stats(regime_series: pd.Series) -> pd.DataFrame:
    """
    Stats table summarizing the regime history.

    Columns: Regime, Days, Pct, Runs, AvgRun, MedRun, MaxRun, LastEntry, Active.
    Sorted by Days descending.
    """
    if regime_series.empty:
        return pd.DataFrame(
            columns=[
                "Regime",
                "Days",
                "Pct",
                "Runs",
                "AvgRun",
                "MedRun",
                "MaxRun",
                "LastEntry",
                "Active",
            ]
        )

    runs = regime_runs(regime_series)
    total = len(regime_series)
    current = regime_series.iloc[-1]

    rows = []
    for bucket, group in runs.groupby("Regime"):
        days = group["Duration"].sum()
        rows.append(
            {
                "Regime": bucket,
                "Days": int(days),
                "Pct": float(days / total * 100),
                "Runs": int(len(group)),
                "AvgRun": float(group["Duration"].mean()),
                "MedRun": float(group["Duration"].median()),
                "MaxRun": int(group["Duration"].max()),
                "LastEntry": group["Start"].max(),
                "Active": bucket == current,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("Days", ascending=False).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Current regime info card
# ---------------------------------------------------------------------------
def current_regime_info(
    regime_series: pd.Series,
    loadings_df: pd.DataFrame,
    persistence_series: pd.Series,
) -> dict:
    """
    Bundle of facts about the most recent regime, for the headline card.

    Includes leader, concentration, sign pattern, days-in-regime, and the
    color to use (driven by leader, or special-state color for Mixed/Transitioning).
    """
    if regime_series.empty or loadings_df.empty:
        return {}

    current = regime_series.iloc[-1]
    last_date = regime_series.index[-1]

    # Walk backward to find when this regime started
    start_date = last_date
    for d in regime_series.index[::-1]:
        if regime_series.loc[d] == current:
            start_date = d
        else:
            break

    days_in = int(
        (regime_series.index <= last_date).sum()
        - (regime_series.index < start_date).sum()
    )

    last_loadings = loadings_df.iloc[-1]
    lead_today = leadership_stats(loadings_df.iloc[[-1]]).iloc[0]

    last_persistence = (
        persistence_series.iloc[-1]
        if not persistence_series.empty and not pd.isna(persistence_series.iloc[-1])
        else None
    )

    # Determine display color
    if current in SPECIAL_COLOR:
        color = SPECIAL_COLOR[current]
    else:
        color = LEADER_COLOR.get(lead_today["Leader"], "#fbbf24")

    return {
        "regime": current,
        "color": color,
        "since": start_date,
        "days_in": days_in,
        "leader": lead_today["Leader"],
        "leader_label": lead_today["LeaderLabel"],
        "leader_load": float(lead_today["LeaderLoad"]),
        "concentration": float(lead_today["Concentration"]),
        "sign_pattern": lead_today["SignPattern"],
        "loadings": {a: float(last_loadings[f"{a}_load"]) for a in ASSETS},
        "expvar": float(last_loadings["ExplainedVar"]),
        "persistence": last_persistence,
    }


# ---------------------------------------------------------------------------
# Transitions log
# ---------------------------------------------------------------------------
def transitions_log(
    regime_series: pd.Series,
    persistence_series: pd.Series,
    last_n: int = 20,
) -> pd.DataFrame:
    """
    Log of regime change events. Each row = one transition.

    Columns: Date, From, To, Persistence (min over 5d), DurationFrom.
    """
    if regime_series.empty or len(regime_series) < 2:
        return pd.DataFrame(
            columns=[
                "Date",
                "From",
                "To",
                "Persistence",
                "DurationFrom",
            ]
        )

    runs = regime_runs(regime_series)
    if len(runs) < 2:
        return pd.DataFrame(
            columns=[
                "Date",
                "From",
                "To",
                "Persistence",
                "DurationFrom",
            ]
        )

    aligned_pers = persistence_series.reindex(regime_series.index)

    transitions = []
    for i in range(1, len(runs)):
        prev_run = runs.iloc[i - 1]
        curr_run = runs.iloc[i]
        transition_date = curr_run["Start"]

        end_idx = aligned_pers.index.get_loc(transition_date)
        start_idx = max(0, end_idx - 4)
        window_pers = aligned_pers.iloc[start_idx : end_idx + 1].dropna()
        pers = float(window_pers.min()) if len(window_pers) > 0 else None

        transitions.append(
            {
                "Date": transition_date,
                "From": prev_run["Regime"],
                "To": curr_run["Regime"],
                "Persistence": pers,
                "DurationFrom": int(prev_run["Duration"]),
            }
        )

    df = pd.DataFrame(transitions)
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    if last_n:
        df = df.head(last_n)
    return df


# ---------------------------------------------------------------------------
# Color helper for regime labels (used by view layer)
# ---------------------------------------------------------------------------
def regime_color(regime_label: str) -> str:
    """
    Resolve a regime label ('Mixed', 'Transitioning', or 'Leader SignPattern')
    to a display color, keyed by the leader asset. Matches the FULL asset
    label (longest first) so multi-word labels like "UST 10Y" or "HY OAS"
    resolve correctly instead of collapsing to their first token.
    """
    if regime_label in SPECIAL_COLOR:
        return SPECIAL_COLOR[regime_label]

    for asset_key, label in sorted(
        ASSET_LABELS.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        if regime_label.startswith(label + " "):
            return LEADER_COLOR.get(asset_key, "#fbbf24")

    return "#fbbf24"


__REGIME_VERSION__ = "ficc-v1.0-2026-05-09"
