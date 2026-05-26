"""
4-regime relative classification for SPX/UST10Y/DXY PCA loadings.

The dominant theme (PC1) is sign-anchored so SPX is the positive reference.
An absolute 8-bucket sign cube is therefore misleading: the four SPX-negative
buckets essentially never occur under that anchor. Instead we classify by
whether UST 10Y and DXY are ALIGNED WITH or OPPOSITE TO SPX, which yields four
meaningful regimes (and works regardless of SPX's absolute sign, since it's a
relative comparison):

    UST 10Y vs SPX | DXY vs SPX | Regime
    ---------------|------------|----------------------------------
    opposite       | opposite   | Risk-on / Growth reflation
    same           | opposite   | Goldilocks / Duration-led risk-on
    opposite       | same       | Inflation / Rates pressure
    same           | same       | Defensive / Dollar squeeze

Mapping rationale (with SPX anchored +):
  - Risk-on / Growth reflation  (10Y +, DXY −): equities and yields rise
    together while the dollar falls — growth optimism / reflationary risk.
  - Goldilocks / Duration-led   (10Y −, DXY −): equities up, yields and dollar
    down — easing financial conditions, soft-landing risk appetite.
  - Inflation / Rates pressure  (10Y +, DXY +): yields and dollar both rise
    against the equity anchor — a rates/inflation-pressure regime.
  - Defensive / Dollar squeeze  (10Y −, DXY +): dollar bid defensively while
    yields fall — risk-off / dollar squeeze / growth stress.

A row is "Mixed" when PC1 is too weak to read (explained variance below
EXP_VAR_THRESHOLD, or a loading magnitude below LOADING_MAGNITUDE_THRESHOLD).
Days with very low day-over-day persistence are relabeled "Transitioning".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ---- Classification thresholds --------------------------------------------
LOADING_MAGNITUDE_THRESHOLD = 0.30
EXP_VAR_THRESHOLD = 0.60

# ---- Persistence threshold for "Transitioning" relabel --------------------
PERSISTENCE_THRESHOLD = 0.85

# ---- The four relative regimes --------------------------------------------
RISK_ON_REFLATION = "Risk-on / Growth reflation"
GOLDILOCKS = "Goldilocks / Duration-led risk-on"
INFLATION_PRESSURE = "Inflation / Rates pressure"
DEFENSIVE_SQUEEZE = "Defensive / Dollar squeeze"

REGIME_ORDER = [
    RISK_ON_REFLATION,
    GOLDILOCKS,
    INFLATION_PRESSURE,
    DEFENSIVE_SQUEEZE,
    "Mixed",
    "Transitioning",
]

# Display labels (the regime names are already readable, so display == name)
BUCKET_DISPLAY = {r: r for r in REGIME_ORDER}

# Color palette — dark-bg friendly, one stable hue per regime.
BUCKET_COLOR = {
    RISK_ON_REFLATION: "#22c55e",  # green  — growth reflation / risk-on
    GOLDILOCKS: "#06b6d4",  # teal   — soft-landing / duration-led
    INFLATION_PRESSURE: "#f97316",  # orange — rates/inflation pressure
    DEFENSIVE_SQUEEZE: "#a855f7",  # purple — defensive / dollar squeeze
    "Mixed": "#525252",  # gray   — no readable signal
    "Transitioning": "#e5e7eb",  # light  — fast rotation
}

# Back-compat alias: some call sites still import BUCKET_ORDER.
BUCKET_ORDER = REGIME_ORDER


def classify_loadings(
    spx_load: float,
    ust10y_load: float,
    dxy_load: float,
    explained_var: float,
    mag_threshold: float = LOADING_MAGNITUDE_THRESHOLD,
    var_threshold: float = EXP_VAR_THRESHOLD,
) -> str:
    """
    Classify one (SPX, UST10Y, DXY, explained_var) tuple into one of the four
    relative regimes, or "Mixed" when the signal is too weak.

    Classification is by the sign of UST 10Y and DXY RELATIVE to SPX, so it is
    independent of SPX's absolute sign.
    """
    if pd.isna(explained_var) or explained_var < var_threshold:
        return "Mixed"

    loads = (spx_load, ust10y_load, dxy_load)
    if any(pd.isna(v) for v in loads):
        return "Mixed"
    if any(abs(v) < mag_threshold for v in loads):
        return "Mixed"

    spx_sign = spx_load >= 0
    ust_same = (ust10y_load >= 0) == spx_sign
    dxy_same = (dxy_load >= 0) == spx_sign

    # Mapping (in absolute signs with SPX anchored +):
    #   10Y +, DXY −  (10Y same, DXY opp)  -> Risk-on / Growth reflation
    #   10Y −, DXY −  (10Y opp,  DXY opp)  -> Goldilocks / Duration-led risk-on
    #   10Y +, DXY +  (10Y same, DXY same) -> Inflation / Rates pressure
    #   10Y −, DXY +  (10Y opp,  DXY same) -> Defensive / Dollar squeeze
    if ust_same and not dxy_same:
        return RISK_ON_REFLATION
    if not ust_same and not dxy_same:
        return GOLDILOCKS
    if ust_same and dxy_same:
        return INFLATION_PRESSURE
    # not ust_same and dxy_same
    return DEFENSIVE_SQUEEZE


def classify_loadings_series(
    loadings_df: pd.DataFrame,
    mag_threshold: float = LOADING_MAGNITUDE_THRESHOLD,
    var_threshold: float = EXP_VAR_THRESHOLD,
) -> pd.Series:
    """
    Apply classify_loadings to each row of a rolling_pca_loadings() frame.
    Expected columns: SPX_load, USGG10YR_load, DXY_load, ExplainedVar.
    """
    if loadings_df.empty:
        return pd.Series(dtype="object", name="Regime")

    def _row_classify(row):
        return classify_loadings(
            row["SPX_load"],
            row["USGG10YR_load"],
            row["DXY_load"],
            row["ExplainedVar"],
            mag_threshold,
            var_threshold,
        )

    return loadings_df.apply(_row_classify, axis=1).rename("Regime")


def cosine_persistence(loadings_df: pd.DataFrame) -> pd.Series:
    """
    Day-over-day cosine similarity of the loading vector. Loadings are unit
    vectors so this is just dot(today, yesterday).

    Returns a Series indexed the same as loadings_df, with NaN on the first
    row (no previous day to compare).

    Range: -1.0 (full sign flip) to +1.0 (perfectly stable). Higher = more stable.
    """
    if loadings_df.empty:
        return pd.Series(dtype=float, name="Persistence")

    cols = ["SPX_load", "USGG10YR_load", "DXY_load"]
    V = loadings_df[cols].values
    if V.shape[0] < 2:
        return pd.Series(
            [np.nan] * V.shape[0], index=loadings_df.index, name="Persistence"
        )

    # dot product of consecutive rows
    dots = np.einsum("ij,ij->i", V[1:], V[:-1])
    norms = np.linalg.norm(V[1:], axis=1) * np.linalg.norm(V[:-1], axis=1)
    cos = np.where(norms > 0, dots / norms, np.nan)
    out = np.concatenate([[np.nan], cos])
    return pd.Series(out, index=loadings_df.index, name="Persistence")


def regime_runs(regime_series: pd.Series) -> pd.DataFrame:
    """
    Convert a per-day regime series into a list of contiguous "runs".

    Returns a DataFrame with columns:
        Regime    : bucket label
        Start     : first date of the run
        End       : last date of the run (inclusive)
        Duration  : number of days

    Useful for both the timeline stripe and the stats table.
    """
    if regime_series.empty:
        return pd.DataFrame(columns=["Regime", "Start", "End", "Duration"])

    s = regime_series.reset_index()
    s.columns = ["Date", "Regime"]
    # Mark transitions where regime changes from previous row
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
    Produce a stats table summarizing the regime history.

    Columns:
        Regime    : bucket label
        Days      : total days in this regime
        Pct       : percentage of total days
        Runs      : number of distinct stretches in this regime
        AvgRun    : mean duration of a run, in days
        MedRun    : median run duration
        MaxRun    : longest run
        LastEntry : most recent date this regime started
        Active    : True if regime is the current (most recent) one

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

    df = pd.DataFrame(rows).sort_values("Days", ascending=False).reset_index(drop=True)
    return df


def current_regime_info(
    regime_series: pd.Series,
    loadings_df: pd.DataFrame,
    persistence_series: pd.Series,
) -> dict:
    """
    Bundle of facts about the most recent (current) regime, for the
    "NOW" headline card on the regime tab.
    """
    if regime_series.empty or loadings_df.empty:
        return {}

    current = regime_series.iloc[-1]
    last_date = regime_series.index[-1]

    # Walk backward to find when the current regime started
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
    last_persistence = (
        persistence_series.iloc[-1]
        if not persistence_series.empty and not pd.isna(persistence_series.iloc[-1])
        else None
    )

    return {
        "regime": current,
        "label": BUCKET_DISPLAY[current],
        "color": BUCKET_COLOR[current],
        "since": start_date,
        "days_in": days_in,
        "spx_load": float(last_loadings["SPX_load"]),
        "ust_load": float(last_loadings["USGG10YR_load"]),
        "dxy_load": float(last_loadings["DXY_load"]),
        "expvar": float(last_loadings["ExplainedVar"]),
        "persistence": last_persistence,
    }


def apply_persistence_filter(
    regime_series: pd.Series,
    persistence_series: pd.Series,
    threshold: float = PERSISTENCE_THRESHOLD,
) -> pd.Series:
    """
    Relabel any day where day-over-day cosine persistence is below `threshold`
    as "Transitioning". Catches 1-day blips during regime rotations where the
    eigenvector momentarily passes through a sign zone before settling.

    Days where persistence is NaN (the very first day, or after gaps) are
    left unchanged.
    """
    if regime_series.empty:
        return regime_series

    out = regime_series.copy()
    aligned = persistence_series.reindex(out.index)
    mask = aligned.notna() & (aligned < threshold)
    out[mask] = "Transitioning"
    return out


def transitions_log(
    regime_series: pd.Series,
    persistence_series: pd.Series,
    last_n: int = 20,
) -> pd.DataFrame:
    """
    Log of regime change events. Each row represents a transition from one
    regime to another, with metadata.

    Columns:
        Date         : day the new regime became active (transition day)
        From         : previous regime label
        To           : new regime label
        Persistence  : MINIMUM persistence over the 5 days leading up to and
                       including the transition. Captures the rotation magnitude
                       even when the transition day itself looks stable.
        DurationFrom : how many days the previous regime lasted

    Returns the most recent `last_n` transitions, sorted newest-first.
    """
    if regime_series.empty or len(regime_series) < 2:
        return pd.DataFrame(
            columns=["Date", "From", "To", "Persistence", "DurationFrom"]
        )

    runs = regime_runs(regime_series)
    if len(runs) < 2:
        return pd.DataFrame(
            columns=["Date", "From", "To", "Persistence", "DurationFrom"]
        )

    aligned_pers = persistence_series.reindex(regime_series.index)

    transitions = []
    for i in range(1, len(runs)):
        prev_run = runs.iloc[i - 1]
        curr_run = runs.iloc[i]
        transition_date = curr_run["Start"]

        # Min persistence over the 5 days ending at the transition day
        # (lookback window for capturing rotation magnitude)
        end_idx = aligned_pers.index.get_loc(transition_date)
        start_idx = max(0, end_idx - 4)
        window_pers = aligned_pers.iloc[start_idx : end_idx + 1].dropna()
        if len(window_pers) > 0:
            pers = float(window_pers.min())
        else:
            pers = None

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


__REGIME_VERSION__ = "v1.5-2026-05-07"
