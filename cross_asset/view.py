"""
Cross-Asset section view — renders correlations + dominant theme.

Reads cross_asset/data/CROSSASSET.xlsx (Date, SPX, USGG10YR, DXY).
Aesthetic matches the OFR-style dark dashboard: amber/cyan/magenta palette
on top of the same #0a0a0a background as the rest of the app.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theming import BG, GRID, TEXT, TEXT_DIM, DARK_LAYOUT
from cross_asset.analytics import (
    compute_returns,
    rolling_pairwise_corrs, latest_pairwise_corrs,
    pca_dominant_theme, rolling_pca_loadings,
    correlation_story, loading_label,
    __ANALYTICS_VERSION__,
)
from cross_asset.regime import (
    classify_loadings_series, cosine_persistence,
    soft_scores, apply_persistence_filter, transitions_log,
    regime_runs, regime_stats, current_regime_info,
    BUCKET_ORDER, BUCKET_DISPLAY, BUCKET_COLOR,
    LOADING_MAGNITUDE_THRESHOLD, EXP_VAR_THRESHOLD, PERSISTENCE_THRESHOLD,
    __REGIME_VERSION__,
)

DATA_PATH = Path(__file__).parent / "data" / "CROSSASSET.xlsx"

# OFR-inspired palette — amber/magenta/cyan/lime
COLOR_SPX     = "#84cc16"   # lime
COLOR_UST10Y  = "#06b6d4"   # cyan
COLOR_DXY     = "#fb923c"   # orange/amber
COLOR_PC1     = "#fbbf24"   # amber (for linkage / dominant theme)

PAIR_COLORS = {
    "SPX_vs_USGG10YR": "#ec4899",   # magenta
    "SPX_vs_DXY":      "#06b6d4",   # cyan
    "USGG10YR_vs_DXY": "#fb923c",   # amber
}
PAIR_LABELS = {
    "SPX_vs_USGG10YR": "SPX vs UST 10Y",
    "SPX_vs_DXY":      "SPX vs DXY",
    "USGG10YR_vs_DXY": "UST 10Y vs DXY",
}


@st.cache_data(show_spinner=False)
def load_prices(path: Path, _mtime: float) -> pd.DataFrame:
    """Read CROSSASSET.xlsx and return clean prices DataFrame indexed by Date.

    Handles common BQL export quirks:
      - First column may be a BQL "ID" column (with header labels in some rows)
      - Calendar days included with weekend rows forward-filled from Friday
      - Duplicate dates from BQL spilling
      - Column names like "SPX Index", "USGG10YR Index", "DXY Curncy"
    """
    raw = pd.read_excel(path)

    # ---- Step 1: identify the columns we care about -------------------
    # Find the date column. The robust trick: we want a column whose values
    # are EITHER already datetime64 dtype, OR strings that parse to dates,
    # but NOT floats (which pandas will happily convert via Excel serials).
    cols = list(raw.columns)
    date_col = None
    for c in cols:
        col = raw[c]
        # Already a datetime?
        if pd.api.types.is_datetime64_any_dtype(col):
            date_col = c
            break
        # Object dtype that parses to dates?
        if col.dtype == object:
            parsed = pd.to_datetime(col, errors="coerce")
            if parsed.notna().sum() / max(len(parsed), 1) > 0.8:
                date_col = c
                break
    if date_col is None:
        # Fallback: scan column NAMES for "date"
        for c in cols:
            if "date" in str(c).lower():
                date_col = c
                break
    if date_col is None:
        raise ValueError(
            f"Couldn't find a Date column in CROSSASSET.xlsx. "
            f"Columns found: {cols}"
        )

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.rename(columns={date_col: "Date"})

    # ---- Step 2: rename SPX / yield / DXY columns ---------------------
    # Match by header substring AND require numeric data (the BQL "ID" column
    # often shares the SPX header name but contains string labels).
    rename_map = {}
    used = set()
    for c in raw.columns:
        if c == "Date" or c in used:
            continue
        col_data = raw[c]
        # Must be numeric (or coercible to numeric) — skip the ID/label column
        try:
            numeric_share = pd.to_numeric(col_data, errors="coerce").notna().mean()
        except Exception:
            numeric_share = 0.0
        if numeric_share < 0.5:
            continue
        cu = str(c).upper().replace(" ", "")
        if "SPX" in cu and "SPX" not in rename_map.values():
            rename_map[c] = "SPX"; used.add(c)
        elif ("USGG10" in cu or "UST10" in cu or "US10" in cu) and "USGG10YR" not in rename_map.values():
            rename_map[c] = "USGG10YR"; used.add(c)
        elif "DXY" in cu and "DXY" not in rename_map.values():
            rename_map[c] = "DXY"; used.add(c)
    raw = raw.rename(columns=rename_map)

    needed = ["SPX", "USGG10YR", "DXY"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CROSSASSET.xlsx is missing columns: {missing}. "
            f"Required: a Date column plus SPX, USGG10YR (or UST10Y), DXY. "
            f"Found: {list(raw.columns)}"
        )

    df = raw[["Date"] + needed].copy()

    # ---- Step 3: clean rows -------------------------------------------
    # 3a. Drop rows where Date or any price is NaN (handles BQL header rows)
    df = df.dropna(subset=["Date"] + needed)

    # 3b. Numeric coercion (in case prices came in as strings)
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)

    # 3c. Drop weekends — BQL exports often include Sat/Sun with prior-Friday
    # values forward-filled, which corrupts return calculations
    df["_dow"] = df["Date"].dt.dayofweek  # Mon=0, Sun=6
    df = df[df["_dow"] < 5].drop(columns="_dow")

    # 3d. De-duplicate on Date (BQL spilling sometimes repeats the last row)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")

    # 3e. Set Date as index
    df = df.set_index("Date")

    return df


# ===========================================================================
# Public render
# ===========================================================================
def render_cross_asset():
    """Render the Cross-Asset section."""

    if not DATA_PATH.exists():
        st.error(
            f"CROSSASSET.xlsx not found at {DATA_PATH}.\n\n"
            "Required columns: **Date, SPX, USGG10YR, DXY** (one row per trading day, "
            "ideally 2-3 years of history). See README for BQL formula."
        )
        return

    mtime = DATA_PATH.stat().st_mtime
    try:
        prices = load_prices(DATA_PATH, mtime)
    except Exception as e:
        st.error(f"Failed to read CROSSASSET.xlsx: {e}")
        return

    last_updated = datetime.fromtimestamp(mtime).strftime("%b %d, %Y · %H:%M")

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="background:#0a0a0a;padding:1rem 1.25rem;margin:0 0 1rem 0;
                    border-bottom:1px solid #1a1a1a;color:#fff;">
          <div style="font-size:11px;color:#fbbf24;letter-spacing:0.2em;
                      text-transform:uppercase;font-weight:600;margin-bottom:0.25rem;">
            ⬢ Cross-Asset Linkages · SPX / UST 10Y / DXY
          </div>
          <div style="font-size:36px;font-weight:800;letter-spacing:-0.01em;
                      font-family:'Inter',sans-serif;line-height:1;">
            CROSS-ASSET MONITOR
          </div>
          <div style="font-size:11px;color:#888;letter-spacing:0.05em;margin-top:0.5rem;
                      text-transform:uppercase;">
            Latest: {last_updated} · {len(prices)} trading days
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # Inline methodology controls — visible on page (no sidebar dependency)
    # -------------------------------------------------------------------
    st.markdown(
        """
        <div style="font-size:10px;color:#fbbf24;letter-spacing:0.15em;
                    text-transform:uppercase;font-weight:600;margin-top:0.25rem;
                    margin-bottom:0.25rem;">
          ⚙ Methodology
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ----- Quick presets — sync both panels to the same methodology
    pc1, pc2, pc3, pc_spacer = st.columns([1, 1, 1, 4])
    with pc1:
        if st.button("⚡ Standard", use_container_width=True, key="ca_preset_std",
                     help="Both panels: window=20, no smoothing, per-day SPX-positive anchor."):
            for ns in ("corr", "pca"):
                st.session_state[f"{ns}_window"] = 20
                st.session_state[f"{ns}_weighting"] = "equal"
            st.session_state["ca_scaling"] = "zscore"
            st.session_state["pca_method"] = "standard"
            st.session_state["pca_presmooth"] = 0
            st.rerun()
    with pc2:
        if st.button("〜 Smooth", use_container_width=True, key="ca_preset_smooth",
                     help="Both panels: window=90, halflife=15 pre-smoothing, Procrustes."):
            for ns in ("corr", "pca"):
                st.session_state[f"{ns}_window"] = 90
                st.session_state[f"{ns}_weighting"] = "equal"
            st.session_state["ca_scaling"] = "zscore"
            st.session_state["pca_method"] = "procrustes"
            st.session_state["pca_presmooth"] = 15
            st.rerun()
    with pc3:
        if st.button("≈ Very Smooth", use_container_width=True, key="ca_preset_vsmooth",
                     help="Both panels: window=120, halflife=20, EWM weighting, Procrustes."):
            for ns in ("corr", "pca"):
                st.session_state[f"{ns}_window"] = 120
                st.session_state[f"{ns}_weighting"] = "ewm"
            st.session_state["ca_scaling"] = "zscore"
            st.session_state["pca_method"] = "procrustes"
            st.session_state["pca_presmooth"] = 20
            st.rerun()

    # Top bar: Return scaling (shared, affects what "returns" means for both panels) + Refresh
    tbc1, tbc2 = st.columns([6, 1])
    with tbc1:
        scaling = st.radio(
            "Return scaling (shared by both panels)",
            options=["zscore", "volscale"],
            index=["zscore", "volscale"].index(st.session_state.get("ca_scaling", "zscore")),
            format_func=lambda x: {
                "zscore":   "Z-score (within window)",
                "volscale": "Vol-scale (trailing-vol divisor)",
            }[x],
            key="ca_scaling",
            horizontal=True,
            help=(
                "Z-score uses each window's own mean/std (correlation matrix). "
                "Vol-scale divides each return by its trailing realized vol — preserves "
                "cross-window magnitude comparability. This setting is global because "
                "it defines the meaning of 'returns' fed into both panels."
            ),
        )
    with tbc2:
        st.markdown("<div style='font-size:10px;color:transparent;'>spacer</div>",
                    unsafe_allow_html=True)
        if st.button("↻ Refresh data", use_container_width=True, key="ca_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Read per-panel settings from session state (defaults same as old "Standard")
    corr_window = st.session_state.get("corr_window", 20)
    corr_weighting = st.session_state.get("corr_weighting", "equal")
    pca_window = st.session_state.get("pca_window", 20)
    pca_weighting = st.session_state.get("pca_weighting", "equal")
    pca_method = st.session_state.get("pca_method", "standard")
    pca_presmooth = st.session_state.get("pca_presmooth", 0)

    # Build a two-line summary of current settings showing both panels' methodology
    pca_smooth_str = f"halflife={pca_presmooth}d" if pca_presmooth > 0 else "off"
    settings_html = f"""
        <div style="background:#0f0f0f;border:1px solid #2a2a2a;border-left:3px solid #fbbf24;
                    padding:0.5rem 0.75rem;margin:0.5rem 0;font-family:'JetBrains Mono',monospace;
                    font-size:11px;color:#ccc;line-height:1.7;">
          <div><span style="color:#fbbf24;font-weight:600;">PAIRWISE CORR:</span>
            window={corr_window}d · weighting={corr_weighting}</div>
          <div><span style="color:#fbbf24;font-weight:600;">DOMINANT THEME:</span>
            window={pca_window}d · weighting={pca_weighting} · sign={pca_method} ·
            pre-smooth={pca_smooth_str}</div>
          <div><span style="color:#fbbf24;font-weight:600;">SHARED:</span>
            scaling={scaling} · filter=strict</div>
        </div>
    """
    st.markdown(settings_html, unsafe_allow_html=True)

    st.caption(
        f"📁 {DATA_PATH.name} · 🕐 {last_updated} · "
        f"{len(prices)} trading days, {prices.index.min().date()} → {prices.index.max().date()} · "
        f"⚙ Analytics: {__ANALYTICS_VERSION__}"
    )

    # Compute returns according to chosen method
    returns = compute_returns(prices, vol_scale=(scaling == "volscale"))

    # Strict filter: drop days where any of the three returns is exactly 0.
    # A zero log-return / zero yield-diff means today's price equals yesterday's,
    # which on a US trading day usually means the BQL pull captured stale data
    # for that asset (US holidays where one market closed, or intraday-pulled
    # data that hadn't updated yet). Including these rows mixes stale-data
    # zeros into the rolling correlation, biasing values toward zero.
    # Removing them brings the math in line with Bloomberg's CORREL function.
    n_before = len(returns)
    returns = returns[(returns["SPX"] != 0) & (returns["USGG10YR"] != 0) & (returns["DXY"] != 0)]
    n_dropped = n_before - len(returns)

    if n_dropped > 0:
        st.caption(
            f"⚠ Strict filter dropped {n_dropped} stale-data rows "
            f"(days where at least one of SPX/UST10Y/DXY had zero return — "
            f"typically US holidays or intraday-pulled data). "
            f"{len(returns)} valid trading days used for correlation/PCA."
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # Sub-tabs inside the Cross-Asset section
    # -------------------------------------------------------------------
    tab_pairwise, tab_regime = st.tabs([
        "📊 Correlations & Theme",
        "🔬 Regime",
    ])

    with tab_pairwise:
        col_left, col_right = st.columns(2)

        with col_left:
            _render_correlations_panel(returns)

        with col_right:
            _render_dominant_theme_panel(returns)

    with tab_regime:
        _render_regime_panel(returns)


# ---------------------------------------------------------------------------
# Panel 1: Pairwise rolling correlations
# ---------------------------------------------------------------------------
def _render_correlations_panel(returns: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          CROSS-ASSET CORRELATIONS
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          How each pair of assets is moving relative to each other.
          Correlation ranges from -1 to +1.
          <span style="color:#84cc16;font-weight:600;">Positive</span> = they move in the same direction.
          <span style="color:#f87171;font-weight:600;">Negative</span> = they move in opposite directions.
          Zero = no consistent relationship.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Panel-local controls (independent of the Dominant Theme panel) ----
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        window = st.slider(
            "Window (trading days)",
            min_value=5, max_value=252, step=1,
            value=st.session_state.get("corr_window", 20),
            key="corr_window",
            help="Rolling window for pairwise correlation. Bloomberg's default is 20.",
        )
    with cc2:
        weighting = st.radio(
            "Weighting",
            options=["equal", "ewm"],
            index=["equal", "ewm"].index(st.session_state.get("corr_weighting", "equal")),
            format_func=lambda x: {"equal": "Equal", "ewm": "Exp (W/3)"}[x],
            key="corr_weighting",
            horizontal=True,
        )

    # ---- Latest pairwise correlations as headline numbers ----
    latest = latest_pairwise_corrs(returns, window=window, weighting=weighting)

    for pair_key, pair_label in PAIR_LABELS.items():
        rho = latest[pair_key]
        story = correlation_story(pair_key, rho)
        sign_color = "#84cc16" if rho > 0.05 else ("#f87171" if rho < -0.05 else "#888")
        sign_str = f"+{rho:.2f}" if rho >= 0 else f"{rho:.2f}"
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.5rem 0;
                        border-bottom:1px solid #161616;">
              <div style="flex:0 0 160px;font-size:13px;color:#fff;font-weight:600;">
                {pair_label}
              </div>
              <div style="flex:0 0 70px;font-family:'JetBrains Mono',monospace;
                          font-size:18px;font-weight:700;color:{sign_color};">
                {sign_str}
              </div>
              <div style="flex:1;font-size:11px;color:#aaa;">
                {story}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Rolling correlation chart ----
    rolled = rolling_pairwise_corrs(returns, window=window, weighting=weighting)

    fig = go.Figure()
    for pair_key in PAIR_LABELS:
        fig.add_trace(go.Scatter(
            x=rolled.index, y=rolled[pair_key],
            mode="lines", name=PAIR_LABELS[pair_key],
            line=dict(color=PAIR_COLORS[pair_key], width=1.4),
            hovertemplate=f"<b>{PAIR_LABELS[pair_key]}</b><br>"
                          + "%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        **DARK_LAYOUT,
        height=360,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc"),
        ),
        xaxis=dict(
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            rangeslider=dict(visible=True, bgcolor="#1a1a1a",
                             bordercolor="#fbbf24", borderwidth=1, thickness=0.04),
            type="date",
        ),
        yaxis=dict(
            range=[-1, 1],
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        margin=dict(l=40, r=20, t=10, b=80),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Panel 2: Dominant theme (PCA)
# ---------------------------------------------------------------------------
def _render_dominant_theme_panel(returns: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          DOMINANT MARKET THEME
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          When all 3 assets move together, there's a common
          <span style="font-weight:600;color:#fff;">"theme"</span> driving them
          (like risk-on/risk-off, or a Fed reaction). The
          <span style="font-weight:600;color:#fff;">loadings</span> show how much each asset
          participates in that theme. A loading of 0.6 means that asset is
          heavily involved; 0.2 means it's barely participating. If two assets have
          the <span style="font-weight:600;color:#fff;">same sign</span>
          (both positive or both negative), they're moving in the same direction within
          the theme. <span style="font-weight:600;color:#fff;">Opposite signs</span> mean
          they're moving against each other.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Panel-local controls (independent of the Pairwise panel) ----
    pc1, pc2 = st.columns([3, 2])
    with pc1:
        window = st.slider(
            "Window (trading days)",
            min_value=5, max_value=252, step=1,
            value=st.session_state.get("pca_window", 20),
            key="pca_window",
            help=(
                "Rolling window for PCA decomposition. Longer windows give "
                "smoother loading curves but lag regime changes."
            ),
        )
    with pc2:
        presmooth_halflife = st.select_slider(
            "Pre-smooth halflife",
            options=[0, 3, 5, 10, 15, 20, 30],
            value=st.session_state.get("pca_presmooth", 0),
            key="pca_presmooth",
            help=(
                "EWMA filter on returns BEFORE PCA. 0=off. Higher=smoother but "
                "more lag. 15-20 produces trending curves seen in published "
                "cross-asset PCA dashboards."
            ),
        )
    pc3, pc4 = st.columns(2)
    with pc3:
        weighting = st.radio(
            "Weighting",
            options=["equal", "ewm"],
            index=["equal", "ewm"].index(st.session_state.get("pca_weighting", "equal")),
            format_func=lambda x: {"equal": "Equal", "ewm": "Exp (W/3)"}[x],
            key="pca_weighting",
            horizontal=True,
        )
    with pc4:
        pca_method = st.radio(
            "Sign convention",
            options=["standard", "procrustes"],
            index=["standard", "procrustes"].index(st.session_state.get("pca_method", "standard")),
            format_func=lambda x: {"standard": "Per-day SPX+", "procrustes": "Procrustes"}[x],
            key="pca_method",
            horizontal=True,
            help=(
                "Per-day SPX+: anchor SPX positive each day. Honest about regime "
                "changes, can show sign jitter. Procrustes: align with previous "
                "day's PC1. Smooth curves, may obscure regime changes."
            ),
        )

    pca = pca_dominant_theme(returns, window=window, weighting=weighting)
    explained = pca["explained_variance"]
    loadings = pca["loadings"]

    # Mixed vs aligned interpretation
    signs = {a: ("pos" if v > 0 else "neg") for a, v in loadings.items()}
    same_sign = len(set(signs.values())) == 1

    # Headline interpretation — only call out a "dominant" asset when one loading
    # meaningfully exceeds the others (gap >= 0.10). When all three are similar,
    # PCA is saying they're equally participating, not that one is leading.
    label_map = {"SPX": "SPX", "USGG10YR": "UST 10Y", "DXY": "DXY"}
    sorted_by_mag = sorted(loadings.items(), key=lambda kv: abs(kv[1]), reverse=True)
    largest_asset, largest_load = sorted_by_mag[0]
    second_load = sorted_by_mag[1][1]
    dominance_gap = abs(largest_load) - abs(second_load)
    DOMINANCE_THRESHOLD = 0.10

    if dominance_gap >= DOMINANCE_THRESHOLD:
        weight_note = (
            f'<b style="color:#fff;">{label_map[largest_asset]}</b> has the largest '
            f'weight in this theme.'
        )
    else:
        weight_note = (
            'All three assets are participating with similar weight — no single '
            'asset dominates the theme.'
        )

    if same_sign:
        direction_note = "All three are moving in the same direction within the theme."
    else:
        direction_note = (
            "The assets have mixed directions — some are moving with and some "
            "against the common theme."
        )

    st.markdown(
        f"""
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">RIGHT NOW:</span>
          The dominant theme explains
          <b style="color:#fff;">{explained*100:.0f}%</b> of cross-asset moves.
          {weight_note}
          {direction_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Loadings as 3 metrics
    cols = st.columns(3)
    for col_widget, asset in zip(cols, ["SPX", "USGG10YR", "DXY"]):
        load = loadings.get(asset, np.nan)
        weight_label = loading_label(load)
        col_color = {
            "SPX": COLOR_SPX, "USGG10YR": COLOR_UST10Y, "DXY": COLOR_DXY,
        }[asset]
        with col_widget:
            sign = "+" if load >= 0 else ""
            st.markdown(
                f"""
                <div style="text-align:center;padding:0.5rem 0;">
                  <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
                              color:{col_color};font-weight:600;">
                    {label_map[asset]} weight
                  </div>
                  <div style="font-size:24px;font-weight:700;
                              font-family:'JetBrains Mono',monospace;color:#fff;">
                    {sign}{load:.2f}
                  </div>
                  <div style="font-size:10px;color:#888;letter-spacing:0.05em;">
                    {weight_label}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---- Rolling loadings chart ----
    roll = rolling_pca_loadings(returns, window=window, weighting=weighting,
                                pca_method=pca_method,
                                presmooth_halflife=presmooth_halflife)

    # Low-confidence mask: when PC1 is not really dominant.
    # Two failure modes flagged with the same gray band:
    #   (1) Math instability: eigenvalue gap < 0.15 — PC1/PC2 are nearly tied,
    #       so the chosen "dominant direction" is essentially arbitrary.
    #   (2) Weak signal: PC1 explained variance < 0.50 — flags only genuinely
    #       uninformative days. Note: this is looser than the regime panel's
    #       0.60 threshold (which decides "Mixed" classification). The two
    #       thresholds answer different questions:
    #         - Gray band: "should I disclaim this loading reading?" (loose)
    #         - Mixed bucket: "should this day get a clean sign-triple label?" (strict)
    low_conf_mask = (roll["EigGap"] < 0.15) | (roll["ExplainedVar"] < 0.50)

    fig = go.Figure()

    # Shade low-confidence regions in the background
    if low_conf_mask.any():
        # Find contiguous low-confidence bands
        in_band = False
        band_start = None
        for date, is_low in zip(roll.index, low_conf_mask):
            if is_low and not in_band:
                band_start = date
                in_band = True
            elif not is_low and in_band:
                fig.add_vrect(
                    x0=band_start, x1=date,
                    fillcolor="rgba(120,120,120,0.12)",
                    line=dict(width=0),
                    layer="below",
                )
                in_band = False
        if in_band:
            fig.add_vrect(
                x0=band_start, x1=roll.index[-1],
                fillcolor="rgba(120,120,120,0.12)",
                line=dict(width=0),
                layer="below",
            )

    # Mask out gray-band (low-confidence) days from the line traces themselves.
    # Plotly draws None/NaN as line breaks, so loadings will only appear on
    # days where they're statistically reliable. The gray rectangles still
    # show the user that those periods exist; the loading values just don't
    # mislead by suggesting precise readings on uncertain days.
    spx_masked = roll["SPX_load"].where(~low_conf_mask)
    ust_masked = roll["USGG10YR_load"].where(~low_conf_mask)
    dxy_masked = roll["DXY_load"].where(~low_conf_mask)

    fig.add_trace(go.Scatter(
        x=roll.index, y=spx_masked, mode="lines",
        name="SPX weight",
        line=dict(color=COLOR_SPX, width=1.4),
        connectgaps=False,
        hovertemplate="<b>SPX</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=ust_masked, mode="lines",
        name="UST 10Y weight",
        line=dict(color=COLOR_UST10Y, width=1.4),
        connectgaps=False,
        hovertemplate="<b>UST 10Y</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=dxy_masked, mode="lines",
        name="DXY weight",
        line=dict(color=COLOR_DXY, width=1.4),
        connectgaps=False,
        hovertemplate="<b>DXY</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        **DARK_LAYOUT,
        height=360,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc"),
        ),
        xaxis=dict(
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            rangeslider=dict(visible=True, bgcolor="#1a1a1a",
                             bordercolor="#fbbf24", borderwidth=1, thickness=0.04),
            type="date",
        ),
        yaxis=dict(
            range=[-1, 1],
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        margin=dict(l=40, r=20, t=10, b=80),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        f"Same sign = moving together in the theme · Opposite sign = diverging · "
        f"PC1 explains {explained*100:.0f}% of variance currently · "
        f"Gray bands = days where loadings are unreliable "
        f"(PC1 explains <50% of variance OR eigenvalue gap <0.15). "
        f"Lines drop out across those days to avoid showing unstable values."
    )


# ---------------------------------------------------------------------------
# Panel: Regime classification (8-bucket sign cube + persistence)
# ---------------------------------------------------------------------------
def _render_regime_panel(returns: pd.DataFrame):
    """
    8-bucket regime classification view. Reads the same window/weighting/sign
    settings as the Dominant Theme panel (single source of truth), then layers
    a categorical regime label on top of the rolling PCA loadings.
    """
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          REGIME CLASSIFICATION (8-BUCKET SIGN CUBE)
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          Mechanical labeling of each day's PC1 loadings by sign pattern. No economic
          interpretation imposed — the bucket name is just the sign triple
          (e.g. <code>+−−</code> means SPX positive, UST10Y negative, DXY negative).
          <br>
          A day is labeled <span style="color:#aaa;font-weight:600;">Mixed</span> when
          the dominant theme is unreliable: PC1 explained variance below
          <strong>{var:.0%}</strong>, or any loading magnitude below
          <strong>{mag:.2f}</strong>.
          A day is labeled <span style="color:#f97316;font-weight:600;">Transitioning</span>
          when day-over-day persistence drops below <strong>{pers:.2f}</strong>
          (high rotation), regardless of sign pattern.
          Soft scores show how close the loadings are to each archetype regime.
        </div>
        """.format(var=EXP_VAR_THRESHOLD, mag=LOADING_MAGNITUDE_THRESHOLD,
                   pers=PERSISTENCE_THRESHOLD),
        unsafe_allow_html=True,
    )

    # Reuse the Dominant Theme panel's settings — single source of truth so the
    # regime view never disagrees with what the user sees in the loading chart.
    pca_window = st.session_state.get("pca_window", 20)
    pca_weighting = st.session_state.get("pca_weighting", "equal")
    pca_method = st.session_state.get("pca_method", "standard")
    pca_presmooth = st.session_state.get("pca_presmooth", 0)

    smooth_str = f"halflife={pca_presmooth}d" if pca_presmooth > 0 else "off"
    st.caption(
        f"Using Dominant Theme settings: window={pca_window}d · "
        f"weighting={pca_weighting} · sign={pca_method} · pre-smooth={smooth_str}"
    )

    # ---- Compute rolling loadings + classify --------------------------
    loadings = rolling_pca_loadings(
        returns,
        window=pca_window,
        weighting=pca_weighting,
        pca_method=pca_method,
        presmooth_halflife=pca_presmooth,
    )

    if loadings.empty or len(loadings) < 2:
        st.warning(
            "Not enough data to compute regimes. Try a shorter window or "
            "wait for more data."
        )
        return

    # ---- Hard classification + persistence + soft scoring + filtered regimes ----
    raw_regimes = classify_loadings_series(loadings)
    persistence = cosine_persistence(loadings)
    scores = soft_scores(loadings)
    # Apply persistence filter: relabel high-rotation days as "Transitioning"
    regimes = apply_persistence_filter(raw_regimes, persistence)
    info = current_regime_info(regimes, loadings, persistence)

    # ---- Section 1: Current regime headline card -----------------------
    if info:
        spx_color = "#84cc16" if info["spx_load"] > 0 else "#f87171"
        ust_color = "#84cc16" if info["ust_load"] > 0 else "#f87171"
        dxy_color = "#84cc16" if info["dxy_load"] > 0 else "#f87171"
        spx_str = f"+{info['spx_load']:.2f}" if info["spx_load"] >= 0 else f"{info['spx_load']:.2f}"
        ust_str = f"+{info['ust_load']:.2f}" if info["ust_load"] >= 0 else f"{info['ust_load']:.2f}"
        dxy_str = f"+{info['dxy_load']:.2f}" if info["dxy_load"] >= 0 else f"{info['dxy_load']:.2f}"

        # Persistence interpretation
        if info["persistence"] is None:
            pers_str = "—"
            pers_label = ""
        else:
            pers_str = f"{info['persistence']:+.3f}"
            p = info["persistence"]
            if p > 0.99:    pers_label = "very stable"
            elif p > 0.95:  pers_label = "stable"
            elif p > 0.85:  pers_label = "drifting"
            elif p > 0.70:  pers_label = "rotating"
            elif p > 0.0:   pers_label = "rotating fast"
            else:           pers_label = "flipped"

        st.markdown(
            f"""
            <div style="background:rgba(251,191,36,0.04);border:1px solid {info['color']};
                        border-left:4px solid {info['color']};
                        padding:0.85rem 1rem;margin-bottom:1rem;">
              <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
                <div>
                  <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                              text-transform:uppercase;">Now in regime</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                              font-weight:700;color:{info['color']};">{info['regime']}</div>
                  <div style="font-size:11px;color:#bbb;">{info['label']}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                              text-transform:uppercase;">Days in regime</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                              font-weight:700;color:#fff;">{info['days_in']}</div>
                  <div style="font-size:11px;color:#bbb;">since {info['since'].strftime('%Y-%m-%d')}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                              text-transform:uppercase;">Loadings</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;">
                    SPX <span style="color:{spx_color};">{spx_str}</span> ·
                    10Y <span style="color:{ust_color};">{ust_str}</span> ·
                    DXY <span style="color:{dxy_color};">{dxy_str}</span>
                  </div>
                  <div style="font-size:11px;color:#bbb;">PC1 explains
                    <span style="color:#fff;font-weight:600;">{info['expvar']*100:.0f}%</span> of variance</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                              text-transform:uppercase;">Persistence (1-day cosine)</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                              font-weight:700;color:#fff;">{pers_str}</div>
                  <div style="font-size:11px;color:#bbb;">{pers_label}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Section 1b: Soft archetype scores (top-3 closest sign-triples) ----
    if not scores.empty and len(scores) > 0:
        latest_scores = scores.iloc[-1].sort_values(ascending=False)
        # Show top 3 archetypes for the latest day
        top3 = latest_scores.head(3)
        st.markdown(
            """
            <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                        margin:1.2rem 0 0.4rem 0;">
              ARCHETYPE SIMILARITY (TOP 3)
            </div>
            <div style="font-size:11px;color:#888;margin-bottom:0.6rem;">
              Cosine similarity of today's loading vector against each
              archetype sign-triple. +1.0 = perfect match, 0 = orthogonal,
              −1.0 = perfect opposite. The top match is the natural
              hard-bucket assignment (subject to thresholds).
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Build mini bars with colored fill proportional to similarity
        bar_html = "<div style='display:flex;flex-direction:column;gap:6px;margin-bottom:1rem;'>"
        for triple, sim in top3.items():
            color = BUCKET_COLOR.get(triple, "#525252")
            # Bar width: similarity ranges -1 to +1, but display 0-100% width.
            # For positive similarity, bar goes right; negative would go left.
            width_pct = max(0, min(100, sim * 100))
            sim_str = f"{sim:+.3f}"
            bar_html += (
                f"<div style='display:flex;align-items:center;gap:10px;'>"
                f"<code style='display:inline-block;width:50px;color:{color};font-weight:600;'>{triple}</code>"
                f"<div style='flex:1;background:#0f0f0f;border:1px solid #1a1a1a;height:18px;"
                f"position:relative;overflow:hidden;'>"
                f"<div style='background:{color};width:{width_pct}%;height:100%;opacity:0.8;'></div>"
                f"</div>"
                f"<code style='display:inline-block;width:60px;text-align:right;color:#ccc;font-size:11px;'>{sim_str}</code>"
                f"</div>"
            )
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

    # ---- Section 2: Regime timeline stripe -----------------------------
    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          REGIME TIMELINE
        </div>
        """,
        unsafe_allow_html=True,
    )

    runs = regime_runs(regimes)

    # Build a Plotly heatmap-style stripe: one row, x-axis is date,
    # cells colored by regime.
    fig_stripe = go.Figure()
    for _, run in runs.iterrows():
        fig_stripe.add_trace(go.Bar(
            x=[run["End"] - run["Start"] + pd.Timedelta(days=1)],
            y=["Regime"],
            base=[run["Start"]],
            orientation='h',
            marker=dict(color=BUCKET_COLOR[run["Regime"]],
                        line=dict(color=BG, width=0.5)),
            hovertemplate=(
                f"<b>{run['Regime']}</b><br>"
                f"{run['Start'].strftime('%Y-%m-%d')} → "
                f"{run['End'].strftime('%Y-%m-%d')}<br>"
                f"{run['Duration']} day(s)<extra></extra>"
            ),
            showlegend=False,
        ))
    fig_stripe.update_layout(
        **{**DARK_LAYOUT,
           "height": 90,
           "barmode": "stack",
           "showlegend": False,
           "margin": dict(l=10, r=10, t=10, b=30),
           "yaxis": dict(visible=False, fixedrange=True),
           "xaxis": dict(
               type="date",
               showgrid=False,
               tickfont=dict(size=10, color=TEXT_DIM),
           ),
        }
    )
    st.plotly_chart(fig_stripe, use_container_width=True,
                    config={"displayModeBar": False})

    # Color legend strip
    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:1rem;font-size:11px;color:#bbb;margin-bottom:1rem;'>"
    bucket_counts = regimes.value_counts()
    for b in BUCKET_ORDER:
        if b not in bucket_counts.index:
            continue
        legend_html += (
            f"<span style='display:flex;align-items:center;gap:5px;'>"
            f"<span style='width:12px;height:12px;background:{BUCKET_COLOR[b]};"
            f"border-radius:2px;display:inline-block;'></span>"
            f"<code style='color:#ddd;'>{b}</code>"
            f"<span style='color:#888;'>({bucket_counts[b]}d)</span>"
            f"</span>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # ---- Section 3: Persistence tracker --------------------------------
    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          PERSISTENCE TRACKER (DAY-OVER-DAY COSINE SIMILARITY)
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:0.6rem;">
          How stable is the loading vector vs yesterday? +1.0 = unchanged,
          0 = orthogonal rotation, −1.0 = full sign flip.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_pers = go.Figure()
    fig_pers.add_trace(go.Scatter(
        x=persistence.index, y=persistence.values,
        mode="lines",
        line=dict(color="#06b6d4", width=1.2),
        name="Persistence",
        hovertemplate="%{x|%Y-%m-%d}<br>cos = %{y:.4f}<extra></extra>",
    ))
    # Reference bands
    for y, label, color in [
        (0.99,  "very stable (≥0.99)", "rgba(132,204,22,0.25)"),
        (0.95,  "stable (≥0.95)",      "rgba(252,211,77,0.20)"),
        (0.85,  "drifting (≥0.85)",    "rgba(251,146,60,0.18)"),
    ]:
        fig_pers.add_hline(y=y, line=dict(color=color, dash="dot", width=1),
                           annotation_text=label, annotation_position="left",
                           annotation_font=dict(color=TEXT_DIM, size=9))
    fig_pers.add_hline(y=0, line=dict(color="rgba(248,113,113,0.3)", width=1, dash="dash"))

    fig_pers.update_layout(
        **{**DARK_LAYOUT, "height": 220, "showlegend": False,
           "margin": dict(l=10, r=10, t=10, b=30),
           "yaxis": dict(range=[-1.05, 1.05], gridcolor=GRID,
                         tickfont=dict(size=10, color=TEXT_DIM),
                         tickvals=[-1, -0.5, 0, 0.5, 1]),
           "xaxis": dict(gridcolor=GRID,
                         tickfont=dict(size=10, color=TEXT_DIM)),
        }
    )
    st.plotly_chart(fig_pers, use_container_width=True,
                    config={"displayModeBar": False})

    # ---- Section 4: Recent transitions log -----------------------------
    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          RECENT TRANSITIONS
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:0.6rem;">
          Last 20 regime changes, newest first. Persistence column shows the
          minimum cosine over the 5-day window ending at the transition —
          captures rotation magnitude even if the transition day itself
          looks stable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    trans = transitions_log(regimes, persistence, last_n=20)
    if trans.empty:
        st.caption("No transitions in this period.")
    else:
        def _fmt_trans_row(row):
            from_color = BUCKET_COLOR.get(row["From"], "#525252")
            to_color = BUCKET_COLOR.get(row["To"], "#525252")
            pers_str = f"{row['Persistence']:+.3f}" if row["Persistence"] is not None else "—"
            # Color the persistence: red if a real rotation happened
            if row["Persistence"] is not None:
                p = row["Persistence"]
                if p < 0.0:    pers_color = "#dc2626"  # full flip
                elif p < 0.5:  pers_color = "#ef4444"  # major rotation
                elif p < 0.85: pers_color = "#f97316"  # transition zone
                else:          pers_color = "#84cc16"  # mild
            else:
                pers_color = "#888"
            return (
                f"<tr>"
                f"<td style='padding:5px 10px;font-family:JetBrains Mono;color:#bbb;'>"
                f"{row['Date'].strftime('%Y-%m-%d')}</td>"
                f"<td style='padding:5px 10px;'>"
                f"<span style='display:inline-block;width:8px;height:8px;background:{from_color};"
                f"border-radius:2px;margin-right:6px;'></span>"
                f"<code style='color:#ddd;'>{row['From']}</code></td>"
                f"<td style='padding:5px 10px;color:#666;text-align:center;'>→</td>"
                f"<td style='padding:5px 10px;'>"
                f"<span style='display:inline-block;width:8px;height:8px;background:{to_color};"
                f"border-radius:2px;margin-right:6px;'></span>"
                f"<code style='color:#fff;font-weight:600;'>{row['To']}</code></td>"
                f"<td style='padding:5px 10px;font-family:JetBrains Mono;text-align:right;color:{pers_color};'>"
                f"{pers_str}</td>"
                f"<td style='padding:5px 10px;font-family:JetBrains Mono;text-align:right;color:#bbb;'>"
                f"{row['DurationFrom']}d</td>"
                f"</tr>"
            )

        trans_rows = "".join(_fmt_trans_row(r) for _, r in trans.iterrows())
        st.markdown(
            f"""
            <table style='width:100%;border-collapse:collapse;font-size:11px;color:#ccc;
                          border:1px solid #1a1a1a;'>
              <thead>
                <tr style='background:#0f0f0f;border-bottom:1px solid #2a2a2a;'>
                  <th style='padding:7px 10px;text-align:left;font-weight:600;color:#fbbf24;
                             letter-spacing:0.04em;'>DATE</th>
                  <th style='padding:7px 10px;text-align:left;font-weight:600;color:#fbbf24;
                             letter-spacing:0.04em;'>FROM</th>
                  <th style='padding:7px 10px;'></th>
                  <th style='padding:7px 10px;text-align:left;font-weight:600;color:#fbbf24;
                             letter-spacing:0.04em;'>TO</th>
                  <th style='padding:7px 10px;text-align:right;font-weight:600;color:#fbbf24;
                             letter-spacing:0.04em;'>MIN PERS</th>
                  <th style='padding:7px 10px;text-align:right;font-weight:600;color:#fbbf24;
                             letter-spacing:0.04em;'>FROM HELD</th>
                </tr>
              </thead>
              <tbody>
                {trans_rows}
              </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

    # ---- Section 5: Regime stats table ---------------------------------
    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          REGIME STATISTICS
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = regime_stats(regimes)
    if stats.empty:
        st.caption("No data to summarize.")
        return

    # Format for display
    def _fmt_stats_row(row):
        active_dot = "<span style='color:#84cc16;'>●</span>" if row["Active"] else ""
        return (
            f"<tr>"
            f"<td style='padding:6px 10px;'><span style='display:inline-block;width:10px;"
            f"height:10px;background:{BUCKET_COLOR[row['Regime']]};border-radius:2px;"
            f"margin-right:8px;'></span>"
            f"<code style='color:#fff;font-weight:600;'>{row['Regime']}</code> "
            f"{active_dot}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['Days']}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['Pct']:.1f}%</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['Runs']}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['AvgRun']:.1f}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['MaxRun']}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;color:#bbb;'>{row['LastEntry'].strftime('%Y-%m-%d')}</td>"
            f"</tr>"
        )

    table_rows = "".join(_fmt_stats_row(r) for _, r in stats.iterrows())
    st.markdown(
        f"""
        <table style='width:100%;border-collapse:collapse;font-size:12px;color:#ccc;
                      border:1px solid #1a1a1a;'>
          <thead>
            <tr style='background:#0f0f0f;border-bottom:1px solid #2a2a2a;'>
              <th style='padding:8px 10px;text-align:left;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>REGIME</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>DAYS</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>%</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>RUNS</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>AVG RUN</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>MAX RUN</th>
              <th style='padding:8px 10px;text-align:left;font-weight:600;color:#fbbf24;
                         letter-spacing:0.04em;'>LAST ENTRY</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Active = currently in this regime (green dot). · "
        f"Pct = % of {len(regimes)} classified days. · "
        f"Avg/Max Run = mean/longest consecutive days in this regime. · "
        f"Regime engine: {__REGIME_VERSION__}"
    )


# ---------------------------------------------------------------------------
# Panel 3: Raw price levels overview
# ---------------------------------------------------------------------------
def _render_price_levels(prices: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          PRICE LEVELS · 3 INDEPENDENT Y-AXES
        </div>
        <div style="font-size:11px;color:#888;letter-spacing:0.04em;margin-bottom:0.5rem;">
          Reference chart: each asset on its own scale so you can see relative moves at a glance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices["SPX"], mode="lines",
        name="SPX (level)", line=dict(color=COLOR_SPX, width=1.2),
        yaxis="y", hovertemplate="<b>SPX</b><br>%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices["USGG10YR"], mode="lines",
        name="UST 10Y (yield %)", line=dict(color=COLOR_UST10Y, width=1.2),
        yaxis="y2", hovertemplate="<b>UST 10Y</b><br>%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices["DXY"], mode="lines",
        name="DXY (level)", line=dict(color=COLOR_DXY, width=1.2),
        yaxis="y3", hovertemplate="<b>DXY</b><br>%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=320,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=1.12,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc"),
        ),
        xaxis=dict(
            domain=[0.05, 0.92],
            showgrid=True, gridcolor=GRID, zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM), type="date",
        ),
        yaxis=dict(
            title=dict(text="SPX", font=dict(color=COLOR_SPX, size=10)),
            tickfont=dict(color=COLOR_SPX, size=9),
            showgrid=True, gridcolor=GRID, zeroline=False,
        ),
        yaxis2=dict(
            title=dict(text="UST 10Y %", font=dict(color=COLOR_UST10Y, size=10)),
            tickfont=dict(color=COLOR_UST10Y, size=9),
            overlaying="y", side="right",
            showgrid=False, zeroline=False,
        ),
        yaxis3=dict(
            title=dict(text="DXY", font=dict(color=COLOR_DXY, size=10)),
            tickfont=dict(color=COLOR_DXY, size=9),
            overlaying="y", side="right", position=1.0, anchor="free",
            showgrid=False, zeroline=False,
        ),
        margin=dict(l=60, r=80, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
