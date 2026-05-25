"""
Cross-Asset FICC section view — 5-asset version of cross_asset/view.

Reads cross_asset_ficc/data/MARKET_DATA.xlsx (sheet: ficc) (Date, SPX, USGG10YR, DXY,
BCOM, LF98OAS). LF98OAS is displayed as "HY OAS".

Three sub-tabs:
  1. Heatmap (primary) — 5 rows × time, intensity = |loading|, hue = sign vs SPX
  2. Drill-down small multiples — 5 line charts of |loadings| over time
  3. Correlations & Theme — 10 pairwise + leadership decomposition
  4. Regime — continuous regime label + transitions

Color palette: colorblind-safe blue/orange diverging scheme throughout.
Blue  = positive PC1 loading / positive correlation / "same side as SPX".
Orange = negative PC1 loading / negative correlation / "opposite SPX".
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from shared.plots import plot_regime_timeline
from shared.data_utils import drop_all_zero_return_rows
from theming import BG, GRID, TEXT, TEXT_DIM, DARK_LAYOUT
from cross_asset_ficc.analytics import (
    ASSETS,
    ASSET_LABELS,
    LOAD_COLS,
    compute_returns,
    rolling_pairwise_corrs,
    latest_pairwise_corrs,
    all_pair_keys,
    pca_dominant_theme,
    rolling_pca_loadings,
    leadership_stats,
    correlation_summary,
    correlation_label,
    loading_label,
    concentration_label,
    __ANALYTICS_VERSION__,
)
from cross_asset_ficc.regime import (
    classify_loadings_series,
    cosine_persistence,
    apply_persistence_filter,
    transitions_log,
    regime_runs,
    regime_stats,
    current_regime_info,
    regime_color,
    LEADER_COLOR,
    SPECIAL_COLOR,
    LOADING_MAGNITUDE_THRESHOLD,
    EXP_VAR_THRESHOLD,
    PERSISTENCE_THRESHOLD,
    __REGIME_VERSION__,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "MARKET_DATA.xlsx"
SHEET_NAME = "ficc"

# ---------------------------------------------------------------------------
# Color palette — colorblind-safe (blue/orange diverging)
# ---------------------------------------------------------------------------
# Diverging palette for sign-of-loading and sign-of-correlation.
# Blue end = positive (same side as SPX, positive correlation).
# Orange end = negative (opposite side, negative correlation).
# Both colors remain distinguishable under deuteranopia, protanopia, and
# tritanopia. Avoid red/green pairings entirely.
COLOR_POS_DARK = "#1e3a8a"  # deep blue (strongest positive)
COLOR_POS_BRIGHT = "#3b82f6"  # mid blue
COLOR_POS_LIGHT = "#93c5fd"  # light blue
COLOR_NEG_DARK = "#9a3412"  # deep orange
COLOR_NEG_BRIGHT = "#f97316"  # mid orange
COLOR_NEG_LIGHT = "#fdba74"  # light orange
COLOR_NEUTRAL = "#1a1a1a"  # near-black (zero / dark center)

# Diverging colorscales for heatmaps (Plotly format: list of [pos, color])
DIVERGING_SCALE = [
    [0.0, COLOR_NEG_BRIGHT],
    [0.5, COLOR_NEUTRAL],
    [1.0, COLOR_POS_BRIGHT],
]

# Grayscale magnitude-only scale: black (zero) → white (|loading|=1).
# Used for heatmaps where sign is encoded separately as +/− text overlay.
# This avoids the smooth-gradient legibility problem entirely — pure
# light/dark contrast is the most readable encoding of magnitude.
GRAYSCALE_MAG = [
    [0.0, "#0a0a0a"],  # near-black at zero
    [0.5, "#737373"],  # mid-gray
    [1.0, "#f5f5f5"],  # near-white at max
]

# Sequential blue scale for one-dimensional severity (used in transitions log
# for rotation magnitude — severity is one-dimensional so we use a single hue
# rather than a diverging scale).
SEVERITY_LIGHT = "#bfdbfe"  # very light blue (mild)
SEVERITY_MILD = "#60a5fa"  # blue
SEVERITY_STRONG = "#2563eb"  # strong blue
SEVERITY_MAX = "#1e40af"  # very dark blue (severe)

# Per-asset accent colors. Kept distinct in hue AND brightness so they remain
# distinguishable under common forms of colorblindness. No red/green pairing.
ASSET_COLOR = {
    "SPX": "#3b82f6",  # blue
    "USGG10YR": "#06b6d4",  # cyan (distinguishable from blue by saturation)
    "DXY": "#f97316",  # orange
    "BCOM": "#a855f7",  # purple
    "LF98OAS": "#facc15",  # yellow (distinct from orange by lightness)
}

# Substring -> canonical column name. Match logic lifted from cross_asset/view.py
# but extended to BCOM and LF98OAS.
SUBSTRING_MAP = [
    (("SPX",), "SPX"),
    (("USGG10", "UST10", "US10"), "USGG10YR"),
    (("DXY",), "DXY"),
    (("BCOM",), "BCOM"),
    (("LF98OAS",), "LF98OAS"),
]


@st.cache_data(show_spinner=False)
def load_prices(path: Path, _mtime: float) -> pd.DataFrame:
    """Read MARKET_DATA.xlsx (sheet: ficc) and return clean prices DataFrame indexed by Date."""
    raw = pd.read_excel(path, sheet_name=SHEET_NAME)

    # ---- Identify Date column ------------------------------------------
    cols = list(raw.columns)
    date_col = None
    for c in cols:
        col = raw[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            date_col = c
            break
        if col.dtype == object:
            parsed = pd.to_datetime(col, errors="coerce")
            if parsed.notna().sum() / max(len(parsed), 1) > 0.8:
                date_col = c
                break
    if date_col is None:
        for c in cols:
            if "date" in str(c).lower():
                date_col = c
                break
    if date_col is None:
        raise ValueError(
            f"Couldn't find a Date column in MARKET_DATA.xlsx (sheet: ficc). Columns: {cols}"
        )

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.rename(columns={date_col: "Date"})

    # ---- Match price columns by substring (numeric-only) ---------------
    rename_map = {}
    used_targets = set()
    for c in raw.columns:
        if c == "Date":
            continue
        col_data = raw[c]
        try:
            numeric_share = pd.to_numeric(col_data, errors="coerce").notna().mean()
        except Exception:
            numeric_share = 0.0
        if numeric_share < 0.5:
            continue
        cu = str(c).upper().replace(" ", "")
        for substrs, target in SUBSTRING_MAP:
            if target in used_targets:
                continue
            if any(s in cu for s in substrs):
                rename_map[c] = target
                used_targets.add(target)
                break
    raw = raw.rename(columns=rename_map)

    needed = ASSETS
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(
            f"MARKET_DATA.xlsx (sheet: ficc) is missing columns: {missing}. "
            f"Required: Date, SPX, USGG10YR (or UST10Y), DXY, BCOM, LF98OAS. "
            f"Found: {list(raw.columns)}"
        )

    df = raw[["Date"] + needed].copy()
    df = df.dropna(subset=["Date"] + needed)
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)

    df["_dow"] = df["Date"].dt.dayofweek
    df = df[df["_dow"] < 5].drop(columns="_dow")

    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")
    df = df.set_index("Date")
    return df


# ===========================================================================
# Public render
# ===========================================================================
def render_cross_asset_ficc():
    """Render the FICC Cross-Asset section."""

    if not DATA_PATH.exists():
        st.error(
            f"MARKET_DATA.xlsx (sheet: ficc) not found at {DATA_PATH}.\n\n"
            "Required columns: **Date, SPX, USGG10YR, DXY, BCOM, LF98OAS** "
            "(one row per trading day). LF98OAS is displayed as 'HY OAS'."
        )
        return

    mtime = DATA_PATH.stat().st_mtime
    try:
        prices = load_prices(DATA_PATH, mtime)
    except Exception as e:
        st.error(f"Failed to read MARKET_DATA.xlsx (sheet: ficc): {e}")
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
            ⬢ FICC Cross-Asset · SPX / UST 10Y / DXY / BCOM / HY OAS
          </div>
          <div style="font-size:36px;font-weight:800;letter-spacing:-0.01em;
                      font-family:'Inter',sans-serif;line-height:1;">
            FICC MONITOR
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
    # Methodology controls
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

    pc1, pc2, pc3, _ = st.columns([1, 1, 1, 4])
    with pc1:
        if st.button(
            "⚡ Standard",
            use_container_width=True,
            key="ficc_preset_std",
            help="window=20, no smoothing, per-day SPX-positive anchor.",
        ):
            for ns in ("ficc_corr", "ficc_pca"):
                st.session_state[f"{ns}_window"] = 20
                st.session_state[f"{ns}_weighting"] = "equal"
            st.session_state["ficc_scaling"] = "zscore"
            st.session_state["ficc_pca_method"] = "standard"
            st.session_state["ficc_pca_presmooth"] = 0
            st.rerun()
    with pc2:
        if st.button(
            "〜 Smooth",
            use_container_width=True,
            key="ficc_preset_smooth",
            help="window=90, halflife=15 pre-smoothing, Procrustes.",
        ):
            for ns in ("ficc_corr", "ficc_pca"):
                st.session_state[f"{ns}_window"] = 90
                st.session_state[f"{ns}_weighting"] = "equal"
            st.session_state["ficc_scaling"] = "zscore"
            st.session_state["ficc_pca_method"] = "procrustes"
            st.session_state["ficc_pca_presmooth"] = 15
            st.rerun()
    with pc3:
        if st.button(
            "≈ Very Smooth",
            use_container_width=True,
            key="ficc_preset_vsmooth",
            help="window=120, halflife=20, EWM, Procrustes.",
        ):
            for ns in ("ficc_corr", "ficc_pca"):
                st.session_state[f"{ns}_window"] = 120
                st.session_state[f"{ns}_weighting"] = "ewm"
            st.session_state["ficc_scaling"] = "zscore"
            st.session_state["ficc_pca_method"] = "procrustes"
            st.session_state["ficc_pca_presmooth"] = 20
            st.rerun()

    tbc1, tbc2 = st.columns([6, 1])
    with tbc1:
        scaling = st.radio(
            "Return scaling (shared by all panels)",
            options=["zscore", "volscale"],
            index=["zscore", "volscale"].index(
                st.session_state.get("ficc_scaling", "zscore")
            ),
            format_func=lambda x: {
                "zscore": "Z-score (within window)",
                "volscale": "Vol-scale (trailing-vol divisor)",
            }[x],
            key="ficc_scaling",
            horizontal=True,
        )
    with tbc2:
        st.markdown(
            "<div style='font-size:10px;color:transparent;'>spacer</div>",
            unsafe_allow_html=True,
        )
        if st.button("↻ Refresh data", use_container_width=True, key="ficc_refresh"):
            st.cache_data.clear()
            st.rerun()

    corr_window = st.session_state.get("ficc_corr_window", 20)
    corr_weighting = st.session_state.get("ficc_corr_weighting", "equal")
    pca_window = st.session_state.get("ficc_pca_window", 20)
    pca_weighting = st.session_state.get("ficc_pca_weighting", "equal")
    pca_method = st.session_state.get("ficc_pca_method", "standard")
    pca_presmooth = st.session_state.get("ficc_pca_presmooth", 0)

    pca_smooth_str = f"halflife={pca_presmooth}d" if pca_presmooth > 0 else "off"
    settings_html = f"""
        <div style="background:#0f0f0f;border:1px solid #2a2a2a;border-left:3px solid #fbbf24;
                    padding:0.5rem 0.75rem;margin:0.5rem 0;font-family:'JetBrains Mono',monospace;
                    font-size:11px;color:#ccc;line-height:1.7;">
          <div><span style="color:#fbbf24;font-weight:600;">PAIRWISE CORR:</span>
            window={corr_window}d · weighting={corr_weighting}</div>
          <div><span style="color:#fbbf24;font-weight:600;">DOMINANT THEME / REGIME:</span>
            window={pca_window}d · weighting={pca_weighting} · sign={pca_method} ·
            pre-smooth={pca_smooth_str}</div>
          <div><span style="color:#fbbf24;font-weight:600;">SHARED:</span>
            scaling={scaling}</div>
        </div>
    """
    st.markdown(settings_html, unsafe_allow_html=True)

    st.caption(
        f"📁 {DATA_PATH.name} · 🕐 {last_updated} · "
        f"{len(prices)} trading days, {prices.index.min().date()} → {prices.index.max().date()} · "
        f"⚙ Analytics: {__ANALYTICS_VERSION__}"
    )

    returns = compute_returns(prices, vol_scale=(scaling == "volscale"))

    returns, n_dropped = drop_all_zero_return_rows(returns)
    if n_dropped > 0:
        st.caption(
            f"Dropped {n_dropped} rows where all assets were unchanged. "
            f"{len(returns)} valid trading days used."
        )

    st.markdown("---")

    tab_heatmap, tab_drill, tab_corr, tab_regime = st.tabs(
        [
            "🔥 Heatmap",
            "📈 Drill-down",
            "📊 Correlations & Theme",
            "🔬 Regime",
        ]
    )

    loadings = rolling_pca_loadings(
        returns,
        window=pca_window,
        weighting=pca_weighting,
        pca_method=pca_method,
        presmooth_halflife=pca_presmooth,
    )

    with tab_heatmap:
        _render_heatmap_panel(loadings, returns)

    with tab_drill:
        _render_drilldown_panel(loadings)

    with tab_corr:
        col_left, col_right = st.columns(2)
        with col_left:
            _render_correlations_panel(returns)
        with col_right:
            _render_dominant_theme_panel(returns)

    with tab_regime:
        _render_regime_panel(loadings)


# ---------------------------------------------------------------------------
# Tab 1: Heatmap (primary view)
# ---------------------------------------------------------------------------
def _render_heatmap_panel(loadings: pd.DataFrame, returns: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          PARTICIPATION HEATMAP
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          5 rows, one per asset. Cell <b>brightness</b> = how much that asset
          is participating in the dominant theme on that day (|PC1 loading|).
          <b style="color:#f5f5f5;">White</b> = strong participation,
          <b style="color:#737373;">gray</b> = moderate, <b>black</b> = barely participating.
          The <b>+</b> / <b>−</b> symbols inside cells indicate sign vs SPX:
          <b>+</b> = same side as SPX (positive PC1 loading),
          <b>−</b> = opposite SPX. SPX itself is always +.
          Regimes appear as <b>vertical bands</b> of consistent brightness and signs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if loadings.empty:
        st.warning("Not enough data to compute loadings yet.")
        return

    # Magnitude matrix (always non-negative, for grayscale intensity)
    Z_mag = np.array([loadings[f"{a}_load"].abs().values for a in ASSETS])

    # Sign matrix as text overlay. Only show symbols on cells with meaningful
    # magnitude (|loading| ≥ 0.20) — below that, the cell is too dark to read
    # a symbol against and the sign isn't very meaningful anyway.
    SIGN_THRESHOLD = 0.20
    Z_signed = np.array([loadings[f"{a}_load"].values for a in ASSETS])
    text_grid = np.where(
        np.abs(Z_signed) >= SIGN_THRESHOLD, np.where(Z_signed >= 0, "+", "−"), ""
    )

    # Custom hover so we still show the signed loading value
    hover_text = np.array(
        [
            [f"loading = {Z_signed[i, j]:+.3f}" for j in range(Z_signed.shape[1])]
            for i in range(Z_signed.shape[0])
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=Z_mag,
            x=loadings.index,
            y=[ASSET_LABELS[a] for a in ASSETS],
            colorscale=GRAYSCALE_MAG,
            zmin=0,
            zmax=1,
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(
                family="JetBrains Mono",
                size=9,
                # Symbol color: dynamic per cell would be ideal but plotly
                # heatmaps don't support per-cell text color. Use a mid-gray
                # that's visible on both dark and light backgrounds.
                color="#fbbf24",
            ),
            customdata=hover_text,
            hovertemplate="<b>%{y}</b><br>%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
            colorbar=dict(
                title=dict(text="|PC1 loading|", font=dict(color=TEXT_DIM, size=10)),
                tickfont=dict(color=TEXT_DIM, size=9),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                len=0.85,
                thickness=12,
            ),
        )
    )

    fig.update_layout(
        **{
            **DARK_LAYOUT,
            "height": 360,
            "showlegend": False,
            "margin": dict(l=80, r=20, t=10, b=40),
            "xaxis": dict(
                type="date",
                showgrid=False,
                tickfont=dict(size=10, color=TEXT_DIM),
            ),
            "yaxis": dict(
                showgrid=False,
                tickfont=dict(size=11, color=TEXT, family="JetBrains Mono"),
                autorange="reversed",
            ),
        }
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Aux strip: explained variance + concentration ----
    fig_aux = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.5, 0.5],
        subplot_titles=("PC1 explained variance", "Leadership concentration"),
    )

    fig_aux.add_trace(
        go.Scatter(
            x=loadings.index,
            y=loadings["ExplainedVar"],
            mode="lines",
            line=dict(color="#fbbf24", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>ExpVar = %{y:.2%}<extra></extra>",
            name="ExplainedVar",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    lead = leadership_stats(loadings)
    fig_aux.add_trace(
        go.Scatter(
            x=lead.index,
            y=lead["Concentration"],
            mode="lines",
            line=dict(color="#06b6d4", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>Conc = %{y:.2%}<extra></extra>",
            name="Concentration",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig_aux.add_hline(
        y=0.20,
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1),
        row=2,
        col=1,
        annotation_text="diffuse floor (0.20)",
        annotation_position="left",
        annotation_font=dict(color=TEXT_DIM, size=9),
    )
    fig_aux.add_hline(
        y=0.50,
        line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1),
        row=2,
        col=1,
    )

    fig_aux.update_layout(
        **{
            **DARK_LAYOUT,
            "height": 240,
            "showlegend": False,
            "margin": dict(l=60, r=20, t=30, b=30),
        },
    )
    fig_aux.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        tickfont=dict(size=9, color=TEXT_DIM),
        type="date",
    )
    fig_aux.update_yaxes(
        row=1,
        col=1,
        range=[0, 1.0],
        showgrid=True,
        gridcolor=GRID,
        tickfont=dict(size=9, color=TEXT_DIM),
        tickformat=".0%",
    )
    fig_aux.update_yaxes(
        row=2,
        col=1,
        range=[0, 1.0],
        showgrid=True,
        gridcolor=GRID,
        tickfont=dict(size=9, color=TEXT_DIM),
        tickformat=".0%",
    )
    for ann in fig_aux["layout"]["annotations"]:
        ann["font"] = dict(size=10, color="#fbbf24")

    st.plotly_chart(fig_aux, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Top: how much variance the dominant theme is explaining. Low → no single theme. "
        "Bottom: how concentrated leadership is. 0.20 = perfectly diffuse "
        "(all 5 assets equal); 1.0 = pure single-asset move."
    )


# ---------------------------------------------------------------------------
# Tab 2: Drill-down — small multiples
# ---------------------------------------------------------------------------
def _render_drilldown_panel(loadings: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          DRILL-DOWN · SMALL MULTIPLES
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          One panel per asset, |loading| on a shared 0–1 axis. Confirms whether a
          band you spotted in the heatmap is real and stable, not just one
          volatile day.
          <span style="color:#3b82f6;font-weight:600;">Blue</span> = same side as SPX,
          <span style="color:#f97316;font-weight:600;">orange</span> = opposite SPX.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if loadings.empty:
        st.warning("Not enough data.")
        return

    full_min = loadings.index.min().date()
    full_max = loadings.index.max().date()
    default_min = max(full_min, (loadings.index.max() - pd.Timedelta(days=365)).date())

    c1, c2 = st.columns(2)
    with c1:
        d_start = st.date_input(
            "From",
            value=default_min,
            min_value=full_min,
            max_value=full_max,
            key="ficc_drill_start",
        )
    with c2:
        d_end = st.date_input(
            "To",
            value=full_max,
            min_value=full_min,
            max_value=full_max,
            key="ficc_drill_end",
        )

    if d_start > d_end:
        st.warning("Start date is after end date.")
        return

    mask = (loadings.index.date >= d_start) & (loadings.index.date <= d_end)
    sub = loadings.loc[mask]

    if sub.empty:
        st.warning("No data in selected range.")
        return

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.16] * 5 + [0.20],
        subplot_titles=tuple(
            [ASSET_LABELS[a] for a in ASSETS] + ["PC1 explained variance"]
        ),
    )

    for i, asset in enumerate(ASSETS):
        load_col = f"{asset}_load"
        abs_load = sub[load_col].abs()
        signed_load = sub[load_col]

        if asset == "SPX":
            # SPX is always blue (anchored positive)
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=abs_load.values,
                    mode="lines",
                    line=dict(color=COLOR_POS_BRIGHT, width=1.4),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.20)",
                    hovertemplate=f"<b>{ASSET_LABELS[asset]}</b><br>"
                    "%{x|%Y-%m-%d}<br>|load| = %{y:.3f}<extra></extra>",
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
        else:
            # Split blue (same side as SPX) vs orange (opposite)
            spx_sign = np.sign(sub["SPX_load"].values)
            asset_sign = np.sign(signed_load.values)
            same_side = (spx_sign == asset_sign).astype(float)
            same_y = np.where(same_side == 1, abs_load.values, np.nan)
            opp_y = np.where(same_side == 0, abs_load.values, np.nan)

            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=same_y,
                    mode="lines",
                    line=dict(color=COLOR_POS_BRIGHT, width=1.2),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.20)",
                    name=ASSET_LABELS[asset],
                    showlegend=False,
                    hovertemplate=f"<b>{ASSET_LABELS[asset]}</b> (same side as SPX)"
                    "<br>%{x|%Y-%m-%d}<br>|load| = %{y:.3f}<extra></extra>",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=opp_y,
                    mode="lines",
                    line=dict(color=COLOR_NEG_BRIGHT, width=1.2),
                    fill="tozeroy",
                    fillcolor="rgba(249,115,22,0.20)",
                    name=ASSET_LABELS[asset],
                    showlegend=False,
                    hovertemplate=f"<b>{ASSET_LABELS[asset]}</b> (opposite SPX)"
                    "<br>%{x|%Y-%m-%d}<br>|load| = %{y:.3f}<extra></extra>",
                ),
                row=i + 1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=sub["ExplainedVar"].values,
            mode="lines",
            line=dict(color="#fbbf24", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(251,191,36,0.10)",
            hovertemplate="%{x|%Y-%m-%d}<br>ExpVar = %{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=6,
        col=1,
    )

    fig.update_layout(
        **{
            **DARK_LAYOUT,
            "height": 700,
            "showlegend": False,
            "margin": dict(l=60, r=20, t=30, b=30),
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        tickfont=dict(size=9, color=TEXT_DIM),
        type="date",
    )
    for r in range(1, 6):
        fig.update_yaxes(
            row=r,
            col=1,
            range=[0, 1.0],
            showgrid=True,
            gridcolor=GRID,
            tickfont=dict(size=8, color=TEXT_DIM),
            tickvals=[0, 0.5, 1.0],
        )
    fig.update_yaxes(
        row=6,
        col=1,
        range=[0, 1.0],
        showgrid=True,
        gridcolor=GRID,
        tickfont=dict(size=8, color=TEXT_DIM),
        tickformat=".0%",
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=10, color="#fbbf24")
        ann["x"] = 0.0
        ann["xanchor"] = "left"

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        f"Selected range: {d_start} → {d_end} · {len(sub)} days · "
        "Blue fill = asset on same side of PC1 as SPX · "
        "Orange fill = opposite side."
    )


# ---------------------------------------------------------------------------
# Tab 3a: Pairwise rolling correlations (10 pairs)
# ---------------------------------------------------------------------------
def _render_correlations_panel(returns: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          PAIRWISE CORRELATIONS (10 PAIRS)
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          All 10 unique pairs. Cell <b>brightness</b> = strength of correlation
          (|ρ|): white = strong, black = weak. The <b>signed value</b> printed
          in each cell tells you direction (+ or −).
        </div>
        """,
        unsafe_allow_html=True,
    )

    cc1, cc2 = st.columns([3, 2])
    with cc1:
        window = st.slider(
            "Window (trading days)",
            min_value=5,
            max_value=252,
            step=1,
            value=st.session_state.get("ficc_corr_window", 20),
            key="ficc_corr_window",
        )
    with cc2:
        weighting = st.radio(
            "Weighting",
            options=["equal", "ewm"],
            index=["equal", "ewm"].index(
                st.session_state.get("ficc_corr_weighting", "equal")
            ),
            format_func=lambda x: {"equal": "Equal", "ewm": "Exp (W/3)"}[x],
            key="ficc_corr_weighting",
            horizontal=True,
        )

    latest = latest_pairwise_corrs(returns, window=window, weighting=weighting)

    n = len(ASSETS)
    M = np.full((n, n), np.nan)
    for i in range(n):
        M[i, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{ASSETS[i]}_vs_{ASSETS[j]}"
            v = latest.get(key, np.nan)
            M[i, j] = v
            M[j, i] = v

    fig_mat = go.Figure(
        data=go.Heatmap(
            z=np.abs(M),
            x=[ASSET_LABELS[a] for a in ASSETS],
            y=[ASSET_LABELS[a] for a in ASSETS],
            colorscale=GRAYSCALE_MAG,
            zmin=0,
            zmax=1,
            text=[[f"{v:+.2f}" if not np.isnan(v) else "" for v in row] for row in M],
            texttemplate="%{text}",
            textfont=dict(color="#fbbf24", size=11, family="JetBrains Mono"),
            colorbar=dict(
                title=dict(text="|ρ|", font=dict(color=TEXT_DIM, size=10)),
                tickfont=dict(color=TEXT_DIM, size=9),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                len=0.85,
                thickness=10,
            ),
            hovertemplate="%{y} vs %{x}<br>ρ = %{text}<extra></extra>",
        )
    )
    fig_mat.update_layout(
        **{
            **DARK_LAYOUT,
            "height": 320,
            "showlegend": False,
            "margin": dict(l=70, r=20, t=10, b=40),
            "xaxis": dict(showgrid=False, tickfont=dict(size=10, color=TEXT)),
            "yaxis": dict(
                showgrid=False, tickfont=dict(size=10, color=TEXT), autorange="reversed"
            ),
        }
    )
    st.plotly_chart(fig_mat, use_container_width=True, config={"displayModeBar": False})

    rolled = rolling_pairwise_corrs(returns, window=window, weighting=weighting)

    # Colorblind-safe palette for 10 pairs: vary hue AND lightness so they
    # remain distinguishable. Avoid red/green pairs.
    PAIR_PALETTE = [
        "#3b82f6",  # blue
        "#06b6d4",  # cyan
        "#f97316",  # orange
        "#a855f7",  # purple
        "#facc15",  # yellow
        "#0ea5e9",  # sky blue
        "#fb923c",  # light orange
        "#7c3aed",  # violet
        "#0891b2",  # teal
        "#fcd34d",  # light yellow
    ]

    fig = go.Figure()
    for i, key in enumerate(all_pair_keys()):
        a, b = key.split("_vs_")
        label = f"{ASSET_LABELS[a]} vs {ASSET_LABELS[b]}"
        fig.add_trace(
            go.Scatter(
                x=rolled.index,
                y=rolled[key],
                mode="lines",
                name=label,
                line=dict(color=PAIR_PALETTE[i % len(PAIR_PALETTE)], width=1.1),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        **DARK_LAYOUT,
        height=380,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9, color="#ccc"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            rangeslider=dict(
                visible=True,
                bgcolor="#1a1a1a",
                bordercolor="#fbbf24",
                borderwidth=1,
                thickness=0.04,
            ),
            type="date",
        ),
        yaxis=dict(
            range=[-1, 1],
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        margin=dict(l=40, r=20, t=10, b=120),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Tab 3b: Dominant theme + leadership
# ---------------------------------------------------------------------------
def _render_dominant_theme_panel(returns: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          DOMINANT THEME · LEADERSHIP
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          PC1 loadings for all 5 assets, the leader (largest |loading|),
          and the concentration of leadership. Concentration floor is 0.20
          (diffuse), ceiling 1.0 (single-asset move).
        </div>
        """,
        unsafe_allow_html=True,
    )

    pc1, pc2 = st.columns([3, 2])
    with pc1:
        window = st.slider(
            "Window (trading days)",
            min_value=5,
            max_value=252,
            step=1,
            value=st.session_state.get("ficc_pca_window", 20),
            key="ficc_pca_window",
        )
    with pc2:
        presmooth_halflife = st.select_slider(
            "Pre-smooth halflife",
            options=[0, 3, 5, 10, 15, 20, 30],
            value=st.session_state.get("ficc_pca_presmooth", 0),
            key="ficc_pca_presmooth",
        )
    pc3, pc4 = st.columns(2)
    with pc3:
        weighting = st.radio(
            "Weighting",
            options=["equal", "ewm"],
            index=["equal", "ewm"].index(
                st.session_state.get("ficc_pca_weighting", "equal")
            ),
            format_func=lambda x: {"equal": "Equal", "ewm": "Exp (W/3)"}[x],
            key="ficc_pca_weighting",
            horizontal=True,
        )
    with pc4:
        pca_method = st.radio(
            "Sign convention",
            options=["standard", "procrustes"],
            index=["standard", "procrustes"].index(
                st.session_state.get("ficc_pca_method", "standard")
            ),
            format_func=lambda x: {
                "standard": "Per-day SPX+",
                "procrustes": "Procrustes",
            }[x],
            key="ficc_pca_method",
            horizontal=True,
        )

    # Single source of truth: the rolling PCA used by the chart and regime
    # below. The headline is derived from its latest row so the card, chart,
    # and regime all reflect the SAME window/weighting/method/presmooth.
    roll = rolling_pca_loadings(
        returns,
        window=window,
        weighting=weighting,
        pca_method=pca_method,
        presmooth_halflife=presmooth_halflife,
    )
    latest = roll.iloc[-1]
    explained = float(latest["ExplainedVar"])
    loadings_today = {a: float(latest[f"{a}_load"]) for a in ASSETS}

    abs_loads = {a: abs(loadings_today.get(a, 0.0)) for a in ASSETS}
    leader = max(abs_loads, key=abs_loads.get)
    sq = {a: loadings_today.get(a, 0.0) ** 2 for a in ASSETS}
    sq_sum = sum(sq.values()) or 1.0
    concentration = sq[leader] / sq_sum

    st.markdown(
        f"""
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">RIGHT NOW:</span>
          PC1 explains <b style="color:#fff;">{explained*100:.0f}%</b> of variance.
          Leader: <b style="color:{ASSET_COLOR[leader]};">{ASSET_LABELS[leader]}</b>
          ({concentration*100:.0f}% of theme variance — {concentration_label(concentration)}).
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    for col_widget, asset in zip(cols, ASSETS):
        load = loadings_today.get(asset, np.nan)
        weight_label = loading_label(load)
        col_color = ASSET_COLOR[asset]
        with col_widget:
            sign = "+" if load >= 0 else ""
            # Color the loading number by sign — blue if positive, orange if negative
            load_color = COLOR_POS_BRIGHT if load >= 0 else COLOR_NEG_BRIGHT
            st.markdown(
                f"""
                <div style="text-align:center;padding:0.5rem 0;">
                  <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;
                              color:{col_color};font-weight:600;">
                    {ASSET_LABELS[asset]}
                  </div>
                  <div style="font-size:20px;font-weight:700;
                              font-family:'JetBrains Mono',monospace;color:{load_color};">
                    {sign}{load:.2f}
                  </div>
                  <div style="font-size:9px;color:#888;letter-spacing:0.05em;">
                    {weight_label}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    low_conf_mask = (roll["EigGap"] < 0.10) | (roll["ExplainedVar"] < 0.35)

    fig = go.Figure()
    if low_conf_mask.any():
        in_band = False
        band_start = None
        for date_, is_low in zip(roll.index, low_conf_mask):
            if is_low and not in_band:
                band_start = date_
                in_band = True
            elif not is_low and in_band:
                fig.add_vrect(
                    x0=band_start,
                    x1=date_,
                    fillcolor="rgba(120,120,120,0.12)",
                    line=dict(width=0),
                    layer="below",
                )
                in_band = False
        if in_band:
            fig.add_vrect(
                x0=band_start,
                x1=roll.index[-1],
                fillcolor="rgba(120,120,120,0.12)",
                line=dict(width=0),
                layer="below",
            )

    for asset in ASSETS:
        fig.add_trace(
            go.Scatter(
                x=roll.index,
                y=roll[f"{asset}_load"],
                mode="lines",
                name=f"{ASSET_LABELS[asset]} weight",
                line=dict(color=ASSET_COLOR[asset], width=1.3),
                hovertemplate=f"<b>{ASSET_LABELS[asset]}</b><br>"
                "%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    fig.update_layout(
        **DARK_LAYOUT,
        height=360,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9, color="#ccc"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            rangeslider=dict(
                visible=True,
                bgcolor="#1a1a1a",
                bordercolor="#fbbf24",
                borderwidth=1,
                thickness=0.04,
            ),
            type="date",
        ),
        yaxis=dict(
            range=[-1, 1],
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(size=9, color=TEXT_DIM),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        margin=dict(l=40, r=20, t=10, b=80),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        f"Same sign = moving together in the theme · Opposite sign = diverging · "
        f"PC1 explains {explained*100:.0f}% of variance currently · "
        f"Gray bands = periods where the dominant theme is weak."
    )


# ---------------------------------------------------------------------------
# Tab 4: Regime
# ---------------------------------------------------------------------------
def _render_regime_panel(loadings: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fbbf24;
                    margin-bottom:0.4rem;">
          REGIME · CONTINUOUS LEADER + SIGN PATTERN
        </div>
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">What this shows:</span>
          A regime is identified by <b>(leader, 5-bit sign pattern)</b>. Two days
          with the same leader AND same sign pattern are the same regime.
          Days where the dominant theme is weak (PC1 explained variance &lt; {var:.0%},
          or leader |loading| &lt; {mag:.2f}) are labeled
          <span style="color:#aaa;font-weight:600;">Mixed</span>.
          Days where day-over-day persistence drops below <b>{pers:.2f}</b> are labeled
          <span style="color:#facc15;font-weight:600;">Transitioning</span>.
        </div>
        """.format(
            var=EXP_VAR_THRESHOLD,
            mag=LOADING_MAGNITUDE_THRESHOLD,
            pers=PERSISTENCE_THRESHOLD,
        ),
        unsafe_allow_html=True,
    )

    if loadings.empty or len(loadings) < 2:
        st.warning("Not enough data to compute regimes.")
        return

    raw_regimes = classify_loadings_series(loadings)
    persistence = cosine_persistence(loadings)
    regimes = apply_persistence_filter(raw_regimes, persistence)
    info = current_regime_info(regimes, loadings, persistence)

    if info:
        if info["persistence"] is None:
            pers_str = "—"
            pers_label = ""
        else:
            p = info["persistence"]
            pers_str = f"{p:+.3f}"
            if p > 0.99:
                pers_label = "very stable"
            elif p > 0.95:
                pers_label = "stable"
            elif p > 0.85:
                pers_label = "drifting"
            elif p > 0.70:
                pers_label = "rotating"
            elif p > 0.0:
                pers_label = "rotating fast"
            else:
                pers_label = "flipped"

        # Loading colors: blue if positive, orange if negative
        loadings_html = " · ".join(
            f"{ASSET_LABELS[a]} <span style='color:{COLOR_POS_BRIGHT if info['loadings'][a] >= 0 else COLOR_NEG_BRIGHT};'>"
            f"{'+' if info['loadings'][a] >= 0 else ''}{info['loadings'][a]:.2f}</span>"
            for a in ASSETS
        )

        st.markdown(
            f"""
            <div style="background:rgba(251,191,36,0.04);border:1px solid {info['color']};
                        border-left:4px solid {info['color']};
                        padding:0.85rem 1rem;margin-bottom:1rem;">
              <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
                <div>
                  <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                              text-transform:uppercase;">Now in regime</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:20px;
                              font-weight:700;color:{info['color']};">{info['regime']}</div>
                  <div style="font-size:11px;color:#bbb;">
                    leader: <b style="color:{info['color']};">{info['leader_label']}</b> ·
                    concentration: <b style="color:#fff;">{info['concentration']*100:.0f}%</b>
                    ({concentration_label(info['concentration'])})
                  </div>
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
                              text-transform:uppercase;">Persistence (1-day cosine)</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
                              font-weight:700;color:#fff;">{pers_str}</div>
                  <div style="font-size:11px;color:#bbb;">{pers_label}</div>
                </div>
              </div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                          color:#bbb;margin-top:0.6rem;border-top:1px solid #1a1a1a;
                          padding-top:0.5rem;">
                {loadings_html} · ExpVar
                <b style="color:#fff;">{info['expvar']*100:.0f}%</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          REGIME TIMELINE
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:0.6rem;">
          Each band is a contiguous run of one regime. Color is determined by leader.
        </div>
        """,
        unsafe_allow_html=True,
    )

    runs = regime_runs(regimes)

    fig_stripe = plot_regime_timeline(
        runs=runs,
        color_func=regime_color,
        x_min=regimes.index.min(),
        x_max=regimes.index.max(),
        height=120,
        dark_layout=DARK_LAYOUT,
        text_dim=TEXT_DIM,
    )
    st.plotly_chart(
        fig_stripe, use_container_width=True, config={"displayModeBar": False}
    )

    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:1rem;font-size:11px;color:#bbb;margin-bottom:1rem;'>"
    for asset in ASSETS:
        legend_html += (
            f"<span style='display:flex;align-items:center;gap:5px;'>"
            f"<span style='width:12px;height:12px;background:{LEADER_COLOR[asset]};"
            f"border-radius:2px;display:inline-block;'></span>"
            f"<code style='color:#ddd;'>{ASSET_LABELS[asset]}-led</code>"
            f"</span>"
        )
    for special, color in SPECIAL_COLOR.items():
        legend_html += (
            f"<span style='display:flex;align-items:center;gap:5px;'>"
            f"<span style='width:12px;height:12px;background:{color};"
            f"border-radius:2px;display:inline-block;'></span>"
            f"<code style='color:#ddd;'>{special}</code>"
            f"</span>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          PERSISTENCE TRACKER (DAY-OVER-DAY COSINE SIMILARITY, 5-D VECTOR)
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_pers = go.Figure()
    fig_pers.add_trace(
        go.Scatter(
            x=persistence.index,
            y=persistence.values,
            mode="lines",
            line=dict(color="#06b6d4", width=1.2),
            name="Persistence",
            hovertemplate="%{x|%Y-%m-%d}<br>cos = %{y:.4f}<extra></extra>",
        )
    )
    # Reference bands — using blue gradient (severity scale, not red/green)
    for y, label, color in [
        (0.99, "very stable (≥0.99)", "rgba(147,197,253,0.30)"),
        (0.95, "stable (≥0.95)", "rgba(96,165,250,0.25)"),
        (0.85, "drifting (≥0.85)", "rgba(37,99,235,0.20)"),
    ]:
        fig_pers.add_hline(
            y=y,
            line=dict(color=color, dash="dot", width=1),
            annotation_text=label,
            annotation_position="left",
            annotation_font=dict(color=TEXT_DIM, size=9),
        )
    fig_pers.add_hline(
        y=0, line=dict(color="rgba(249,115,22,0.4)", width=1, dash="dash")
    )

    fig_pers.update_layout(
        **{
            **DARK_LAYOUT,
            "height": 220,
            "showlegend": False,
            "margin": dict(l=10, r=10, t=10, b=30),
            "yaxis": dict(
                range=[-1.05, 1.05],
                gridcolor=GRID,
                tickfont=dict(size=10, color=TEXT_DIM),
                tickvals=[-1, -0.5, 0, 0.5, 1],
            ),
            "xaxis": dict(gridcolor=GRID, tickfont=dict(size=10, color=TEXT_DIM)),
        }
    )
    st.plotly_chart(
        fig_pers, use_container_width=True, config={"displayModeBar": False}
    )

    st.markdown(
        """
        <div style="font-size:12px;font-weight:600;letter-spacing:0.06em;color:#fbbf24;
                    margin:1.2rem 0 0.4rem 0;">
          RECENT TRANSITIONS
        </div>
        """,
        unsafe_allow_html=True,
    )

    trans = transitions_log(regimes, persistence, last_n=20)
    if trans.empty:
        st.caption("No transitions in this period.")
    else:

        def _fmt_trans_row(row):
            from_color = regime_color(row["From"])
            to_color = regime_color(row["To"])
            pers_str = (
                f"{row['Persistence']:+.3f}" if row["Persistence"] is not None else "—"
            )
            # Severity color uses single-hue blue scale (severity is 1-D, not diverging)
            if row["Persistence"] is not None:
                p = row["Persistence"]
                if p < 0.0:
                    pers_color = SEVERITY_MAX  # full flip — darkest blue
                elif p < 0.5:
                    pers_color = SEVERITY_STRONG  # major rotation
                elif p < 0.85:
                    pers_color = SEVERITY_MILD  # transition zone
                else:
                    pers_color = SEVERITY_LIGHT  # mild
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
                  <th style='padding:7px 10px;text-align:left;font-weight:600;color:#fbbf24;'>DATE</th>
                  <th style='padding:7px 10px;text-align:left;font-weight:600;color:#fbbf24;'>FROM</th>
                  <th style='padding:7px 10px;'></th>
                  <th style='padding:7px 10px;text-align:left;font-weight:600;color:#fbbf24;'>TO</th>
                  <th style='padding:7px 10px;text-align:right;font-weight:600;color:#fbbf24;'>MIN PERS</th>
                  <th style='padding:7px 10px;text-align:right;font-weight:600;color:#fbbf24;'>FROM HELD</th>
                </tr>
              </thead>
              <tbody>{trans_rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

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

    def _fmt_stats_row(row):
        active_dot = "<span style='color:#3b82f6;'>●</span>" if row["Active"] else ""
        return (
            f"<tr>"
            f"<td style='padding:6px 10px;'><span style='display:inline-block;width:10px;"
            f"height:10px;background:{regime_color(row['Regime'])};border-radius:2px;"
            f"margin-right:8px;'></span>"
            f"<code style='color:#fff;font-weight:600;'>{row['Regime']}</code> "
            f"{active_dot}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['Days']}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['Pct']:.1f}%</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['Runs']}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['AvgRun']:.1f}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;text-align:right;'>{row['MaxRun']}</td>"
            f"<td style='padding:6px 10px;font-family:JetBrains Mono;color:#bbb;'>"
            f"{row['LastEntry'].strftime('%Y-%m-%d')}</td>"
            f"</tr>"
        )

    table_rows = "".join(_fmt_stats_row(r) for _, r in stats.iterrows())
    st.markdown(
        f"""
        <table style='width:100%;border-collapse:collapse;font-size:12px;color:#ccc;
                      border:1px solid #1a1a1a;'>
          <thead>
            <tr style='background:#0f0f0f;border-bottom:1px solid #2a2a2a;'>
              <th style='padding:8px 10px;text-align:left;font-weight:600;color:#fbbf24;'>REGIME</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;'>DAYS</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;'>%</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;'>RUNS</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;'>AVG RUN</th>
              <th style='padding:8px 10px;text-align:right;font-weight:600;color:#fbbf24;'>MAX RUN</th>
              <th style='padding:8px 10px;text-align:left;font-weight:600;color:#fbbf24;'>LAST ENTRY</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Active = currently in this regime (blue dot). · "
        f"Pct = % of {len(regimes)} classified days. · "
        f"Avg/Max Run = mean/longest consecutive days in this regime. · "
        f"Regime engine: {__REGIME_VERSION__}"
    )
