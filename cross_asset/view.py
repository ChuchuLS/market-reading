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
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    with cc1:
        window = st.select_slider(
            "Rolling window (trading days)",
            options=[10, 15, 20, 25, 30, 42, 60, 90, 126, 252],
            value=20,
            key="ca_window",
        )
    with cc2:
        scaling = st.radio(
            "Return scaling",
            options=["zscore", "volscale"],
            index=0,
            format_func=lambda x: {
                "zscore":   "Z-score (within window)",
                "volscale": "Vol-scale (trailing-vol divisor)",
            }[x],
            key="ca_scaling",
            horizontal=True,
            help=(
                "Z-score uses each window's own mean/std (PCA on correlation matrix). "
                "Vol-scale divides each return by its trailing realized vol — preserves "
                "cross-window magnitude comparability."
            ),
        )
    with cc3:
        weighting = st.radio(
            "Window weighting",
            options=["equal", "ewm"],
            index=0,
            format_func=lambda x: {
                "equal": "Equal",
                "ewm":   f"Exponential (halflife=W/3)",
            }[x],
            key="ca_weighting",
            horizontal=True,
            help=(
                "Equal = standard rolling. EWM = exponential decay so recent days carry more weight."
            ),
        )
    with cc4:
        st.markdown("<div style='font-size:10px;color:transparent;'>spacer</div>",
                    unsafe_allow_html=True)
        if st.button("↻ Refresh data", use_container_width=True, key="ca_refresh"):
            st.cache_data.clear()
            st.rerun()

    st.caption(
        f"📁 {DATA_PATH.name} · 🕐 {last_updated} · "
        f"{len(prices)} trading days, {prices.index.min().date()} → {prices.index.max().date()} · "
        f"⚙ Analytics: {__ANALYTICS_VERSION__}"
    )

    # Compute returns according to chosen method
    returns = compute_returns(prices, vol_scale=(scaling == "volscale"))

    st.markdown("---")

    # -------------------------------------------------------------------
    # Two-column layout: Pairwise Correlations | Dominant Theme
    # -------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        _render_correlations_panel(returns, window, weighting)

    with col_right:
        _render_dominant_theme_panel(returns, window, weighting)


# ---------------------------------------------------------------------------
# Panel 1: Pairwise rolling correlations
# ---------------------------------------------------------------------------
def _render_correlations_panel(returns: pd.DataFrame, window: int, weighting: str):
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
def _render_dominant_theme_panel(returns: pd.DataFrame, window: int, weighting: str):
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

    pca = pca_dominant_theme(returns, window=window, weighting=weighting)
    explained = pca["explained_variance"]
    loadings = pca["loadings"]

    # Mixed vs aligned interpretation
    signs = {a: ("pos" if v > 0 else "neg") for a, v in loadings.items()}
    same_sign = len(set(signs.values())) == 1

    # Headline interpretation
    biggest = max(loadings, key=lambda k: abs(loadings[k]))
    label_map = {"SPX": "SPX", "USGG10YR": "UST 10Y", "DXY": "DXY"}
    biggest_label = label_map[biggest]

    if same_sign:
        direction_note = "All three assets are moving in the same direction within this theme."
    else:
        direction_note = "The assets have mixed directions — some are moving with and some against the common theme."

    st.markdown(
        f"""
        <div style="background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.2);
                    padding:0.75rem 1rem;font-size:11px;color:#ccc;line-height:1.6;
                    margin-bottom:1rem;">
          <span style="color:#fbbf24;font-weight:600;">RIGHT NOW:</span>
          The dominant theme explains
          <b style="color:#fff;">{explained*100:.0f}%</b> of cross-asset moves.
          <b style="color:#fff;">{biggest_label}</b> has the largest weight in this theme.
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
    roll = rolling_pca_loadings(returns, window=window, weighting=weighting)

    # Low-confidence mask: when PC1 is not really dominant
    # (eigenvalue gap is small OR explained variance is low)
    low_conf_mask = (roll["EigGap"] < 0.15) | (roll["ExplainedVar"] < 0.45)

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

    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["SPX_load"], mode="lines",
        name="SPX weight",
        line=dict(color=COLOR_SPX, width=1.4),
        hovertemplate="<b>SPX</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["USGG10YR_load"], mode="lines",
        name="UST 10Y weight",
        line=dict(color=COLOR_UST10Y, width=1.4),
        hovertemplate="<b>UST 10Y</b><br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll.index, y=roll["DXY_load"], mode="lines",
        name="DXY weight",
        line=dict(color=COLOR_DXY, width=1.4),
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
        f"Gray bands = periods where the dominant theme is weak (loadings unreliable)."
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
