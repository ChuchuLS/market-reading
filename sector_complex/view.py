"""
Sector complex view — inter-sector analysis on 11 US GICS sector ETFs.

Three sections combined:
  1. Inter-sector correlation regime (macro market vs stock-picker's market)
  2. Sector PCA dominant theme + standard regime classification panel
  3. Relative-strength leadership ranking + cyclical/defensive read

Data source: tries the 'SPDRIndex' sheet of MARKET_DATA.xlsx first (BQL pull),
falls back to yfinance if that sheet is absent. The yfinance path works in
deployment (open network); in a restricted sandbox it will simply error and
the panel shows a friendly message.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theming import BG, GRID, TEXT, TEXT_DIM, DARK_LAYOUT
from sector_complex.analytics import (
    ASSETS, ASSET_LABELS, ANCHOR, BENCHMARK,
    compute_returns, benchmark_returns,
    avg_pairwise_correlation, correlation_regime_summary,
    relative_strength, cyclical_defensive_spread,
    rolling_cyc_def_spread, classify_sector_regime, smooth_regime,
    sector_regime_info, sector_regime_color,
    __ANALYTICS_VERSION__,
)
from sector_complex.regime import (
    regime_runs, transitions_log,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "MARKET_DATA.xlsx"
SHEET_NAME = "SPDRIndex"


# ---------------------------------------------------------------------------
# Data loading: BQL 'sectors' sheet of MARKET_DATA.xlsx
# ---------------------------------------------------------------------------
# Expected sheet layout (matching the existing 'ficc' sheet convention):
#   - First column = dates (BQL exports this as "Unnamed: 0")
#   - One column per instrument, named with the full Bloomberg ticker:
#       XLK US Equity, XLF US Equity, XLE US Equity, XLV US Equity,
#       XLI US Equity, XLY US Equity, XLP US Equity, XLU US Equity,
#       XLB US Equity, XLRE US Equity, XLC US Equity, SPY US Equity
#   Column order does not matter; matching is by ticker prefix.
@st.cache_data(show_spinner=False)
def load_prices(path_str: str, _mtime: float) -> pd.DataFrame:
    """
    Read the 'SPDRIndex' sheet from MARKET_DATA.xlsx.

    Matches columns to tickers by prefix, so "XLK US Equity" → XLK.
    Raises ValueError with an actionable message if the sheet is missing
    or doesn't contain all 11 sectors.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"MARKET_DATA.xlsx not found at {path}")

    xl = pd.ExcelFile(path)
    if SHEET_NAME not in xl.sheet_names:
        raise ValueError(
            f"No '{SHEET_NAME}' sheet in MARKET_DATA.xlsx. "
            f"Found sheets: {xl.sheet_names}. "
            f"Add a '{SHEET_NAME}' sheet to your BQL pull with columns: "
            f"XLK US Equity, XLF US Equity, XLE US Equity, XLV US Equity, "
            f"XLI US Equity, XLY US Equity, XLP US Equity, XLU US Equity, "
            f"XLB US Equity, XLRE US Equity, XLC US Equity, SPY US Equity."
        )

    raw = pd.read_excel(path, sheet_name=SHEET_NAME)
    cols = list(raw.columns)

    # Find the date column
    date_col = None
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(raw[c]):
            date_col = c
            break
        if raw[c].dtype == object:
            parsed = pd.to_datetime(raw[c], errors="coerce")
            if parsed.notna().sum() / max(len(parsed), 1) > 0.8:
                date_col = c
                break
    if date_col is None:
        for c in cols:
            if "date" in str(c).lower() or str(c).startswith("Unnamed"):
                date_col = c
                break
    if date_col is None:
        raise ValueError(
            f"Couldn't find a date column in the '{SHEET_NAME}' sheet. "
            f"Columns: {cols}"
        )

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.rename(columns={date_col: "Date"})

    # Map columns to tickers by prefix. Match longer tickers first so
    # "XLRE" isn't shadowed by a hypothetical "XL" match.
    targets = sorted(ASSETS + [BENCHMARK], key=len, reverse=True)
    rename_map = {}
    used = set()
    for c in raw.columns:
        if c == "Date":
            continue
        cu = str(c).upper().replace(" ", "")
        for t in targets:
            if t in used:
                continue
            if cu.startswith(t):
                rename_map[c] = t
                used.add(t)
                break
    raw = raw.rename(columns=rename_map)

    present = [t for t in ASSETS if t in raw.columns]
    missing = [t for t in ASSETS if t not in raw.columns]
    if missing:
        raise ValueError(
            f"The '{SHEET_NAME}' sheet is missing these sectors: {missing}. "
            f"Required tickers (as Bloomberg columns): "
            f"{', '.join(a + ' US Equity' for a in ASSETS)}. "
            f"Found columns: {list(raw.columns)}"
        )

    keep = ["Date"] + present + ([BENCHMARK] if BENCHMARK in raw.columns else [])
    df = raw[keep].copy()
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"] + present)
    df["_dow"] = df["Date"].dt.dayofweek
    df = df[df["_dow"] < 5].drop(columns="_dow")
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")
    return df


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render_sector_complex():
    st.markdown(
        "<div style='font-size:20px;font-weight:700;letter-spacing:0.04em;"
        "color:#fbbf24;margin-bottom:0.2rem;'>INTER-SECTOR ANALYSIS</div>"
        "<div style='font-size:12px;color:#888;margin-bottom:1rem;'>"
        "11 US GICS sectors · correlation regime · sector PCA · relative strength"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load data
    try:
        mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
        prices = load_prices(str(DATA_PATH), mtime)
    except Exception as e:
        st.error(
            "Couldn't load sector data.\n\n"
            f"{e}\n\n"
            "To fix: add a sheet named 'SPDRIndex' to your BQL pull "
            "(MARKET_DATA.xlsx) with one column per sector ETF using the full "
            "Bloomberg ticker (e.g. 'XLK US Equity'), plus 'SPY US Equity' as "
            "the benchmark, and a date column in the first position."
        )
        return

    if prices is None or len(prices) < 120:
        st.warning("Not enough sector history to compute (need ~120 days).")
        return

    st.caption(
        f"Source: Bloomberg · MARKET_DATA.xlsx · sectors sheet · "
        f"{len(prices)} days · "
        f"{prices.index.min().date()} → {prices.index.max().date()} · "
        f"analytics {__ANALYTICS_VERSION__}"
    )

    returns = compute_returns(prices)
    bench = benchmark_returns(prices)

    # === Section 1: Correlation regime (headline) ===
    _render_correlation_regime(returns)

    # === Section 2: Sector PCA + regime panel ===
    _render_sector_regime(returns)

    # === Section 3: Leadership ranking ===
    _render_leadership(returns, bench)


# ---------------------------------------------------------------------------
# Section 1: Inter-sector correlation regime
# ---------------------------------------------------------------------------
def _render_correlation_regime(returns: pd.DataFrame):
    st.markdown(
        "<div style='font-size:14px;font-weight:600;letter-spacing:0.06em;"
        "color:#fbbf24;margin:1rem 0 0.4rem 0;'>1 · INTER-SECTOR CORRELATION "
        "REGIME</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:11px;color:#999;margin-bottom:0.8rem;'>"
        "Average pairwise correlation across the 11 sectors. <b>High</b> = sectors "
        "moving together (macro-driven market, top-down/factor bets favored). "
        "<b>Low</b> = sectors diverging (stock-picker's market, security selection "
        "favored). Shown as a percentile vs trailing ~2y because average "
        "correlation drifts structurally over time.</div>",
        unsafe_allow_html=True,
    )

    corr = avg_pairwise_correlation(returns, window=60)
    if corr.empty:
        st.caption("Not enough data for correlation regime.")
        return

    summ = correlation_regime_summary(corr)

    # Headline card
    pct = summ["percentile"]
    if pct >= 60:
        card_color = "#f97316"   # macro = orange
    elif pct <= 40:
        card_color = "#22c55e"   # micro = green
    else:
        card_color = "#888"
    st.markdown(
        f"<div style='border:1px solid {card_color};border-radius:6px;"
        f"padding:0.8rem 1.2rem;margin-bottom:1rem;'>"
        f"<span style='font-size:11px;color:#888;letter-spacing:0.08em;'>"
        f"CURRENT REGIME</span><br>"
        f"<span style='font-size:22px;font-weight:700;color:{card_color};"
        f"font-family:monospace;'>{summ['label']}</span>"
        f"<span style='font-size:13px;color:#aaa;margin-left:1rem;'>"
        f"{summ['percentile']:.0f}th percentile · avg corr {summ['latest']:.2f} "
        f"(median {summ['median']:.2f})</span><br>"
        f"<span style='font-size:12px;color:#999;'>{summ['desc']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Time series of avg correlation
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr.index, y=corr["AvgCorr"], mode="lines",
        line=dict(color="#fbbf24", width=1.4),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        name="Avg pairwise corr",
    ))
    # Median reference line
    med = summ["median"]
    fig.add_hline(y=med, line=dict(color="#666", width=1, dash="dash"),
                  annotation_text="trailing median", annotation_position="right",
                  annotation_font=dict(size=9, color=TEXT_DIM))
    fig.update_layout(
        **{**DARK_LAYOUT, "height": 240, "showlegend": False,
           "margin": dict(l=40, r=20, t=10, b=30),
           "yaxis": dict(gridcolor=GRID, zeroline=False,
                         tickfont=dict(size=9, color=TEXT_DIM)),
           "xaxis": dict(showgrid=False,
                         tickfont=dict(size=9, color=TEXT_DIM)),
        }
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Section 2: Sector PCA + regime panel
# ---------------------------------------------------------------------------
def _render_sector_regime(returns):
    st.markdown(
        "<div style='font-size:14px;font-weight:600;letter-spacing:0.06em;"
        "color:#fbbf24;margin:1.5rem 0 0.4rem 0;'>2 \u00b7 SECTOR REGIME "
        "(RISK \u00d7 STRUCTURE)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:11px;color:#999;margin-bottom:0.8rem;'>"
        "Two axes. <b>Risk</b>: cyclical vs defensive tilt \u2014 cyclicals "
        "leading = Risk-On, defensives leading = Risk-Off. <b>Structure</b>: "
        "inter-sector correlation percentile \u2014 high = Macro (broad, "
        "top-down market), low = Micro (dispersed, stock-picker's market). "
        "Quadrants: Risk-On/Macro = broad rally, Risk-On/Micro = narrow risk, "
        "Risk-Off/Macro = correlated selloff, Risk-Off/Micro = dispersed "
        "defensive. Flips shorter than 3 days are smoothed out.</div>",
        unsafe_allow_html=True,
    )

    spread_series = rolling_cyc_def_spread(returns, window=60)
    corr_series = avg_pairwise_correlation(returns, window=60)
    if spread_series.empty or corr_series.empty:
        st.caption("Not enough data for sector regime.")
        return

    raw_regimes = classify_sector_regime(spread_series, corr_series)
    regimes = smooth_regime(raw_regimes, min_days=3)
    if regimes.empty:
        st.caption("Not enough data for sector regime.")
        return

    corr_summ = correlation_regime_summary(corr_series)
    cd = cyclical_defensive_spread(returns, window=60)
    info = sector_regime_info(regimes, spread_series, corr_summ, cd)

    color = sector_regime_color(info["regime"])
    sp = info.get("spread")
    sp_txt = f"{sp*100:+.1f}%" if sp is not None and not pd.isna(sp) else "\u2014"
    pct = info.get("corr_percentile")
    pct_txt = f"{pct:.0f}th" if pct is not None else "\u2014"
    cyc_r = info.get("cyclical_return")
    dfn_r = info.get("defensive_return")
    detail = ""
    if cyc_r is not None and dfn_r is not None:
        detail = f"cyclicals {cyc_r*100:+.1f}% vs defensives {dfn_r*100:+.1f}% (60d)"
    st.markdown(
        f"<div style='border:1px solid {color};border-radius:6px;"
        f"padding:0.8rem 1.2rem;margin-bottom:1rem;'>"
        f"<table style='width:100%;font-size:12px;color:#aaa;'><tr>"
        f"<td><span style='font-size:10px;color:#888;'>NOW IN REGIME</span><br>"
        f"<span style='font-size:18px;font-weight:700;color:{color};"
        f"font-family:monospace;'>{info['regime']}</span></td>"
        f"<td><span style='font-size:10px;color:#888;'>DAYS IN REGIME</span><br>"
        f"<span style='font-size:18px;font-weight:700;color:#eee;'>"
        f"{info['days_in']}</span><br>"
        f"<span style='font-size:11px;'>since {info['since'].strftime('%Y-%m-%d')}"
        f"</span></td>"
        f"<td><span style='font-size:10px;color:#888;'>RISK / STRUCTURE</span><br>"
        f"<span style='font-size:13px;color:#eee;font-family:monospace;'>"
        f"spread {sp_txt} \u00b7 corr {pct_txt}</span><br>"
        f"<span style='font-size:11px;'>{detail}</span></td>"
        f"</tr></table></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:#fbbf24;"
        "margin:0.5rem 0;'>REGIME TIMELINE</div>",
        unsafe_allow_html=True,
    )
    runs = regime_runs(regimes)
    fig_stripe = go.Figure()
    for _, run in runs.iterrows():
        width_ms = (run["End"] - run["Start"] + pd.Timedelta(days=1)).total_seconds() * 1000.0
        fig_stripe.add_trace(go.Bar(
            x=[width_ms],
            y=["Regime"],
            base=run["Start"].isoformat(),
            orientation="h",
            marker=dict(color=sector_regime_color(run["Regime"]),
                        line=dict(color=BG, width=0.5)),
            hovertemplate=(
                f"<b>{run['Regime']}</b><br>"
                f"{run['Start'].strftime('%Y-%m-%d')} \u2192 "
                f"{run['End'].strftime('%Y-%m-%d')}<br>"
                f"{run['Duration']} day(s)<extra></extra>"
            ),
            showlegend=False,
        ))
    fig_stripe.update_layout(
        **{**DARK_LAYOUT,
           "height": 90, "barmode": "overlay", "showlegend": False,
           "margin": dict(l=10, r=10, t=10, b=30),
           "yaxis": dict(visible=False, fixedrange=True),
           "xaxis": dict(type="date", showgrid=False,
                         tickfont=dict(size=10, color=TEXT_DIM)),
        }
    )
    st.plotly_chart(fig_stripe, use_container_width=True,
                    config={"displayModeBar": False})

    present = list(runs["Regime"].unique())
    legend_html = ("<div style='display:flex;flex-wrap:wrap;gap:0.8rem;"
                   "font-size:10px;color:#bbb;margin-bottom:1rem;'>")
    for r in ["Risk-On / Macro", "Risk-On / Micro", "Risk-On / Bal",
              "Risk-Off / Macro", "Risk-Off / Micro", "Risk-Off / Bal",
              "Neutral / Macro", "Neutral / Micro", "Neutral / Bal"]:
        if r in present:
            legend_html += (
                f"<span style='display:flex;align-items:center;gap:4px;'>"
                f"<span style='width:10px;height:10px;background:"
                f"{sector_regime_color(r)};display:inline-block;'></span>"
                f"{r}</span>"
            )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:#fbbf24;"
        "margin:0.5rem 0;'>CYCLICAL \u2212 DEFENSIVE SPREAD (60d, risk axis)</div>",
        unsafe_allow_html=True,
    )
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(
        x=spread_series.index, y=spread_series.values * 100, mode="lines",
        line=dict(color="#22c55e", width=1.2),
        hovertemplate="%{x|%Y-%m-%d}: %{y:+.1f}%<extra></extra>",
    ))
    fig_sp.add_hline(y=0, line=dict(color="#666", width=1, dash="dash"))
    fig_sp.update_layout(
        **{**DARK_LAYOUT, "height": 200, "showlegend": False,
           "margin": dict(l=40, r=20, t=10, b=30),
           "yaxis": dict(gridcolor=GRID, zeroline=False,
                         tickfont=dict(size=9, color=TEXT_DIM)),
           "xaxis": dict(showgrid=False,
                         tickfont=dict(size=9, color=TEXT_DIM)),
        }
    )
    st.plotly_chart(fig_sp, use_container_width=True,
                    config={"displayModeBar": False})

    trans = transitions_log(regimes, pd.Series(dtype=float), last_n=10)
    if not trans.empty:
        st.markdown(
            "<div style='font-size:12px;font-weight:600;color:#fbbf24;"
            "margin:0.5rem 0;'>RECENT TRANSITIONS</div>",
            unsafe_allow_html=True,
        )
        disp = trans[["Date", "From", "To", "DurationFrom"]].copy()
        disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
        disp = disp.rename(columns={"DurationFrom": "From Held (d)"})
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 3: Relative-strength leadership ranking
# ---------------------------------------------------------------------------
def _render_leadership(returns: pd.DataFrame, bench: pd.Series):
    st.markdown(
        "<div style='font-size:14px;font-weight:600;letter-spacing:0.06em;"
        "color:#fbbf24;margin:1.5rem 0 0.4rem 0;'>3 · SECTOR LEADERSHIP "
        "(RELATIVE STRENGTH vs SPY)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:11px;color:#999;margin-bottom:0.8rem;'>"
        "Trailing 60-day return of each sector minus SPY. Positive = "
        "outperforming the index; negative = lagging. The actionable "
        "'which sectors lead right now' view.</div>",
        unsafe_allow_html=True,
    )

    rs = relative_strength(returns, bench, window=60)
    if rs.empty:
        st.caption("Not enough data for relative strength (SPY may be missing).")
        return

    # Horizontal bar chart, sorted
    colors = ["#22c55e" if v >= 0 else "#f97316" for v in rs["RelStrength"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=rs["Label"][::-1],
        x=(rs["RelStrength"] * 100)[::-1],
        orientation="h",
        marker=dict(color=colors[::-1]),
        hovertemplate="%{y}: %{x:+.1f}% vs SPY<extra></extra>",
    ))
    fig.update_layout(
        **{**DARK_LAYOUT, "height": 340, "showlegend": False,
           "margin": dict(l=90, r=20, t=10, b=30),
           "yaxis": dict(tickfont=dict(size=10, color=TEXT)),
           "xaxis": dict(title=dict(text="relative strength vs SPY (%, 60d)",
                                    font=dict(size=10, color=TEXT_DIM)),
                         gridcolor=GRID, zeroline=True, zerolinecolor="#666",
                         tickfont=dict(size=9, color=TEXT_DIM)),
        }
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    # Cyclical / defensive read
    cd = cyclical_defensive_spread(returns, window=60)
    if cd:
        spread = cd["spread"]
        cd_color = ("#22c55e" if spread > 0.02
                    else "#f97316" if spread < -0.02 else "#888")
        st.markdown(
            f"<div style='border:1px solid {cd_color};border-radius:6px;"
            f"padding:0.6rem 1rem;font-size:12px;color:#bbb;'>"
            f"<b style='color:{cd_color};'>{cd['label']}</b> · "
            f"cyclicals {cd['cyclical_return']*100:+.1f}% vs defensives "
            f"{cd['defensive_return']*100:+.1f}% over 60d "
            f"(spread {spread*100:+.1f}%)</div>",
            unsafe_allow_html=True,
        )
