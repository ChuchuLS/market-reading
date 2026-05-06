"""
Data layer for the Stock Heatmap tab — uses yfinance.

Strategy:
  - Constituents from Wikipedia (S&P 500) or hardcoded snapshots (NDX, Dow).
  - Fundamentals (sector, market cap) cached for 24h.
  - Prices cached for 15min, batched. Returns computed from history.
  - Aggressive error reporting so users see what's failing.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Index constituent lists (snapshots — drift over time but adequate)
# ---------------------------------------------------------------------------
DOW_30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]

NDX_100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "PEP", "COST", "ADBE", "CSCO", "CMCSA", "TMUS", "NFLX", "AMD", "INTC",
    "INTU", "TXN", "QCOM", "AMGN", "HON", "AMAT", "ISRG", "BKNG", "VRTX",
    "ADP", "GILD", "ADI", "MU", "REGN", "PANW", "LRCX", "MDLZ", "SBUX",
    "KLAC", "PYPL", "SNPS", "MELI", "CDNS", "ABNB", "CRWD", "MAR", "ORLY",
    "MNST", "FTNT", "ASML", "CSX", "PCAR", "ROP", "CHTR", "WDAY", "ADSK",
    "NXPI", "DXCM", "PAYX", "AEP", "MRVL", "FANG", "ROST", "ODFL", "FAST",
    "CTAS", "KDP", "EXC", "BKR", "TEAM", "CPRT", "VRSK", "EA", "DDOG",
    "GEHC", "BIIB", "ON", "XEL", "CSGP", "IDXX", "ZS", "CCEP", "TTWO",
    "DLTR", "ANSS", "MCHP", "WBD", "TTD", "CDW", "MDB", "DASH", "ILMN",
    "WBA", "LULU", "ALGN",
]

RUSSELL_2000_PROXY = [
    "SMCI", "MSTR", "FTAI", "RKLB", "FIX", "AIT", "SFM", "INSM", "MLI",
    "ENSG", "CRS", "ANF", "CVLT", "CHX", "BMI", "KTB", "MUR", "RDN",
    "WTS", "CHRD", "EME", "BOOT", "PI", "WERN", "TGNA", "DCI", "RGEN",
    "SPSC", "PJT", "COKE", "ROAD", "AAON", "ESI", "ATKR", "ITRI", "MTH",
    "MMSI", "LNTH", "TFIN", "AVAV", "ESNT", "CSWI", "EXLS", "WTFC", "FSS",
    "WAFD", "BCC", "ALG", "OFG", "TPH",
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_sp500_constituents() -> list[str]:
    """Scrape S&P 500 ticker list from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
        return sorted(set(tickers))
    except Exception as e:
        st.warning(f"Wikipedia S&P 500 scrape failed: {e}. Using Nasdaq 100 fallback.")
        return sorted(set(NDX_100))


def get_index_tickers(index_name: str) -> list[str]:
    if index_name == "S&P 500":
        return get_sp500_constituents()
    elif index_name == "Nasdaq 100":
        return NDX_100
    elif index_name == "Dow Jones 30":
        return DOW_30
    elif index_name == "Russell 2000 (Top 50 proxy)":
        return RUSSELL_2000_PROXY
    return []


# ---------------------------------------------------------------------------
# Prices (fast path — batched yf.download)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Pull 1-year daily closes for all tickers in batched calls.
    Returns DataFrame indexed by date, columns = ticker symbols.
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame()

    CHUNK = 25
    chunks = [list(tickers[i:i + CHUNK]) for i in range(0, len(tickers), CHUNK)]
    frames = []
    errors = []

    progress = st.progress(0.0, text=f"Fetching prices for {len(tickers)} tickers…")
    for i, chunk in enumerate(chunks):
        try:
            raw = yf.download(
                tickers=chunk,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                top = raw.columns.get_level_values(0).unique()
                close_cols = {}
                for t in chunk:
                    if t in top and "Close" in raw[t].columns:
                        close_cols[t] = raw[t]["Close"]
                if close_cols:
                    frames.append(pd.DataFrame(close_cols))
            elif "Close" in raw.columns:
                frames.append(raw[["Close"]].rename(columns={"Close": chunk[0]}))
        except Exception as e:
            errors.append(f"chunk {i+1} ({chunk[0]}…{chunk[-1]}): {e}")
        progress.progress(
            (i + 1) / len(chunks),
            text=f"Fetching prices… {min((i+1)*CHUNK, len(tickers))}/{len(tickers)}",
        )
    progress.empty()

    if errors:
        st.warning("Some price chunks failed:\n" + "\n".join(errors[:3]))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


# ---------------------------------------------------------------------------
# Fundamentals (slow path — one .info call per ticker)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch sector, industry, market cap, name per ticker. Cached 24h."""
    import yfinance as yf

    rows = []
    progress = st.progress(0.0, text=f"Fetching fundamentals for {len(tickers)} tickers…")
    n = len(tickers)
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info or {}
            rows.append({
                "Ticker": t,
                "Name": info.get("shortName") or info.get("longName") or t,
                "Sector": info.get("sector") or "Unknown",
                "Industry": info.get("industry") or "Unknown",
                "MarketCap": info.get("marketCap") or 0,
                "AvgVolume": info.get("averageVolume") or 0,
            })
        except Exception:
            rows.append({
                "Ticker": t, "Name": t, "Sector": "Unknown",
                "Industry": "Unknown", "MarketCap": 0, "AvgVolume": 0,
            })
        if (i + 1) % 5 == 0 or i == n - 1:
            progress.progress((i + 1) / n, text=f"Fundamentals… {i+1}/{n}")
    progress.empty()
    return pd.DataFrame(rows)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 1D / 1W / 1M / 3M / YTD returns from a price panel."""
    if prices.empty:
        return pd.DataFrame()

    last_date = prices.index.max()
    last = prices.iloc[-1]

    def lookback(days: int) -> pd.Series:
        target = last_date - timedelta(days=days)
        valid = prices.index[prices.index <= target]
        if len(valid) == 0:
            return pd.Series(index=prices.columns, dtype=float)
        ref = prices.loc[valid[-1]]
        return (last / ref - 1.0) * 100.0

    year_start = pd.Timestamp(year=last_date.year, month=1, day=1)
    valid_ytd = prices.index[prices.index >= year_start]
    if len(valid_ytd):
        ytd_ref = prices.loc[valid_ytd[0]]
        ytd_ret = (last / ytd_ref - 1.0) * 100.0
    else:
        ytd_ret = pd.Series(index=prices.columns, dtype=float)

    out = pd.DataFrame({
        "1D": lookback(1),
        "1W": lookback(7),
        "1M": lookback(30),
        "3M": lookback(90),
        "YTD": ytd_ret,
    })
    out.index.name = "Ticker"
    return out.reset_index()


# ---------------------------------------------------------------------------
# Public entry point — bundles everything with diagnostics
# ---------------------------------------------------------------------------
def get_heatmap_data(index_name: str) -> pd.DataFrame:
    """
    Returns a DataFrame with: Ticker, Name, Sector, Industry, MarketCap,
    AvgVolume, 1D, 1W, 1M, 3M, YTD.
    Surfaces failures via st.info/st.warning/st.error so users can diagnose.
    """
    tickers = get_index_tickers(index_name)
    if not tickers:
        st.error(f"No tickers found for index '{index_name}'.")
        return pd.DataFrame()

    st.info(f"📡 Fetching {index_name}: {len(tickers)} tickers …")

    tickers_t = tuple(sorted(tickers))

    # Step 1: prices
    prices = fetch_prices(tickers_t)
    if prices.empty:
        st.error(
            "❌ Yahoo Finance returned no price data. "
            "Most likely yfinance is being rate-limited from Streamlit Cloud's "
            "shared IP. Try again in a few minutes, or test locally first."
        )
        return pd.DataFrame()

    n_with_prices = prices.shape[1]
    st.success(f"✅ Got prices for {n_with_prices}/{len(tickers)} tickers")

    # Step 2: returns (local)
    returns = compute_returns(prices)

    # Step 3: fundamentals
    fundamentals = fetch_fundamentals(tickers_t)

    df = fundamentals.merge(returns, on="Ticker", how="inner")
    if df.empty:
        st.error("❌ Merge between fundamentals and returns produced 0 rows.")
        return pd.DataFrame()

    return df
