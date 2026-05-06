"""
Data layer for the Stock Heatmap tab — uses yfinance.

Strategy:
  - Index constituents fetched from Wikipedia (S&P 500) or hardcoded (NDX, Dow).
  - Fundamentals (sector, industry, market cap) cached for 24h via Streamlit.
  - Prices cached for 15min via Streamlit, batched in chunks of 50 tickers.
  - Returns computed from a 1-year history (covers 1D / 1W / 1M / 3M / YTD).

All public functions are decorated with @st.cache_data so the heavy network
calls only happen on first load or after TTL expiry.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Index constituent lists
# ---------------------------------------------------------------------------
# Hardcoded fallback for cases where Wikipedia scrape fails.
# Updated 2025; will drift over time but adequate for a heatmap.

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
    "WBA", "SPLK", "LULU", "ALGN", "SIRI", "ENPH", "JD", "PDD", "MRNA",
    "OKTA", "ZM",
]

# A mid-cap small-cap proxy when "Russell 2000" is requested. We use IWM's
# top holdings rather than the full 2000-name list (which would crush yfinance).
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
    """Scrape S&P 500 ticker list from Wikipedia. Falls back if scrape fails."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        # Wikipedia uses dots in tickers (BRK.B), yfinance uses dashes (BRK-B)
        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
        return sorted(set(tickers))
    except Exception:
        # Minimal fallback — S&P 500 won't fit here so return Nasdaq 100 as a poor man's substitute
        return sorted(set(NDX_100))


def get_index_tickers(index_name: str) -> list[str]:
    """Return the ticker list for the chosen index."""
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
# Fundamentals (sector, industry, market cap)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Fetch sector, industry, market cap, name for a list of tickers.
    Uses yf.Ticker(...).info per ticker — slow first time, then cached 24h.
    """
    import yfinance as yf

    rows = []
    progress = st.progress(0.0, text="Fetching fundamentals…")
    n = len(tickers)
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
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
            progress.progress((i + 1) / n, text=f"Fetching fundamentals… {i+1}/{n}")
    progress.empty()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prices & returns
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Pull 1-year of daily closes for all tickers.
    yfinance batches well — pass space-separated ticker string.
    Returns a DataFrame indexed by date with one column per ticker (close prices).
    """
    import yfinance as yf

    # Chunk into groups of ~50 to avoid massive single requests
    CHUNK = 50
    chunks = [tickers[i:i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    frames = []

    progress = st.progress(0.0, text="Fetching prices…")
    for i, chunk in enumerate(chunks):
        try:
            df = yf.download(
                tickers=" ".join(chunk),
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # When batch returns multi-index columns, extract Close per ticker
            if isinstance(df.columns, pd.MultiIndex):
                close = pd.DataFrame({
                    t: df[t]["Close"] for t in chunk if t in df.columns.levels[0]
                })
            else:
                # Single ticker case — columns are flat
                close = df[["Close"]].rename(columns={"Close": chunk[0]})
            frames.append(close)
        except Exception as e:
            st.warning(f"Price fetch failed for chunk {i+1}: {e}")
        progress.progress((i + 1) / len(chunks),
                          text=f"Fetching prices… {(i+1)*CHUNK}/{len(tickers)}")
    progress.empty()

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    From a 1-year price panel, compute 1D / 1W / 1M / 3M / YTD returns per ticker.
    Returns a DataFrame: index=Ticker, columns=['1D','1W','1M','3M','YTD'].
    """
    if prices.empty:
        return pd.DataFrame()

    last_date = prices.index.max()
    last = prices.iloc[-1]

    def lookback(days: int) -> pd.Series:
        target = last_date - timedelta(days=days)
        # Find the closest trading day on or before target
        valid = prices.index[prices.index <= target]
        if len(valid) == 0:
            return pd.Series(index=prices.columns, dtype=float)
        ref = prices.loc[valid[-1]]
        return (last / ref - 1.0) * 100.0

    # YTD: first trading day of current year
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
# Convenience: full bundle for a chosen index
# ---------------------------------------------------------------------------
def get_heatmap_data(index_name: str) -> pd.DataFrame:
    """
    Return a single dataframe with: Ticker, Name, Sector, Industry,
    MarketCap, AvgVolume, 1D, 1W, 1M, 3M, YTD — ready for the treemap.
    """
    tickers = get_index_tickers(index_name)
    if not tickers:
        return pd.DataFrame()
    tickers_t = tuple(sorted(tickers))

    fundamentals = fetch_fundamentals(tickers_t)
    prices = fetch_prices(tickers_t)
    returns = compute_returns(prices)

    df = fundamentals.merge(returns, on="Ticker", how="left")
    return df
