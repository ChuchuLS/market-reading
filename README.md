# Market Reading

Unified Streamlit dashboard combining cross-asset macro tracking, a
TradingView-style stock heatmap, and a suite of PCA-based "complex" regime
analyzers (rates, credit, FX, equity, commodities, sectors).

## Navigation

The app uses **sidebar navigation** (`st.sidebar.radio`) — only the selected
page is rendered, so Streamlit no longer computes every page on each run.

Pages:

- **Macro Tracker** — BQL-driven snapshot across instruments (themes, equity
  ETFs, commodity futures) with sub-tabs: Heatmap, Leaders & Laggards,
  Category Rotation, Cross-Timeframe rotation map.
- **Stock Heatmap** — interactive Plotly treemap of S&P 500 / Nasdaq 100 /
  Dow / Russell 2000 (proxy) constituents, sized by market cap or volume,
  colored by 1D / 1W / 1M / 3M / YTD return. Powered by Yahoo Finance.
- **3-Asset Classic** — SPX / UST 10Y / DXY rolling correlation + PCA.
- **FICC** — 5-asset cross-asset PCA (SPX, UST 10Y, DXY, BCOM, HY OAS).
- **Rates** — rates complex (curve, breakevens, real yields, MOVE).
- **Credit** — credit complex (HY OAS, IG OAS, iTraxx).
- **FX** — FX complex (DXY, EM FX, cross-currency basis).
- **Equity** — equity internals (SPX cap, SPW equal, VIX).
- **Commodities** — commodity complex (BCOM, WTI, copper, gold, soybeans).
- **Sectors** — inter-sector analysis across the 11 US GICS sector ETFs
  (correlation regime, sector PCA, relative-strength leadership).

Each complex page shows a Dominant Theme panel (rolling PC1 loadings), a
Regime Timeline (continuous colored bands via the shared Scatter helper), a
persistence tracker, and a recent-transitions table.

## Repo layout

```
market-reading/
├── app.py                      # entry point — sidebar navigation
├── theming.py                  # global CSS, color tokens, DARK_LAYOUT
├── requirements.txt
├── README.md
│
├── shared/
│   ├── __init__.py
│   ├── plots.py                # plot_regime_timeline() — shared Scatter helper
│   └── data_utils.py           # drop_all_zero_return_rows()
│
├── data/
│   └── MARKET_DATA.xlsx        # central BQL workbook (multiple sheets)
│
├── macro_tracker/              # reads data/MARKET_DATA.xlsx (sheet: macro_tracker)
├── stock_heatmap/              # yfinance-powered treemap (data_yf.py)
│
├── cross_asset/                # 3-asset classic; reads cross_asset/data/CROSSASSET.xlsx
├── cross_asset_ficc/           # reads data/MARKET_DATA.xlsx (sheet: ficc)
├── rates_complex/              # reads data/MARKET_DATA.xlsx (sheet: ficc)
├── credit_complex/             # reads data/MARKET_DATA.xlsx (sheet: ficc)
├── fx_complex/                 # reads data/MARKET_DATA.xlsx (sheet: ficc)
├── equity_complex/             # reads data/MARKET_DATA.xlsx (sheet: ficc)
├── comdty_complex/             # reads data/MARKET_DATA.xlsx (sheet: ficc)
└── sector_complex/             # reads data/MARKET_DATA.xlsx (sheet: SPDRIndex)
```

Each complex module follows the same shape: `analytics.py` (returns, PCA,
regime math), `regime.py` (classification, persistence, colors), `view.py`
(Streamlit rendering + data loading).

## Data: `data/MARKET_DATA.xlsx`

A single Excel workbook with one sheet per data domain. The loaders match
columns to tickers by prefix, so full Bloomberg column names like
`SPX Index`, `USGG10YR Index`, `XLK US Equity` work directly. The first
column is the date (BQL exports it as `Unnamed: 0`).

Sheets currently used:

| Sheet         | Used by                                              |
|---------------|------------------------------------------------------|
| `macro_tracker` | Macro Tracker                                      |
| `ficc`        | FICC, Rates, Credit, FX, Equity, Commodities (each reads its own subset of columns) |
| `SPDRIndex`   | Sectors (11 sector ETFs `XLK..XLC US Equity` + `SPY US Equity`) |

The 3-Asset Classic page reads its own file, `cross_asset/data/CROSSASSET.xlsx`
(columns `SPX Index`, `DXY Curncy`, `USGG10YR Index`).

> Note: the `cross_asset` sheet inside MARKET_DATA.xlsx exists but the
> 3-Asset Classic page currently reads the standalone CROSSASSET.xlsx file.

## Setup (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

## Setup (Streamlit Cloud)

Point the deployment at this repo's root with `app.py` as the entry point.
Streamlit Cloud reads `requirements.txt` and installs everything.

## Refreshing data

- **Complex pages + Macro Tracker** read sheets from `data/MARKET_DATA.xlsx`.
  To refresh: regenerate the BQL sheets, save over the workbook, and use the
  sidebar refresh control. The loaders cache on file mtime, so saving the
  file invalidates the cache.
- **Stock Heatmap** fetches live from Yahoo Finance on demand. First load per
  index takes 30–90s; subsequent loads use cached data (15min TTL for prices,
  24h for fundamentals).

## Notes

- A zero daily return is valid for spreads, curves, OAS, breakevens, and vol
  indices, so the stale-data filter only drops rows where **all** assets are
  unchanged (a fully forward-filled holiday), not rows where a single series
  is flat. See `shared/data_utils.drop_all_zero_return_rows`.
- yfinance is rate-limited and occasionally flaky. If the heatmap fails to
  load, wait a minute and retry.
- Theme ETF 1D returns may show 0% if a BQL sheet was refreshed outside US
  trading hours; refresh during US hours for live 1D values.
