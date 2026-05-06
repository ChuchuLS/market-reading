# Market Reading

Unified Streamlit dashboard combining cross-asset macro tracking with a TradingView-style stock heatmap.

## Top-level sections

- **📊 Macro Tracker** — BQL-driven snapshot across 57 instruments (themes, equity ETFs, commodity futures) with 4 sub-tabs: Heatmap, Leaders & Laggards, Category Rotation, Cross-Timeframe rotation map.
- **🌐 Stock Heatmap** — interactive Plotly treemap of S&P 500 / Nasdaq 100 / Dow / Russell 2000 (top-50 proxy) constituents, sized by market cap or volume, colored by 1D / 1W / 1M / 3M / YTD return. Powered by Yahoo Finance.

## Repo layout

```
market_reading/
├── app.py                      # entry point — top-level tabs
├── theming.py                  # global CSS, color tokens
├── requirements.txt
├── README.md
│
├── macro_tracker/
│   ├── __init__.py
│   ├── view.py                 # 4 BQL sub-tabs + ticker map
│   └── data/
│       └── DATA.xlsx           # BQL output
│
└── stock_heatmap/
    ├── __init__.py
    ├── view.py                 # treemap + filters
    └── data_yf.py              # yfinance fetching layer
```

## Setup (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

## Setup (Streamlit Cloud)

Point the deployment at this repo's root, with `app.py` as the entry point. Streamlit Cloud will read `requirements.txt` and install everything automatically.

## Refreshing data

- **Macro Tracker** reads `macro_tracker/data/DATA.xlsx`. To refresh: regenerate the BQL sheet, save over the file, click the sidebar **↻ Refresh BQL data** button.
- **Stock Heatmap** fetches live from Yahoo Finance on demand. Click **⟳ Load / Refresh data** in the tab. First load per index takes 30-90s (especially S&P 500); subsequent loads use cached data (15min TTL for prices, 24h for fundamentals).

## Notes

- yfinance is rate-limited and occasionally flaky. If the heatmap fails to load, wait a minute and retry. The Russell 2000 option uses a 50-name proxy because the full 2000-constituent fetch crushes yfinance.
- All theme ETF 1D returns may show 0% if the BQL sheet was refreshed outside US trading hours (Asia/Europe morning). The Macro Tracker auto-detects this and shows a warning. Refresh during US hours for live 1D values.
