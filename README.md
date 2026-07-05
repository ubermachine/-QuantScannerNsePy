# QuantScannerNsePy

Nifty 50 stock screening, sector rotation (RRG), and portfolio backtesting for the Indian market. Built with Streamlit, DuckDB, and NumPy.

## Features

- **Scan Dashboard** — Screen Nifty 500 stocks across multiple strategies (RSI, ADX, EMA crossover, Bollinger, etc.) with score breakdowns
- **Sector Rotation** — RRG (Relative Rotation Graph) bubble chart with quadrant analysis, per-sector unique colors, and rotation backtesting vs. Nifty 50
- **Daily Scanner** — Scan all stocks for 4 mechanical daily-chart strategies: JNSAR (Stop & Reverse), J10SAR (EMA10 envelope), MA Crossover (8/21 with 144 EMA bias), LRHR (61.8% retracement). Results filterable by direction (LONG/SHORT) with signal counts per strategy
- **Portfolio Backtesting** — Single-strategy and multi-strategy comparison backtests with equity curves, trade logs, and performance metrics
- **Chart Analysis** — Interactive candlestick charts with indicator overlays (EMA, JNSAR, MACD, Fibonacci retracements)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app starts in headless mode automatically (configured in `.streamlit/config.toml`).

## Data Sync

Data is fetched from Yahoo Finance on-demand within the app:

- **Scan page** — Click "Sync Yahoo Finance Data" to download daily bars for all ~867 Nifty 500 tickers (takes 5–10 minutes)
- **Sector page** — Sector-level data has its own sync button
- Data is stored locally in a DuckDB database (`database/quantscanner.duckdb`)

### Cold start

On first clone, the repo includes pre-exported **Parquet** files (`database/*.parquet`).
When the app runs and finds no DuckDB file, it automatically creates one from the
Parquet data — no immediate sync needed. Sync any time to get fresh data.

> If the app loads slowly on first run, use the sync buttons in each page to populate the database.

## Project Structure

```
app.py                  # Entry point — sidebar nav, page routing
core.py                 # Business logic — scanning, backtesting, sector rotation, data sync
indicators.py           # 30+ NumPy indicator functions (EMA, RSI, ADX, MACD, etc.)
pages_app/
  scan.py               # Scan Dashboard
  sector.py             # Sector Rotation (RRG)
  dailyscanner.py        # Daily Scanner (4 daily-chart strategies)
  backtest.py           # Portfolio Simulation
  chart.py              # Chart Analysis
database/               # Parquet files (tracked in git) + DuckDB (auto-generated)
.streamlit/
  config.toml           # Streamlit config (headless mode, no usage stats)
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Key libraries: Streamlit, DuckDB, NumPy, Pandas (display only), Plotly, yfinance.

## License

MIT
