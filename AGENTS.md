# AGENTS.md — QuantScannerPy

> Nifty 50 stock screening, sector rotation (RRG), and portfolio backtesting for the Indian market. Streamlit UI with DuckDB + NumPy backend.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app starts in headless mode (`.streamlit/config.toml` sets `headless = true`).

### Dependencies

- `streamlit>=1.35` — UI framework
- `duckdb>=1.0` — embedded OLAP database
- `pandas>=2.0` — display / DataFrame rendering only (analysis uses NumPy)
- `numpy>=1.24` — all computation
- `plotly>=5.18` — charts (candlestick, RRG, equity curves)
- `yfinance>=0.2.28` — data ingestion from Yahoo Finance

### Commands

| Action | Command |
|---|---|
| Run app | `streamlit run app.py` |
| Install deps | `pip install -r requirements.txt` |
| Lint / format | Not configured — no formatter or linter in the project |

There are **no tests**, no test runner, and no CI/CD pipeline.

## Architecture

### Module tree

```
app.py                     # Streamlit entry point: set_page_config + sidebar nav + page routing
core.py                    # Pure functions returning plain dicts — scan engine, backtests, data sync (~1210
lines)
indicators.py              # Pure NumPy indicator functions — no classes, no state (~426 lines)
pages_app/
  scan.py                  # Scan Dashboard — market regime badge + results table + score breakdown
  sector.py                # Sector Rotation — RRG bubble chart + quadrant tables + rotation backtest
  backtest.py              # Portfolio Simulation — strategy backtest config + equity curve + trade log
  chart.py                 # Chart Analysis — ticker candlestick with indicator overlays (EMA, JNSAR, MACD,
Fib)
database/
  quantscanner.duckdb      # DuckDB database file (pre-populated)
.streamlit/
  config.toml              # Streamlit config (headless, usage stats off)
```

### Data flow

```
Yahoo Finance (yfinance)
    ↓  sync_yahoo_data() / sync_sector_data()
DuckDB tables: DailyBars, WeeklyBars, StockMetadatas, SectorDailyBars
    ↓  core.py: _conn() → _closes_hlv() / _batch_load_all()
NumPy arrays (closes, highs, lows, volumes, dates)
    ↓  indicators.py: pure functions
Scored results, backtest equity curves, RRG quadrant data
    ↓  pages_app/*.py: Streamlit + Plotly
Streamlit UI
```

### Key design decisions

- **No classes, no ORM.** Every function returns a plain dict, making it trivially exposeable as an LLM tool.
- **Batch DB loading.** `_batch_load_all()` loads all ticker OHLCV in a single SQL query instead of N separate queries (~1700→1).
- **`_last` function variants.** Every indicator has a `*_last()` variant that computes only the final value (avoids full array allocation) — used in the scanner for per-ticker speed.
- **NumPy over pandas for computation.** Pandas is used only for `st.dataframe` rendering and `yfinance` output. All indicator math and backtest logic uses raw NumPy arrays.
- **`safe_round()`** utility wraps `round()` with NaN/inf protection.

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Entry point, sidebar nav routing. 4 pages: Scan, Sector, Backtest, Chart |
| `core.py` | All business logic — scanning, backtesting, sector rotation, data sync. Largest file. |
| `indicators.py` | 30+ indicator functions: EMA, RSI, ADX, MACD, Bollinger, Keltner, CMF, OBV, ATR, JNSAR, Chandelier, etc. |
| `pages_app/scan.py` | Scan results with strategy filter, score breakdown expander |
| `pages_app/sector.py` | RRG bubble chart, quadrant tables, rotation backtest with Nifty comparison |
| `pages_app/backtest.py` | Single-strategy and multi-strategy comparison backtests |
| `pages_app/chart.py` | Candlestick chart with EMA/JNSAR/Fib/MACD overlays |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit config (headless, no usage stats) |

## Coding Conventions

### Naming

- **Functions**: `snake_case`, descriptive verb prefix: `get_*`, `run_*`, `sync_*`, `_closes_hlv`, `_batch_load_all`.
- **Private helpers**: prefixed with `_` (`_conn()`, `_load_nifty()`).
- **Indicator functions**: `ema()`, `rsi_last()`, `multi_ema_last()`, `adx()` — matching TradingView / Pine Script conventions.
- **Page modules**: each exports a single `show()` function called by `app.py`.
- **Constants**: `UPPER_CASE` (`DB`, `NIFTY_TICKER`).

### Patterns

- **Pure functions only.** No classes, no global state, no ORM. Functions take arrays / params and return dicts.
- **DuckDB connections** are created and closed within each function (pattern: `con = _conn()` → use → `con.close()`). Not pooled.
- **`sys.path.insert(0, ...)`** in every page file to import from parent directory.
- **Exception handling** is minimal — `try/except` used only in the Yahoo sync loops and `safe_round`.

### Important gotchas

- **`yfinance` MultiIndex columns**: `sync_yahoo_data()` and `sync_sector_data()` both handle the `yfinance` MultiIndex bug with `if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)`.
- **Rate limiting**: `sync_yahoo_data()` sleeps `0.3s` between tickers. Full sync of ~867 tickers takes 5–10 minutes.
- **DuckDB date handling**: dates from DuckDB are `datetime` objects; `ytd_vwap()` checks `hasattr(d, 'year')` for safety.
- **No `.NS` suffix in ticker symbols for the chart page**: `_get_all_tickers()` strips `.NS` from tickers for display, but the database stores them with `.NS` (Indian Yahoo Finance convention).

## Navigation assistance

| You want... | Look in... |
|---|---|
| Add/edit a screening strategy | `core.py`: `get_stock_scan()` strategy matching section (~line 253-290), update `get_strategies()` list |
| Add/edit a technical indicator | `indicators.py` |
| Add a new Streamlit page | Create `pages_app/my_page.py` with a `show()` function, add route in `app.py` |
| Tweak backtest logic | `core.py`: `run_backtest()` (~line 600-900) and `run_rotation_backtest()` (~line 472-600) |
| Add a new data source | `core.py`: follow `sync_yahoo_data()` / `sync_sector_data()` pattern |
| Modify the database schema | `core.py`: `sync_yahoo_data()` has `CREATE TABLE IF NOT EXISTS` statements |
| Fix a chart layout issue | `pages_app/chart.py` (Plotly config) or `pages_app/sector.py` (RRG chart) |

## Tips for AI Agents

1. **`core.py` is the heart of the project** (~1210 lines). Before editing, use `_batch_load_all` and `_closes_hlv` — they're the primary data loading paths.
2. **No ORM.** All SQL is raw DuckDB queries in string literals. Do not look for SQLAlchemy or Django ORM patterns.
3. **Backtest simulation is monthly-rebalance logic** inside `run_backtest()` — about 300 lines of tick-level position management with entry/exit/stop/trim logic. Read it carefully before modifying.
4. **The scan page calls sync on-demand** — clicking "Sync Yahoo Finance Data" runs `sync_yahoo_data()` which downloads all 867 tickers. The Sector page has its own `sync_sector_data()`.
5. **No tests exist.** Any verification must be done by running the app and checking the UI, or by calling functions from a Python REPL.
6. **Streamlit rerun cycle**: after sync completes, `st.rerun()` is called. Be aware that `with st.spinner` blocks are synchronous, not async.
7. **Plotly charts use `plotly_dark` template** — any new chart should follow this.

## Agent Coding Rules

These rules must be followed for every code change.

---

### 1. Think Before Coding
Reason explicitly. Never silently assume.

- **Clarify ambiguity** — If something is unclear, stop and ask the user for clarification. Do not guess.
- **Surface tradeoffs** — When multiple valid approaches exist, briefly present the options and let the user decide (unless one is clearly better).
- **Name confusion** — If you're unsure what to do, state what's confusing and ask for direction.

---

### 2. Simplicity First
Ship the minimal solution that fulfills the request.

- **No extras** — Don't add features, configurability, or "flexibility" that wasn't asked for.
- **No speculative code** — Skip error handling for impossible scenarios; don't abstract single-use code.
- **Prefer clarity** — If the code could be shorter and clearer, make it so. Ask: *Would a senior engineer call this overcomplicated?* If yes, simplify.

---

### 3. Surgical Changes
Edit only what is necessary to meet the goal.

- **Don't touch adjacent code** — No reformatting, refactoring, or "fixing" things that aren't broken.
- **Match existing style** — Even if you'd do it differently, follow the patterns you see.
- **Clean up your own mess** — Remove imports, variables, or functions that *your* changes made unused.
- **Leave pre-existing dead code** — Do not delete anything unrelated unless explicitly asked. If you notice an issue, mention it in a comment, but don't fix it.

---

### 4. Goal-Driven Execution
Define success criteria, then loop until verified.

- **Make it testable** — Transform vague instructions into concrete checks:
  - "Add validation" → "Write tests for invalid inputs, then make them pass."
  - "Fix the bug" → "Write a test that reproduces it, then make it pass."
- **Plan with checkpoints** — For multi-step tasks, outline:
  1. Step → verify: [specific check]
  2. Step → verify: [specific check]
- **Close the loop** — After the change, run the relevant tests and confirm the original problem is truly solved. If not, iterate.

---

---

### 5. Vibe Coder Mode (token‑minimal Python)

When the user's request leans toward "just code" or you're in a fast‑iteration flow, switch to this mode.

**Output rules**
- Code only, unless I explicitly ask a question.
- No greetings, no "Sure!", no "Here is…", no docstrings, no type hints.
- Edits as line‑based shorthand: `+L`, `*L`, `-L`, or unified diff.
- Use Python's terse features: comprehensions, lambdas, walrus, `import as`.

**Python style**
- `from os import getenv`, not `import os`
- `d = {k:v for k,v in data if v}` not multi‑step loops
- `lambda` for tiny functions
- `try: … except: pass` is fine
- No `if __name__ == "__main__"` unless requested
- Short variable names when meaning is clear (`f` for file, `r` for response, `d` for data)
- **Density rule**: write Python like you're typing on a 10‑key phone.

**Self-review (silent)**
1. Is my answer only code? If not, delete the words.
2. Did I use shortest Python idiom? (map/list comp, implicit bool, etc.)
3. Edits: did I only show the diff?
4. No docstrings, no type hints, no comments unless clarifying tricky logic.

---

**When in doubt, ask.** It's faster to clarify once than to redo incorrect work.
