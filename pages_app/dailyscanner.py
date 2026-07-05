"""Daily Scanner — 4 mechanical daily-chart strategies: JNSAR, J10SAR, MA Crossover, LRHR."""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import _batch_load_all, _load_nifty, _conn
from indicators import *
import numpy as np


def _scan_ticker(closes, highs, lows, vols, dates, opens=None):
    """Run all 4 daily strategies on one ticker. Returns dict of signals or None."""
    price = float(closes[-1])
    if price <= 0:
        return None

    # Common indicators
    ema8, ema10, ema21, ema50, ema200 = multi_ema_last(closes)
    jnsar_val = jnsar_last(closes, highs, lows)
    atr14 = float(atr(highs, lows, closes)[-1]) if len(closes) >= 14 else 0

    # --- Strategy 4: JNSAR Stop & Reverse ---
    jnsar_sig = "LONG" if price > jnsar_val else ("SHORT" if price < jnsar_val else "NONE")

    # Whipsaw filter: 0.3% buffer on first 2 days (approx — we use current bar)
    jnsar_filtered = jnsar_sig
    if jnsar_sig == "LONG" and price < jnsar_val * 1.003:
        jnsar_filtered = "NONE"
    elif jnsar_sig == "SHORT" and price > jnsar_val * 0.997:
        jnsar_filtered = "NONE"

    # --- Strategy 1: J10SAR (EMA10 + 2.5% envelope) ---
    mid, upper, lower = j10sar(closes)
    j10_sig = "LONG" if price > mid and lows[-1] <= lower else \
              ("SHORT" if price < mid and highs[-1] >= upper else "NONE")

    # --- Strategy 2: MA Crossover (8/21 with 144 EMA bias) ---
    ema144 = ema_last(closes, 144)
    ma_sig = "LONG" if ema8 > ema21 and price > ema144 else \
             ("SHORT" if ema8 < ema21 and price < ema144 else "NONE")

    # --- Strategy 3: LRHR (61.8% retracement) ---
    golden_zone, swh, swl = daily_swing_fib618(closes, highs, lows, lookback=60)
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    lrhr_long = lows[-1] <= golden_zone and closes[-1] > opens[-1]
    lrhr_short = highs[-1] >= golden_zone and closes[-1] < opens[-1]
    lrhr_sig = "LONG" if lrhr_long else ("SHORT" if lrhr_short else "NONE")

    # Stop loss calculations
    if len(closes) >= 20:
        sl_20_low = float(np.min(lows[-20:]))
        sl_20_high = float(np.max(highs[-20:]))
    else:
        sl_20_low = float(np.min(lows))
        sl_20_high = float(np.max(highs))

    return {
        "price": price, "atr14": atr14,
        "jnsar": jnsar_val, "jnsar_sig": jnsar_filtered,
        "j10_sig": j10_sig, "j10_mid": mid, "j10_upper": upper, "j10_lower": lower,
        "ma_sig": ma_sig, "ema8": ema8, "ema21": ema21, "ema144": ema144,
        "lrhr_sig": lrhr_sig, "golden_zone": golden_zone, "swing_h": swh, "swing_l": swl,
        "sl_long": sl_20_low - 1, "sl_short": sl_20_high + 1,
    }


def show():
    st.title("📅 Daily Scanner")
    st.caption("4 mechanical daily-chart strategies: JNSAR, J10SAR, MA Crossover (8/21×144), LRHR 61.8%")

    with st.sidebar:
        st.markdown("### 🔄 Data")
        st.caption("Uses daily bars from DuckDB")

    strategy_tab = st.radio("Strategy", ["JNSAR (S&R)", "J10SAR", "MA Crossover", "LRHR (61.8%)"], horizontal=True)
    dir_filter = st.radio("Direction", ["All", "LONG", "SHORT"], horizontal=True, label_visibility="collapsed")

    if st.button("Run Daily Scan", type="primary", use_container_width=True):
        with st.spinner("Loading daily data for all stocks..."):
            all_data = _batch_load_all(min_bars=200, lookback=250)
            if not all_data:
                st.error("No data available. Run sync from Scan Dashboard first.")
                return

        con = _conn()
        stocks = con.execute("SELECT Ticker, Sector FROM StockMetadatas").fetchall()
        sectors = {r[0]: r[1] or "NSE" for r in stocks}
        con.close()

        rows = []
        for ticker, (closes, highs, lows, vols, dates) in all_data.items():
            opens = np.concatenate([[closes[0]], closes[:-1]])
            result = _scan_ticker(closes, highs, lows, vols, dates)
            if result is None:
                continue
            sector = sectors.get(ticker, "NSE")

            sig_map = {"JNSAR (S&R)": result["jnsar_sig"], "J10SAR": result["j10_sig"],
                        "MA Crossover": result["ma_sig"], "LRHR (61.8%)": result["lrhr_sig"]}
            sig = sig_map.get(strategy_tab, "NONE")
            if sig == "NONE":
                continue

            row = {"Ticker": ticker.replace(".NS", ""), "Sector": sector, "Price": result["price"],
                   "Signal": sig, "SL": result["sl_long"] if sig == "LONG" else result["sl_short"],
                   "A TR": result["atr14"]}

            if strategy_tab == "JNSAR (S&R)":
                row["JNSAR"] = result["jnsar"]
            elif strategy_tab == "J10SAR":
                row["Mid"], row["Upper"], row["Lower"] = result["j10_mid"], result["j10_upper"], result["j10_lower"]
            elif strategy_tab == "MA Crossover":
                row["EMA8"], row["EMA21"], row["EMA144"] = result["ema8"], result["ema21"], result["ema144"]
            elif strategy_tab == "LRHR (61.8%)":
                row["Golden Z"], row["Swing H"], row["Swing L"] = result["golden_zone"], result["swing_h"], result["swing_l"]
            rows.append(row)

        if dir_filter != "All":
            rows = [r for r in rows if r["Signal"] == dir_filter]

        if not rows:
            st.info(f"No {strategy_tab} signals currently. Try another strategy tab.")
            return

        df = pd.DataFrame(rows).sort_values("Ticker")
        st.success(f"{len(df)} {strategy_tab} signals found")

        col_config = {"Price": st.column_config.NumberColumn(format="%.2f"),
                      "SL": st.column_config.NumberColumn(format="%.2f"),
                      "A TR": st.column_config.NumberColumn(format="%.2f")}
        for extra in ["JNSAR", "Mid", "Upper", "Lower", "EMA8", "EMA21", "EMA144",
                       "Golden Z", "Swing H", "Swing L"]:
            if extra in df.columns:
                col_config[extra] = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(df, use_container_width=True, height=500, column_config=col_config)

        # Summary counts
        st.divider()
        st.caption("Signal counts across all strategies:")
        c1, c2, c3, c4 = st.columns(4)
        for col, label, key in [(c1, "JNSAR", "jnsar_sig"), (c2, "J10SAR", "j10_sig"),
                                 (c3, "MA Xover", "ma_sig"), (c4, "LRHR", "lrhr_sig")]:
            cnt = sum(1 for _, (closes, highs, lows, vols, dates) in all_data.items()
                      if (r := _scan_ticker(closes, highs, lows, vols, dates)) and r[key] != "NONE")
            col.metric(label, cnt)
    else:
        st.info("Select a strategy and click **Run Daily Scan** to scan all stocks for signals.")
