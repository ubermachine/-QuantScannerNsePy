"""Daily Scanner — 4 mechanical daily-chart strategies: JNSAR, J10SAR, MA Crossover, LRHR."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import _batch_load_all, _conn, get_stock_chart
from indicators import *
import numpy as np


@st.dialog("📈 Chart")
def chart_dialog(ticker):
    st.caption(f"**{ticker}** — Candlestick with EMA 8/21, JNSAR, MACD")
    with st.spinner(f"Loading chart for {ticker}..."):
        chart_data = get_stock_chart(ticker, 100)

    if "error" in chart_data or not chart_data.get("candles"):
        st.error(chart_data.get("error", "No chart data available"))
    else:
        candles = chart_data["candles"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=[c["date"] for c in candles], open=[c["open"] for c in candles],
            high=[c["high"] for c in candles], low=[c["low"] for c in candles],
            close=[c["close"] for c in candles], name=ticker,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=[c["date"] for c in candles], y=[c["ema8"] for c in candles],
            mode="lines", name="EMA 8", line=dict(color="#636efa", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[c["date"] for c in candles], y=[c["ema21"] for c in candles],
            mode="lines", name="EMA 21", line=dict(color="#ef553b", width=1)), row=1, col=1)

        jnsar_vals = [c["jnsar"] for c in candles]
        fig.add_trace(go.Scatter(x=[c["date"] for c in candles], y=jnsar_vals,
            mode="lines", name="JNSAR", line=dict(color="#ffa15a", width=1, dash="dot")), row=1, col=1)

        dates_m = [c["date"] for c in candles]
        macd_l = [c["macd_line"] for c in candles]
        macd_s = [c["macd_signal"] for c in candles]
        macd_h = [c["macd_histogram"] for c in candles]
        colors = ["#00cc96" if h >= 0 else "#ef553b" for h in macd_h]
        fig.add_trace(go.Bar(x=dates_m, y=macd_h, name="MACD Hist", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates_m, y=macd_l, mode="lines", name="MACD Line",
            line=dict(color="#636efa", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates_m, y=macd_s, mode="lines", name="Signal",
            line=dict(color="#ef553b", width=1)), row=2, col=1)

        fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                          template="plotly_dark", hovermode="x unified")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        last = candles[-1]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Close", last["close"])
        mcol2.metric("High", last["high"])
        mcol3.metric("Low", last["low"])
        mcol4.metric("Volume", f"{last['volume']:,}")

    if st.button("✕ Close"):
        st.session_state.daily_chart_ticker = None
        st.rerun()


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
    st.session_state.setdefault("daily_chart_ticker", None)
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
                st.session_state.daily_results = None

        if all_data:
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
                       "ATR": result["atr14"]}

                if strategy_tab == "JNSAR (S&R)":
                    row["JNSAR"] = result["jnsar"]
                elif strategy_tab == "J10SAR":
                    row["Mid"], row["Upper"], row["Lower"] = result["j10_mid"], result["j10_upper"], result["j10_lower"]
                elif strategy_tab == "MA Crossover":
                    row["EMA8"], row["EMA21"], row["EMA144"] = result["ema8"], result["ema21"], result["ema144"]
                elif strategy_tab == "LRHR (61.8%)":
                    row["GoldenZ"], row["SwingH"], row["SwingL"] = result["golden_zone"], result["swing_h"], result["swing_l"]
                rows.append(row)

            if dir_filter != "All":
                rows = [r for r in rows if r["Signal"] == dir_filter]

            if not rows:
                st.info(f"No {strategy_tab} signals currently. Try another strategy tab.")
                st.session_state.daily_results = None
            else:
                st.success(f"{len(rows)} {strategy_tab} signals found")
                st.session_state.daily_results = rows
                st.session_state.daily_all_data = all_data
    else:
        st.info("Select a strategy and click **Run Daily Scan** to scan all stocks for signals.")

    # Results table — click any row to open chart popup
    if st.session_state.get("daily_results"):
        rows = st.session_state.daily_results
        extra_cols = [k for k in ["JNSAR", "Mid", "Upper", "Lower", "EMA8", "EMA21", "EMA144",
                                   "GoldenZ", "SwingH", "SwingL"] if k in rows[0]]

        df = pd.DataFrame(rows)
        col_config = {"Price": st.column_config.NumberColumn(format="%.2f"),
                      "SL": st.column_config.NumberColumn(format="%.2f"),
                      "ATR": st.column_config.NumberColumn(format="%.2f")}
        for k in extra_cols:
            col_config[k] = st.column_config.NumberColumn(format="%.2f")
        tbl_key = f"daily_tbl_{st.session_state.get('_dt', 0)}"
        sel = st.dataframe(df, key=tbl_key, use_container_width=True, height=500,
                           column_config=col_config,
                           on_select="rerun", selection_mode="single-row")
        if sel and hasattr(sel, 'selection') and sel.selection and sel.selection.rows:
            row_idx = sel.selection.rows[0]
            ticker = rows[row_idx]["Ticker"]
            st.session_state.daily_chart_ticker = ticker
            st.session_state._dt = st.session_state.get("_dt", 0) + 1
            st.rerun()

        # Summary counts
        all_data = st.session_state.get("daily_all_data")
        if all_data:
            st.divider()
            st.caption("Signal counts across all strategies:")
            c1, c2, c3, c4 = st.columns(4)
            for col, label, key in [(c1, "JNSAR", "jnsar_sig"), (c2, "J10SAR", "j10_sig"),
                                     (c3, "MA Xover", "ma_sig"), (c4, "LRHR", "lrhr_sig")]:
                cnt = sum(1 for _, (closes, highs, lows, vols, dates) in all_data.items()
                          if (r := _scan_ticker(closes, highs, lows, vols, dates)) and r[key] != "NONE")
                col.metric(label, cnt)

    # Open chart popup when a ticker is selected
    if st.session_state.daily_chart_ticker:
        chart_dialog(st.session_state.daily_chart_ticker)
