"""Chart Analysis — ticker selector + candlestick chart with indicator overlays."""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import get_stock_chart, _conn


def _get_all_tickers() -> list:
    con = _conn()
    rows = con.execute("SELECT Ticker FROM StockMetadatas ORDER BY Ticker").fetchall()
    con.close()
    return [r[0].replace(".NS", "") for r in rows]


def show():
    st.title("📈 Chart Analysis")

    tickers = _get_all_tickers()
    ticker = st.selectbox("Select Ticker", tickers, index=tickers.index("RELIANCE") if "RELIANCE" in tickers else 0)
    limit = st.slider("Bars to show", 50, 500, 250, 50)

    # Indicator toggles
    col1, col2, col3, col4 = st.columns(4)
    show_ema = col1.checkbox("EMA 8/21", True)
    show_jnsar = col2.checkbox("JNSAR", True)
    show_fib = col3.checkbox("Fib 618", True)
    show_macd = col4.checkbox("MACD", True)

    with st.spinner(f"Loading chart data for {ticker}..."):
        data = get_stock_chart(ticker, limit)

    if "error" in data:
        st.error(data["error"])
        return

    candles = data["candles"]
    if not candles:
        st.warning("No chart data available")
        return

    n_indicators = 1 if show_macd else 0
    fig = make_subplots(
        rows=1 + n_indicators, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7] + [0.3] * n_indicators,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=[c["date"] for c in candles],
        open=[c["open"] for c in candles],
        high=[c["high"] for c in candles],
        low=[c["low"] for c in candles],
        close=[c["close"] for c in candles],
        name=ticker,
    ), row=1, col=1)

    # EMAs
    if show_ema:
        ema8_vals = [c["ema8"] for c in candles]
        ema21_vals = [c["ema21"] for c in candles]
        fig.add_trace(go.Scatter(x=[c["date"] for c in candles], y=ema8_vals,
                                 mode="lines", name="EMA 8", line=dict(color="#636efa", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[c["date"] for c in candles], y=ema21_vals,
                                 mode="lines", name="EMA 21", line=dict(color="#ef553b", width=1)), row=1, col=1)

    # JNSAR
    if show_jnsar:
        jnsar_vals = [c["jnsar"] for c in candles]
        fig.add_trace(go.Scatter(x=[c["date"] for c in candles], y=jnsar_vals,
                                 mode="lines", name="JNSAR", line=dict(color="#ffa15a", width=1, dash="dot")), row=1, col=1)

    # Fib 618
    if show_fib:
        fib_val = candles[-1]["fib618"]
        if fib_val:
            fig.add_hline(y=fib_val, line=dict(color="#00cc96", dash="dash", width=1),
                          annotation_text=f"Fib 618: {fib_val:.2f}", row=1, col=1)

    # MACD
    if show_macd:
        dates = [c["date"] for c in candles]
        macd_l = [c["macd_line"] for c in candles]
        macd_s = [c["macd_signal"] for c in candles]
        macd_h = [c["macd_histogram"] for c in candles]
        colors = ["#00cc96" if h >= 0 else "#ef553b" for h in macd_h]
        fig.add_trace(go.Bar(x=dates, y=macd_h, name="MACD Hist", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=macd_l, mode="lines", name="MACD Line",
                                 line=dict(color="#636efa", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=macd_s, mode="lines", name="Signal",
                                 line=dict(color="#ef553b", width=1)), row=2, col=1)

    fig.update_layout(
        height=700 if show_macd else 500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_macd:
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Quick stats
    last = candles[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Close", last["close"])
    col2.metric("High", last["high"])
    col3.metric("Low", last["low"])
    col4.metric("Volume", f"{last['volume']:,}")
    col5.metric("MACD Hist", f"{last.get('macd_histogram', 0):.2f}")
