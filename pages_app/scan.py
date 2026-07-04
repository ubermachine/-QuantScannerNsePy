"""Scan Dashboard — market regime badge + results table + score breakdown."""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import get_market_regime, get_stock_scan, get_strategies, sync_yahoo_data


def show():
    st.title("🔍 Scan Dashboard")

    # Sync trigger in sidebar
    with st.sidebar:
        st.markdown("### 🔄 Data Sync")
        st.caption(f"DB: `database/quantscanner.duckdb`")
        if st.button("Sync Yahoo Finance Data", type="secondary", use_container_width=True,
                     help="Downloads ~1yr of daily data for all 867 stocks from Yahoo Finance"):
            with st.spinner("Syncing all stocks from Yahoo Finance... (this takes ~5-10 min)"):
                result = sync_yahoo_data()
            if result["status"] == "completed":
                st.success(f"✅ Synced {result['synced']}/{result['total']} tickers")
                if result["errors"]:
                    with st.expander(f"⚠️ {len(result['errors'])} errors"):
                        for e in result["errors"][:20]:
                            st.code(e)
            else:
                st.error("Sync failed")
            st.rerun()

    with st.spinner("Loading market regime..."):
        regime = get_market_regime()

    col1, col2, col3 = st.columns(3)
    r = regime["market_regime"]
    badge = "🟢 BULLISH" if r == "BULLISH" else ("🔴 BEARISH" if r == "BEARISH" else "⚪ UNKNOWN")
    col1.metric("Market Regime", badge)
    col2.metric("Nifty 50", regime["index_close"])
    col3.metric("200 EMA", regime["index_ema200"])

    strategies = get_strategies()
    selected = st.selectbox("Strategy Filter", strategies, index=0)

    if st.button("Run Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning 867 stocks..."):
            data = get_stock_scan(selected)

        if not data["results"]:
            st.warning("No results match the selected strategy.")
            return

        st.success(f"Found {len(data['results'])} matching stocks (scored {data['total_scored']} total)")

        # Build display table
        rows = []
        for r in data["results"]:
            rows.append({
                "Ticker": r["ticker"],
                "Sector": r["sector"],
                "Price": r["price"],
                "Score": r["score"],
                "Strategy": r["strategy"],
                "Conviction": r["conviction"],
                "RSI": r["rsi14"],
                "ADX": r["adx14"],
                "Z-Score": r["z_score"],
                "52W Disc%": r["discount_52w"],
                "Stop": r["stop_loss"],
                "Target": r["target1"],
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=500,
                     column_config={
                         "Score": st.column_config.NumberColumn(format="%d"),
                         "Price": st.column_config.NumberColumn(format="%.2f"),
                         "RSI": st.column_config.NumberColumn(format="%.1f"),
                         "ADX": st.column_config.NumberColumn(format="%.1f"),
                         "Z-Score": st.column_config.NumberColumn(format="%.2f"),
                         "52W Disc%": st.column_config.NumberColumn(format="%.1f%%"),
                         "Stop": st.column_config.NumberColumn(format="%.2f"),
                         "Target": st.column_config.NumberColumn(format="%.2f"),
                     })

        # Detail expander for first few tickers
        with st.expander("📊 Score Breakdown (Top 5)"):
            for r in data["results"][:5]:
                cols = st.columns(7)
                cols[0].metric("Trend", r["trend_score"], help="Max 20")
                cols[1].metric("RS", r["rs_score"], help="Max 20")
                cols[2].metric("Vol Acc", r["vol_acc_score"], help="Max 10")
                cols[3].metric("Vol Setup", r["vol_setup_score"], help="Max 10")
                cols[4].metric("Momentum", r["momentum_score"], help="Max 10")
                cols[5].metric("Institutional", r["inst_score"], help="Max 10")
                cols[6].metric("Total", r["score"])
                st.caption(f"**{r['ticker']}** — {r['strategy']} ({r['conviction']})")
                st.divider()
    else:
        st.info("Select a strategy and click **Run Scan** to begin.")
