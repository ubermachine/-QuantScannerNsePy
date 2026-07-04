"""Backtest — portfolio simulation config sidebar + equity curve + trade log.
Supports single strategy run and multi-strategy comparison."""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import run_backtest, run_backtest_multi, get_strategies


def show():
    st.title("📊 Backtest — Portfolio Simulation")

    strategies = get_strategies()
    with st.sidebar:
        st.markdown("### ⚙️ Backtest Config")
        compare_all = st.checkbox("Compare All Strategies", value=False,
                                  help="Run all 5 strategies and overlay equity curves")
        if not compare_all:
            strategy = st.selectbox("Strategy", [s for s in strategies if s != "All"], index=2)
        capital = st.number_input("Starting Capital", min_value=10000, value=100000, step=10000)
        max_pos = st.slider("Max Positions", 1, 20, 5)
        risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0, 0.5)
        tx_cost = st.slider("Transaction Cost %", 0.0, 1.0, 0.1, 0.05)
        slippage = st.slider("Slippage %", 0.0, 1.0, 0.1, 0.05)
        run = st.button("Run Backtest", type="primary", use_container_width=True)

    if run:
        base_params = {
            "starting_capital": capital,
            "max_positions": max_pos,
            "risk_per_trade_pct": risk_pct,
            "transaction_cost_pct": tx_cost,
            "slippage_pct": slippage,
        }

        if compare_all:
            with st.spinner("Running all 5 strategies..."):
                multi = run_backtest_multi({**base_params})

            if not multi["strategies"]:
                st.error("No strategy results returned")
                return

            # Summary comparison table
            st.subheader("📋 Strategy Comparison")
            cols = st.columns(len(multi["strategies"]) + 1)
            cols[0].metric("Metric", "Nifty 50 B&H")
            for i, sline in enumerate(multi["strategies"]):
                cols[i + 1].metric(sline["strategyName"],
                                    f'{sline["summary"]["return_pct"]:.1f}%')

            # Comparison table
            import pandas as pd
            rows = []
            for sline in multi["strategies"]:
                s = sline["summary"]
                rows.append({
                    "Strategy": sline["strategyName"],
                    "Return %": s["return_pct"],
                    "Sharpe": s["sharpe_ratio"],
                    "Max DD %": s["max_drawdown_pct"],
                    "Win Rate %": s["win_rate"],
                    "Trades": s["total_trades"],
                    "End Capital ₹": s["ending_capital"],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True,
                         column_config={
                             "Return %": st.column_config.NumberColumn(format="%.2f%%"),
                             "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                             "Max DD %": st.column_config.NumberColumn(format="%.1f%%"),
                             "Win Rate %": st.column_config.NumberColumn(format="%.1f%%"),
                             "End Capital ₹": st.column_config.NumberColumn(format="₹%.0f"),
                         })

            # Overlay equity curves
            fig = go.Figure()
            colors = ["#00cc96", "#636efa", "#ef553b", "#ab63fa", "#ffa15a", "#ff6692"]
            for i, sline in enumerate(multi["strategies"]):
                eq = sline["equity_curve"]
                if eq:
                    fig.add_trace(go.Scatter(
                        x=[e["date"] for e in eq],
                        y=[e["balance"] for e in eq],
                        mode="lines", name=sline["strategyName"],
                        line=dict(color=colors[i % len(colors)]),
                    ))

            # Nifty buy & hold curve
            nifty_line = multi["strategies"][0] if multi["strategies"] else None
            if nifty_line and nifty_line.get("nifty_curve"):
                n_eq = nifty_line["nifty_curve"]
                n_vals = [capital * (1 + v / 100) for v in n_eq]
                first_eq = multi["strategies"][0]["equity_curve"]
                fig.add_trace(go.Scatter(
                    x=[e["date"] for e in first_eq],
                    y=n_vals,
                    mode="lines", name="Nifty 50 (Buy & Hold)",
                    line=dict(color="gray", dash="dot"),
                ))

            fig.update_layout(height=450, yaxis_title="Portfolio Value",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        else:
            with st.spinner("Running portfolio simulation..."):
                params = {**base_params, "strategy": strategy}
                result = run_backtest(params)

            if "error" in result:
                st.error(result["error"])
                return

            # Summary
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total Return", f"{result['return_pct']:.2f}%")
            col2.metric("Nifty Return", f"{result['nifty_return']:.2f}%")
            col3.metric("Sharpe", f"{result['sharpe_ratio']:.2f}")
            col4.metric("Max DD", f"{result['max_drawdown_pct']:.1f}%")
            col5.metric("Win Rate", f"{result['win_rate']:.1f}%")
            col6.metric("Trades", result["total_trades"])

            # Equity curve
            eq = result["equity_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[e["date"] for e in eq],
                y=[e["balance"] for e in eq],
                mode="lines", name="Portfolio", line=dict(color="#00cc96"),
            ))
            n_eq = result.get("nifty_curve", [])
            if n_eq:
                base = capital
                n_vals = [base * (1 + v / 100) for v in n_eq]
                fig.add_trace(go.Scatter(
                    x=[e["date"] for e in eq],
                    y=n_vals,
                    mode="lines", name="Nifty 50 (Buy & Hold)", line=dict(color="#636efa", dash="dot"),
                ))
            fig.update_layout(height=400, yaxis_title="Portfolio Value")
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown
            st.subheader("📉 Drawdown")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[e["date"] for e in eq],
                y=[-e["drawdown_pct"] for e in eq],
                mode="lines", fill="tozeroy",
                line=dict(color="#ef553b"),
            ))
            fig2.update_layout(height=200, yaxis_title="Drawdown %")
            st.plotly_chart(fig2, use_container_width=True)

            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Starting Capital", f"₹{result['starting_capital']:,.0f}")
            col2.metric("Ending Capital", f"₹{result['ending_capital']:,.0f}")
            col3.metric("Total Profit", f"₹{result['total_profit']:,.0f}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Win", f"₹{result['avg_win']:,.0f}")
            col2.metric("Avg Loss", f"₹{result['avg_loss']:,.0f}")
            col3.metric("Profit Factor", f"{result['profit_factor']:.2f}")
            col4.metric("Wins/Losses", f"{result['winning_trades']}/{result['losing_trades']}")

            # Trade log
            with st.expander("📋 Trade Log", expanded=False):
                if result["trades"]:
                    import pandas as pd
                    df = pd.DataFrame(result["trades"])
                    st.dataframe(df, use_container_width=True,
                                 column_config={
                                     "profit": st.column_config.NumberColumn(format="₹%.0f"),
                                     "profit_pct": st.column_config.NumberColumn(format="%.2f%%"),
                                     "entry_price": st.column_config.NumberColumn(format="%.2f"),
                                     "exit_price": st.column_config.NumberColumn(format="%.2f"),
                                 })
                else:
                    st.caption("No trades generated")
    else:
        st.info("Configure parameters in the sidebar and click **Run Backtest**.")
