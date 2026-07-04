"""Sector Rotation — RRG bubble chart + quadrant tables + rotation signals + backtest."""
import streamlit as st
import plotly.graph_objects as go
import sys, os, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import get_sector_rotation, run_rotation_backtest, sync_sector_data, export_sector_data_csv


def show():
    st.title("🔄 Sector Rotation — RRG")

    # Sync button in sidebar
    with st.sidebar:
        st.markdown("### 📥 Sector Index Data")
        st.caption("Downloads 5 years of daily data for all sector indices from Yahoo Finance")
        if st.button("Sync Sector Indices", type="secondary", use_container_width=True,
                     help="Downloads ~5yr daily data for 16 sector indices"):
            with st.spinner("Syncing sector indices from Yahoo Finance... (takes ~1-2 min)"):
                result = sync_sector_data("5y")
            if result["status"] == "completed":
                st.success(f"✅ Synced {result['synced']}/{result['total']} indices")
                if result["errors"]:
                    with st.expander(f"⚠️ {len(result['errors'])} errors"):
                        for e in result["errors"][:10]:
                            st.code(e)
            else:
                st.error("Sync failed")
            st.rerun()

        st.divider()
        if st.button("📥 Export CSV", type="secondary", use_container_width=True,
                     help="Download all sector index data as CSV"):
            csv = export_sector_data_csv()
            if csv:
                st.download_button("⬇️ Download", data=csv, file_name="sector_indices.csv",
                                   mime="text/csv", use_container_width=True)
            else:
                st.warning("No sector data to export")

    with st.spinner("Loading sector rotation data..."):
        data = get_sector_rotation()

    if not data["sectors"]:
        st.warning("No sector data available. Click **Sync Sector Indices** in the sidebar.")
        return

    if data["rotation_signal"]:
        st.info("🔄 **Rotation Signal Detected** — Capital is rotating between sectors!")
    else:
        st.success("✅ No rotation signal — market trending normally.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Leading", len(data["leading"]), help="RS Ratio > 0, RS Momentum > 0")
    col2.metric("Improving", len(data["improving"]), help="RS Ratio < 0, RS Momentum > 0")
    col3.metric("Weakening", len(data["weakening"]), help="RS Ratio > 0, RS Momentum < 0")
    col4.metric("Lagging", len(data["lagging"]), help="RS Ratio < 0, RS Momentum < 0")

    # RRG Chart
    fig = go.Figure()
    colors = {"Leading": "#00cc96", "Improving": "#636efa", "Weakening": "#ef553b", "Lagging": "#ab63fa"}

    for s in data["sectors"]:
        # Historical trail
        trail = s.get("trail", [])
        if len(trail) >= 2:
            tx = [p["rs_ratio"] for p in trail]
            ty = [p["rs_momentum"] for p in trail]
            fig.add_trace(go.Scatter(
                x=tx, y=ty,
                mode="lines+markers",
                marker=dict(size=4, color=colors.get(s["quadrant"], "#888"), opacity=0.5),
                line=dict(width=1.5, color=colors.get(s["quadrant"], "#888"), shape="spline"),
                name=f"{s['name']} Trail",
                showlegend=False,
                hoverinfo="skip",
            ))
        # Current dot
        fig.add_trace(go.Scatter(
            x=[s["rs_ratio"]], y=[s["rs_momentum"]],
            mode="markers+text",
            text=s["name"],
            textposition="top center",
            marker=dict(size=12, color=colors.get(s["quadrant"], "#888")),
            name=s["name"],
            hovertemplate=f"<b>{s['name']}</b><br>RS-Ratio: {s['rs_ratio']:.2f}<br>RS-Momentum: {s['rs_momentum']:.2f}<br>Quadrant: {s['quadrant']}<br>Trail: {len(trail)} weeks",
        ))

    fig.add_hline(y=0, line_color="gray", line_dash="dash", opacity=0.5)
    fig.add_vline(x=0, line_color="gray", line_dash="dash", opacity=0.5)
    fig.update_layout(
        height=600,
        xaxis_title="RS-Ratio (z-score)",
        yaxis_title="RS-Momentum (z-score)",
        showlegend=False,
        annotations=[
            dict(x=2, y=2.5, text="Leading", showarrow=False, font=dict(color="#00cc96", size=16)),
            dict(x=-2, y=2.5, text="Improving", showarrow=False, font=dict(color="#636efa", size=16)),
            dict(x=2, y=-2.5, text="Weakening", showarrow=False, font=dict(color="#ef553b", size=16)),
            dict(x=-2, y=-2.5, text="Lagging", showarrow=False, font=dict(color="#ab63fa", size=16)),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quadrant tables
    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Leading", "📈 Improving", "📉 Weakening", "⚪ Lagging"])
    for tab, qname, qlist in [
        (tab1, "Leading", data["leading"]),
        (tab2, "Improving", data["improving"]),
        (tab3, "Weakening", data["weakening"]),
        (tab4, "Lagging", data["lagging"]),
    ]:
        with tab:
            if qlist:
                df = pd.DataFrame(qlist)
                st.dataframe(df[["ticker", "name", "rs_ratio", "rs_momentum", "price"]],
                             use_container_width=True,
                             column_config={
                                 "rs_ratio": st.column_config.NumberColumn("RS-Ratio", format="%.2f"),
                                 "rs_momentum": st.column_config.NumberColumn("RS-Momentum", format="%.2f"),
                                 "price": st.column_config.NumberColumn(format="%.2f"),
                             })
            else:
                st.caption(f"No sectors in {qname} quadrant")

    # Rotation Backtest
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📈 Rotation Strategy Backtest")
    with col2:
        rot_capital = st.number_input("Capital", min_value=10000, value=100000, step=10000, key="rot_cap",
                                       label_visibility="collapsed")
    run_rot = st.button("Run Rotation Backtest", type="primary", use_container_width=False)

    if run_rot:
        with st.spinner("Running sector rotation backtest..."):
            result = run_rotation_backtest({"starting_capital": rot_capital})

        if "error" in result:
            st.error(result["error"])
            return

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{result['return_pct']:.2f}%")
        col2.metric("Nifty Return", f"{result['nifty_return']:.2f}%")
        col3.metric("Max DD", f"{result['max_drawdown_pct']:.1f}%")
        col4.metric("Win Rate", f"{result['win_rate']:.1f}%")
        col5.metric("Trades", result["total_trades"])

        # Equity curve
        eq = result["equity_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[e["date"] for e in eq],
            y=[e["balance"] for e in eq],
            mode="lines", name="Rotation Strategy", line=dict(color="#00cc96"),
        ))
        n_eq = result.get("nifty_curve", [])
        if n_eq:
            base = rot_capital
            n_vals = [base * (1 + v / 100) for v in n_eq]
            fig.add_trace(go.Scatter(
                x=[e["date"] for e in eq],
                y=n_vals,
                mode="lines", name="Nifty 50 (Buy & Hold)", line=dict(color="#636efa", dash="dot"),
            ))
        fig.update_layout(height=400, yaxis_title="Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)

        # Trade log
        with st.expander("📋 Trade Log", expanded=False):
            if result["trades"]:
                df = pd.DataFrame(result["trades"])
                st.dataframe(df, use_container_width=True,
                             column_config={
                                 "return_pct": st.column_config.NumberColumn(format="%.2f%%"),
                                 "entry_price": st.column_config.NumberColumn(format="%.2f"),
                                 "exit_price": st.column_config.NumberColumn(format="%.2f"),
                             })
            else:
                st.caption("No trades generated")
