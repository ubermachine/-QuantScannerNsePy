"""Sector Rotation — Premium RRG dashboard inspired by DEXT T3, with 100-based axes, dynamic checkboxes, trail length slider, direction arrows, and clockwise lifecycle guide."""
import streamlit as st
import plotly.graph_objects as go
import sys, os, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import get_sector_rotation, run_rotation_backtest, sync_sector_data, export_sector_data_csv


def show():
    st.subheader("🔄 Sector Rotation — RRG")

    # Dynamic explanation guide for RRG clockwise rotation lifecycle
    with st.expander("ℹ️ How to Read this Sector Rotation (RRG) Chart", expanded=False):
        st.markdown("""
        **Relative Rotation Graphs (RRG)** track the strength and momentum of sectors relative to the benchmark index (Nifty 50).
        
        Sectors typically rotate in a **clockwise** cycle through the four quadrants:
        1. **📈 Recovering (Top-Left):** Sector momentum is turning positive, but trend strength is still lagging. *Add to Watchlist.*
        2. **🚀 Accelerating (Top-Right):** Sector has both strong trend and strong momentum. *Strong Buy/Hold candidate.*
        3. **📉 Decelerating (Bottom-Right):** Sector trend remains strong, but upward momentum is fading. *Consider locking in profits.*
        4. **⚪ Underperforming (Bottom-Left):** Both trend and momentum are weak. *Avoid or exit positions.*
        
        *The Vector Arrow (↗, ↖, ↙, ↘) next to the symbol shows its weekly rotation direction. Northeast movement (↗) is a strong bullish indicator.*
        """)

    # Sync button in sidebar
    with st.sidebar:
        st.markdown("### 📥 Sector Index Data")
        st.caption("Downloads 5 years of daily data for all sector indices from Yahoo Finance")
        if st.button("Sync Sector Indices", type="secondary", width="stretch",
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
        if st.button("📥 Export CSV", type="secondary", width="stretch",
                     help="Download all sector index data as CSV"):
            csv = export_sector_data_csv()
            if csv:
                st.download_button("⬇️ Download", data=csv, file_name="sector_indices.csv",
                                   mime="text/csv", width="stretch")
            else:
                st.warning("No sector data to export")

    with st.spinner("Loading sector rotation data..."):
        data = get_sector_rotation()

    if not data["sectors"]:
        st.warning("No sector data available. Click **Sync Sector Indices** in the sidebar.")
        return

    # Initialize selected sectors in session_state if not present
    if "selected_sectors" not in st.session_state:
        st.session_state.selected_sectors = {s["name"]: True for s in data["sectors"]}

    # Create Side-by-Side split screen layout matching DEXT T3
    col_left, col_right = st.columns([1.2, 2.5])

    # Left Column: Sector selection table with checkboxes, live quadrant pill, trend, and momentum values
    with col_left:
        st.markdown("#### Symbols")
        
        # Table Header
        h1, h2, h3, h4, h5 = st.columns([0.12, 0.33, 0.25, 0.15, 0.15])
        h1.markdown("")
        h2.markdown("<span style='font-size: 12px; font-weight: bold; color: #888;'>Symbol</span>", unsafe_allow_html=True)
        h3.markdown("<span style='font-size: 12px; font-weight: bold; color: #888;'>Type</span>", unsafe_allow_html=True)
        h4.markdown("<span style='font-size: 12px; font-weight: bold; color: #888;'>Trend</span>", unsafe_allow_html=True)
        h5.markdown("<span style='font-size: 12px; font-weight: bold; color: #888;'>Mom.</span>", unsafe_allow_html=True)
        
        st.divider()
        
        for s in data["sectors"]:
            r1, r2, r3, r4, r5 = st.columns([0.12, 0.33, 0.25, 0.15, 0.15])
            with r1:
                # Retrieve check state
                is_checked = st.checkbox("", value=st.session_state.selected_sectors.get(s["name"], True), 
                                         key=f"check_{s['name']}", label_visibility="collapsed")
                st.session_state.selected_sectors[s["name"]] = is_checked
            with r2:
                # Symbol Name and Direction Arrow indicator
                symbol_html = f"<span style='font-size: 13px; font-weight: bold;'>{s['name']} {s['direction']}</span>"
                st.markdown(symbol_html, unsafe_allow_html=True)
            with r3:
                # Live Quadrant badge
                quad = s["quadrant"]
                if quad == "Accelerating":
                    st.markdown("<span style='font-size: 11px; color:#00cc96; background-color:rgba(0,204,150,0.1); padding:2px 6px; border-radius:4px; font-weight:bold;'>Accelerating</span>", unsafe_allow_html=True)
                elif quad == "Recovering":
                    st.markdown("<span style='font-size: 11px; color:#ab63fa; background-color:rgba(171,99,250,0.1); padding:2px 6px; border-radius:4px; font-weight:bold;'>Recovering</span>", unsafe_allow_html=True)
                elif quad == "Decelerating":
                    st.markdown("<span style='font-size: 11px; color:#ffa15a; background-color:rgba(255,161,90,0.1); padding:2px 6px; border-radius:4px; font-weight:bold;'>Decelerating</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='font-size: 11px; color:#ef553b; background-color:rgba(239,85,59,0.1); padding:2px 6px; border-radius:4px; font-weight:bold;'>Lagging</span>", unsafe_allow_html=True)
            with r4:
                # Scale Trend (z-score converted to 100 center)
                val_trend = s["rs_ratio"] * 5 + 100
                st.markdown(f"<span style='font-size: 12px; color: #888;'>{val_trend:.2f}</span>", unsafe_allow_html=True)
            with r5:
                # Scale Momentum
                val_mom = s["rs_momentum"] * 5 + 100
                st.markdown(f"<span style='font-size: 12px; color: #888;'>{val_mom:.2f}</span>", unsafe_allow_html=True)

    # Right Column: Line chart + RRG bubble chart with color shading
    with col_right:
        # 1. Top Chart: Nifty 50 Benchmark Line Chart
        st.markdown("#### Nifty 50 Benchmark History")
        benchmark = data.get("benchmark_history", [])
        if benchmark:
            df_bench = pd.DataFrame(benchmark)
            df_bench["date"] = pd.to_datetime(df_bench["date"])
            
            fig_bench = go.Figure()
            fig_bench.add_trace(go.Scatter(
                x=df_bench["date"],
                y=df_bench["close"],
                mode="lines",
                line=dict(color="#19d3f3", width=1.5),
                name="Nifty 50",
                hoverinfo="x+y"
            ))
            fig_bench.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=False, showticklabels=True, color="#888", gridcolor="#222"),
                yaxis=dict(showgrid=True, gridcolor="#222", showticklabels=True, color="#888"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        st.divider()

        # 2. Trail length slider above bottom RRG plot to control clutter
        st.markdown("#### Relative Cycle Graph")
        trail_weeks = st.slider("Historical Trail Length (Weeks)", min_value=1, max_value=20, value=5, step=1)

        fig_rrg = go.Figure()
        
        # Color palette for checked symbols
        palette = ["#636efa","#ef553b","#00cc96","#ab63fa","#ffa15a","#19d3f3","#ff6692","#b6e880","#ff97ff","#fecb52",
                   "#d62728","#2ca02c","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        sector_colors = {s["name"]: palette[i % len(palette)] for i, s in enumerate(data["sectors"])}

        # Plot each checked sector
        for s in data["sectors"]:
            # Skip if unchecked in the left control panel
            if not st.session_state.selected_sectors.get(s["name"], True):
                continue
                
            trail = s.get("trail", [])
            # Slice historical trail to the slider value
            trail_sliced = trail[-trail_weeks:] if len(trail) >= trail_weeks else trail
            
            if len(trail_sliced) >= 2:
                # Plot 100-based shifted values: coordinate = z_score * 5 + 100
                tx = [p["rs_ratio"] * 5 + 100 for p in trail_sliced]
                ty = [p["rs_momentum"] * 5 + 100 for p in trail_sliced]
                fig_rrg.add_trace(go.Scatter(
                    x=tx, y=ty,
                    mode="lines+markers",
                    marker=dict(size=4, color=sector_colors[s["name"]], opacity=0.4),
                    line=dict(width=1.5, color=sector_colors[s["name"]], shape="spline"),
                    showlegend=False,
                    hoverinfo="skip"
                ))
            
            # Current value dot mapped to 100-based axes
            curr_x_100 = s["rs_ratio"] * 5 + 100
            curr_y_100 = s["rs_momentum"] * 5 + 100
            fig_rrg.add_trace(go.Scatter(
                x=[curr_x_100], y=[curr_y_100],
                mode="markers+text",
                text=s["name"],
                textposition="top center",
                marker=dict(size=12, color=sector_colors[s["name"]]),
                name=s["name"],
                hovertemplate=f"<b>{s['name']}</b><br>Trend: {curr_x_100:.2f}<br>Mom: {curr_y_100:.2f}<br>Quadrant: {s['quadrant']}",
            ))

        # Color-coded quadrant backgrounds split at exactly 100.00
        quadrant_shapes = [
            # Top-Right (Accelerating) - Green
            dict(type="rect", xref="x", yref="y", x0=100, y0=100, x1=120, y1=120, fillcolor="rgba(0, 204, 150, 0.03)", line_width=0, layer="below"),
            # Top-Left (Recovering) - Purple
            dict(type="rect", xref="x", yref="y", x0=80, y0=100, x1=100, y1=120, fillcolor="rgba(171, 99, 250, 0.03)", line_width=0, layer="below"),
            # Bottom-Left (Lagging/Underperforming) - Red
            dict(type="rect", xref="x", yref="y", x0=80, y0=80, x1=100, y1=100, fillcolor="rgba(239, 85, 59, 0.03)", line_width=0, layer="below"),
            # Bottom-Right (Decelerating) - Yellow
            dict(type="rect", xref="x", yref="y", x0=100, y0=80, x1=120, y1=100, fillcolor="rgba(255, 161, 90, 0.03)", line_width=0, layer="below")
        ]

        # Draw grid dividing lines crossing exactly at 100.00
        fig_rrg.add_hline(y=100, line_color="#444", line_dash="dash", opacity=0.7)
        fig_rrg.add_vline(x=100, line_color="#444", line_dash="dash", opacity=0.7)

        fig_rrg.update_layout(
            height=500,
            xaxis=dict(title="Strength Trend (RS-Ratio)", range=[85, 115], gridcolor="#222", showgrid=True, zeroline=False),
            yaxis=dict(title="Strength Momentum (RS-Momentum)", range=[85, 115], gridcolor="#222", showgrid=True, zeroline=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            shapes=quadrant_shapes,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[
                dict(x=110, y=112.5, text="Accelerating", showarrow=False, font=dict(color="#00cc96", size=14, weight="bold")),
                dict(x=90, y=112.5, text="Recovering", showarrow=False, font=dict(color="#ab63fa", size=14, weight="bold")),
                dict(x=110, y=87.5, text="Decelerating", showarrow=False, font=dict(color="#ffa15a", size=14, weight="bold")),
                dict(x=90, y=87.5, text="Underperforming", showarrow=False, font=dict(color="#ef553b", size=14, weight="bold")),
            ]
        )
        st.plotly_chart(fig_rrg, use_container_width=True)

    # Quadrant tables
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Accelerating", "📈 Recovering", "📉 Decelerating", "⚪ Underperforming"])
    for tab, qname, qlist in [
        (tab1, "Accelerating", data["accelerating"]),
        (tab2, "Recovering", data["recovering"]),
        (tab3, "Decelerating", data["decelerating"]),
        (tab4, "Underperforming", data["underperforming"]),
    ]:
        with tab:
            if qlist:
                df = pd.DataFrame(qlist)
                # Display shifted values in table to stay consistent with chart
                df["rs_ratio"] = df["rs_ratio"] * 5 + 100
                df["rs_momentum"] = df["rs_momentum"] * 5 + 100
                
                st.dataframe(df[["ticker", "name", "rs_ratio", "rs_momentum", "price"]],
                             use_container_width=True,
                             column_config={
                                 "rs_ratio": st.column_config.NumberColumn("Trend (RS-Ratio)", format="%.2f"),
                                 "rs_momentum": st.column_config.NumberColumn("Momentum (RS-Momentum)", format="%.2f"),
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
    run_rot = st.button("Run Rotation Backtest", type="primary", width="stretch")

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
