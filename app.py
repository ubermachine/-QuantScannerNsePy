"""app.py — Streamlit entry point. Sidebar nav, page routing, global config."""
import streamlit as st

st.set_page_config(
    page_title="QuantScanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## 📈 QuantScanner")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Scan Dashboard", "Sector Rotation", "Backtest", "Chart Analysis"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("v2.0 · Streamlit")

if page == "Scan Dashboard":
    from pages_app.scan import show
    show()
elif page == "Sector Rotation":
    from pages_app.sector import show
    show()
elif page == "Backtest":
    from pages_app.backtest import show
    show()
elif page == "Chart Analysis":
    from pages_app.chart import show
    show()

