import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_data
from prophet_model import run_prophet
from sarima_model import run_sarima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title  = "EnergyIQ · Forecast",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* global */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* hide streamlit default header */
#MainMenu, footer, header { visibility: hidden; }

/* main container */
.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* hero header */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #0d1117 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.3rem 0;
    letter-spacing: -1px;
}
.hero p {
    color: #8888aa;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}
.hero-accent { color: #63b3ed; }

/* metric cards */
.metric-card {
    background: #12121e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #63b3ed; }
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8888aa;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #ffffff;
}
.metric-sub {
    font-size: 0.75rem;
    color: #8888aa;
    margin-top: 4px;
}
.metric-good  { color: #68d391; }
.metric-warn  { color: #f6ad55; }

/* section title */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8888aa;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #2a2a4a;
}

/* winner banner */
.winner-banner {
    background: linear-gradient(135deg, #1a2a1a, #0d1a0d);
    border: 1px solid #276749;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1rem;
    font-weight: 500;
    color: #68d391;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #0d0d1a;
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

/* selectbox & slider */
.stSelectbox > div, .stSlider > div { color: #e8e8f0; }
</style>
""", unsafe_allow_html=True)


# ── hero header ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Live Forecast</div>
    <h1>Energy<span class="hero-accent">IQ</span></h1>
    <p>Household power consumption forecasting · Prophet vs SARIMA</p>
</div>
""", unsafe_allow_html=True)


# sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    model_choice  = st.selectbox("Model", ["Both", "Prophet", "SARIMA"])
    forecast_days = st.slider("Forecast Window (days)", 7, 60, 30)
    show_raw      = st.checkbox("Show Raw Data Table")
    st.markdown("---")

# load data 
@st.cache_data
def get_data():
    return load_data("data/household_power_consumption.txt")

with st.spinner("Loading dataset..."):
    df = get_data()


#  top stats 
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Days</div>
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-sub">in dataset</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Power</div>
        <div class="metric-value">{df['y'].mean():.2f}</div>
        <div class="metric-sub">kW daily avg</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Peak Power</div>
        <div class="metric-value metric-warn">{df['y'].max():.2f}</div>
        <div class="metric-sub">kW max recorded</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecast Window</div>
        <div class="metric-value metric-good">{forecast_days}</div>
        <div class="metric-sub">days ahead</div>
    </div>""", unsafe_allow_html=True)


# ── run models 
actual = df["y"][-30:].values

if model_choice in ["Prophet", "Both"]:
    with st.spinner("Running Prophet..."):
        prophet_forecast = run_prophet(df)
        prophet_pred     = prophet_forecast["yhat"][-30:].values

if model_choice in ["SARIMA", "Both"]:
    with st.spinner("Running SARIMA..."):
        sarima_pred, _ = run_sarima(df)
        sarima_pred    = sarima_pred.values


#  forecast chart 
st.markdown('<div class="section-title">Forecast Chart</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#12121e")
ax.set_facecolor("#12121e")

# actual data - show last 90 days for clarity
ax.plot(df["ds"][-90:], df["y"][-90:],
        color="#8888aa", linewidth=1.2, label="Actual", alpha=0.8)

if model_choice in ["Prophet", "Both"]:
    ax.plot(prophet_forecast["ds"][-30:], prophet_forecast["yhat"][-30:],
            color="#63b3ed", linewidth=2, label="Prophet", linestyle="--")

if model_choice in ["SARIMA", "Both"]:
    ax.plot(df["ds"][-30:], sarima_pred,
            color="#68d391", linewidth=2, label="SARIMA", linestyle="-.")

# styling
ax.tick_params(colors="#8888aa", labelsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for spine in ax.spines.values():
    spine.set_edgecolor("#2a2a4a")
ax.set_xlabel("Date", color="#8888aa", fontsize=10)
ax.set_ylabel("Power (kW)", color="#8888aa", fontsize=10)
ax.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a",
          labelcolor="#e8e8f0", fontsize=10)
ax.grid(axis="y", color="#2a2a4a", linewidth=0.5, alpha=0.5)
fig.tight_layout()
st.pyplot(fig)


#  model metrics
st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

if model_choice in ["Prophet", "Both"]:
    p_mae  = mean_absolute_error(actual, prophet_pred)
    p_rmse = np.sqrt(mean_squared_error(actual, prophet_pred))
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"> Prophet</div>
            <div style="display:flex; justify-content:space-around; margin-top:1rem;">
                <div>
                    <div class="metric-label">MAE</div>
                    <div class="metric-value" style="font-size:1.4rem; color:#63b3ed">{p_mae:.4f}</div>
                </div>
                <div>
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value" style="font-size:1.4rem; color:#63b3ed">{p_rmse:.4f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

if model_choice in ["SARIMA", "Both"]:
    s_mae  = mean_absolute_error(actual, sarima_pred)
    s_rmse = np.sqrt(mean_squared_error(actual, sarima_pred))
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"> SARIMA</div>
            <div style="display:flex; justify-content:space-around; margin-top:1rem;">
                <div>
                    <div class="metric-label">MAE</div>
                    <div class="metric-value" style="font-size:1.4rem; color:#68d391">{s_mae:.4f}</div>
                </div>
                <div>
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value" style="font-size:1.4rem; color:#68d391">{s_rmse:.4f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


#  winner 
if model_choice == "Both":
    winner = " Prophet" if p_mae < s_mae else " SARIMA"
    st.markdown(f"""
    <div class="winner-banner">
         &nbsp; <strong>{winner}</strong> &nbsp; wins with lower MAE on this forecast window
    </div>""", unsafe_allow_html=True)


# raw data 
if show_raw:
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(
        df.tail(50).style.background_gradient(subset=["y"], cmap="Blues"),
        use_container_width=True
    )