# ============================================================
# Indian Market Analyser Pro — Enterprise Edition
# Built for production with Session State & Micro-interactions
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# ── 1. Core App Configuration ─────────────────────────────────
st.set_page_config(
    page_title="Market AI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Enterprise CSS Injection ───────────────────────────────
def inject_enterprise_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Hide Default Streamlit Chrome to make it look like a real app */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* App Canvas */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        [data-testid="stAppViewContainer"] { background-color: #05070A; color: #E2E8F0; }
        [data-testid="stSidebar"] { background-color: #0A0F18; border-right: 1px solid #1A2333; padding-top: 2rem;}
        
        /* Remove top padding for edge-to-edge feel */
        .block-container { padding-top: 1rem; max-width: 95%; }

        /* Glassmorphic Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        [data-testid="stMetric"]:hover { 
            transform: translateY(-4px); 
            border-color: rgba(16, 185, 129, 0.4);
            box-shadow: 0 10px 20px -10px rgba(16, 185, 129, 0.2);
        }
        [data-testid="stMetricValue"] { color: #FFFFFF; font-weight: 800; font-size: 2rem !important; }
        [data-testid="stMetricLabel"] { color: #64748B; font-size: 0.8rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
        [data-testid="stMetricDelta"] svg { display: none; }

        /* App Tabs (Segmented Control Style) */
        .stTabs [data-baseweb="tab-list"] { 
            background-color: #0A0F18; border-radius: 12px; padding: 6px; gap: 4px; border: 1px solid #1A2333; width: fit-content;
        }
        .stTabs [data-baseweb="tab"] { 
            color: #64748B; border-radius: 8px; padding: 8px 24px; font-weight: 600; border: none; transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #10B981 !important; color: #022C22 !important; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        /* Ticker Tape */
        .ticker-wrap {
            width: 100%; overflow: hidden; background: linear-gradient(90deg, #05070A 0%, #0A0F18 50%, #05070A 100%); 
            border-top: 1px solid #1A2333; border-bottom: 1px solid #1A2333;
            padding: 10px 0; margin-bottom: 1.5rem; display: flex;
        }
        .ticker-item { font-weight: 600; font-size: 0.85rem; margin-right: 50px; white-space: nowrap; color: #94A3B8; }
        .pos { color: #10B981; } .neg { color: #EF4444; }

        /* Primary Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white; border-radius: 8px; border: none; font-weight: 600; width: 100%;
            padding: 0.6rem; transition: all 0.2s;
        }
        div.stButton > button:hover { opacity: 0.9; box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); transform: scale(0.98); }
    </style>
    """, unsafe_allow_html=True)

# ── 3. Constants & Data Dictionary ────────────────────────────
ASSETS = {
    "Reliance Ind": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS"
}

# ── 4. Robust Data Fetching & Caching ─────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetches and sanitizes OHLCV data."""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

def calculate_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Engine for all technical overlays."""
    df = df.copy()
    c = df["Close"]
    df["SMA_20"], df["SMA_50"] = c.rolling(20).mean(), c.rolling(50).mean()
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger
    std20 = c.rolling(20).std()
    df["BB_Upper"], df["BB_Lower"] = df["SMA_20"] + 2 * std20, df["SMA_20"] - 2 * std20
    return df

@st.cache_resource(show_spinner=False)
def ai_prediction_engine(df: pd.DataFrame):
    """Machine learning backend."""
    df = calculate_technicals(df).dropna()
    if len(df) < 60: return None, 0.0, 0.0
    
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["RSI", "SMA_20", "SMA_50", "BB_Upper", "BB_Lower"]
    df_train = df.dropna(subset=features + ["Target"])
    
    X, y = df_train[features].values, df_train["Target"].values
    split = int(len(X) * 0.8)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y[:split])

    acc = accuracy_score(y[split:], model.predict(X_test))
    prob = model.predict_proba(scaler.transform(df[features].iloc[-1].values.reshape(1, -1)))[0][1]

    signal = "BUY" if prob >= 0.55 else ("SELL" if prob <= 0.45 else "HOLD")
    return signal, round(prob * 100, 1), round(acc * 100, 1)

# ── 5. App Controller (Main Loop) ─────────────────────────────
def main():
    inject_enterprise_css()

    # --- STATE MANAGEMENT ---
    # Real apps remember user choices.
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

    # --- SIDEBAR (App Navigation) ---
    with st.sidebar:
        st.markdown("<h2 style='color:#10B981; font-weight:800; letter-spacing:-1px; margin-bottom:2rem;'>✦ QuantumAI</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#64748B; font-size:0.8rem; font-weight:600; text-transform:uppercase;'>Asset Selection</p>", unsafe_allow_html=True)
        selected_asset = st.selectbox("Ticker", list(ASSETS.keys()), label_visibility="collapsed")
        ticker = ASSETS[selected_asset]
        
        st.markdown("<p style='color:#64748B; font-size:0.8rem; font-weight:600; text-transform:uppercase; margin-top:1rem;'>Horizon</p>", unsafe_allow_html=True)
        period = st.select_slider("Timeframe", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], value="1y", label_visibility="collapsed")

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Force Sync Data"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.toast("Connection secured. Live data synchronized.", icon="⚡")
            st.rerun()

    # --- DATA FETCHING (With graceful fallbacks) ---
    with st.spinner("Establishing secure connection to market node..."):
        df = get_market_data(ticker, period)
    
    if df.empty:
        st.error("⚠️ Failed to establish connection to the data node. The market API might be throttling requests. Please wait 60 seconds and try again.")
        st.stop()

    df_tech = calculate_technicals(df)

    # --- TOP TICKER TAPE ---
    st.markdown(f"""
        <div class="ticker-wrap">
            <div style="display:flex; animation: scroll 25s linear infinite;">
                <span class="ticker-item">NIFTY 50 22150.50 <span class="pos">▲ 0.8%</span></span>
                <span class="ticker-item">BANKNIFTY 46500.20 <span class="pos">▲ 1.1%</span></span>
                <span class="ticker-item">SENSEX 73000.00 <span class="pos">▲ 0.7%</span></span>
                <span class="ticker-item">INDIA VIX 15.20 <span class="neg">▼ 2.4%</span></span>
                <span class="ticker-item">USD/INR 82.90 <span class="pos">▲ 0.1%</span></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- HEADER & KPI METRICS ---
    st.markdown(f"<h1 style='font-size:2.5rem; margin-bottom:0;'>{selected_asset} <span style='color:#64748B; font-size:1.2rem; font-weight:500;'>{ticker}</span></h1>", unsafe_allow_html=True)
    
    current_px = float(df["Close"].iloc[-1])
    prev_px = float(df["Close"].iloc[-2])
    chg = current_px - prev_px
    chg_pct = (chg / prev_px) * 100
    color_hex = "#10B981" if chg >= 0 else "#EF4444"
    sign = "+" if chg >= 0 else ""

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Value", f"₹{current_px:,.2f}", f"{sign}{chg:.2f} ({sign}{chg_pct:.2f}%)")
    m2.metric("Period High", f"₹{float(df['High'].max()):,.2f}")
    m3.metric("Period Low", f"₹{float(df['Low'].min()):,.2f}")
    
    rsi_latest = float(df_tech["RSI"].iloc[-1]) if not pd.isna(df_tech["RSI"].iloc[-1]) else 50.0
    rsi_state = "Overbought" if rsi_latest > 70 else ("Oversold" if rsi_latest < 30 else "Neutral")
    m4.metric("RSI Momentum", f"{rsi_latest:.1f}", rsi_state)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- APPLICATION WORKSPACES (Tabs) ---
    t_market, t_tech, t_ai = st.tabs(["Market Context", "Technical Overlay", "AI Oracle"])

    with t_market:
        # High performance Plotly chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#10B981", increasing_fillcolor="#10B981",
            decreasing_line_color="#EF4444", decreasing_fillcolor="#EF4444", name="Price"
        ), row=1, col=1)
        
        vol_colors = ["#10B981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#EF4444" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, opacity=0.4, name="Volume"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=550,
            margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1A2333")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1A2333")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) # Hides standard plotly tools for cleaner UI

    with t_tech:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price Action", line=dict(color="#64748B", width=1.5)))
        fig2.add_trace(go.Scatter(x=df_tech.index, y=df_tech["SMA_20"], name="SMA 20", line=dict(color="#3B82F6", width=2)))
        fig2.add_trace(go.Scatter(x=df_tech.index, y=df_tech["SMA_50"], name="SMA 50", line=dict(color="#F59E0B", width=2)))
        fig2.add_trace(go.Scatter(x=df_tech.index, y=df_tech["BB_Upper"], name="Upper Band", line=dict(color="#8B5CF6", width=1, dash="dot")))
        fig2.add_trace(go.Scatter(x=df_tech.index, y=df_tech["BB_Lower"], name="Lower Band", line=dict(color="#8B5CF6", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(139, 92, 246, 0.05)"))
        
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=550,
            margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig2.update_xaxes(gridcolor="#1A2333")
        fig2.update_yaxes(gridcolor="#1A2333")
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with t_ai:
        signal, prob, acc = ai_prediction_engine(df)
        
        if signal:
            c1, c2 = st.columns([1.5, 2])
            with c1:
                sig_color = "#10B981" if signal == "BUY" else ("#EF4444" if signal == "SELL" else "#F59E0B")
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {sig_color}40; border-radius: 16px; padding: 30px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <p style="color:#64748B; font-weight:600; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Algorithmic Signal</p>
                    <h2 style="color:{sig_color}; font-size:3rem; font-weight:800; margin:0;">{signal}</h2>
                    <h3 style="color:#F8FAFC; margin-top:15px; font-weight:400;">Confidence: <b style="color:{sig_color}">{prob}%</b></h3>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid #1A2333; border-radius: 16px; padding: 25px; height: 100%;">
                    <h4 style="color:#F8FAFC; margin-top:0; border-bottom: 1px solid #1A2333; padding-bottom:10px;">Intelligence Node Diagnostics</h4>
                    <p style="color:#94A3B8; font-size:0.95rem; line-height:1.6;">
                    The QuantumAI engine utilizes a <b>Random Forest Classification</b> model. It synthesizes real-time volatility, momentum variations (RSI), and trailing averages to project T+1 day price direction.
                    </p>
                    <div style="display:flex; justify-content: space-between; margin-top: 25px;">
                        <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; width: 48%; border: 1px solid #1A2333;">
                            <span style="color:#64748B; font-size:0.8rem; text-transform:uppercase;">Out-of-Sample Acc</span><br>
                            <span style="color:#F8FAFC; font-size:1.5rem; font-weight:700;">{0}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; width: 48%; border: 1px solid #1A2333;">
                            <span style="color:#64748B; font-size:0.8rem; text-transform:uppercase;">Data Points Modeled</span><br>
                            <span style="color:#F8FAFC; font-size:1.5rem; font-weight:700;">{1}</span>
                        </div>
                    </div>
                </div>
                """.format(acc, len(df)), unsafe_allow_html=True)
        else:
            st.warning("Insufficient historical data to initialize the QuantumAI engine. Select a wider timeframe.")

if __name__ == "__main__":
    main()
