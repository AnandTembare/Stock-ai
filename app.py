# ============================================================
# Indian Stock Market Analyser — Premium Dark Edition
# Streamlit App | Live NSE/BSE data via yfinance
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

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Market Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
NSE_STOCKS = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "L&T": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS",
}

INDICES = {"Nifty 50": "^NSEI", "Sensex": "^BSESN", "Nifty Bank": "^NSEBANK"}

# ── CSS Styling (Premium Dark Mode) ───────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base Theme */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        [data-testid="stAppViewContainer"] { background-color: #070B14; color: #E2E8F0; }
        [data-testid="stSidebar"] { background-color: #0F1523; border-right: 1px solid #1E293B; }
        [data-testid="stHeader"] { background-color: transparent; }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #131B2F, #0B101D);
            border: 1px solid #1E293B;
            border-radius: 16px;
            padding: 1.2rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #34D399; }
        [data-testid="stMetricValue"] { color: #F8FAFC; font-weight: 700; font-size: 1.8rem !important; }
        [data-testid="stMetricLabel"] { color: #94A3B8; font-size: 0.85rem !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
        
        /* Delta Colors (Neon Green/Red) */
        [data-testid="stMetricDelta"] svg { display: none; } /* Hide default arrows */
        [data-testid="stMetricDelta"] { font-weight: 600; }

        /* Tabs (Pill Style from Image) */
        .stTabs [data-baseweb="tab-list"] { 
            background-color: transparent; gap: 12px; padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab"] { 
            background-color: #131B2F; color: #94A3B8; 
            border-radius: 24px; padding: 8px 24px; 
            border: 1px solid #1E293B; font-weight: 500;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #064E3B !important; 
            color: #34D399 !important; 
            border-color: #059669 !important; 
            box-shadow: 0 0 12px rgba(52, 211, 153, 0.2);
        }

        /* Buttons */
        div.stButton > button {
            background-color: #10B981; color: #022C22; 
            border-radius: 24px; border: none; font-weight: 700;
            padding: 0.5rem 1.5rem; transition: all 0.2s;
        }
        div.stButton > button:hover { background-color: #34D399; box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }

        /* Headers */
        .header-title { font-size: 2.2rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0; display: flex; align-items: center; gap: 10px;}
        .header-title span { color: #10B981; } /* Accent color */
        .header-sub { color: #64748B; font-size: 0.95rem; margin-top: 5px; }

        /* Custom Ticker Tape */
        .ticker-wrap {
            width: 100%; overflow: hidden; background-color: #0B101D; 
            border-top: 1px solid #1E293B; border-bottom: 1px solid #1E293B;
            padding: 12px 0; margin: 20px 0; display: flex;
        }
        .ticker-item {
            color: #E2E8F0; font-weight: 600; font-size: 0.95rem; margin-right: 40px; white-space: nowrap;
        }
        .ticker-up { color: #10B981; }
        .ticker-down { color: #EF4444; }

        /* Signal Box (ML Prediction) */
        .signal-box {
            border-radius: 16px; padding: 2rem; text-align: center; font-size: 1.5rem; font-weight: 700;
            background: linear-gradient(145deg, #131B2F, #0B101D); border: 1px solid #1E293B;
        }
        .buy { border-color: #10B981; color: #10B981; box-shadow: 0 0 20px rgba(16, 185, 129, 0.1); }
        .sell { border-color: #EF4444; color: #EF4444; box-shadow: 0 0 20px rgba(239, 68, 68, 0.1); }
        .hold { border-color: #F59E0B; color: #F59E0B; }
    </style>
    """, unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).fast_info
        return {"price": round(info.last_price, 2), "prev_close": round(info.previous_close, 2)}
    except: return {"price": None, "prev_close": None}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    
    df["Vol_MA"] = df["Volume"].rolling(20).mean()
    return df

@st.cache_resource(show_spinner=False)
def ml_predict(df: pd.DataFrame) -> tuple:
    df = add_indicators(df).dropna().copy()
    if len(df) < 50: return "Need Data", 0.0, 0.0
    
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["RSI", "SMA_20", "SMA_50", "BB_Upper", "BB_Lower"]
    df_train = df.dropna(subset=features + ["Target"])
    
    X, y = df_train[features].values, df_train["Target"].values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    prob = model.predict_proba(scaler.transform(df[features].iloc[-1].values.reshape(1, -1)))[0][1]

    signal = "BUY" if prob >= 0.55 else ("SELL" if prob <= 0.45 else "HOLD")
    return signal, round(prob * 100, 1), round(acc * 100, 1)

# ── Main Application ──────────────────────────────────────────
def main():
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:#F8FAFC; font-weight:700;'>◭ Analyser</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        stock_name = st.selectbox("Search Asset", list(NSE_STOCKS.keys()), index=0)
        ticker = NSE_STOCKS[stock_name]
        period = st.selectbox("Timeframe", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<div class="header-title"><span>✦</span> {stock_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="header-sub">{ticker} &nbsp;•&nbsp; Live Market Data</div>', unsafe_allow_html=True)
    with col2:
        st.write("") # Spacer
        if st.button("✦ Quick Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Fetch Data
    with st.spinner("Analyzing market structure..."):
        df = fetch_stock_data(ticker, period)
        live = fetch_live_price(ticker)

    if df.empty:
        st.error("Connection interrupted. Please try again.")
        st.stop()

    df_ind = add_indicators(df)

    # Calculate current metrics
    try:
        price = live["price"] if live["price"] else float(df["Close"].iloc[-1])
        prev = live["prev_close"] if live["prev_close"] else float(df["Close"].iloc[-2])
        change = price - prev
        change_p = (change / prev) * 100
        sign = "+" if change >= 0 else ""
    except:
        price, prev, change, change_p, sign = 0.0, 0.0, 0.0, 0.0, ""

    # Simulated Ticker Tape Layout (Static CSS marquee style)
    st.markdown(f"""
        <div class="ticker-wrap">
            <div style="display:flex; animation: scroll 20s linear infinite;">
                <span class="ticker-item">RELIANCE 2930.50 <span class="ticker-up">▲ 1.2%</span></span>
                <span class="ticker-item">HDFCBANK 1435.20 <span class="ticker-down">▼ 0.8%</span></span>
                <span class="ticker-item">INFY 1420.00 <span class="ticker-up">▲ 2.1%</span></span>
                <span class="ticker-item">TCS 3890.10 <span class="ticker-up">▲ 0.5%</span></span>
                <span class="ticker-item">ITC 410.25 <span class="ticker-down">▼ 1.1%</span></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Top Metrics Grid
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"₹{price:,.2f}", f"{sign}{change:.2f} ({sign}{change_p:.2f}%)")
    m2.metric("Period High", f"₹{float(df['High'].max()):,.2f}")
    m3.metric("Period Low", f"₹{float(df['Low'].min()):,.2f}")
    rsi_val = float(df_ind["RSI"].iloc[-1]) if not pd.isna(df_ind["RSI"].iloc[-1]) else 0.0
    m4.metric("Technical RSI", f"{rsi_val:.1f}", "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))

    st.markdown("<br>", unsafe_allow_html=True)

    # Beautiful Tabs
    tab1, tab2, tab3 = st.tabs(["✨ Quick Insights", "📈 Technical Analysis", "🤖 AI Price Prediction"])

    with tab1:
        st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 20px;'>Market Structure</h4>", unsafe_allow_html=True)
        # Main Chart styled with Neon Colors
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#10B981", increasing_fillcolor="#10B981", # Neon Green
            decreasing_line_color="#EF4444", decreasing_fillcolor="#EF4444", # Bright Red
            name="Price"
        ), row=1, col=1)

        colors = ["#10B981" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#EF4444" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.5, name="Volume"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=500, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(15, 21, 35, 0.8)")
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1E293B", zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1E293B", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 20px;'>Advanced Overlays</h4>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#94A3B8", width=1.5)))
        fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_20"], name="SMA 20", line=dict(color="#3B82F6", width=2)))
        fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"], name="BB Upper", line=dict(color="#8B5CF6", width=1, dash="dot")))
        fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"], name="BB Lower", line=dict(color="#8B5CF6", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(139, 92, 246, 0.05)"))
        
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450,
            margin=dict(l=0, r=0, t=10, b=0), legend=dict(bgcolor="rgba(15, 21, 35, 0.8)")
        )
        fig2.update_xaxes(gridcolor="#1E293B")
        fig2.update_yaxes(gridcolor="#1E293B")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        signal, prob, acc = ml_predict(df)
        
        st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 20px;'>AI Strategy Builder</h4>", unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            css_class = "buy" if signal == "BUY" else ("sell" if signal == "SELL" else "hold")
            st.markdown(f'''
                <div class="signal-box {css_class}">
                    <div style="font-size:0.9rem; color:#94A3B8; font-weight:500; text-transform:uppercase;">AI Signal</div>
                    {signal}<br>
                    <div style="font-size:1.1rem; margin-top:10px;">{prob}% Confidence</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with col_s2:
            st.markdown(f"""
            <div style="background-color: #131B2F; border: 1px solid #1E293B; border-radius: 16px; padding: 20px; height: 100%;">
                <h5 style="color:#10B981; margin-top:0;">Model Diagnostics</h5>
                <p style="color:#94A3B8; font-size:0.95rem;">
                The AI model analyzes historical patterns using Random Forest classification. 
                It evaluates volatility (Bollinger Bands), momentum (RSI), and trend trajectories (SMAs).
                </p>
                <div style="display:flex; justify-content: space-between; border-top: 1px solid #1E293B; padding-top: 15px; margin-top:15px;">
                    <div><span style="color:#64748B; font-size:0.85rem;">Test Accuracy</span><br><b style="color:#E2E8F0;">{acc}%</b></div>
                    <div><span style="color:#64748B; font-size:0.85rem;">Engine</span><br><b style="color:#E2E8F0;">RandomForest</b></div>
                    <div><span style="color:#64748B; font-size:0.85rem;">Horizon</span><br><b style="color:#E2E8F0;">1 Day (T+1)</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
