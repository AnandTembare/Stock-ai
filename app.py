# ============================================================
#  Indian Stock Market Analyser — Anand Tembare
#  Single-file Streamlit app | Live NSE/BSE data via yfinance
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Market Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricValue"] { color: #f0fdf4; font-family: 'Space Mono', monospace; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

    .stTabs [data-baseweb="tab-list"] { background: #0f172a; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 8px; padding: 6px 18px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }

    .signal-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem 0;
    }
    .buy  { background: #052e16; border: 2px solid #16a34a; color: #4ade80; }
    .sell { background: #450a0a; border: 2px solid #dc2626; color: #f87171; }
    .hold { background: #1c1917; border: 2px solid #ca8a04; color: #facc15; }

    .header-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .header-sub { color: #64748b; font-size: 0.9rem; margin-top: 0.2rem; }

    div[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
    div[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# ── Top 50 NSE Stocks ─────────────────────────────────────────
NSE_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "L&T": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Titan Company": "TITAN.NS",
    "NTPC": "NTPC.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Power Grid": "POWERGRID.NS",
    "Nestle India": "NESTLEIND.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Divis Lab": "DIVISLAB.NS",
    "Cipla": "CIPLA.NS",
    "Dr Reddys": "DRREDDY.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "BPCL": "BPCL.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Shree Cement": "SHREECEM.NS",
    "Grasim": "GRASIM.NS",
    "Hindalco": "HINDALCO.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Divi's": "DIVISLAB.NS",
    "Britannia": "BRITANNIA.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Tata Consultancy": "TCS.NS",
    "Nifty 50 Index": "^NSEI",
}

INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
}

# ── Helper Functions ──────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).fast_info
        return {
            "price": round(info.last_price, 2),
            "prev_close": round(info.previous_close, 2),
            "market_cap": info.market_cap,
        }
    except Exception:
        return {"price": None, "prev_close": None, "market_cap": None}

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["EMA_20"]  = close.ewm(span=20, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20

    # ATR
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - close.shift()).abs()
    lpc = (df["Low"]  - close.shift()).abs()
    df["ATR"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # Volume MA
    df["Vol_MA"] = df["Volume"].rolling(20).mean()

    return df

def ml_predict(df: pd.DataFrame) -> tuple[str, float, float]:
    df = add_indicators(df).dropna().copy()
    if len(df) < 100:
        return "Not enough data", 0.0, 0.0

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["RSI", "MACD", "MACD_Signal", "ATR",
                "SMA_20", "SMA_50", "EMA_20", "BB_Upper", "BB_Lower"]
    df = df.dropna(subset=features + ["Target"])

    X = df[features].values
    y = df["Target"].values

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc   = accuracy_score(y_test, model.predict(X_test))
    prob  = model.predict_proba(scaler.transform(X[-1:].reshape(1, -1)))[0][1]

    signal = "BUY 🟢" if prob >= 0.55 else ("SELL 🔴" if prob <= 0.45 else "HOLD 🟡")
    return signal, round(prob * 100, 1), round(acc * 100, 1)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Indian Market Analyser")
    st.markdown("---")

    stock_name = st.selectbox("Select Stock", list(NSE_STOCKS.keys()), index=0)
    ticker     = NSE_STOCKS[stock_name]
    period     = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    st.markdown("---")
    st.markdown("**Compare with Index**")
    show_index   = st.checkbox("Show Index Overlay", value=False)
    index_choice = st.selectbox("Index", list(INDICES.keys()), index=0)

    st.markdown("---")
    st.caption("Data: Yahoo Finance (NSE/BSE)\nBuilt by Anand Tembare")

# ── Header ────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f'<div class="header-title">📊 {stock_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="header-sub">NSE Ticker: {ticker} &nbsp;|&nbsp; Live Indian Market Data</div>', unsafe_allow_html=True)
with col_h2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ── Fetch Data ────────────────────────────────────────────────
with st.spinner(f"Fetching live data for {stock_name}..."):
    df   = fetch_stock_data(ticker, period)
    live = fetch_live_price(ticker)

if df.empty:
    st.error("Could not fetch data. Check your internet or try another stock.")
    st.stop()

df_ind = add_indicators(df)

# ── Key Metrics ───────────────────────────────────────────────
price     = live["price"] or float(df["Close"].iloc[-1])
prev      = live["prev_close"] or float(df["Close"].iloc[-2])
change    = price - prev
change_p  = (change / prev) * 100
high_52   = float(df["High"].max())
low_52    = float(df["Low"].min())
avg_vol   = int(df["Volume"].mean())
rsi_val   = float(df_ind["RSI"].iloc[-1]) if "RSI" in df_ind.columns else 0.0

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Live Price", f"₹{price:,.2f}", f"{change:+.2f} ({change_p:+.2f}%)")
m2.metric("52W High",   f"₹{high_52:,.2f}")
m3.metric("52W Low",    f"₹{low_52:,.2f}")
m4.metric("RSI (14)",   f"{rsi_val:.1f}", "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
m5.metric("Avg Volume", f"{avg_vol:,}")
m6.metric("Data Points", f"{len(df):,}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Indicators", "🤖 ML Prediction", "📋 Raw Data"])

# ── TAB 1: Price Chart ────────────────────────────────────────
with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_20"], name="SMA 20",
                             line=dict(color="#60a5fa", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_50"], name="SMA 50",
                             line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"], name="BB Upper",
                             line=dict(color="#a78bfa", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"], name="BB Lower",
                             line=dict(color="#a78bfa", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(167,139,250,0.05)"), row=1, col=1)

    # Volume
    colors = ["#22c55e" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef4444"
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Vol_MA"], name="Vol MA",
                             line=dict(color="#f59e0b", width=1.5)), row=2, col=1)

    # Index overlay
    if show_index:
        idx_df = fetch_stock_data(INDICES[index_choice], period)
        if not idx_df.empty:
            norm_idx   = idx_df["Close"] / idx_df["Close"].iloc[0]
            norm_stock = df["Close"] / df["Close"].iloc[0]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=norm_stock.index, y=norm_stock, name=stock_name,
                                      line=dict(color="#60a5fa", width=2)))
            fig2.add_trace(go.Scatter(x=norm_idx.index, y=norm_idx, name=index_choice,
                                      line=dict(color="#f59e0b", width=2)))
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#0f172a",
                               plot_bgcolor="#0f172a", height=300,
                               title="Normalised Performance vs Index",
                               legend=dict(bgcolor="#1e293b"))
            st.plotly_chart(fig2, use_container_width=True)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=560,
        showlegend=True,
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(gridcolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Indicators ─────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        # RSI Chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"],
                                     name="RSI", line=dict(color="#60a5fa", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought 70")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="Oversold 30")
        fig_rsi.update_layout(template="plotly_dark", paper_bgcolor="#0f172a",
                               plot_bgcolor="#0f172a", height=280, title="RSI (14)",
                               yaxis=dict(range=[0, 100], gridcolor="#1e293b"),
                               xaxis=dict(gridcolor="#1e293b"), margin=dict(t=40, b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_b:
        # MACD Chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"],
                                      name="MACD", line=dict(color="#60a5fa", width=2)))
        fig_macd.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_Signal"],
                                      name="Signal", line=dict(color="#f59e0b", width=1.5)))
        hist_colors = ["#22c55e" if v >= 0 else "#ef4444"
                       for v in (df_ind["MACD"] - df_ind["MACD_Signal"])]
        fig_macd.add_trace(go.Bar(x=df_ind.index,
                                   y=df_ind["MACD"] - df_ind["MACD_Signal"],
                                   name="Histogram", marker_color=hist_colors, opacity=0.6))
        fig_macd.update_layout(template="plotly_dark", paper_bgcolor="#0f172a",
                                plot_bgcolor="#0f172a", height=280, title="MACD",
                                yaxis=dict(gridcolor="#1e293b"),
                                xaxis=dict(gridcolor="#1e293b"), margin=dict(t=40, b=10))
        st.plotly_chart(fig_macd, use_container_width=True)

    # Indicator Summary Table
    st.markdown("#### 📋 Indicator Summary")
    latest = df_ind.iloc[-1]

    def rsi_signal(r):
        if r > 70: return "⚠️ Overbought"
        if r < 30: return "✅ Oversold (Buy zone)"
        return "➡️ Neutral"

    def macd_signal(m, s):
        return "✅ Bullish (MACD > Signal)" if m > s else "🔴 Bearish (MACD < Signal)"

    def sma_signal(price, sma):
        return "✅ Bullish (Above SMA)" if price > sma else "🔴 Bearish (Below SMA)"

    summary_data = {
        "Indicator": ["RSI (14)", "MACD", "SMA 20", "SMA 50", "ATR (14)"],
        "Value": [
            f"{latest['RSI']:.2f}",
            f"{latest['MACD']:.2f}",
            f"₹{latest['SMA_20']:.2f}",
            f"₹{latest['SMA_50']:.2f}",
            f"₹{latest['ATR']:.2f}",
        ],
        "Signal": [
            rsi_signal(latest["RSI"]),
            macd_signal(latest["MACD"], latest["MACD_Signal"]),
            sma_signal(float(df["Close"].iloc[-1]), latest["SMA_20"]),
            sma_signal(float(df["Close"].iloc[-1]), latest["SMA_50"]),
            "Volatility measure",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

# ── TAB 3: ML Prediction ──────────────────────────────────────
with tab3:
    st.markdown("#### 🤖 Machine Learning Price Direction Predictor")
    st.caption("Trained on RSI, MACD, ATR, SMA, Bollinger Band features using Random Forest (80/20 time-series split)")

    with st.spinner("Training model on historical data..."):
        signal, prob, acc = ml_predict(df)

    col_s1, col_s2, col_s3 = st.columns(3)

    css_class = "buy" if "BUY" in signal else ("sell" if "SELL" in signal else "hold")
    col_s1.markdown(f'<div class="signal-box {css_class}">Signal<br>{signal}</div>', unsafe_allow_html=True)
    col_s2.metric("Buy Probability", f"{prob}%",
                  "↑ Strong signal" if prob >= 60 else ("↓ Weak signal" if prob <= 40 else "Neutral"))
    col_s3.metric("Model Accuracy", f"{acc}%",
                  "on out-of-sample test data")

    st.markdown("---")
    st.markdown("#### How this works")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.info("**1. Features Used**\nRSI, MACD, MACD Signal, ATR, SMA 20, SMA 50, EMA 20, Bollinger Bands")
    with col_e2:
        st.info("**2. Model**\nRandom Forest with 100 trees trained on 80% historical data. Test on remaining 20%.")
    with col_e3:
        st.info("**3. Output**\nProbability that tomorrow's close will be HIGHER than today's close.")

    st.warning("⚠️ This is for educational purposes only. Not financial advice. Always do your own research before investing.")

    # Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Buy Probability %", "font": {"color": "#f8fafc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar":  {"color": "#1d4ed8"},
            "steps": [
                {"range": [0,  40], "color": "#450a0a"},
                {"range": [40, 55], "color": "#1c1917"},
                {"range": [55, 100], "color": "#052e16"},
            ],
            "threshold": {"line": {"color": "#f8fafc", "width": 3}, "value": prob},
        },
        number={"suffix": "%", "font": {"color": "#f8fafc", "size": 36}},
    ))
    fig_gauge.update_layout(paper_bgcolor="#0f172a", font_color="#f8fafc", height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── TAB 4: Raw Data ───────────────────────────────────────────
with tab4:
    st.markdown(f"#### {stock_name} — Last 100 Trading Days")
    display_df = df.tail(100).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df["Daily Change %"] = ((display_df["Close"] - display_df["Open"]) / display_df["Open"] * 100).round(2)
    display_df = display_df.round(2)
    st.dataframe(display_df[::-1], use_container_width=True)

    csv = display_df.to_csv().encode("utf-8")
    st.download_button("⬇️ Download CSV", csv,
                       file_name=f"{stock_name.replace(' ', '_')}_data.csv",
                       mime="text/csv")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:0.82rem;">'
    'Built by <b style="color:#60a5fa">Anand Tembare</b> &nbsp;|&nbsp; '
    'Data Science Portfolio Project &nbsp;|&nbsp; Live NSE/BSE Data via Yahoo Finance'
    '</div>',
    unsafe_allow_html=True,
)
