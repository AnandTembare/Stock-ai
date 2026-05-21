from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stock_app.backtest import run_signal_backtest
from stock_app.data import fetch_yahoo_data, generate_demo_ohlcv, normalize_ohlcv
from stock_app.features import DEFAULT_FEATURES, add_technical_indicators, build_feature_frame
from stock_app.models import feature_importance, latest_probability, train_time_series_classifier
from stock_app.plots import confusion_matrix_chart, drawdown_chart, equity_curve_chart, price_chart, probability_chart, roc_chart
from stock_app.risk import position_size, risk_profile


st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
    }
    .small-note {color: #64748b; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_data(symbol: str, years: int) -> pd.DataFrame:
    return generate_demo_ohlcv(symbol=symbol, years=years)


@st.cache_data(show_spinner=False, ttl=900)
def load_yahoo_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return fetch_yahoo_data(symbol=symbol, period=period, interval=interval)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def num(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def app_sidebar() -> dict:
    with st.sidebar:
        st.title("Control Panel")
        source = st.selectbox("Data source", ["Demo data", "Yahoo Finance", "Upload CSV"])
        symbol = st.text_input("Symbol", value="AAPL").strip().upper() or "AAPL"

        demo_years = st.slider("Demo history", 2, 10, 6)
        period = st.selectbox("Yahoo period", ["1y", "2y", "5y", "10y", "max"], index=2)
        interval = st.selectbox("Yahoo interval", ["1d", "1wk"], index=0)
        uploaded_file = st.file_uploader("CSV file", type=["csv"])

        st.divider()
        model_name = st.selectbox(
            "Model",
            ["Soft Voting Ensemble", "Random Forest", "Gradient Boosting", "Logistic Regression"],
        )
        horizon = st.slider("Forecast horizon", 1, 10, 1)
        test_size = st.slider("Out-of-sample test size", 0.15, 0.40, 0.25, 0.05)
        threshold = st.slider("Entry probability", 0.50, 0.80, 0.56, 0.01)
        sentiment_score = st.slider("Sentiment input", -1.0, 1.0, 0.0, 0.05)

        st.divider()
        starting_capital = st.number_input("Starting capital", min_value=1_000, value=100_000, step=5_000)
        risk_per_trade = st.slider("Risk per trade", 0.0025, 0.05, 0.01, 0.0025)
        max_position_pct = st.slider("Max position", 0.05, 1.0, 0.25, 0.05)
        atr_multiplier = st.slider("ATR stop multiplier", 0.5, 5.0, 2.0, 0.25)
        transaction_cost_bps = st.slider("Transaction cost bps", 0, 50, 5)

    return {
        "source": source,
        "symbol": symbol,
        "demo_years": demo_years,
        "period": period,
        "interval": interval,
        "uploaded_file": uploaded_file,
        "model_name": model_name,
        "horizon": horizon,
        "test_size": test_size,
        "threshold": threshold,
        "sentiment_score": sentiment_score,
        "starting_capital": float(starting_capital),
        "risk_per_trade": float(risk_per_trade),
        "max_position_pct": float(max_position_pct),
        "atr_multiplier": float(atr_multiplier),
        "transaction_cost_bps": float(transaction_cost_bps),
    }


def load_selected_data(config: dict) -> pd.DataFrame:
    source = config["source"]
    if source == "Demo data":
        return load_demo_data(config["symbol"], config["demo_years"])

    if source == "Yahoo Finance":
        try:
            return load_yahoo_data(config["symbol"], config["period"], config["interval"])
        except Exception as exc:
            st.warning(f"Yahoo Finance could not load this symbol, so demo data is shown instead. Details: {exc}")
            return load_demo_data(config["symbol"], config["demo_years"])

    uploaded_file = config["uploaded_file"]
    if uploaded_file is None:
        st.info("Upload a CSV to use the CSV data source, or switch to demo data.")
        st.stop()

    return normalize_ohlcv(pd.read_csv(uploaded_file))


def main() -> None:
    config = app_sidebar()

    st.title("Stock Analysis & Prediction App")
    st.markdown(
        '<span class="small-note">Research dashboard for market indicators, ML direction signals, backtesting, and ATR-based risk sizing. Not financial advice.</span>',
        unsafe_allow_html=True,
    )

    raw_data = load_selected_data(config)
    feature_frame = build_feature_frame(raw_data, horizon=config["horizon"], sentiment_score=config["sentiment_score"])
    if len(feature_frame) < 160:
        st.error("The selected dataset is too small after indicator warm-up. Use a longer period or a larger CSV.")
        st.stop()

    feature_columns = [column for column in DEFAULT_FEATURES if column in feature_frame.columns]
    model_result = train_time_series_classifier(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        model_name=config["model_name"],
        test_size=config["test_size"],
    )

    latest_features = feature_frame[feature_columns].iloc[[-1]]
    latest_buy_probability = latest_probability(model_result.model, latest_features)
    latest_close = float(feature_frame["Close"].iloc[-1])
    latest_atr = float(feature_frame["ATR_14"].iloc[-1])
    signal = "Long setup" if latest_buy_probability >= config["threshold"] else "Cash watch"

    backtest = run_signal_backtest(
        feature_frame=feature_frame,
        probabilities=model_result.probabilities,
        threshold=config["threshold"],
        starting_capital=config["starting_capital"],
        transaction_cost_bps=config["transaction_cost_bps"],
        risk_per_trade=config["risk_per_trade"],
        atr_multiplier=config["atr_multiplier"],
        max_position_pct=config["max_position_pct"],
    )
    position_plan = position_size(
        capital=config["starting_capital"],
        entry_price=latest_close,
        atr=latest_atr,
        risk_per_trade=config["risk_per_trade"],
        atr_multiplier=config["atr_multiplier"],
        max_position_pct=config["max_position_pct"],
    )
    profile = risk_profile(
        latest_buy_probability,
        backtest.metrics["max_drawdown"],
        backtest.metrics["sharpe"],
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Latest close", num(latest_close))
    metric_cols[1].metric("Buy probability", pct(latest_buy_probability))
    metric_cols[2].metric("Signal", signal)
    metric_cols[3].metric("ROC-AUC", num(model_result.metrics["roc_auc"], 3) if model_result.metrics["roc_auc"] is not None else "n/a")
    metric_cols[4].metric("Sharpe", num(backtest.metrics["sharpe"]))
    metric_cols[5].metric("Risk profile", profile)

    chart_frame = add_technical_indicators(raw_data).dropna().tail(420)
    tabs = st.tabs(["Market", "Model", "Backtest", "Risk", "Data"])

    with tabs[0]:
        st.plotly_chart(price_chart(chart_frame, f"{config['symbol']} price and moving averages"), use_container_width=True)
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(probability_chart(model_result.probabilities, config["threshold"]), use_container_width=True)
        with right:
            latest_table = pd.DataFrame(
                {
                    "Metric": ["Close", "ATR 14", "RSI 14", "SMA 50 ratio", "SMA 200 ratio"],
                    "Value": [
                        latest_close,
                        latest_atr,
                        float(feature_frame["RSI_14"].iloc[-1]),
                        float(feature_frame["SMA_Ratio_50"].iloc[-1]),
                        float(feature_frame["SMA_Ratio_200"].iloc[-1]),
                    ],
                }
            )
            st.dataframe(latest_table, hide_index=True, use_container_width=True)

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.subheader("ROC Curve")
            st.plotly_chart(roc_chart(model_result.roc_curve, model_result.metrics["roc_auc"]), use_container_width=True)
        with right:
            st.subheader("Confusion Matrix")
            st.plotly_chart(confusion_matrix_chart(model_result.confusion_matrix), use_container_width=True)

        metrics_table = pd.DataFrame(
            [
                {"Metric": "Accuracy", "Value": pct(model_result.metrics["accuracy"])},
                {"Metric": "Precision", "Value": pct(model_result.metrics["precision"])},
                {"Metric": "Recall", "Value": pct(model_result.metrics["recall"])},
                {"Metric": "F1 score", "Value": num(model_result.metrics["f1"], 3)},
                {"Metric": "Model used", "Value": model_result.model_name},
                {"Metric": "Training rows", "Value": num(len(model_result.train_index), 0)},
                {"Metric": "Test rows", "Value": num(len(model_result.test_index), 0)},
            ]
        )
        st.dataframe(metrics_table, hide_index=True, use_container_width=True)

        importance = feature_importance(model_result.model, feature_columns)
        if not importance.empty:
            st.subheader("Feature Importance")
            st.bar_chart(importance.set_index("Feature"))
        else:
            st.caption("Feature importance is not available for this model configuration.")

    with tabs[2]:
        st.plotly_chart(equity_curve_chart(backtest.equity), use_container_width=True)
        st.plotly_chart(drawdown_chart(backtest.equity), use_container_width=True)
        cols = st.columns(6)
        cols[0].metric("Final value", f"${backtest.metrics['final_value']:,.0f}")
        cols[1].metric("Annual return", pct(backtest.metrics["annual_return"]))
        cols[2].metric("Buy-hold annual", pct(backtest.metrics["buy_hold_annual_return"]))
        cols[3].metric("Max drawdown", pct(backtest.metrics["max_drawdown"]))
        cols[4].metric("Win rate", pct(backtest.metrics["win_rate"]))
        cols[5].metric("Trades", num(backtest.metrics["trades"], 0))

        st.download_button(
            "Download backtest signals",
            data=backtest.signals.to_csv().encode("utf-8"),
            file_name=f"{config['symbol'].lower()}_signals.csv",
            mime="text/csv",
        )
        st.dataframe(backtest.signals.tail(25), use_container_width=True)

    with tabs[3]:
        st.subheader("ATR Position Plan")
        plan = position_plan.as_dict()
        risk_table = pd.DataFrame(
            [
                {"Item": "Entry price", "Value": f"${plan['entry_price']:,.2f}"},
                {"Item": "Stop price", "Value": f"${plan['stop_price']:,.2f}"},
                {"Item": "Shares", "Value": f"{plan['shares']:,}"},
                {"Item": "Exposure", "Value": f"${plan['exposure']:,.2f}"},
                {"Item": "Exposure percent", "Value": pct(plan["exposure_pct"])},
                {"Item": "Planned risk", "Value": f"${plan['planned_risk']:,.2f}"},
                {"Item": "Planned risk percent", "Value": pct(plan["planned_risk_pct"])},
            ]
        )
        st.dataframe(risk_table, hide_index=True, use_container_width=True)

        st.subheader("Guardrails")
        st.write(
            "- Keep position size capped when ATR widens.\n"
            "- Recheck the model after major market regime changes.\n"
            "- Treat high probability as a signal filter, not a guarantee.\n"
            "- Include broker fees, taxes, spread, and slippage before live use.\n"
            "- Use paper trading before connecting any real execution system."
        )

    with tabs[4]:
        st.subheader("Clean OHLCV Data")
        st.dataframe(raw_data.tail(200), use_container_width=True)
        st.subheader("Feature Frame")
        st.dataframe(feature_frame.tail(200), use_container_width=True)


if __name__ == "__main__":
    main()

