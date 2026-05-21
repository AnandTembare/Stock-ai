# Stock Analysis & Prediction Streamlit App

A GitHub-ready Streamlit dashboard for stock analysis, directional prediction, backtesting, and risk sizing. It is built from the technical overview in this workspace, but packaged as a practical app that can run with demo data, Yahoo Finance data, or an uploaded CSV.

This project is for research and education. It is not financial advice.

## Features

- Interactive Streamlit dashboard with market, model, backtest, risk, and data tabs
- Demo-data mode so the app works without API keys or internet access
- Optional Yahoo Finance loader through `yfinance`
- CSV upload support for OHLCV datasets
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, volatility, momentum, and volume features
- Directional ML classifiers: logistic regression, random forest, gradient boosting, and soft-voting ensemble
- Time-series train/test split to reduce lookahead bias
- Classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve
- Trading backtest with probability threshold, transaction cost, dynamic exposure cap, Sharpe ratio, max drawdown, and equity curve
- Risk sizing module with ATR-based stop, risk per trade, max exposure, estimated shares, and capital at risk
- Dockerfile and GitHub Actions workflow for deployment and CI

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If you use macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Options

### Demo Data

Use this first to confirm the app works. It creates realistic synthetic OHLCV data with trend, volatility regimes, and volume spikes.

### Yahoo Finance

Select `Yahoo Finance` in the sidebar and enter a symbol such as `AAPL`, `MSFT`, `SPY`, or `RELIANCE.NS`. This requires internet access at runtime.

### CSV Upload

Upload a CSV with these columns:

```text
Date, Open, High, Low, Close, Volume
```

The parser also accepts lowercase column names and `Adj Close`.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── Dockerfile
├── pyproject.toml
├── README.md
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── src/stock_app/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   ├── risk.py
│   └── plots.py
└── tests/
    └── test_core.py
```

## Docker

```bash
docker build -t stock-streamlit-app .
docker run -p 8501:8501 stock-streamlit-app
```

Then open `http://localhost:8501`.

## Streamlit Cloud Deployment

1. Push this folder to GitHub.
2. Create a new Streamlit Cloud app from the repository.
3. Set the main file path to `app.py`.
4. Deploy.

No secrets are required for demo data or Yahoo Finance. If you later add paid market data or news APIs, store keys in Streamlit secrets.

## Development Notes

- The model target is whether the close price is higher after the selected forecast horizon.
- The split is chronological, not random, to better match real trading conditions.
- The backtest applies the model signal with a one-bar delay and subtracts transaction costs.
- Dynamic exposure is based on ATR stop distance and capped by the maximum position setting.
- Backtest numbers are illustrative and depend strongly on data quality, costs, slippage, and market regime.

## Roadmap

- Add Alpha Vantage or broker-data connectors
- Add model registry support with MLflow
- Add walk-forward retraining and rolling out-of-sample windows
- Add sentiment ingestion from news APIs
- Add strategy templates for conservative, balanced, trend, momentum, and aggressive profiles
- Add Kubernetes manifests and Prometheus metrics endpoint for production deployments

