from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "Return_1D",
    "Return_5D",
    "Return_21D",
    "Volatility_21D",
    "SMA_Ratio_20",
    "SMA_Ratio_50",
    "SMA_Ratio_200",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Bollinger_Position",
    "ATR_Pct",
    "Volume_Z",
    "Momentum_10D",
    "Momentum_21D",
    "Sentiment_Score",
]


def add_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["Return_1D"] = close.pct_change()
    df["Return_5D"] = close.pct_change(5)
    df["Return_21D"] = close.pct_change(21)
    df["Log_Return"] = np.log(close).diff()
    df["Volatility_21D"] = df["Log_Return"].rolling(21).std() * np.sqrt(252)

    for window in (10, 20, 50, 200):
        df[f"SMA_{window}"] = close.rolling(window).mean()
        df[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

    df["SMA_Ratio_20"] = close / df["SMA_20"] - 1
    df["SMA_Ratio_50"] = close / df["SMA_50"] - 1
    df["SMA_Ratio_200"] = close / df["SMA_200"] - 1

    df["RSI_14"] = relative_strength_index(close)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    band_width = (upper_band - lower_band).replace(0, np.nan)
    df["Bollinger_Position"] = (close - lower_band) / band_width

    df["ATR_14"] = average_true_range(high, low, close)
    df["ATR_Pct"] = df["ATR_14"] / close

    volume_mean = volume.rolling(21).mean()
    volume_std = volume.rolling(21).std().replace(0, np.nan)
    df["Volume_Z"] = (volume - volume_mean) / volume_std
    df["Momentum_10D"] = close / close.shift(10) - 1
    df["Momentum_21D"] = close / close.shift(21) - 1

    return df


def build_feature_frame(ohlcv: pd.DataFrame, horizon: int = 1, sentiment_score: float = 0.0) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    df = add_technical_indicators(ohlcv)
    df["Sentiment_Score"] = float(sentiment_score)
    df["Forward_Return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["Target"] = (df["Forward_Return"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=DEFAULT_FEATURES + ["Target", "Forward_Return", "ATR_14"])
    return df


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

