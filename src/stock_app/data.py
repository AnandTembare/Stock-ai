from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _stable_symbol_seed(symbol: str, base_seed: int = 17) -> int:
    return base_seed + sum((index + 1) * ord(char) for index, char in enumerate(symbol.upper()))


def generate_demo_ohlcv(symbol: str = "DEMO", years: int = 6, seed: int | None = None) -> pd.DataFrame:
    """Create realistic synthetic OHLCV data for local demos and tests."""
    periods = max(252, int(years * 252))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    rng = np.random.default_rng(seed if seed is not None else _stable_symbol_seed(symbol))

    market_cycle = np.sin(np.linspace(0, 8 * np.pi, periods))
    volatility = 0.009 + 0.008 * (market_cycle > 0).astype(float) + rng.uniform(0, 0.004, periods)
    drift = 0.00025 + 0.00018 * np.sin(np.linspace(0, 3 * np.pi, periods))
    shocks = rng.normal(0, volatility, periods)
    log_returns = drift + shocks

    close = 100 * np.exp(np.cumsum(log_returns))
    overnight_gap = rng.normal(0, 0.0035, periods)
    open_ = np.r_[close[0], close[:-1]] * (1 + overnight_gap)
    intraday_range = np.abs(rng.normal(0.008, 0.004, periods))
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)
    volume_base = rng.lognormal(mean=14.2, sigma=0.35, size=periods)
    volume = volume_base * (1 + 8 * np.abs(log_returns))

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume.astype(int),
        },
        index=dates,
    )
    df.index.name = "Date"
    df.attrs["symbol"] = symbol.upper()
    return df


def fetch_yahoo_data(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance through yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to use the Yahoo Finance data source.") from exc

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError(f"No data returned for symbol {symbol!r}.")

    normalized = normalize_ohlcv(data)
    normalized.attrs["symbol"] = symbol.upper()
    return normalized


def normalize_ohlcv(data: pd.DataFrame, required: Iterable[str] = OHLCV_COLUMNS) -> pd.DataFrame:
    """Normalize common market-data formats into Date-indexed OHLCV columns."""
    if data.empty:
        raise ValueError("The provided dataset is empty.")

    df = data.copy()
    df = _flatten_market_columns(df)
    df = _normalize_date_index(df)

    rename_map = {}
    for column in df.columns:
        clean = str(column).strip().lower().replace("_", " ")
        if clean in {"open", "high", "low", "close", "volume"}:
            rename_map[column] = clean.title()
        elif clean in {"adj close", "adjusted close"} and "Close" not in rename_map.values():
            rename_map[column] = "Close"

    df = df.rename(columns=rename_map)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    df = df[list(required)].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.empty:
        raise ValueError("No valid OHLCV rows remain after cleaning.")

    df.index.name = "Date"
    return df


def _flatten_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    price_names = {"open", "high", "low", "close", "adj close", "volume"}
    for level in range(df.columns.nlevels):
        values = {str(value).strip().lower() for value in df.columns.get_level_values(level)}
        if values & price_names:
            flattened = df.copy()
            flattened.columns = df.columns.get_level_values(level)
            return flattened

    flattened = df.copy()
    flattened.columns = ["_".join(str(part) for part in column if str(part)) for column in df.columns]
    return flattened


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    candidate_names = {"date", "datetime", "timestamp", "time"}
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    date_column = next((lower_columns[name] for name in candidate_names if name in lower_columns), None)

    if date_column is not None:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column]).set_index(date_column)
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

    return df

