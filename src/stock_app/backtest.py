from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict[str, float | int]


def run_signal_backtest(
    feature_frame: pd.DataFrame,
    probabilities: pd.Series,
    threshold: float = 0.55,
    starting_capital: float = 100_000,
    transaction_cost_bps: float = 5,
    risk_per_trade: float = 0.01,
    atr_multiplier: float = 2.0,
    max_position_pct: float = 0.25,
) -> BacktestResult:
    aligned = feature_frame.loc[probabilities.index].copy()
    close = aligned["Close"].astype(float)
    atr = aligned.get("ATR_14", close * 0.02).astype(float).fillna(close * 0.02)
    probabilities = probabilities.astype(float).reindex(close.index).ffill().fillna(0.5)

    stop_pct = (atr_multiplier * atr / close).clip(lower=0.002)
    exposure_pct = (risk_per_trade / stop_pct).clip(upper=max_position_pct)
    desired_position = pd.Series(
        np.where(probabilities >= threshold, exposure_pct, 0.0),
        index=close.index,
        name="Exposure",
    )

    daily_return = close.pct_change().fillna(0)
    held_position = desired_position.shift(1).fillna(0)
    turnover = desired_position.diff().abs().fillna(desired_position.abs())
    transaction_cost = turnover * (transaction_cost_bps / 10_000)
    strategy_return = held_position * daily_return - transaction_cost
    buy_hold_return = daily_return

    equity = pd.DataFrame(
        {
            "Strategy": starting_capital * (1 + strategy_return).cumprod(),
            "Buy & Hold": starting_capital * (1 + buy_hold_return).cumprod(),
            "Strategy Return": strategy_return,
            "Buy & Hold Return": buy_hold_return,
            "Exposure": desired_position,
        },
        index=close.index,
    )
    equity["Drawdown"] = equity["Strategy"] / equity["Strategy"].cummax() - 1

    signals = pd.DataFrame(
        {
            "Close": close,
            "Buy_Probability": probabilities,
            "Signal": np.where(probabilities >= threshold, "Long", "Cash"),
            "Exposure": desired_position,
            "ATR_Stop": close - atr_multiplier * atr,
            "Forward_Return": aligned["Forward_Return"],
        },
        index=close.index,
    )

    return BacktestResult(
        equity=equity,
        signals=signals,
        metrics=performance_metrics(equity, starting_capital, turnover),
    )


def performance_metrics(equity: pd.DataFrame, starting_capital: float, turnover: pd.Series) -> dict[str, float | int]:
    returns = equity["Strategy Return"].fillna(0)
    buy_hold_returns = equity["Buy & Hold Return"].fillna(0)
    periods = max(len(returns), 1)
    annual_factor = 252

    final_value = float(equity["Strategy"].iloc[-1])
    buy_hold_value = float(equity["Buy & Hold"].iloc[-1])
    annual_return = (final_value / starting_capital) ** (annual_factor / periods) - 1
    buy_hold_annual_return = (buy_hold_value / starting_capital) ** (annual_factor / periods) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(annual_factor) if returns.std() else 0.0
    volatility = returns.std() * np.sqrt(annual_factor) if returns.std() else 0.0
    max_drawdown = float(equity["Drawdown"].min())
    active_returns = returns[returns != 0]
    win_rate = float((active_returns > 0).mean()) if len(active_returns) else 0.0
    positive = returns[returns > 0].sum()
    negative = abs(returns[returns < 0].sum())
    profit_factor = float(positive / negative) if negative else float("inf")
    trades = int((turnover > 0).sum())

    return {
        "final_value": final_value,
        "annual_return": float(annual_return),
        "buy_hold_annual_return": float(buy_hold_annual_return),
        "sharpe": float(sharpe),
        "volatility": float(volatility),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": trades,
    }

