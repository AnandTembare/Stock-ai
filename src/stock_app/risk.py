from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor


@dataclass
class PositionPlan:
    entry_price: float
    stop_price: float
    stop_distance: float
    shares: int
    exposure: float
    exposure_pct: float
    planned_risk: float
    planned_risk_pct: float
    max_loss_per_share: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def position_size(
    capital: float,
    entry_price: float,
    atr: float,
    risk_per_trade: float = 0.01,
    atr_multiplier: float = 2.0,
    max_position_pct: float = 0.25,
) -> PositionPlan:
    if capital <= 0:
        raise ValueError("Capital must be positive.")
    if entry_price <= 0:
        raise ValueError("Entry price must be positive.")
    if atr <= 0:
        atr = entry_price * 0.02

    stop_distance = max(atr * atr_multiplier, entry_price * 0.005)
    stop_price = max(entry_price - stop_distance, 0)
    risk_budget = capital * risk_per_trade
    raw_shares = floor(risk_budget / stop_distance)
    max_shares = floor((capital * max_position_pct) / entry_price)
    shares = max(0, min(raw_shares, max_shares))
    exposure = shares * entry_price
    planned_risk = shares * stop_distance

    return PositionPlan(
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        stop_distance=float(stop_distance),
        shares=int(shares),
        exposure=float(exposure),
        exposure_pct=float(exposure / capital),
        planned_risk=float(planned_risk),
        planned_risk_pct=float(planned_risk / capital),
        max_loss_per_share=float(stop_distance),
    )


def risk_profile(probability: float, max_drawdown: float, sharpe: float) -> str:
    if probability >= 0.65 and sharpe >= 1 and max_drawdown > -0.2:
        return "Constructive"
    if probability >= 0.55 and max_drawdown > -0.3:
        return "Selective"
    return "Defensive"

