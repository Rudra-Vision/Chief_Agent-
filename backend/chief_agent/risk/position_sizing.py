"""Position sizing.

No fixed arbitrary quantities, ever. Quantity is derived from:

    RiskCapital = AccountEquity x RiskPerTrade
    Quantity    = RiskCapital / |Entry - Stop|

…then reduced by every applicable constraint (exposure cap, margin, cash,
liquidity, correlation) and finally rounded DOWN to the exchange lot/tick rule.

Rounding is always DOWN so the realised risk can never exceed the intended risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..indicators.core import round_to_tick
from ..logging_setup import get_logger

log = get_logger(__name__, component="position_sizing")


@dataclass
class SizingConstraints:
    """Everything that can reduce the naive size."""

    account_equity: float
    risk_pct: float
    max_risk_pct: float = 0.005
    entry_price: float = 0.0
    stop_price: float = 0.0
    lot_size: int = 1
    tick_size: float = 0.05
    max_position_exposure_pct: float = 0.25
    max_gross_exposure_pct: float = 1.0
    available_cash: Optional[float] = None
    current_gross_exposure: float = 0.0
    max_total_open_risk_pct: float = 0.0075
    current_open_risk_pct: float = 0.0
    average_daily_volume: Optional[float] = None
    max_participation_pct: float = 0.01
    margin_per_share: Optional[float] = None
    price_circuit_pct: Optional[float] = None


@dataclass
class SizingResult:
    quantity: int
    risk_amount: float
    risk_pct: float
    notional: float
    binding_constraint: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantity": self.quantity,
            "risk_amount": round(self.risk_amount, 2),
            "risk_pct": round(self.risk_pct, 6),
            "notional": round(self.notional, 2),
            "binding_constraint": self.binding_constraint,
            "steps": self.steps,
            "warnings": self.warnings,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


def compute_position_size(constraints: SizingConstraints) -> SizingResult:
    """Risk-first position sizing with a full audit trail."""
    steps: List[Dict[str, Any]] = []
    warnings: List[str] = []

    equity = float(constraints.account_equity)
    entry = float(constraints.entry_price)
    stop = float(constraints.stop_price)

    if equity <= 0:
        return SizingResult(0, 0.0, 0.0, 0.0, "no_equity", steps, warnings, True, "account equity is zero")
    if entry <= 0:
        return SizingResult(0, 0.0, 0.0, 0.0, "no_price", steps, warnings, True, "entry price is not positive")

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return SizingResult(
            0, 0.0, 0.0, 0.0, "zero_stop_distance", steps, warnings, True,
            "stop distance is zero - a trade without defined risk is never taken",
        )

    # --- 1. risk budget -----------------------------------------------------
    risk_pct = min(float(constraints.risk_pct), float(constraints.max_risk_pct))
    if risk_pct < constraints.risk_pct:
        warnings.append(
            f"risk per trade capped at {constraints.max_risk_pct:.2%} (requested {constraints.risk_pct:.2%})"
        )
    risk_capital = equity * risk_pct
    naive_quantity = risk_capital / risk_per_share
    steps.append(
        {
            "step": "risk_budget",
            "equity": round(equity, 2),
            "risk_pct": risk_pct,
            "risk_capital": round(risk_capital, 2),
            "risk_per_share": round(risk_per_share, 4),
            "quantity": math.floor(naive_quantity),
        }
    )
    quantity = math.floor(naive_quantity)

    # --- 2. per-position exposure cap ---------------------------------------
    max_position_notional = equity * float(constraints.max_position_exposure_pct)
    exposure_quantity = math.floor(max_position_notional / entry) if entry > 0 else 0
    if exposure_quantity < quantity:
        quantity = exposure_quantity
        binding = "max_position_exposure"
        steps.append({"step": "position_exposure_cap", "limit_notional": round(max_position_notional, 2),
                      "quantity": quantity})
    else:
        binding = "risk_budget"

    # --- 3. gross exposure cap ----------------------------------------------
    gross_headroom_pct = float(constraints.max_gross_exposure_pct) - float(constraints.current_gross_exposure)
    gross_headroom = max(0.0, equity * gross_headroom_pct)
    gross_quantity = math.floor(gross_headroom / entry) if entry > 0 else 0
    if gross_quantity < quantity:
        quantity = gross_quantity
        binding = "max_gross_exposure"
        steps.append({"step": "gross_exposure_cap", "headroom": round(gross_headroom, 2), "quantity": quantity})

    # --- 4. aggregate open risk cap -----------------------------------------
    open_risk_headroom = max(0.0, (float(constraints.max_total_open_risk_pct) - float(constraints.current_open_risk_pct)) * equity)
    open_risk_quantity = math.floor(open_risk_headroom / risk_per_share)
    if open_risk_quantity < quantity:
        quantity = open_risk_quantity
        binding = "max_total_open_risk"
        steps.append({"step": "open_risk_cap", "headroom_amount": round(open_risk_headroom, 2), "quantity": quantity})

    # --- 5. cash / margin ----------------------------------------------------
    if constraints.available_cash is not None:
        cash = float(constraints.available_cash)
        cash_quantity = math.floor(cash / entry) if entry > 0 else 0
        if constraints.margin_per_share:
            cash_quantity = math.floor(cash / float(constraints.margin_per_share))
        if cash_quantity < quantity:
            quantity = cash_quantity
            binding = "available_cash"
            steps.append({"step": "cash_cap", "available": round(cash, 2), "quantity": quantity})

    # --- 6. liquidity / participation ---------------------------------------
    if constraints.average_daily_volume:
        max_participation = float(constraints.average_daily_volume) * float(constraints.max_participation_pct)
        liquidity_quantity = math.floor(max_participation)
        if liquidity_quantity < quantity:
            quantity = liquidity_quantity
            binding = "liquidity"
            steps.append({"step": "liquidity_cap", "max_participation": round(max_participation, 0), "quantity": quantity})

    # --- 7. lot rounding (always DOWN) --------------------------------------
    lot = max(1, int(constraints.lot_size or 1))
    rounded = (quantity // lot) * lot
    if rounded != quantity:
        steps.append({"step": "lot_rounding", "lot_size": lot, "before": quantity, "after": rounded})
        quantity = rounded

    if quantity <= 0:
        return SizingResult(
            0, 0.0, 0.0, 0.0, binding, steps, warnings, True,
            f"sizing produced zero quantity (binding constraint: {binding})",
        )

    actual_risk = quantity * risk_per_share
    return SizingResult(
        quantity=quantity,
        risk_amount=actual_risk,
        risk_pct=actual_risk / equity,
        notional=quantity * entry,
        binding_constraint=binding,
        steps=steps,
        warnings=warnings,
    )


def stop_and_target_prices(
    entry: float,
    stop: float,
    direction: str,
    tick_size: float = 0.05,
) -> Dict[str, float]:
    """Round stop/target to valid ticks in the direction that preserves risk."""
    if direction.upper() == "LONG":
        return {
            "stop": round_to_tick(stop, tick_size, "down"),
            "entry": round_to_tick(entry, tick_size, "nearest"),
        }
    return {
        "stop": round_to_tick(stop, tick_size, "up"),
        "entry": round_to_tick(entry, tick_size, "nearest"),
    }


__all__ = ["SizingConstraints", "SizingResult", "compute_position_size", "stop_and_target_prices"]
