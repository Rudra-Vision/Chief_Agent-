"""Paper-trading engine.

Simulates the full order lifecycle against live (or simulated) prices without
ever touching the broker:

    order -> latency -> acknowledgement -> fill (possibly partial) -> position
          -> stop / target / trailing -> exit -> journal

Realism modelled (all configurable in ``config/execution.yaml``):

* submit/ack latency,
* market-order slippage and limit-order fill requirements,
* partial fills,
* rejections (with realistic reasons),
* bid-ask spread,
* stop triggers, including gap-through fills at the open,
* real transaction costs from the shared :class:`CostModel`.

The engine is deliberately conservative: a limit order only fills when the bar
actually trades through the limit, and a stop only fills when the bar trades at
or beyond the trigger.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..broker.upstox_market_data import Candle
from ..costs.transaction_costs import CostModel, Product
from ..logging_setup import get_logger
from ..settings import get_config_store
from ..timeutil import IST, now_ist

log = get_logger(__name__, component="paper")


def _parse_ts(value: Any) -> dt.datetime:
    """Parse an ISO timestamp from a fill record back into an IST datetime."""
    if isinstance(value, dt.datetime):
        return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    try:
        parsed = dt.datetime.fromisoformat(str(value))
        return parsed.astimezone(IST) if parsed.tzinfo else parsed.replace(tzinfo=IST)
    except (TypeError, ValueError):
        return now_ist()


@dataclass
class PaperOrder:
    order_id: str
    instrument_key: str
    symbol: str
    transaction_type: str            # BUY | SELL
    quantity: int
    order_type: str = "LIMIT"
    price: float = 0.0
    trigger_price: float = 0.0
    product: str = "I"
    tag: str = ""
    leg: str = "ENTRY"
    status: str = "PENDING_NEW"
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    submitted_at: Optional[dt.datetime] = None
    acknowledged_at: Optional[dt.datetime] = None
    completed_at: Optional[dt.datetime] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)
    rejection_reason: str = ""
    latency_ms: float = 0.0
    fees: float = 0.0
    slippage_cost: float = 0.0
    trade_id: str = ""
    strategy_version: str = ""

    @property
    def pending_quantity(self) -> int:
        return max(0, self.quantity - self.filled_quantity)

    @property
    def is_open(self) -> bool:
        return self.status in ("PENDING_NEW", "OPEN", "PARTIALLY_FILLED", "TRIGGER_PENDING")

    @property
    def is_complete(self) -> bool:
        return self.status == "COMPLETE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "pending_quantity": self.pending_quantity,
            "order_type": self.order_type,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "product": self.product,
            "tag": self.tag,
            "leg": self.leg,
            "status": self.status,
            "average_fill_price": round(self.average_fill_price, 4),
            "fees": round(self.fees, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "latency_ms": round(self.latency_ms, 1),
            "rejection_reason": self.rejection_reason,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "fill_count": len(self.fills),
        }


@dataclass
class PaperPosition:
    position_id: str
    trade_id: str
    instrument_key: str
    symbol: str
    sector: Optional[str]
    direction: str
    quantity: int
    entry_price: float
    entry_ts: dt.datetime
    initial_stop: float
    current_stop: float
    target_1: float
    target_2: float
    initial_risk_per_share: float
    strategy_version: str
    regime: Optional[str] = None
    entry_fees: float = 0.0
    entry_slippage: float = 0.0
    ltp: Optional[float] = None
    mae_price: float = 0.0
    mfe_price: float = 0.0
    remaining_fraction: float = 1.0
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    status: str = "OPEN"
    closed_at: Optional[dt.datetime] = None
    realized_pnl: float = 0.0
    exit_reason: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    partial_booked: bool = False

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    @property
    def risk_per_share(self) -> float:
        return max(self.initial_risk_per_share, 1e-9)

    @property
    def unrealized_pnl(self) -> float:
        if self.ltp is None:
            return 0.0
        quantity = self.quantity * self.remaining_fraction
        if self.direction == "LONG":
            return (self.ltp - self.entry_price) * quantity
        return (self.entry_price - self.ltp) * quantity

    @property
    def r_multiple(self) -> float:
        if self.ltp is None:
            return 0.0
        move = (self.ltp - self.entry_price) if self.direction == "LONG" else (self.entry_price - self.ltp)
        return move / self.risk_per_share

    @property
    def exposure(self) -> float:
        return (self.ltp or self.entry_price) * self.quantity * self.remaining_fraction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "trade_id": self.trade_id,
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "sector": self.sector,
            "direction": self.direction,
            "quantity": self.quantity,
            "remaining_fraction": round(self.remaining_fraction, 4),
            "entry_price": round(self.entry_price, 4),
            "entry_ts": self.entry_ts.isoformat(),
            "ltp": round(self.ltp, 4) if self.ltp else None,
            "initial_stop": round(self.initial_stop, 4),
            "current_stop": round(self.current_stop, 4),
            "target_1": round(self.target_1, 4),
            "target_2": round(self.target_2, 4),
            "risk_per_share": round(self.risk_per_share, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "r_multiple": round(self.r_multiple, 3),
            "mae_r": round(
                ((self.entry_price - self.mae_price) / self.risk_per_share)
                if self.direction == "LONG"
                else ((self.mae_price - self.entry_price) / self.risk_per_share),
                3,
            ),
            "mfe_r": round(
                ((self.mfe_price - self.entry_price) / self.risk_per_share)
                if self.direction == "LONG"
                else ((self.entry_price - self.mfe_price) / self.risk_per_share),
                3,
            ),
            "strategy_version": self.strategy_version,
            "regime": self.regime,
            "status": self.status,
            "opened_at": self.entry_ts.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_reason": self.exit_reason,
            "hold_minutes": round(
                ((self.closed_at or now_ist()) - self.entry_ts).total_seconds() / 60.0, 1
            ),
        }


@dataclass
class PaperAccountState:
    account_id: str = "paper-default"
    initial_capital: float = 500_000.0
    cash: float = 500_000.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    slippage_paid: float = 0.0
    peak_equity: float = 500_000.0

    def equity(self, open_positions_value: float = 0.0) -> float:
        """Cash plus the current mark-to-market value of open positions.

        Longs contribute ``+qty x ltp`` (cash already paid for them); shorts
        contribute ``-qty x ltp`` (cash was credited, the liability remains).
        """
        return self.cash + open_positions_value

    def to_dict(self, unrealized: float = 0.0, positions_value: float = 0.0) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "initial_capital": round(self.initial_capital, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(positions_value, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "equity": round(self.equity(positions_value), 2),
            "fees_paid": round(self.fees_paid, 2),
            "slippage_paid": round(self.slippage_paid, 2),
            "peak_equity": round(self.peak_equity, 2),
        }


class PaperTradingEngine:
    """In-memory + persistable paper broker."""

    def __init__(
        self,
        *,
        initial_capital: float = 500_000.0,
        cost_model: Optional[CostModel] = None,
        config: Optional[Dict[str, Any]] = None,
        rng_seed: int = 991,
    ) -> None:
        cfg = config or (get_config_store().load("execution") or {})
        self.cfg = cfg
        self.paper_cfg: Dict[str, Any] = cfg.get("paper", {}) or {}
        self.cost_model = cost_model or CostModel(cfg.get("costs"))
        self.rng = np.random.default_rng(rng_seed)

        self.account = PaperAccountState(
            initial_capital=float(initial_capital),
            cash=float(initial_capital),
            peak_equity=float(initial_capital),
        )
        self.orders: Dict[str, PaperOrder] = {}
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []

    # --------------------------------------------------------------- settings
    def _latency_ms(self) -> float:
        latency = self.paper_cfg.get("latency", {}) or {}
        mean = float(latency.get("submit_ms_mean", 120))
        stddev = float(latency.get("submit_ms_stddev", 45))
        return max(1.0, float(self.rng.normal(mean, stddev)))

    def _fill_cfg(self) -> Dict[str, Any]:
        return self.paper_cfg.get("fills", {}) or {}

    # ------------------------------------------------------------------ orders
    def place_order(
        self,
        *,
        instrument_key: str,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "LIMIT",
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "I",
        tag: str = "",
        leg: str = "ENTRY",
        trade_id: str = "",
        strategy_version: str = "",
        reference_price: Optional[float] = None,
        atr_pct: Optional[float] = None,
        force_immediate_fill: bool = False,
    ) -> PaperOrder:
        order = PaperOrder(
            order_id=f"PAPER-{uuid.uuid4().hex[:12].upper()}",
            instrument_key=instrument_key,
            symbol=symbol,
            transaction_type=transaction_type.upper(),
            quantity=int(quantity),
            order_type=order_type.upper(),
            price=float(price),
            trigger_price=float(trigger_price),
            product=product,
            tag=tag,
            leg=leg,
            trade_id=trade_id,
            strategy_version=strategy_version,
            submitted_at=now_ist(),
        )
        self.orders[order.order_id] = order

        # --- simulate rejection -------------------------------------------------
        reference = reference_price or price or 0.0
        fill_cfg = self._fill_cfg()
        rejection_probability = float(fill_cfg.get("rejection_probability", 0.002))
        if float(self.rng.random()) < rejection_probability:
            reasons = fill_cfg.get("reject_reasons") or ["RMS_BLOCKED"]
            order.status = "REJECTED"
            order.rejection_reason = str(self.rng.choice(reasons))
            order.completed_at = now_ist()
            self.rejections.append({"order_id": order.order_id, "reason": order.rejection_reason, "symbol": symbol})
            log.info("paper order rejected", context={"symbol": symbol, "reason": order.rejection_reason})
            return order

        # --- latency + acknowledgement -----------------------------------------
        order.latency_ms = self._latency_ms()
        order.acknowledged_at = now_ist() + dt.timedelta(milliseconds=order.latency_ms)
        order.status = "TRIGGER_PENDING" if order.order_type in ("SL", "SL-M") and not force_immediate_fill else "OPEN"

        # --- immediate fills -----------------------------------------------------
        if order.order_type == "MARKET" or force_immediate_fill:
            self._fill_market(order, reference, atr_pct)
        elif order.order_type == "LIMIT" and reference and self._limit_crosses(order, reference):
            self._fill_limit(order, reference, atr_pct)
        return order

    def _limit_crosses(self, order: PaperOrder, reference: float) -> bool:
        """A buy limit fills when the market trades at or below it."""
        if order.transaction_type == "BUY":
            return reference <= order.price
        return reference >= order.price

    def _fill_market(self, order: PaperOrder, reference: float, atr_pct: Optional[float]) -> None:
        if reference <= 0:
            order.status = "REJECTED"
            order.rejection_reason = "NO_REFERENCE_PRICE"
            self.rejections.append({"order_id": order.order_id, "reason": order.rejection_reason})
            return
        fill_cfg = self._fill_cfg()
        slippage_bps = float(fill_cfg.get("market_order_slippage_bps", 4))
        is_buy = order.transaction_type == "BUY"
        fraction = slippage_bps / 10_000.0
        fill_price = round(reference * (1 + fraction) if is_buy else reference * (1 - fraction), 4)

        quantity = order.quantity
        if float(self.rng.random()) < float(fill_cfg.get("partial_fill_probability", 0.15)):
            low, high = fill_cfg.get("partial_fill_fraction_range", [0.3, 0.8])
            fraction_filled = float(self.rng.uniform(low, high))
            quantity = max(1, int(math.floor(order.quantity * fraction_filled)))

        self._record_fill(order, quantity, fill_price, reference, atr_pct, "MARKET")
        order.status = "COMPLETE" if order.filled_quantity >= order.quantity else "PARTIALLY_FILLED"
        if order.filled_quantity >= order.quantity:
            order.completed_at = now_ist()

    def _fill_limit(self, order: PaperOrder, reference: float, atr_pct: Optional[float]) -> None:
        fill_cfg = self._fill_cfg()
        probability = float(fill_cfg.get("limit_order_fill_probability", 0.85))
        if float(self.rng.random()) > probability:
            order.status = "OPEN"
            return
        quantity = order.quantity
        if float(self.rng.random()) < float(fill_cfg.get("partial_fill_probability", 0.15)):
            low, high = fill_cfg.get("partial_fill_fraction_range", [0.3, 0.8])
            quantity = max(1, int(math.floor(order.quantity * float(self.rng.uniform(low, high)))))
        # Limit orders fill at the limit (or the reference if it is better).
        fill_price = min(order.price, reference) if order.transaction_type == "BUY" else max(order.price, reference)
        self._record_fill(order, quantity, round(fill_price, 4), reference, atr_pct, "LIMIT")
        order.status = "COMPLETE" if order.filled_quantity >= order.quantity else "PARTIALLY_FILLED"
        if order.filled_quantity >= order.quantity:
            order.completed_at = now_ist()

    def _record_fill(
        self,
        order: PaperOrder,
        quantity: int,
        fill_price: float,
        reference: float,
        atr_pct: Optional[float],
        fill_kind: str,
    ) -> None:
        previous = order.average_fill_price * order.filled_quantity
        order.filled_quantity += quantity
        order.average_fill_price = (previous + fill_price * quantity) / max(1, order.filled_quantity)

        cost = self.cost_model.compute(
            price=fill_price,
            quantity=quantity,
            side=order.transaction_type,
            product=order.product,
            is_stop_exit=order.leg in ("STOP", "EXIT") and order.order_type in ("SL", "SL-M"),
            atr_pct=atr_pct,
            slippage_bps_override=0.0,
        )
        slippage = self.cost_model.slippage_amount(
            fill_price, quantity, atr_pct=atr_pct,
            is_stop_exit=order.leg in ("STOP", "EXIT") and order.order_type in ("SL", "SL-M"),
        )
        order.fees += cost.charges_only
        order.slippage_cost += slippage
        self.account.fees_paid += cost.charges_only
        self.account.slippage_paid += slippage
        self.account.cash -= cost.charges_only

        slippage_bps = ((fill_price - reference) / reference * 10_000.0) if reference else 0.0
        if order.transaction_type == "SELL" and reference:
            slippage_bps = ((reference - fill_price) / reference * 10_000.0)
        order.fills.append(
            {
                "quantity": quantity,
                "price": fill_price,
                "reference_price": reference,
                "slippage_bps": round(slippage_bps, 4),
                "fees": round(cost.charges_only, 4),
                "kind": fill_kind,
                "ts": now_ist().isoformat(),
            }
        )

    # -------------------------------------------------------------- positions
    def open_position_from_order(
        self,
        order: PaperOrder,
        *,
        direction: str,
        stop_price: float,
        target_1: float,
        target_2: float,
        sector: Optional[str] = None,
        regime: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> Optional[PaperPosition]:
        if not order.is_complete and order.filled_quantity <= 0:
            return None
        quantity = order.filled_quantity
        entry_price = order.average_fill_price
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return None

        position = PaperPosition(
            position_id=f"PP-{uuid.uuid4().hex[:10].upper()}",
            trade_id=order.trade_id or f"PT-{uuid.uuid4().hex[:10].upper()}",
            instrument_key=order.instrument_key,
            symbol=order.symbol,
            sector=sector,
            direction=direction.upper(),
            quantity=quantity,
            entry_price=entry_price,
            entry_ts=_parse_ts(order.fills[-1]["ts"]) if order.fills else (order.acknowledged_at or now_ist()),
            initial_stop=stop_price,
            current_stop=stop_price,
            target_1=target_1,
            target_2=target_2,
            initial_risk_per_share=risk_per_share,
            strategy_version=order.strategy_version,
            regime=regime,
            entry_fees=order.fees,
            entry_slippage=order.slippage_cost,
            ltp=entry_price,
            mae_price=entry_price,
            mfe_price=entry_price,
            features=features or {},
        )
        self.positions[position.position_id] = position
        # Cash effect: buying consumes cash, shorting adds it.
        notional = entry_price * quantity
        self.account.cash += (-notional if position.direction == "LONG" else notional)
        log.info(
            "paper position opened",
            context={
                "symbol": position.symbol,
                "direction": position.direction,
                "quantity": quantity,
                "entry": entry_price,
                "stop": stop_price,
            },
        )
        return position

    def mark_to_market(self, prices: Mapping[str, float], ts: Optional[dt.datetime] = None) -> None:
        """Update LTP, excursions and trailing stops for every open position."""
        for position in self.positions.values():
            if not position.is_open:
                continue
            price = prices.get(position.instrument_key)
            if price is None or price <= 0:
                continue
            position.ltp = float(price)
            if position.direction == "LONG":
                position.mae_price = min(position.mae_price, float(price))
                position.mfe_price = max(position.mfe_price, float(price))
            else:
                position.mae_price = max(position.mae_price, float(price))
                position.mfe_price = min(position.mfe_price, float(price))
        self.account.peak_equity = max(self.account.peak_equity, self.equity())

    def evaluate_exits(
        self,
        candles: Mapping[str, Candle],
        *,
        ts: dt.datetime,
        strategy_config: Optional[Dict[str, Any]] = None,
        minutes_into_session: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Check stops / targets / trailing / square-off against the latest bar.

        Returns the list of closed-trade records produced by this call.
        """
        closed: List[Dict[str, Any]] = []
        cfg = strategy_config or {}
        square_off_at = 375 - int((cfg.get("exits") or {}).get("force_square_off_minutes_before_close", 10))

        for position in list(self.positions.values()):
            if not position.is_open:
                continue
            bar = candles.get(position.instrument_key)
            if bar is None:
                continue

            exit_price: Optional[float] = None
            reason = ""
            direction = position.direction

            # forced square-off / time stop
            if minutes_into_session is not None and minutes_into_session >= square_off_at:
                exit_price, reason = float(bar.close), "SQUARE_OFF"
            else:
                time_stop = (cfg.get("exits") or {}).get("time_stop_minutes") or 0
                if time_stop:
                    held = (ts - position.entry_ts).total_seconds() / 60.0
                    if held >= float(time_stop):
                        exit_price, reason = float(bar.close), "TIME_STOP"

            if exit_price is None:
                if direction == "LONG":
                    if bar.low <= position.current_stop:
                        exit_price, reason = min(float(bar.open), position.current_stop), "STOP_LOSS"
                    elif bar.high >= position.target_1:
                        exit_price, reason = float(position.target_1), "TARGET_1"
                else:
                    if bar.high >= position.current_stop:
                        exit_price, reason = max(float(bar.open), position.current_stop), "STOP_LOSS"
                    elif bar.low <= position.target_1:
                        exit_price, reason = float(position.target_1), "TARGET_1"

            if exit_price is None:
                self._apply_trailing(position, bar, cfg)
                continue

            closed.append(self.close_position(position, exit_price, ts, reason))
        return closed

    def _apply_trailing(self, position: PaperPosition, bar: Candle, cfg: Dict[str, Any]) -> None:
        trailing = cfg.get("trailing") or {}
        if not trailing.get("enabled", False):
            return
        model = trailing.get("model", "none")
        if model == "none":
            return
        risk = position.risk_per_share
        close = float(bar.close)
        r = (close - position.entry_price) / risk if position.direction == "LONG" else (position.entry_price - close) / risk

        be_r = float(trailing.get("breakeven_after_r", 0) or 0)
        if be_r and r >= be_r:
            position.current_stop = (
                max(position.current_stop, position.entry_price)
                if position.direction == "LONG"
                else min(position.current_stop, position.entry_price)
            )
        if r < float(trailing.get("trail_activation_r", 0) or 0):
            return
        if model == "atr":
            atr_value = cfg.get("_atr") or 0
            if atr_value:
                multiplier = float(trailing.get("atr_multiplier", 2.0))
                candidate = close - multiplier * atr_value if position.direction == "LONG" else close + multiplier * atr_value
                position.current_stop = (
                    max(position.current_stop, candidate)
                    if position.direction == "LONG"
                    else min(position.current_stop, candidate)
                )
        elif model == "r_multiple":
            step = float(trailing.get("r_multiple_step", 0.5)) or 0.5
            locked_r = math.floor(r / step) * step
            candidate = (
                position.entry_price + locked_r * risk
                if position.direction == "LONG"
                else position.entry_price - locked_r * risk
            )
            position.current_stop = (
                max(position.current_stop, candidate)
                if position.direction == "LONG"
                else min(position.current_stop, candidate)
            )

    def close_position(
        self,
        position: PaperPosition,
        exit_price: float,
        ts: dt.datetime,
        reason: str,
        *,
        fraction: float = 1.0,
    ) -> Dict[str, Any]:
        """Close (all or part of) a position and return the trade record."""
        fraction = max(0.0, min(1.0, float(fraction)))
        quantity = max(1, int(math.floor(position.quantity * position.remaining_fraction * fraction)))
        is_stop = reason in ("STOP_LOSS", "HARD_DAILY_STOP")
        exit_cost = self.cost_model.compute(
            price=exit_price,
            quantity=quantity,
            side="SELL" if position.direction == "LONG" else "BUY",
            product=Product.INTRADAY.value,
            is_stop_exit=is_stop,
        )
        exit_slippage = self.cost_model.slippage_amount(exit_price, quantity, is_stop_exit=is_stop)
        self.account.fees_paid += exit_cost.charges_only
        self.account.slippage_paid += exit_slippage
        self.account.cash -= exit_cost.charges_only

        gross = (
            (exit_price - position.entry_price) * quantity
            if position.direction == "LONG"
            else (position.entry_price - exit_price) * quantity
        )
        allocated_entry_fees = position.entry_fees * (quantity / max(1, position.quantity))
        allocated_entry_slippage = position.entry_slippage * (quantity / max(1, position.quantity))
        fees = allocated_entry_fees + exit_cost.charges_only
        slippage = allocated_entry_slippage + exit_slippage
        net = gross - fees

        # Settle the trade in cash. A long close returns the sale proceeds; a
        # short close pays for the buy-back. P&L therefore flows through cash,
        # which is what keeps `equity = cash + position value` exactly correct
        # (no double counting of the cost basis).
        if position.direction == "LONG":
            self.account.cash += exit_price * quantity
        else:
            self.account.cash -= exit_price * quantity

        position.realized_pnl += net
        self.account.realized_pnl += net
        position.remaining_fraction -= fraction
        closed_fraction = fraction
        if position.remaining_fraction <= 1e-9:
            position.status = "CLOSED"
            position.closed_at = ts
            position.exit_reason = reason

        risk_amount = position.initial_risk_per_share * quantity
        holding = (ts - position.entry_ts).total_seconds() / 60.0
        record = {
            "trade_id": position.trade_id,
            "position_id": position.position_id,
            "instrument_key": position.instrument_key,
            "symbol": position.symbol,
            "sector": position.sector,
            "direction": position.direction,
            "quantity": quantity,
            "entry_price": round(position.entry_price, 4),
            "exit_price": round(exit_price, 4),
            "entry_ts": position.entry_ts.isoformat(),
            "exit_ts": ts.isoformat(),
            "holding_minutes": round(holding, 2),
            "gross_pnl": round(gross, 2),
            "fees": round(fees, 2),
            "slippage_cost": round(slippage, 2),
            "net_pnl": round(net, 2),
            "initial_risk": round(risk_amount, 2),
            "r_multiple": round(net / risk_amount, 4) if risk_amount > 0 else 0.0,
            "mae_r": round(
                ((position.entry_price - position.mae_price) / position.risk_per_share)
                if position.direction == "LONG"
                else ((position.mae_price - position.entry_price) / position.risk_per_share),
                4,
            ),
            "mfe_r": round(
                ((position.mfe_price - position.entry_price) / position.risk_per_share)
                if position.direction == "LONG"
                else ((position.entry_price - position.mfe_price) / position.risk_per_share),
                4,
            ),
            "exit_reason": reason,
            "regime_at_entry": position.regime,
            "strategy_version": position.strategy_version,
            "features": position.features,
            "partial": closed_fraction < 0.999,
            "mode": "PAPER",
        }
        if closed_fraction >= 0.999 or position.status == "CLOSED":
            self.closed_trades.append(record)
        log.info(
            "paper position closed",
            context={
                "symbol": position.symbol,
                "reason": reason,
                "net_pnl": record["net_pnl"],
                "r_multiple": record["r_multiple"],
            },
        )
        return record

    # ---------------------------------------------------------------- queries
    def open_positions(self) -> List[PaperPosition]:
        return [p for p in self.positions.values() if p.is_open]

    def total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions())

    def positions_value(self) -> float:
        """Signed mark-to-market value of open positions (see equity docs)."""
        total = 0.0
        for position in self.open_positions():
            price = position.ltp or position.entry_price
            quantity = position.quantity * position.remaining_fraction
            total += quantity * price if position.direction == "LONG" else -quantity * price
        return total

    def equity(self) -> float:
        return self.account.equity(self.positions_value())

    def snapshot(self) -> Dict[str, Any]:
        unrealized = self.total_unrealized()
        positions_value = self.positions_value()
        equity = self.account.equity(positions_value)
        return {
            "account": self.account.to_dict(unrealized, positions_value),
            "open_positions": [p.to_dict() for p in self.open_positions()],
            "open_position_count": len(self.open_positions()),
            "orders_today": len(self.orders),
            "rejections": len(self.rejections),
            "closed_trades": len(self.closed_trades),
            "drawdown_pct": round(
                (equity - self.account.peak_equity) / self.account.peak_equity, 6
            ) if self.account.peak_equity else 0.0,
        }


__all__ = ["PaperTradingEngine", "PaperOrder", "PaperPosition", "PaperAccountState"]
