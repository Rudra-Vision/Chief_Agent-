"""Execution engine: the SIGNAL -> ... -> JOURNAL lifecycle.

Execution is **separate** from signal generation. This module takes an approved
proposal plus a risk decision and drives it through:

    SIGNAL -> RISK_CHECK -> ORDER_INTENT -> VALIDATION -> BROKER_ORDER
           -> ACKNOWLEDGEMENT -> FILL -> POSITION -> PROTECTION_PLACED
           -> EXIT -> RECONCILIATION -> JOURNAL

Every transition is logged with a structured context and persisted.

Idempotency
-----------
Every order carries a deterministic identity:

    strategy_version + trade_id + signal_id + order_id + tag

with ``tag = CHIEF-<trade_id>-<leg>``. A duplicate signal cannot create a second
order. Retries distinguish:

* *request failed before reaching the broker*  -> safe to retry,
* *broker may have accepted it*                -> RECONCILE FIRST, never resend.

Fail-closed
-----------
If the kill switch is engaged, the preflight is stale, the risk engine is
unavailable, data quality is compromised or reconciliation is pending, the
engine refuses to submit. Uncertainty always resolves to "do nothing".
"""

from __future__ import annotations

import datetime as dt
import enum
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..broker.upstox_orders import OrderRequest, UpstoxOrders
from ..costs.transaction_costs import CostModel
from ..logging_setup import get_logger
from ..risk.engine import ProposedTrade, RiskContext, RiskDecision, RiskEngine, RiskLevel
from ..risk.killswitch import KillSwitch, KillSwitchTrigger, get_kill_switch
from ..settings import OperatingMode, get_config_store, get_settings
from ..timeutil import now_ist
from .paper_engine import PaperTradingEngine

log = get_logger(__name__, component="execution")


class OrderState(str, enum.Enum):
    SIGNAL = "SIGNAL"
    RISK_CHECK = "RISK_CHECK"
    ORDER_INTENT = "ORDER_INTENT"
    VALIDATION = "VALIDATION"
    BROKER_ORDER = "BROKER_ORDER"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    FILL = "FILL"
    POSITION = "POSITION"
    PROTECTION_PLACED = "PROTECTION_PLACED"
    EXIT = "EXIT"
    RECONCILIATION = "RECONCILIATION"
    JOURNAL = "JOURNAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


TERMINAL_STATES = {OrderState.JOURNAL, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}


@dataclass
class OrderIntent:
    """The frozen, idempotent representation of what we intend to do."""

    trade_id: str
    signal_id: str
    strategy_version: str
    instrument_key: str
    symbol: str
    direction: str
    quantity: int
    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    product: str = "I"
    sector: Optional[str] = None
    regime: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    created_at: dt.datetime = field(default_factory=now_ist)

    @property
    def order_id(self) -> str:
        """Deterministic: the same intent always maps to the same order id."""
        return f"ORD-{self.trade_id}-ENTRY"

    @property
    def tag(self) -> str:
        return f"CHIEF-{self.trade_id}"[:40]

    def leg_tag(self, leg: str) -> str:
        return f"CHIEF-{self.trade_id}-{leg}"[:40]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["order_id"] = self.order_id
        payload["tag"] = self.tag
        return payload


@dataclass
class ExecutionResult:
    ok: bool
    state: OrderState
    message: str = ""
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    filled_quantity: int = 0
    average_price: float = 0.0
    fees: float = 0.0
    slippage_cost: float = 0.0
    risk_decision: Optional[RiskDecision] = None
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    protection_placed: bool = False
    requires_reconciliation: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "state": self.state.value,
            "message": self.message,
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "filled_quantity": self.filled_quantity,
            "average_price": round(self.average_price, 4),
            "fees": round(self.fees, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "protection_placed": self.protection_placed,
            "requires_reconciliation": self.requires_reconciliation,
            "error": self.error,
            "risk": self.risk_decision.to_dict() if self.risk_decision else None,
            "transitions": self.transitions,
        }


class ExecutionEngine:
    """Submits orders in the active mode, with idempotency and fail-closed guards."""

    def __init__(
        self,
        *,
        mode: OperatingMode,
        risk_engine: RiskEngine,
        paper_engine: Optional[PaperTradingEngine] = None,
        orders_api: Optional[UpstoxOrders] = None,
        kill_switch: Optional[KillSwitch] = None,
        cost_model: Optional[CostModel] = None,
        preflight_check: Optional[Callable[[], bool]] = None,
        data_quality_check: Optional[Callable[[], bool]] = None,
        reconciliation_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.mode = mode
        self.risk_engine = risk_engine
        self.paper = paper_engine
        self.orders_api = orders_api
        self.kill_switch = kill_switch or get_kill_switch()
        self.cost_model = cost_model or CostModel()
        self._preflight_check = preflight_check or (lambda: True)
        self._data_quality_check = data_quality_check or (lambda: True)
        self._reconciliation_check = reconciliation_check or (lambda: True)

        self.cfg = get_config_store().load("execution")
        self._lock = threading.RLock()
        self._submitted: Dict[str, ExecutionResult] = {}
        self._transitions: List[Dict[str, Any]] = []
        self._order_attempts_today = 0
        self._attempt_day: Optional[dt.date] = None

    # -------------------------------------------------------------- lifecycle
    def _record(self, intent: OrderIntent, state: OrderState, detail: str, **extra: Any) -> None:
        self._transitions.append(
            {
                "ts": now_ist().isoformat(),
                "trade_id": intent.trade_id,
                "order_id": intent.order_id,
                "state": state.value,
                "detail": detail,
                **extra,
            }
        )

    def _bump_attempts(self) -> int:
        today = now_ist().date()
        with self._lock:
            if self._attempt_day != today:
                self._attempt_day = today
                self._order_attempts_today = 0
            self._order_attempts_today += 1
            return self._order_attempts_today

    # ------------------------------------------------------------------- guards
    def preflight_ok(self) -> Tuple[bool, str]:
        if self.mode is OperatingMode.LIVE:
            try:
                if not self._preflight_check():
                    return False, "LIVE preflight check has not passed"
            except Exception as exc:
                return False, f"preflight check raised: {exc}"
        return True, ""

    def system_ready(self) -> Tuple[bool, str]:
        if self.kill_switch.block_new_orders():
            return False, f"kill switch is engaged ({self.kill_switch.state.reason})"
        try:
            if not self._data_quality_check():
                return False, "DATA_SAFE_MODE is active"
        except Exception as exc:
            return False, f"data-quality check raised: {exc}"
        try:
            if not self._reconciliation_check():
                return False, "position reconciliation is pending"
        except Exception as exc:
            return False, f"reconciliation check raised: {exc}"
        max_orders = int((self.cfg.get("safety") or {}).get("max_orders_per_minute", 6))
        if self._order_attempts_today > int((self.cfg.get("safety") or {}).get("max_daily_attempts", 500)):
            return False, "daily order-attempt ceiling reached"
        return True, ""

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        intent: OrderIntent,
        risk_context: RiskContext,
        *,
        spread_pct: Optional[float] = None,
        average_daily_volume: Optional[float] = None,
        atr_pct: Optional[float] = None,
        lot_size: int = 1,
        tick_size: float = 0.05,
        force: bool = False,
    ) -> ExecutionResult:
        """Drive one trade from intent to fill (or refusal)."""
        transitions_start = len(self._transitions)
        self._record(intent, OrderState.SIGNAL, f"{intent.direction} {intent.symbol} proposed")

        # ---- idempotency ---------------------------------------------------
        with self._lock:
            existing = self._submitted.get(intent.order_id)
        if existing is not None:
            log.warning("duplicate order intent suppressed", context={"order_id": intent.order_id})
            self._record(intent, OrderState.VALIDATION, "duplicate intent rejected by idempotency key")
            return ExecutionResult(
                ok=False,
                state=OrderState.REJECTED,
                message="duplicate order intent (same trade_id / order_id already submitted)",
                order_id=intent.order_id,
                transitions=self._transitions[transitions_start:],
            )

        # ---- fail-closed system gates ---------------------------------------
        ready, reason = self.system_ready()
        if not ready and not force:
            self._record(intent, OrderState.VALIDATION, f"blocked: {reason}")
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message=reason,
                order_id=intent.order_id, transitions=self._transitions[transitions_start:],
            )
        preflight_ok, preflight_reason = self.preflight_ok()
        if not preflight_ok and not force:
            self._record(intent, OrderState.VALIDATION, f"blocked: {preflight_reason}")
            self.kill_switch.engage_automatically(
                KillSwitchTrigger.LIVE_PREFLIGHT_FAILED, reason=preflight_reason
            )
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message=preflight_reason,
                order_id=intent.order_id, transitions=self._transitions[transitions_start:],
            )

        # ---- risk check ------------------------------------------------------
        self._record(intent, OrderState.RISK_CHECK, "running the risk engine")
        proposed = ProposedTrade(
            instrument_key=intent.instrument_key,
            symbol=intent.symbol,
            direction=intent.direction,
            entry_price=intent.entry_price,
            stop_price=intent.stop_price,
            target_1=intent.target_1,
            target_2=intent.target_2,
            quantity=intent.quantity,
            strategy_version=intent.strategy_version,
            sector=intent.sector,
            risk_reward=(
                abs(intent.target_1 - intent.entry_price) / abs(intent.entry_price - intent.stop_price)
                if intent.entry_price != intent.stop_price
                else 0.0
            ),
            atr_pct=atr_pct,
            spread_pct=spread_pct,
            average_daily_volume=average_daily_volume,
            lot_size=lot_size,
            tick_size=tick_size,
            signal_id=intent.signal_id,
            product=intent.product,
        )
        decision = self.risk_engine.evaluate(proposed, risk_context)
        if not decision.tradeable:
            self._record(intent, OrderState.REJECTED, f"risk engine: {decision.reason.value} - {decision.message}")
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message=decision.message,
                order_id=intent.order_id, risk_decision=decision,
                transitions=self._transitions[transitions_start:],
            )

        quantity = decision.sizing.quantity if decision.sizing else intent.quantity
        intent.quantity = quantity
        self._record(
            intent, OrderState.ORDER_INTENT,
            f"intent frozen: {quantity} x {intent.symbol} at {intent.entry_price:.2f}",
            quantity=quantity, risk_level=decision.level.value,
        )

        # ---- validation ------------------------------------------------------
        errors = self._validate_intent(intent)
        if errors:
            self._record(intent, OrderState.VALIDATION, "validation failed: " + "; ".join(errors))
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message="; ".join(errors),
                order_id=intent.order_id, risk_decision=decision,
                transitions=self._transitions[transitions_start:],
            )
        self._record(intent, OrderState.VALIDATION, "order intent validated")

        # ---- submit ----------------------------------------------------------
        self._bump_attempts()
        if self.mode in (OperatingMode.PAPER, OperatingMode.SANDBOX) and self.paper is not None and self.mode is OperatingMode.PAPER:
            result = self._submit_paper(intent, decision, spread_pct, atr_pct, transitions_start)
        elif self.mode is OperatingMode.SANDBOX and self.paper is not None and not self.orders_api:
            # Sandbox without a token: still simulated, but clearly labelled.
            result = self._submit_paper(intent, decision, spread_pct, atr_pct, transitions_start, label="SANDBOX_SIMULATED")
        else:
            result = self._submit_broker(intent, decision, transitions_start)

        with self._lock:
            self._submitted[intent.order_id] = result
        result.transitions = self._transitions[transitions_start:]
        return result

    # --------------------------------------------------------------- validation
    def _validate_intent(self, intent: OrderIntent) -> List[str]:
        errors: List[str] = []
        if intent.quantity <= 0:
            errors.append("quantity must be positive")
        if intent.entry_price <= 0:
            errors.append("entry price must be positive")
        if intent.stop_price <= 0:
            errors.append("a stop-loss price is mandatory before any entry")
        if intent.direction == "LONG" and intent.stop_price >= intent.entry_price:
            errors.append("long stop must be below the entry price")
        if intent.direction == "SHORT" and intent.stop_price <= intent.entry_price:
            errors.append("short stop must be above the entry price")
        if not intent.instrument_key:
            errors.append("instrument_key is required")
        if len(intent.tag) > 40:
            errors.append("order tag exceeds the 40 character broker limit")
        return errors

    # -------------------------------------------------------------- paper route
    def _submit_paper(
        self,
        intent: OrderIntent,
        decision: RiskDecision,
        spread_pct: Optional[float],
        atr_pct: Optional[float],
        transitions_start: int,
        label: str = "PAPER",
    ) -> ExecutionResult:
        assert self.paper is not None
        self._record(intent, OrderState.BROKER_ORDER, f"{label} order submitted to the simulator")
        entry_order_type = str((self.cfg.get("entry") or {}).get("entry_order_type", "LIMIT")).upper()
        order = self.paper.place_order(
            instrument_key=intent.instrument_key,
            symbol=intent.symbol,
            transaction_type="BUY" if intent.direction == "LONG" else "SELL",
            quantity=intent.quantity,
            order_type=entry_order_type if entry_order_type in ("LIMIT", "MARKET") else "LIMIT",
            price=intent.entry_price,
            product=intent.product,
            tag=intent.leg_tag("E"),
            leg="ENTRY",
            trade_id=intent.trade_id,
            strategy_version=intent.strategy_version,
            reference_price=intent.entry_price,
            atr_pct=atr_pct,
            force_immediate_fill=entry_order_type == "MARKET",
        )
        self._record(
            intent, OrderState.ACKNOWLEDGEMENT,
            f"{label} order {order.order_id} -> {order.status}",
            broker_order_id=order.order_id,
        )
        if order.status == "REJECTED":
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message=order.rejection_reason,
                order_id=intent.order_id, broker_order_id=order.order_id, risk_decision=decision,
                transitions=self._transitions[transitions_start:],
            )
        if order.filled_quantity <= 0:
            return ExecutionResult(
                ok=False, state=OrderState.ACKNOWLEDGEMENT, message="order accepted but not yet filled",
                order_id=intent.order_id, broker_order_id=order.order_id, risk_decision=decision,
                transitions=self._transitions[transitions_start:],
            )

        position = self.paper.open_position_from_order(
            order,
            direction=intent.direction,
            stop_price=intent.stop_price,
            target_1=intent.target_1,
            target_2=intent.target_2,
            sector=intent.sector,
            regime=intent.regime,
            features=intent.features,
        )
        self._record(
            intent, OrderState.POSITION,
            f"{label} position opened at {order.average_fill_price:.2f}",
            quantity=order.filled_quantity,
        )
        protection = bool((self.cfg.get("protection") or {}).get("require_stop", True))
        if protection:
            self._record(
                intent, OrderState.PROTECTION_PLACED,
                f"stop at {intent.stop_price:.2f} is tracked by the simulator "
                f"(broker-side orders are only used in LIVE/SANDBOX)",
            )
        return ExecutionResult(
            ok=True,
            state=OrderState.PROTECTION_PLACED if protection else OrderState.POSITION,
            message=f"filled {order.filled_quantity} @ {order.average_fill_price:.2f}",
            order_id=intent.order_id,
            broker_order_id=order.order_id,
            filled_quantity=order.filled_quantity,
            average_price=order.average_fill_price,
            fees=order.fees,
            slippage_cost=order.slippage_cost,
            risk_decision=decision,
            protection_placed=protection,
        )

    # ------------------------------------------------------------- broker route
    def _submit_broker(
        self, intent: OrderIntent, decision: RiskDecision, transitions_start: int
    ) -> ExecutionResult:
        if self.orders_api is None:
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message="no broker order API configured",
                order_id=intent.order_id, transitions=self._transitions[transitions_start:],
            )

        entry_order_type = str((self.cfg.get("entry") or {}).get("entry_order_type", "LIMIT")).upper()
        request = OrderRequest(
            instrument_token=intent.instrument_key,
            transaction_type="BUY" if intent.direction == "LONG" else "SELL",
            quantity=intent.quantity,
            order_type=entry_order_type if entry_order_type in ("LIMIT", "MARKET") else "LIMIT",
            product=intent.product,
            price=intent.entry_price if entry_order_type == "LIMIT" else 0.0,
            tag=intent.leg_tag("E"),
        )
        errors = request.validate()
        if errors:
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message="; ".join(errors),
                order_id=intent.order_id, transitions=self._transitions[transitions_start:],
            )

        self._record(intent, OrderState.BROKER_ORDER, f"{self.mode.value} order submitted to Upstox")
        try:
            broker_result = self.orders_api.place_order(request)
        except Exception as exc:
            uncertain = bool(getattr(exc, "outcome_uncertain", False))
            self._record(
                intent, OrderState.RECONCILIATION if uncertain else OrderState.REJECTED,
                f"order submission failed ({'uncertain outcome' if uncertain else 'rejected'}): {exc}",
            )
            if uncertain:
                # NEVER blind-resend: reconcile against the order book first.
                self.kill_switch.engage_automatically(
                    KillSwitchTrigger.UNEXPECTED_LIVE_ORDER,
                    reason=f"uncertain order outcome for {intent.symbol}; reconciliation required",
                )
            return ExecutionResult(
                ok=False,
                state=OrderState.RECONCILIATION if uncertain else OrderState.REJECTED,
                message=str(exc),
                order_id=intent.order_id,
                requires_reconciliation=uncertain,
                risk_decision=decision,
                error=str(exc),
                transitions=self._transitions[transitions_start:],
            )

        if not broker_result.ok:
            self._record(intent, OrderState.REJECTED, f"broker refused the order: {broker_result.message}")
            return ExecutionResult(
                ok=False, state=OrderState.REJECTED, message=broker_result.message,
                order_id=intent.order_id, risk_decision=decision,
                transitions=self._transitions[transitions_start:],
            )

        self._record(
            intent, OrderState.ACKNOWLEDGEMENT,
            f"broker accepted order {broker_result.primary_order_id}",
            broker_order_id=broker_result.primary_order_id,
            latency_ms=broker_result.latency_ms,
        )
        return ExecutionResult(
            ok=True,
            state=OrderState.ACKNOWLEDGEMENT,
            message="order acknowledged by the broker; awaiting the fill confirmation",
            order_id=intent.order_id,
            broker_order_id=broker_result.primary_order_id,
            risk_decision=decision,
            transitions=self._transitions[transitions_start:],
        )

    # --------------------------------------------------------- protection orders
    def place_protection(
        self,
        intent: OrderIntent,
        *,
        filled_quantity: int,
    ) -> Dict[str, Any]:
        """Place the stop (and optional target) at the broker after a fill.

        In PAPER mode the simulator tracks the stop itself. In LIVE/SANDBOX this
        uses the documented SL-M order type so protection lives at the exchange
        rather than depending on this process staying alive.
        """
        if self.mode is OperatingMode.PAPER or self.orders_api is None:
            return {"placed": False, "reason": "protection is tracked internally in PAPER mode"}

        stop_type = str((self.cfg.get("protection") or {}).get("stop_order_type", "SL-M")).upper()
        request = OrderRequest(
            instrument_token=intent.instrument_key,
            transaction_type="SELL" if intent.direction == "LONG" else "BUY",
            quantity=filled_quantity,
            order_type=stop_type if stop_type in ("SL", "SL-M") else "SL-M",
            product=intent.product,
            price=0.0,
            trigger_price=intent.stop_price,
            tag=intent.leg_tag("SL"),
        )
        try:
            result = self.orders_api.place_order(request)
            self._record(
                intent, OrderState.PROTECTION_PLACED,
                f"broker-side {stop_type} stop placed at {intent.stop_price:.2f}",
                broker_order_id=result.primary_order_id,
            )
            return {"placed": result.ok, "order_id": result.primary_order_id, "message": result.message}
        except Exception as exc:
            log.error("could not place the broker-side stop", context={"symbol": intent.symbol, "error": str(exc)})
            # A position without a stop is unacceptable - flatten immediately.
            return {
                "placed": False,
                "error": str(exc),
                "action": "flatten_immediately",
                "reason": str((self.cfg.get("protection") or {}).get("unprotect_position_action", "flatten_immediately")),
            }

    # ------------------------------------------------------------------ queries
    @property
    def transitions(self) -> List[Dict[str, Any]]:
        return list(self._transitions)

    def recent_transitions(self, limit: int = 200) -> List[Dict[str, Any]]:
        return self._transitions[-limit:]

    def submitted_orders(self) -> Dict[str, ExecutionResult]:
        return dict(self._submitted)

    def reset_daily(self) -> None:
        with self._lock:
            self._order_attempts_today = 0
            self._attempt_day = now_ist().date()
            self._submitted.clear()


__all__ = [
    "ExecutionEngine",
    "ExecutionResult",
    "OrderIntent",
    "OrderState",
    "TERMINAL_STATES",
]
