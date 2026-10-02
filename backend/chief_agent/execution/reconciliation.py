"""Position reconciliation and slippage monitoring.

Reconciliation
--------------
Continuously compares the internal position ledger with the broker's positions.
Any mismatch:

    * stops new entries (fail closed),
    * raises an alert,
    * enters RECONCILIATION mode,
    * **never auto-heals a position we do not recognise** - that requires a human.

Slippage monitor
----------------
Tracks expected vs actual fills by symbol, time bucket, order type, volatility
regime and size bucket. Persistent abnormal slippage reduces size and then
suspends the affected strategy.
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..logging_setup import get_logger
from ..settings import get_config_store
from ..timeutil import IST, minutes_from_open, now_ist

log = get_logger(__name__, component="reconciliation")


@dataclass
class Mismatch:
    kind: str
    instrument_key: str
    internal: float
    broker: float
    delta: float
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "instrument_key": self.instrument_key,
            "internal": self.internal,
            "broker": self.broker,
            "delta": self.delta,
            "detail": self.detail,
        }


@dataclass
class ReconciliationReport:
    ts: Any
    matched: bool
    mismatches: List[Mismatch] = field(default_factory=list)
    internal_positions: Dict[str, float] = field(default_factory=dict)
    broker_positions: Dict[str, float] = field(default_factory=dict)
    action_taken: str = "NONE"
    mode_entered: Optional[str] = None

    @property
    def action(self) -> str:
        """What the execution engine must do as a result of this check."""
        return self.action_taken

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "matched": self.matched,
            "mismatch_count": len(self.mismatches),
            "mismatches": [m.to_dict() for m in self.mismatches],
            "internal_positions": self.internal_positions,
            "broker_positions": self.broker_positions,
            "action_taken": self.action_taken,
            "mode_entered": self.mode_entered,
        }


class PositionReconciler:
    """Compares internal and broker positions; fails closed on any difference."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or (get_config_store().load("execution").get("reconciliation") or {})
        self._pending = False
        self._last: Optional[ReconciliationReport] = None
        self._history: List[ReconciliationReport] = []

    @property
    def pending(self) -> bool:
        """True while a mismatch is unresolved - the execution engine blocks on this."""
        return self._pending

    @property
    def last_report(self) -> Optional[ReconciliationReport]:
        return self._last

    def reconcile(
        self,
        internal: Mapping[str, float],
        broker: Mapping[str, float],
        *,
        tolerance: Optional[float] = None,
        mode: str = "PAPER",
    ) -> ReconciliationReport:
        tolerance = float(self.cfg.get("quantity_tolerance", 0) if tolerance is None else tolerance)
        mismatches: List[Mismatch] = []

        for instrument_key, expected in internal.items():
            actual = float(broker.get(instrument_key, 0) or 0)
            if abs(actual - float(expected)) > tolerance:
                kind = "QUANTITY_MISMATCH" if instrument_key in broker else "MISSING_AT_BROKER"
                mismatches.append(
                    Mismatch(
                        kind=kind,
                        instrument_key=instrument_key,
                        internal=float(expected),
                        broker=actual,
                        delta=actual - float(expected),
                        detail="Internal ledger and broker disagree",
                    )
                )

        for instrument_key, actual in broker.items():
            if instrument_key not in internal and abs(float(actual)) > tolerance:
                mismatches.append(
                    Mismatch(
                        kind="UNKNOWN_BROKER_POSITION",
                        instrument_key=instrument_key,
                        internal=0.0,
                        broker=float(actual),
                        delta=float(actual),
                        detail="Position exists at the broker that this system never opened. "
                               "It will NOT be auto-healed; a human must review it.",
                    )
                )

        matched = not mismatches
        action = "NONE"
        entered: Optional[str] = None
        if mismatches:
            # Fail closed: stop new entries, keep managing what we know.
            action = str(self.cfg.get("on_mismatch", {}).get("action", "halt_new_entries")).upper()
            entered = str(self.cfg.get("on_mismatch", {}).get("enter_mode", "RECONCILIATION")).upper()
            log.error(
                "position reconciliation mismatch",
                context={"count": len(mismatches), "kinds": sorted({m.kind for m in mismatches})},
            )

        self._pending = not matched
        report = ReconciliationReport(
            ts=now_ist(),
            matched=matched,
            mismatches=mismatches,
            internal_positions={k: float(v) for k, v in internal.items()},
            broker_positions={k: float(v) for k, v in broker.items()},
            action_taken=action,
            mode_entered=entered,
        )
        self._last = report
        self._history.append(report)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        return report

    def clear(self) -> None:
        """Manual override after a human has reviewed and resolved a mismatch."""
        log.warning("reconciliation mismatch cleared manually")
        self._pending = False

    def status(self) -> Dict[str, Any]:
        return {
            "pending": self._pending,
            "last_report": self._last.to_dict() if self._last else None,
            "checks_performed": len(self._history),
        }


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #
@dataclass
class SlippageRecord:
    ts: Any
    instrument_key: str
    symbol: str
    order_type: str
    expected_price: float
    actual_price: float
    quantity: int
    slippage_bps: float
    direction: str = ""
    atr_pct: Optional[float] = None
    trade_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "order_type": self.order_type,
            "expected_price": round(self.expected_price, 4),
            "actual_price": round(self.actual_price, 4),
            "quantity": self.quantity,
            "slippage_bps": round(self.slippage_bps, 2),
            "direction": self.direction,
            "trade_id": self.trade_id,
        }


@dataclass
class SlippageSummary:
    samples: int = 0
    median_bps: float = 0.0
    mean_bps: float = 0.0
    p90_bps: float = 0.0
    worst_bps: float = 0.0
    total_cost: float = 0.0
    level: str = "OK"          # OK | WARN | REDUCE_SIZE | SUSPEND
    by_symbol: Dict[str, float] = field(default_factory=dict)
    by_order_type: Dict[str, float] = field(default_factory=dict)
    by_time_bucket: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": self.samples,
            "median_bps": round(self.median_bps, 2),
            "mean_bps": round(self.mean_bps, 2),
            "p90_bps": round(self.p90_bps, 2),
            "worst_bps": round(self.worst_bps, 2),
            "total_cost": round(self.total_cost, 2),
            "level": self.level,
            "by_symbol": {k: round(v, 2) for k, v in self.by_symbol.items()},
            "by_order_type": {k: round(v, 2) for k, v in self.by_order_type.items()},
            "by_time_bucket": {k: round(v, 2) for k, v in self.by_time_bucket.items()},
        }


class SlippageMonitor:
    """Tracks execution slippage and escalates when it becomes abnormal."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (get_config_store().load("execution").get("slippage") or {}) if config is None else config
        self.cfg = cfg or {}
        self.records: List[SlippageRecord] = []
        self.suspended_symbols: Dict[str, Any] = {}

    def record(
        self,
        *,
        instrument_key: str,
        symbol: str,
        expected_price: float,
        actual_price: float,
        quantity: int,
        order_type: str = "MARKET",
        direction: str = "",
        atr_pct: Optional[float] = None,
        trade_id: str = "",
        ts: Optional[Any] = None,
    ) -> SlippageRecord:
        ts = ts or now_ist()
        slippage_bps = 0.0
        if expected_price > 0:
            raw = (actual_price - expected_price) / expected_price * 10_000.0
            # A long pays more (bad), a short receives less (bad).
            slippage_bps = raw if direction.upper() in ("", "LONG") else -raw
        record = SlippageRecord(
            ts=ts,
            instrument_key=instrument_key,
            symbol=symbol,
            order_type=order_type,
            expected_price=float(expected_price),
            actual_price=float(actual_price),
            quantity=int(quantity),
            slippage_bps=slippage_bps,
            direction=direction,
            atr_pct=atr_pct,
            trade_id=trade_id,
        )
        self.records.append(record)
        if len(self.records) > 20_000:
            self.records = self.records[-10_000:]
        return record

    def summary(self, window: Optional[int] = None) -> SlippageSummary:
        window = int(window or self.cfg.get("rolling_window_trades", 50))
        recent = self.records[-window:]
        summary = SlippageSummary()
        if not recent:
            return summary

        values = [r.slippage_bps for r in recent]
        summary.samples = len(values)
        summary.median_bps = statistics.median(values)
        summary.mean_bps = statistics.fmean(values)
        ordered = sorted(values)
        summary.p90_bps = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        summary.worst_bps = max(values)
        summary.total_cost = sum(
            max(0.0, (r.actual_price - r.expected_price) * r.quantity * (1 if r.direction.upper() != "SHORT" else -1))
            for r in recent
        )

        warn = float(self.cfg.get("warn_bps", 25))
        reduce = float(self.cfg.get("reduce_size_bps", 40))
        suspend = float(self.cfg.get("suspend_strategy_bps", 60))
        if summary.median_bps >= suspend:
            summary.level = "SUSPEND"
        elif summary.median_bps >= reduce:
            summary.level = "REDUCE_SIZE"
        elif summary.median_bps >= warn:
            summary.level = "WARN"

        def group(key: str) -> Dict[str, float]:
            buckets: Dict[str, List[float]] = {}
            for record in recent:
                if key == "symbol":
                    name = record.symbol
                elif key == "order_type":
                    name = record.order_type
                else:
                    local = record.ts.astimezone(IST) if hasattr(record.ts, "astimezone") else record.ts
                    try:
                        minute = minutes_from_open(local)
                        name = f"{int(minute // 30) * 30:03d}m"
                    except Exception:
                        name = "UNKNOWN"
                buckets.setdefault(name, []).append(record.slippage_bps)
            return {name: statistics.fmean(values) for name, values in buckets.items()}

        summary.by_symbol = group("symbol")
        summary.by_order_type = group("order_type")
        summary.by_time_bucket = group("time_bucket")
        return summary

    def should_suspend(self, symbol: str) -> bool:
        summary = self.summary()
        per_symbol = summary.by_symbol.get(symbol)
        if per_symbol is not None and per_symbol >= float(self.cfg.get("suspend_strategy_bps", 60)):
            return True
        return summary.level == "SUSPEND"

    def size_multiplier(self) -> float:
        """How much to scale position size given current execution quality."""
        level = self.summary().level
        if level == "SUSPEND":
            return 0.0
        if level == "REDUCE_SIZE":
            return 0.5
        return 1.0

    def status(self) -> Dict[str, Any]:
        return {"summary": self.summary().to_dict(), "records": len(self.records)}


__all__ = [
    "PositionReconciler",
    "ReconciliationReport",
    "Mismatch",
    "SlippageMonitor",
    "SlippageRecord",
    "SlippageSummary",
]
