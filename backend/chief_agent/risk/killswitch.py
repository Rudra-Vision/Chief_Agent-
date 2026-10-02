"""Emergency kill switch.

Two layers, both fail-closed:

1. **Internal (instant, local).** A process-wide flag that stops new orders
   immediately, persisting a :class:`KillSwitchEvent` row and broadcasting a
   notification. This is what the giant STOP TRADING button uses. It works even
   when the broker is unreachable.

2. **Broker-side (Upstox Kill Switch, LIVE only).** An extra account-level
   backstop via ``POST /v2/user/kill-switch``. Documented platform rules are
   respected, never worked around: positions must be closed first, open orders
   are cancelled by the broker, a 12-hour cooling period applies before
   re-enabling, and the access token must be regenerated for the change to take
   effect.

Automatic engagement triggers are defined in ``config/risk.yaml`` and are wired
to this module by the monitoring layer.
"""

from __future__ import annotations

import datetime as dt
import enum
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import OperatingMode, get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="killswitch")


class KillSwitchSource(str, enum.Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    BROKER = "broker"
    SYSTEM = "system"


class KillSwitchTrigger(str, enum.Enum):
    MANUAL_BUTTON = "MANUAL_BUTTON"
    DAILY_MAX_LOSS = "DAILY_MAX_LOSS"
    HARD_DAILY_LOSS = "HARD_DAILY_LOSS"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    BROKER_FEED_UNAVAILABLE = "BROKER_FEED_UNAVAILABLE"
    DATA_FAILURE = "DATA_FAILURE"
    MARKET_DATA_STALE = "MARKET_DATA_STALE"
    DATABASE_CORRUPTED = "DATABASE_CORRUPTED"
    POSITION_MISMATCH = "POSITION_MISMATCH"
    ACCOUNT_MISMATCH = "ACCOUNT_MISMATCH"
    UNEXPECTED_LIVE_ORDER = "UNEXPECTED_LIVE_ORDER"
    RISK_ENGINE_FAILURE = "RISK_ENGINE_FAILURE"
    ORDER_REJECTION_LOOP = "ORDER_REJECTION_LOOP"
    ABNORMAL_SLIPPAGE = "ABNORMAL_SLIPPAGE"
    STRATEGY_OUT_OF_SPEC = "STRATEGY_OUT_OF_SPEC"
    LIVE_PREFLIGHT_FAILED = "LIVE_PREFLIGHT_FAILED"


@dataclass
class KillSwitchState:
    engaged: bool = False
    source: Optional[KillSwitchSource] = None
    trigger: Optional[KillSwitchTrigger] = None
    reason: str = ""
    engaged_at: Optional[dt.datetime] = None
    engaged_by: str = "system"
    broker_kill_switch_used: bool = False
    segments: List[str] = field(default_factory=list)
    release_allowed_after: Optional[dt.datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engaged": self.engaged,
            "source": self.source.value if self.source else None,
            "trigger": self.trigger.value if self.trigger else None,
            "reason": self.reason,
            "engaged_at": self.engaged_at.isoformat() if self.engaged_at else None,
            "engaged_by": self.engaged_by,
            "broker_kill_switch_used": self.broker_kill_switch_used,
            "segments": self.segments,
            "release_allowed_after": self.release_allowed_after.isoformat() if self.release_allowed_after else None,
        }


class KillSwitch:
    """Process-wide emergency stop."""

    #: Automatic triggers -> human-readable explanation. Used by the monitoring
    #: layer and shown on the dashboard.
    AUTO_TRIGGERS: Dict[KillSwitchTrigger, str] = {
        KillSwitchTrigger.DAILY_MAX_LOSS: "Daily maximum loss reached",
        KillSwitchTrigger.HARD_DAILY_LOSS: "Hard daily loss limit reached",
        KillSwitchTrigger.MAX_CONSECUTIVE_LOSSES: "Maximum consecutive losses reached",
        KillSwitchTrigger.BROKER_FEED_UNAVAILABLE: "Broker feed unavailable",
        KillSwitchTrigger.DATA_FAILURE: "Data failure detected",
        KillSwitchTrigger.MARKET_DATA_STALE: "Market-data feed is stale",
        KillSwitchTrigger.DATABASE_CORRUPTED: "Database operation failed",
        KillSwitchTrigger.POSITION_MISMATCH: "Position mismatch detected",
        KillSwitchTrigger.ACCOUNT_MISMATCH: "Account mismatch detected",
        KillSwitchTrigger.UNEXPECTED_LIVE_ORDER: "Unexpected live order detected",
        KillSwitchTrigger.RISK_ENGINE_FAILURE: "Risk engine failure",
        KillSwitchTrigger.ORDER_REJECTION_LOOP: "Order rejection loop",
        KillSwitchTrigger.ABNORMAL_SLIPPAGE: "Abnormal slippage detected",
        KillSwitchTrigger.STRATEGY_OUT_OF_SPEC: "Strategy behaving outside its specification",
        KillSwitchTrigger.LIVE_PREFLIGHT_FAILED: "LIVE preflight check failed",
    }

    def __init__(self) -> None:
        self._state = KillSwitchState()
        self._lock = threading.RLock()
        self._listeners: List[Callable[[KillSwitchState], None]] = []

    # ------------------------------------------------------------------ state
    @property
    def engaged(self) -> bool:
        with self._lock:
            return self._state.engaged

    @property
    def state(self) -> KillSwitchState:
        with self._lock:
            return KillSwitchState(**self._state.__dict__)

    def add_listener(self, listener: Callable[[KillSwitchState], None]) -> None:
        self._listeners.append(listener)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            try:
                listener(self.state)
            except Exception:  # pragma: no cover - a listener must never break the switch
                log.exception("kill switch listener failed")

    # ---------------------------------------------------------------- engage
    def engage(
        self,
        trigger: KillSwitchTrigger,
        *,
        reason: str = "",
        source: KillSwitchSource = KillSwitchSource.MANUAL,
        actor: str = "user",
        segments: Optional[Sequence[str]] = None,
        broker_confirmed: bool = False,
    ) -> KillSwitchState:
        with self._lock:
            if self._state.engaged:
                log.info("kill switch already engaged", context={"trigger": trigger.value})
                return self.state
            self._state = KillSwitchState(
                engaged=True,
                source=source,
                trigger=trigger,
                reason=reason or self.AUTO_TRIGGERS.get(trigger, trigger.value),
                engaged_at=now_ist(),
                engaged_by=actor,
                broker_kill_switch_used=broker_confirmed,
                segments=list(segments or []),
            )
        log.error(
            "KILL SWITCH ENGAGED",
            context={
                "trigger": trigger.value,
                "source": source.value,
                "reason": self._state.reason,
                "actor": actor,
            },
        )
        self._notify()
        return self.state

    def engage_automatically(
        self,
        trigger: KillSwitchTrigger,
        *,
        reason: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> KillSwitchState:
        state = self.engage(
            trigger,
            reason=reason,
            source=KillSwitchSource.AUTOMATIC,
            actor="risk_engine",
        )
        if context:
            log.warning("kill switch context", context=context)
        return state

    def engage_with_broker(
        self,
        trigger: KillSwitchTrigger,
        broker_risk: Any,
        *,
        reason: str = "",
        actor: str = "user",
        segments: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Engage locally, then attempt the Upstox account-level switch (LIVE only)."""
        settings = get_settings()
        local_state = self.engage(
            trigger, reason=reason, source=KillSwitchSource.MANUAL, actor=actor, segments=segments
        )
        result: Dict[str, Any] = {"local": local_state.to_dict(), "broker": None}

        if settings.effective_mode() is not OperatingMode.LIVE:
            result["broker"] = {
                "ok": False,
                "skipped": True,
                "reason": "broker kill switch is only used in LIVE mode",
            }
            return result

        try:
            broker_result = broker_risk.engage(segments=segments, reason=reason or trigger.value)
            result["broker"] = broker_result.to_dict()
            if broker_result.ok:
                with self._lock:
                    self._state.broker_kill_switch_used = True
                    self._state.segments = segments and list(segments) or self._state.segments
        except Exception as exc:  # pragma: no cover - defensive
            result["broker"] = {"ok": False, "error": str(exc)}
        return result

    # ---------------------------------------------------------------- release
    def release(self, actor: str = "user", broker_risk: Any = None) -> Dict[str, Any]:
        """Release the LOCAL switch. The broker switch has its own cooling rules."""
        with self._lock:
            if not self._state.engaged:
                return {"ok": False, "reason": "kill switch is not engaged"}
            if self._state.broker_kill_switch_used and self._state.release_allowed_after:
                if now_ist() < self._state.release_allowed_after:
                    return {
                        "ok": False,
                        "reason": "broker cooling period is still active",
                        "allowed_after": self._state.release_allowed_after.isoformat(),
                    }
            previous = KillSwitchState(**self._state.__dict__)
            self._state = KillSwitchState(engaged=False)

        result: Dict[str, Any] = {"ok": True, "released": True, "actor": actor, "previous": previous.to_dict()}
        settings = get_settings()
        if broker_risk is not None and settings.effective_mode() is OperatingMode.LIVE and previous.broker_kill_switch_used:
            try:
                release_result = broker_risk.release(segments=previous.segments or None, reason="manual release")
                result["broker"] = release_result.to_dict()
            except Exception as exc:  # pragma: no cover
                result["broker"] = {"ok": False, "error": str(exc)}
        log.warning("kill switch released", context=result)
        self._notify()
        return result

    # ------------------------------------------------------------------ guards
    def block_new_orders(self) -> bool:
        """The single question the execution engine asks before submitting."""
        return self.engaged

    def to_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()


# --------------------------------------------------------------------------- #
# Process singleton
# --------------------------------------------------------------------------- #
_kill_switch: Optional[KillSwitch] = None
_lock = threading.Lock()


def get_kill_switch() -> KillSwitch:
    global _kill_switch
    with _lock:
        if _kill_switch is None:
            _kill_switch = KillSwitch()
        return _kill_switch


def reset_kill_switch() -> None:
    """Test helper."""
    global _kill_switch
    with _lock:
        _kill_switch = None


__all__ = [
    "KillSwitch",
    "KillSwitchState",
    "KillSwitchSource",
    "KillSwitchTrigger",
    "get_kill_switch",
    "reset_kill_switch",
]
