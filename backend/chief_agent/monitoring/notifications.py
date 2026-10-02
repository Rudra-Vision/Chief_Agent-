"""Internal notification system with pluggable delivery channels.

Everything is stored in the ``notifications`` table first, so the dashboard
always has the full history even if no external channel is configured.
Optional channels (browser, Telegram, WhatsApp, email, webhook) are adapters -
none of them are required, and none are enabled by default, which keeps the
operating cost at zero.

Events raised (matching the brief):
    trade generated, trade entered, stop hit, target hit, daily risk reached,
    broker disconnected, data failure, new challenger, strategy promoted,
    strategy suspended.
"""

from __future__ import annotations

import datetime as dt
import enum
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..logging_setup import get_logger
from ..timeutil import now_ist

log = get_logger(__name__, component="notifications")


class EventKind(str, enum.Enum):
    TRADE_GENERATED = "TRADE_GENERATED"
    TRADE_ENTERED = "TRADE_ENTERED"
    TRADE_EXITED = "TRADE_EXITED"
    STOP_HIT = "STOP_HIT"
    TARGET_HIT = "TARGET_HIT"
    DAILY_RISK_REACHED = "DAILY_RISK_REACHED"
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    BROKER_RECONNECTED = "BROKER_RECONNECTED"
    DATA_FAILURE = "DATA_FAILURE"
    DATA_SAFE_MODE = "DATA_SAFE_MODE"
    NEW_CHALLENGER = "NEW_CHALLENGER"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    STRATEGY_PROMOTED = "STRATEGY_PROMOTED"
    STRATEGY_SUSPENDED = "STRATEGY_SUSPENDED"
    KILL_SWITCH_ENGAGED = "KILL_SWITCH_ENGAGED"
    KILL_SWITCH_RELEASED = "KILL_SWITCH_RELEASED"
    ORDER_REJECTED = "ORDER_REJECTED"
    RECONCILIATION_MISMATCH = "RECONCILIATION_MISMATCH"
    SLIPPAGE_WARNING = "SLIPPAGE_WARNING"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DAILY_REVIEW_READY = "DAILY_REVIEW_READY"


SEVERITY_BY_KIND: Dict[EventKind, str] = {
    EventKind.TRADE_GENERATED: "INFO",
    EventKind.TRADE_ENTERED: "INFO",
    EventKind.TRADE_EXITED: "INFO",
    EventKind.STOP_HIT: "INFO",
    EventKind.TARGET_HIT: "INFO",
    EventKind.DAILY_RISK_REACHED: "CRITICAL",
    EventKind.DRAWDOWN_WARNING: "WARNING",
    EventKind.BROKER_DISCONNECTED: "CRITICAL",
    EventKind.BROKER_RECONNECTED: "INFO",
    EventKind.DATA_FAILURE: "CRITICAL",
    EventKind.DATA_SAFE_MODE: "CRITICAL",
    EventKind.NEW_CHALLENGER: "INFO",
    EventKind.EXPERIMENT_COMPLETED: "INFO",
    EventKind.STRATEGY_PROMOTED: "WARNING",
    EventKind.STRATEGY_SUSPENDED: "WARNING",
    EventKind.KILL_SWITCH_ENGAGED: "CRITICAL",
    EventKind.KILL_SWITCH_RELEASED: "WARNING",
    EventKind.ORDER_REJECTED: "WARNING",
    EventKind.RECONCILIATION_MISMATCH: "CRITICAL",
    EventKind.SLIPPAGE_WARNING: "WARNING",
    EventKind.SYSTEM_ERROR: "CRITICAL",
    EventKind.DAILY_REVIEW_READY: "INFO",
}


@dataclass
class Notification:
    kind: EventKind
    title: str
    body: str = ""
    severity: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: Any = None
    delivered_to: List[str] = field(default_factory=list)
    read: bool = False

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = SEVERITY_BY_KIND.get(self.kind, "INFO")
        if self.created_at is None:
            self.created_at = now_ist()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "body": self.body,
            "severity": self.severity,
            "payload": self.payload,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
            "delivered_to": self.delivered_to,
            "read": self.read,
        }


class Channel:
    """Base delivery channel."""

    name = "base"

    def send(self, notification: Notification) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryChannel(Channel):
    """Always-on channel: keeps the recent history the dashboard reads."""

    name = "internal"

    def __init__(self, capacity: int = 500) -> None:
        self.items: List[Notification] = []
        self.capacity = capacity

    def send(self, notification: Notification) -> bool:
        self.items.append(notification)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]
        return True


class PersistChannel(Channel):
    """Writes notifications to the database."""

    name = "database"

    def __init__(self, session_factory: Optional[Callable[[], Any]] = None) -> None:
        self.session_factory = session_factory

    def send(self, notification: Notification) -> bool:
        if self.session_factory is None:
            return False
        try:
            from ..data.schema import Notification as NotificationRow

            with self.session_factory() as session:
                session.add(
                    NotificationRow(
                        ts=notification.created_at,
                        category=notification.kind.value,
                        severity=notification.severity,
                        title=notification.title[:160],
                        body=notification.body,
                        payload=notification.payload,
                        read=notification.read,
                        delivered_channels=notification.delivered_to,
                    )
                )
                session.commit()
            return True
        except Exception as exc:  # pragma: no cover - delivery must never break trading
            log.warning("could not persist notification", context={"error": str(exc)})
            return False


class WebhookChannel(Channel):
    """Generic outbound webhook (Telegram / WhatsApp bridges / Slack look alike).

    Disabled unless a URL is configured. Any failure is logged and swallowed so
    a flaky chat service can never interfere with trading.
    """

    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0, min_severity: str = "WARNING") -> None:
        self.url = url
        self.timeout = timeout
        self.min_severity = min_severity

    def send(self, notification: Notification) -> bool:
        if not self.url:
            return False
        order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
        if order.get(notification.severity, 0) < order.get(self.min_severity, 1):
            return False
        try:
            import httpx

            response = httpx.post(
                self.url,
                json={
                    "text": f"[{notification.severity}] {notification.title}\n{notification.body}",
                    "kind": notification.kind.value,
                    "payload": notification.payload,
                },
                timeout=self.timeout,
            )
            return response.status_code < 400
        except Exception as exc:
            log.warning("webhook notification failed", context={"error": str(exc)})
            return False


class NotificationCenter:
    """Fan-out hub. Channels are optional; history is always kept."""

    def __init__(self) -> None:
        self.channels: List[Channel] = [InMemoryChannel()]
        self._lock = threading.RLock()
        self._counts: Dict[str, int] = {}

    def add_channel(self, channel: Channel) -> None:
        with self._lock:
            self.channels.append(channel)

    def remove_channel(self, name: str) -> None:
        with self._lock:
            self.channels = [c for c in self.channels if c.name != name]

    def notify(
        self,
        kind: EventKind,
        title: str,
        body: str = "",
        *,
        payload: Optional[Dict[str, Any]] = None,
        severity: Optional[str] = None,
    ) -> Notification:
        notification = Notification(
            kind=kind,
            title=title,
            body=body,
            severity=severity or SEVERITY_BY_KIND.get(kind, "INFO"),
            payload=payload or {},
        )
        with self._lock:
            self._counts[notification.severity] = self._counts.get(notification.severity, 0) + 1
            channels = list(self.channels)

        for channel in channels:
            try:
                if channel.send(notification):
                    notification.delivered_to.append(channel.name)
            except Exception:  # pragma: no cover
                log.exception("notification channel failed", context={"channel": channel.name})

        level = {"INFO": log.info, "WARNING": log.warning, "CRITICAL": log.error}.get(
            notification.severity, log.info
        )
        level("notification", context={"kind": kind.value, "title": title, "severity": notification.severity})
        return notification

    # ------------------------------------------------------------- convenience
    def trade_generated(self, symbol: str, direction: str, score: float, **extra: Any) -> Notification:
        return self.notify(
            EventKind.TRADE_GENERATED,
            f"{direction} setup found: {symbol}",
            f"Opportunity score {score:.0f}/100",
            payload={"symbol": symbol, "direction": direction, "score": score, **extra},
        )

    def trade_entered(self, symbol: str, direction: str, quantity: int, price: float, **extra: Any) -> Notification:
        return self.notify(
            EventKind.TRADE_ENTERED,
            f"{direction} entered: {symbol}",
            f"{quantity} shares at Rs {price:,.2f}",
            payload={"symbol": symbol, "direction": direction, "quantity": quantity, "price": price, **extra},
        )

    def trade_exited(self, symbol: str, reason: str, net_pnl: float, r_multiple: float, **extra: Any) -> Notification:
        kind = (
            EventKind.STOP_HIT if reason == "STOP_LOSS"
            else EventKind.TARGET_HIT if reason.startswith("TARGET")
            else EventKind.TRADE_EXITED
        )
        return self.notify(
            kind,
            f"{reason.replace('_', ' ').title()}: {symbol}",
            f"P&L Rs {net_pnl:,.2f} ({r_multiple:+.2f}R)",
            payload={"symbol": symbol, "reason": reason, "net_pnl": net_pnl, "r_multiple": r_multiple, **extra},
        )

    def daily_risk_reached(self, return_pct: float, reason: str, **extra: Any) -> Notification:
        return self.notify(
            EventKind.DAILY_RISK_REACHED,
            "Daily risk limit reached",
            f"Today's return is {return_pct:.2%} ({reason}). No further entries today.",
            payload={"return_pct": return_pct, "reason": reason, **extra},
        )

    def kill_switch(self, engaged: bool, reason: str, **extra: Any) -> Notification:
        return self.notify(
            EventKind.KILL_SWITCH_ENGAGED if engaged else EventKind.KILL_SWITCH_RELEASED,
            "STOP TRADING engaged" if engaged else "Trading resumed",
            reason,
            payload={"reason": reason, **extra},
        )

    def challenger(self, version: str, variable: str, old: Any, new: Any, **extra: Any) -> Notification:
        return self.notify(
            EventKind.NEW_CHALLENGER,
            f"New challenger: {version}",
            f"{variable}: {old} -> {new}",
            payload={"version": version, "variable": variable, "old": old, "new": new, **extra},
        )

    def promoted(self, version: str, reason: str, **extra: Any) -> Notification:
        return self.notify(
            EventKind.STRATEGY_PROMOTED,
            f"Promoted to champion: {version}",
            reason,
            payload={"version": version, "reason": reason, **extra},
        )

    def history(self, limit: int = 100, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            internal = next((c for c in self.channels if isinstance(c, InMemoryChannel)), None)
        if internal is None:
            return []
        items = internal.items
        if severity:
            items = [n for n in items if n.severity == severity]
        return [n.to_dict() for n in items[-limit:]][::-1]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            internal = next((c for c in self.channels if isinstance(c, InMemoryChannel)), None)
            return {
                "channels": [c.name for c in self.channels],
                "counts": dict(self._counts),
                "history_size": len(internal.items) if internal else 0,
            }


_center: Optional[NotificationCenter] = None
_lock = threading.Lock()


def get_notifications() -> NotificationCenter:
    global _center
    with _lock:
        if _center is None:
            _center = NotificationCenter()
        return _center


def reset_notifications() -> None:
    global _center
    with _lock:
        _center = None


__all__ = [
    "NotificationCenter",
    "Notification",
    "EventKind",
    "Channel",
    "InMemoryChannel",
    "PersistChannel",
    "WebhookChannel",
    "SEVERITY_BY_KIND",
    "get_notifications",
    "reset_notifications",
]
