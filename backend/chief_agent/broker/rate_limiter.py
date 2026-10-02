"""Client-side rate limiting that mirrors the published Upstox limits.

The system must never intentionally bypass broker or exchange limits, so we
enforce the published budgets *client-side* at a configurable safety factor
(default 70% of the published allowance) and back off immediately on HTTP 429.

Published limits (per API, per user) - see config/broker.yaml -> rate_limits:

* Order placement APIs (place, modify, cancel, multi-order, GTT)
    regular algo           : 10 / second, 500 / minute, 2000 / 30 minutes
    SEBI-registered algo   : 50 / second, 500 / minute, 2000 / 30 minutes
* Other standard APIs      : 50 / second, 500 / minute, 2000 / 30 minutes
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from ..logging_setup import get_logger

log = get_logger(__name__, component="rate_limiter")


@dataclass(frozen=True)
class LimitSpec:
    """A sliding-window budget over three co-operating time windows."""

    per_second: int
    per_minute: int
    per_30_minutes: int
    name: str = "default"


class SlidingWindowCounter:
    """Thread-safe sliding-window counter."""

    __slots__ = ("_events", "_lock")

    def __init__(self) -> None:
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float, window: float) -> None:
        cutoff = now - window
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def count(self, window: float, now: Optional[float] = None) -> int:
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._trim(now, window)
            return len(self._events)

    def add(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._events.append(now)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


class RateLimiter:
    """Blocks the calling thread until a slot in every window is available."""

    def __init__(self, spec: LimitSpec, safety_factor: float = 0.7) -> None:
        self.spec = spec
        self.safety_factor = max(0.05, min(1.0, safety_factor))
        self._counter = SlidingWindowCounter()
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self._stats: Dict[str, int] = {"waits": 0, "throttled_429": 0, "acquired": 0}

    # ------------------------------------------------------------------ budgets
    @property
    def effective(self) -> Tuple[int, int, int]:
        f = self.safety_factor
        return (
            max(1, int(self.spec.per_second * f)),
            max(1, int(self.spec.per_minute * f)),
            max(1, int(self.spec.per_30_minutes * f)),
        )

    # ------------------------------------------------------------------- checks
    def _wait_time(self, now: float) -> float:
        per_s, per_m, per_30m = self.effective
        waits = [0.0]

        for window, cap in ((1.0, per_s), (60.0, per_m), (1800.0, per_30m)):
            if self._counter.count(window, now) >= cap:
                with self._counter._lock:  # noqa: SLF001 - deliberate tight coupling
                    self._counter._trim(now, window)
                    if self._counter._events:
                        waits.append(max(0.0, self._counter._events[0] + window - now) + 0.01)
        return max(waits)

    def acquire(self, timeout: Optional[float] = 30.0) -> bool:
        """Block until a request slot is free. Returns False on timeout."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            with self._lock:
                now = time.monotonic()
                if now < self._blocked_until:
                    wait = self._blocked_until - now
                else:
                    wait = self._wait_time(now)
                    if wait <= 0:
                        self._counter.add(now)
                        self._stats["acquired"] += 1
                        return True
                self._stats["waits"] += 1

            if deadline is not None and time.monotonic() + wait > deadline:
                log.warning(
                    "rate limiter acquire timed out",
                    context={"limit": self.spec.name, "next_slot_in": round(wait, 3)},
                )
                return False
            time.sleep(min(wait, 0.25))

    def note_429(self, retry_after_seconds: float = 2.0) -> None:
        """Called when the broker returns HTTP 429 (or an equivalent signal).

        We immediately and globally clamp this limiter for a cooling period
        rather than pretending the limit does not exist.
        """
        with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + retry_after_seconds)
            self._stats["throttled_429"] += 1
        log.warning(
            "broker rate limit hit - cooling down",
            context={"limit": self.spec.name, "cooldown_s": retry_after_seconds},
        )

    def stats(self) -> Dict[str, object]:
        return {
            "limit": self.spec.name,
            "safety_factor": self.safety_factor,
            "effective_budget": {
                "per_second": self.effective[0],
                "per_minute": self.effective[1],
                "per_30_minutes": self.effective[2],
            },
            **self._stats,
        }


# --------------------------------------------------------------------------- #
# Registry - one limiter per logical API class, built from config/broker.yaml
# --------------------------------------------------------------------------- #
DEFAULT_SPECS: Dict[str, LimitSpec] = {
    "order_placement": LimitSpec(10, 500, 2000, "order_placement"),
    "order_placement_registered_algo": LimitSpec(50, 500, 2000, "order_placement_registered_algo"),
    "standard": LimitSpec(50, 500, 2000, "standard"),
    "payout": LimitSpec(10, 500, 2000, "payout"),
}


class RateLimiterRegistry:
    """Holds one :class:`RateLimiter` per API class, shared across the process."""

    def __init__(self, specs: Optional[Dict[str, LimitSpec]] = None, safety_factor: float = 0.7) -> None:
        self._limiters: Dict[str, RateLimiter] = {}
        for name, spec in (specs or DEFAULT_SPECS).items():
            self._limiters[name] = RateLimiter(spec, safety_factor=safety_factor)
        self._safety_factor = safety_factor

    def get(self, name: str) -> RateLimiter:
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(DEFAULT_SPECS["standard"], self._safety_factor)
        return self._limiters[name]

    def configure_from_dict(self, raw: Dict[str, object]) -> None:
        """Apply ``config/broker.yaml`` -> ``rate_limits``."""
        if not isinstance(raw, dict):
            return
        factor = float(raw.get("safety_factor", self._safety_factor) or self._safety_factor)
        for name in ("order_placement", "order_placement_registered_algo", "standard"):
            spec_raw = raw.get(name)
            if isinstance(spec_raw, dict):
                self._limiters[name] = RateLimiter(
                    LimitSpec(
                        per_second=int(spec_raw.get("per_second", 10)),
                        per_minute=int(spec_raw.get("per_minute", 500)),
                        per_30_minutes=int(spec_raw.get("per_30_minutes", 2000)),
                        name=name,
                    ),
                    safety_factor=factor,
                )

    def stats(self) -> Dict[str, object]:
        return {name: limiter.stats() for name, limiter in self._limiters.items()}


_registry: Optional[RateLimiterRegistry] = None


def get_rate_limiters() -> RateLimiterRegistry:
    global _registry
    if _registry is None:
        _registry = RateLimiterRegistry()
        try:
            from ..settings import get_config_store

            _registry.configure_from_dict(get_config_store().load("broker").get("rate_limits", {}))
        except Exception:  # pragma: no cover - config is optional at import time
            pass
    return _registry


def reset_rate_limiters() -> None:
    """Test helper."""
    global _registry
    _registry = None


__all__ = [
    "LimitSpec",
    "RateLimiter",
    "RateLimiterRegistry",
    "get_rate_limiters",
    "reset_rate_limiters",
    "DEFAULT_SPECS",
]
