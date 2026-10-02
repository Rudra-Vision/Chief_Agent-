"""System health.

``/health`` reports the state of every subsystem the brief requires:

    API, database, Redis (if used), Upstox, market WebSocket, order stream,
    strategy engine, risk engine, scheduler.

The dashboard renders HEALTHY / DEGRADED / STOPPED plus the individual checks,
and the execution engine refuses LIVE orders while any required component is
unhealthy.
"""

from __future__ import annotations

import datetime as dt
import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..logging_setup import get_logger
from ..settings import get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="health")


class HealthStatus(str, enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"

    @property
    def rank(self) -> int:
        return {"HEALTHY": 0, "UNKNOWN": 1, "DEGRADED": 2, "STOPPED": 3}[self.value]


@dataclass
class CheckResult:
    name: str
    status: HealthStatus
    detail: str = ""
    required: bool = True
    latency_ms: Optional[float] = None
    checked_at: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "required": self.required,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "checked_at": (self.checked_at or now_ist()).isoformat(),
        }


@dataclass
class HealthReport:
    status: HealthStatus
    checks: List[CheckResult] = field(default_factory=list)
    generated_at: Any = None
    duration_ms: float = 0.0
    mode: str = ""
    version: str = ""

    @property
    def healthy(self) -> bool:
        return self.status is HealthStatus.HEALTHY

    @property
    def can_trade_live(self) -> bool:
        return all(c.status is HealthStatus.HEALTHY for c in self.checks if c.required)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "generated_at": (self.generated_at or now_ist()).isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "mode": self.mode,
            "version": self.version,
            "can_trade_live": self.can_trade_live,
            "checks": [c.to_dict() for c in self.checks],
            "failed_checks": [c.name for c in self.checks if c.status is HealthStatus.STOPPED],
            "degraded_checks": [c.name for c in self.checks if c.status is HealthStatus.DEGRADED],
        }


class HealthMonitor:
    """Collects a health check per subsystem."""

    def __init__(self) -> None:
        self._providers: Dict[str, Callable[[], CheckResult]] = {}
        self._last: Optional[HealthReport] = None
        self._lock = threading.RLock()

    def register(self, name: str, provider: Callable[[], CheckResult]) -> None:
        self._providers[name] = provider

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)

    @property
    def last_report(self) -> Optional[HealthReport]:
        with self._lock:
            return self._last

    def run(self) -> HealthReport:
        started = time.perf_counter()
        settings = get_settings()
        checks: List[CheckResult] = []

        for name, provider in sorted(self._providers.items()):
            check_started = time.perf_counter()
            try:
                result = provider()
            except Exception as exc:  # a crashing check is a STOPPED check
                result = CheckResult(name=name, status=HealthStatus.STOPPED, detail=f"check raised: {exc}")
            if result.latency_ms is None:
                result.latency_ms = (time.perf_counter() - check_started) * 1000.0
            result.checked_at = now_ist()
            checks.append(result)

        required_statuses = [c.status for c in checks if c.required]
        all_statuses = [c.status for c in checks] or [HealthStatus.UNKNOWN]
        overall = max(required_statuses or all_statuses, key=lambda s: s.rank)

        report = HealthReport(
            status=overall,
            checks=checks,
            generated_at=now_ist(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            mode=settings.effective_mode().value,
            version=settings.app_version,
        )
        with self._lock:
            self._last = report
        return report

    # ---------------------------------------------------------- default checks
    def install_default_checks(self, components: Dict[str, Any]) -> None:
        """Wire the standard subsystem checks from live components."""
        settings = get_settings()

        def database_check() -> CheckResult:
            try:
                from ..data.db import healthcheck

                result = healthcheck()
                if result.get("ok"):
                    return CheckResult("database", HealthStatus.HEALTHY, str(result.get("engine")))
                return CheckResult("database", HealthStatus.STOPPED, str(result.get("error")))
            except Exception as exc:
                return CheckResult("database", HealthStatus.STOPPED, str(exc))

        def redis_check() -> CheckResult:
            url = settings.redis_url
            if not url:
                return CheckResult(
                    "redis", HealthStatus.HEALTHY,
                    "not configured - the system runs without Redis (in-process cache)", required=False,
                )
            try:
                import redis  # type: ignore

                client = redis.Redis.from_url(url, socket_timeout=2)
                client.ping()
                return CheckResult("redis", HealthStatus.HEALTHY, "ping ok", required=settings.redis_required)
            except Exception as exc:
                status = HealthStatus.STOPPED if settings.redis_required else HealthStatus.DEGRADED
                return CheckResult("redis", status, f"{exc} (REDIS_REQUIRED={settings.redis_required})",
                                   required=settings.redis_required)

        def broker_check() -> CheckResult:
            broker = components.get("broker")
            if broker is None:
                return CheckResult("upstox", HealthStatus.UNKNOWN, "broker not initialised")
            status = broker.auth.status()
            if not status.get("authenticated"):
                return CheckResult(
                    "upstox", HealthStatus.DEGRADED,
                    "not connected - running on simulated/paper data", required=False,
                )
            if not status.get("token_probably_valid_today"):
                return CheckResult(
                    "upstox", HealthStatus.DEGRADED,
                    "token was not obtained today; Upstox tokens expire daily", required=False,
                )
            return CheckResult("upstox", HealthStatus.HEALTHY, f"source={status.get('token_source')}")

        def market_feed_check() -> CheckResult:
            feed = components.get("market_feed")
            calendar = components.get("calendar")
            tradable = bool(calendar.is_tradable_now()) if calendar is not None else True
            if feed is None:
                return CheckResult("market_data_feed", HealthStatus.HEALTHY,
                                   "REST polling feed active (no websocket configured)", required=False)
            status = feed.status()
            if status.get("connected"):
                return CheckResult("market_data_feed", HealthStatus.HEALTHY,
                                   f"{status.get('subscriptions')} subscriptions")
            if not status.get("decoder_available"):
                return CheckResult(
                    "market_data_feed", HealthStatus.HEALTHY,
                    "protobuf decoder not installed; using the REST polling feed instead", required=False,
                )
            if tradable:
                return CheckResult("market_data_feed", HealthStatus.DEGRADED, "websocket is not connected")
            return CheckResult("market_data_feed", HealthStatus.HEALTHY, "market closed", required=False)

        def order_stream_check() -> CheckResult:
            stream = components.get("order_stream")
            if stream is None:
                return CheckResult("order_stream", HealthStatus.HEALTHY,
                                   "portfolio stream not configured", required=False)
            status = stream.status()
            if status.get("connected"):
                return CheckResult("order_stream", HealthStatus.HEALTHY, "connected")
            return CheckResult("order_stream", HealthStatus.DEGRADED, "not connected", required=False)

        def risk_engine_check() -> CheckResult:
            engine = components.get("risk_engine")
            if engine is None:
                return CheckResult("risk_engine", HealthStatus.STOPPED, "risk engine not initialised")
            limits = engine.limits_summary()
            return CheckResult(
                "risk_engine", HealthStatus.HEALTHY,
                f"risk/trade {limits['risk_per_trade_pct']:.2%}, max positions {limits['max_simultaneous_positions']}",
            )

        def strategy_engine_check() -> CheckResult:
            champion = components.get("champion")
            if champion is None:
                return CheckResult("strategy_engine", HealthStatus.DEGRADED,
                                   "no champion strategy loaded", required=False)
            version = champion.get("version") if isinstance(champion, dict) else str(champion)
            return CheckResult("strategy_engine", HealthStatus.HEALTHY, f"champion {version}")

        def kill_switch_check() -> CheckResult:
            switch = components.get("kill_switch")
            if switch is None:
                return CheckResult("kill_switch", HealthStatus.STOPPED, "kill switch unavailable")
            state = switch.state
            if state.engaged:
                return CheckResult("kill_switch", HealthStatus.DEGRADED, f"ENGAGED: {state.reason}")
            return CheckResult("kill_switch", HealthStatus.HEALTHY, "armed and released")

        def scheduler_check() -> CheckResult:
            scheduler = components.get("scheduler")
            if scheduler is None:
                return CheckResult("scheduler", HealthStatus.HEALTHY, "not started", required=False)
            status = scheduler.status() if hasattr(scheduler, "status") else {}
            if status.get("running"):
                return CheckResult("scheduler", HealthStatus.HEALTHY,
                                   f"{status.get('jobs', 0)} scheduled job(s)")
            return CheckResult("scheduler", HealthStatus.DEGRADED, "scheduler is not running", required=False)

        def data_quality_check() -> CheckResult:
            quality = components.get("data_quality")
            if quality is None:
                return CheckResult("data_quality", HealthStatus.HEALTHY, "not evaluated yet", required=False)
            if quality.data_safe_mode:
                report = quality.last_report
                return CheckResult("data_quality", HealthStatus.DEGRADED,
                                   report.summary if report else "DATA_SAFE_MODE active")
            return CheckResult("data_quality", HealthStatus.HEALTHY, "market data checks passed")

        def reconciliation_check() -> CheckResult:
            reconciler = components.get("reconciler")
            if reconciler is None:
                return CheckResult("reconciliation", HealthStatus.HEALTHY, "not run yet", required=False)
            if reconciler.pending:
                report = reconciler.last_report
                return CheckResult(
                    "reconciliation", HealthStatus.DEGRADED,
                    f"{len(report.mismatches)} position mismatch(es)" if report else "mismatch pending",
                )
            return CheckResult("reconciliation", HealthStatus.HEALTHY, "internal positions match the broker")

        for name, provider in (
            ("database", database_check),
            ("redis", redis_check),
            ("upstox", broker_check),
            ("market_data_feed", market_feed_check),
            ("order_stream", order_stream_check),
            ("strategy_engine", strategy_engine_check),
            ("risk_engine", risk_engine_check),
            ("kill_switch", kill_switch_check),
            ("data_quality", data_quality_check),
            ("reconciliation", reconciliation_check),
            ("scheduler", scheduler_check),
        ):
            self.register(name, provider)


__all__ = ["HealthMonitor", "HealthReport", "CheckResult", "HealthStatus"]
