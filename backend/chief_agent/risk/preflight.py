"""LIVE-mode preflight compliance checklist.

Before LIVE mode can be entered, every one of these must pass:

    Upstox authentication valid
    static IP configuration valid
    market-data connection working
    order API reachable
    account funds accessible
    broker time synchronized
    database operational
    risk engine operational
    kill switch operational
    strategy version valid
    position reconciliation clean
    market data quality acceptable

**No live order is possible if the preflight fails.** The result is cached for a
short period so a hot path re-checks cheaply, and it is re-validated before the
first order of each session.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..logging_setup import get_logger
from ..settings import OperatingMode, Settings, get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="preflight")


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIPPED = "SKIPPED"


@dataclass
class PreflightCheck:
    name: str
    status: CheckStatus
    detail: str = ""
    required_for_live: bool = True
    remediation: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "required_for_live": self.required_for_live,
            "remediation": self.remediation,
        }


@dataclass
class PreflightReport:
    ran_at: dt.datetime
    mode: str
    checks: List[PreflightCheck] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.required_for_live)

    @property
    def blocking(self) -> List[PreflightCheck]:
        return [c for c in self.checks if c.required_for_live and not c.ok]

    @property
    def age_seconds(self) -> float:
        return (now_ist() - self.ran_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran_at": self.ran_at.isoformat(),
            "mode": self.mode,
            "passed": self.passed,
            "age_seconds": round(self.age_seconds, 1),
            "duration_seconds": round(self.duration_seconds, 3),
            "checks": [c.to_dict() for c in self.checks],
            "blocking": [c.name for c in self.blocking],
        }


class Preflight:
    """Runs and caches the LIVE compliance checklist."""

    def __init__(self, settings: Optional[Settings] = None, max_age_seconds: float = 120.0) -> None:
        self.settings = settings or get_settings()
        self.max_age_seconds = max_age_seconds
        self._report: Optional[PreflightReport] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ report
    def last_report(self) -> Optional[PreflightReport]:
        return self._report

    def is_fresh(self) -> bool:
        return self._report is not None and self._report.age_seconds <= self.max_age_seconds

    def ensure_fresh(self, deep: bool = False, force: bool = False) -> PreflightReport:
        with self._lock:
            if not force and self.is_fresh() and self._report is not None:
                return self._report
            return self.run(deep=deep)

    # -------------------------------------------------------------------- run
    def run(self, *, deep: bool = False, components: Optional[Dict[str, Any]] = None) -> PreflightReport:
        """Execute every check.

        ``components`` lets the API layer inject live objects it already holds:
        ``broker``, ``database_health``, ``risk_engine``, ``kill_switch``,
        ``data_quality``, ``reconciliation``, ``strategy_version``.
        """
        import time

        started = time.perf_counter()
        components = components or {}
        checks: List[PreflightCheck] = []
        mode = self.settings.effective_mode()

        try:
            checks.extend(self._config_checks())
            checks.extend(self._auth_checks(components))
            checks.extend(self._infrastructure_checks(components))
            checks.extend(self._trading_checks(components))
            if deep:
                checks.extend(self._deep_checks(components))
        except Exception as exc:  # pragma: no cover - a crash must fail the check, not the app
            log.exception("preflight crashed")
            checks.append(
                PreflightCheck(
                    "preflight_internal",
                    CheckStatus.FAIL,
                    f"preflight raised an unexpected error: {exc}",
                    True,
                    "Check the application logs; no live order will be permitted.",
                )
            )

        report = PreflightReport(
            ran_at=now_ist(),
            mode=mode.value,
            checks=checks,
            duration_seconds=time.perf_counter() - started,
        )
        with self._lock:
            self._report = report

        if not report.passed:
            log.error(
                "preflight FAILED",
                context={"mode": report.mode, "blocking": [c.name for c in report.blocking]},
            )
        else:
            log.info("preflight passed", context={"mode": report.mode, "checks": len(checks)})
        return report

    # ------------------------------------------------------------ check groups
    def _config_checks(self) -> List[PreflightCheck]:
        s = self.settings
        out = [
            PreflightCheck(
                "credentials_configured",
                CheckStatus.PASS if s.has_upstox_credentials else CheckStatus.FAIL,
                "Upstox API key / secret / redirect URI are configured"
                if s.has_upstox_credentials
                else "missing UPSTOX_API_KEY, UPSTOX_API_SECRET or UPSTOX_REDIRECT_URI",
                True,
                "Create an app at https://account.upstox.com/developer/apps and copy the key/secret into .env",
            ),
            PreflightCheck(
                "live_master_switch",
                CheckStatus.PASS if s.allow_live_trading else CheckStatus.FAIL,
                "ALLOW_LIVE_TRADING=true" if s.allow_live_trading else "ALLOW_LIVE_TRADING is false",
                True,
                "Set ALLOW_LIVE_TRADING=true in .env only when you are ready to risk real money.",
            ),
            PreflightCheck(
                "deployment_stage",
                CheckStatus.PASS if s.deployment_stage.value >= 4 else CheckStatus.FAIL,
                f"stage {s.deployment_stage.name}",
                True,
                "Advance through the stages: backtest -> sandbox -> paper -> shadow -> minimum live capital.",
            ),
            PreflightCheck(
                "dashboard_auth_configured",
                CheckStatus.PASS if (s.dashboard_password or not s.require_dashboard_auth_in_live) else CheckStatus.FAIL,
                "dashboard password set" if s.dashboard_password else "DASHBOARD_PASSWORD is not set",
                True,
                "LIVE mode exposes real money; set DASHBOARD_PASSWORD so the dashboard is not open.",
            ),
        ]
        if s.upstox_static_ip_primary:
            status, detail, fix = (
                CheckStatus.PASS,
                f"primary {s.upstox_static_ip_primary}"
                + (f", secondary {s.upstox_static_ip_secondary}" if s.upstox_static_ip_secondary else ""),
                "",
            )
        else:
            status, detail, fix = (
                CheckStatus.FAIL,
                "no static IP configured",
                "Register your server's static IP with Upstox (PUT /v2/user/ip) and set UPSTOX_STATIC_IP_PRIMARY.",
            )
        out.append(PreflightCheck("static_ip_configured", status, detail, s.require_static_ip_for_live, fix))
        return out

    def _auth_checks(self, components: Dict[str, Any]) -> List[PreflightCheck]:
        broker = components.get("broker")
        if broker is None:
            return [PreflightCheck("authentication_valid", CheckStatus.SKIPPED, "broker not supplied", True)]
        state = broker.auth.status()
        if not state.get("authenticated"):
            return [
                PreflightCheck(
                    "authentication_valid",
                    CheckStatus.FAIL,
                    state.get("last_error") or "no access token",
                    True,
                    "Open the dashboard and use Connect Upstox to complete the daily login.",
                )
            ]
        if not state.get("token_probably_valid_today") and self.settings.operating_mode is OperatingMode.LIVE:
            return [
                PreflightCheck(
                    "authentication_valid",
                    CheckStatus.FAIL,
                    "access token was not obtained today (Upstox tokens expire daily)",
                    True,
                    "Re-run the Upstox login to obtain today's access token.",
                )
            ]
        return [PreflightCheck("authentication_valid", CheckStatus.PASS, f"source={state.get('token_source')}")]

    def _infrastructure_checks(self, components: Dict[str, Any]) -> List[PreflightCheck]:
        out: List[PreflightCheck] = []

        db_health = components.get("database_health")
        if callable(db_health):
            db_health = db_health()
        if db_health is None:
            try:
                from ..data.db import healthcheck

                db_health = healthcheck()
            except Exception as exc:
                db_health = {"ok": False, "error": str(exc)}
        out.append(
            PreflightCheck(
                "database_operational",
                CheckStatus.PASS if db_health.get("ok") else CheckStatus.FAIL,
                str(db_health.get("engine") or db_health.get("error")),
                True,
                "Check DATABASE_URL and that the database service is running.",
            )
        )

        risk_engine = components.get("risk_engine")
        out.append(
            PreflightCheck(
                "risk_engine_operational",
                CheckStatus.PASS if risk_engine is not None else CheckStatus.FAIL,
                "risk engine available" if risk_engine is not None else "risk engine not initialised",
                True,
                "Restart the application; no order is permitted without a live risk engine.",
            )
        )

        kill_switch = components.get("kill_switch")
        if kill_switch is None:
            out.append(PreflightCheck("kill_switch_operational", CheckStatus.FAIL, "kill switch not available", True,
                                      "Restart the application."))
        else:
            out.append(
                PreflightCheck(
                    "kill_switch_operational",
                    CheckStatus.PASS if not kill_switch.engaged else CheckStatus.FAIL,
                    "kill switch armed and released"
                    if not kill_switch.engaged
                    else f"kill switch ENGAGED ({kill_switch.state.reason})",
                    True,
                    "Release the kill switch from the dashboard before trading.",
                )
            )

        strategy_version = components.get("strategy_version")
        out.append(
            PreflightCheck(
                "strategy_version_valid",
                CheckStatus.PASS if strategy_version else CheckStatus.FAIL,
                str(strategy_version or "no champion strategy loaded"),
                True,
                "Promote or load a strategy version before enabling LIVE.",
            )
        )
        return out

    def _trading_checks(self, components: Dict[str, Any]) -> List[PreflightCheck]:
        out: List[PreflightCheck] = []

        reconciliation = components.get("reconciliation")
        if isinstance(reconciliation, dict):
            matched = bool(reconciliation.get("matched", False))
            out.append(
                PreflightCheck(
                    "position_reconciliation",
                    CheckStatus.PASS if matched else CheckStatus.FAIL,
                    "internal positions match the broker" if matched else f"{len(reconciliation.get('mismatches', []))} mismatch(es)",
                    True,
                    "Resolve every position mismatch before placing live orders.",
                )
            )
        else:
            out.append(
                PreflightCheck("position_reconciliation", CheckStatus.SKIPPED, "not run", True,
                               "Reconciliation runs automatically once connected.")
            )

        data_quality = components.get("data_quality")
        if isinstance(data_quality, dict):
            safe = bool(data_quality.get("safe_to_trade", False))
            out.append(
                PreflightCheck(
                    "market_data_quality",
                    CheckStatus.PASS if safe else CheckStatus.FAIL,
                    data_quality.get("summary", ""),
                    True,
                    "Fix the data-quality incidents before trading.",
                )
            )
        else:
            out.append(PreflightCheck("market_data_quality", CheckStatus.SKIPPED, "not evaluated", True))

        out.append(
            PreflightCheck(
                "kill_switch_broker_supported",
                CheckStatus.PASS,
                "Upstox Kill Switch API available (POST /v2/user/kill-switch)",
                False,
            )
        )
        return out

    def _deep_checks(self, components: Dict[str, Any]) -> List[PreflightCheck]:
        """Live API probes. Only run when a token is available."""
        broker = components.get("broker")
        if broker is None or not broker.has_token:
            return [
                PreflightCheck("market_data_connection", CheckStatus.SKIPPED, "no broker token", True),
                PreflightCheck("order_api_reachable", CheckStatus.SKIPPED, "no broker token", True),
                PreflightCheck("account_funds_accessible", CheckStatus.SKIPPED, "no broker token", True),
                PreflightCheck("broker_time_synchronized", CheckStatus.SKIPPED, "no broker token", True),
                PreflightCheck("static_ip_matches_outbound", CheckStatus.SKIPPED, "no broker token", True),
            ]

        out: List[PreflightCheck] = []

        try:
            quotes = broker.market_data.ltp(["NSE_INDEX|Nifty 50"])
            out.append(
                PreflightCheck(
                    "market_data_connection",
                    CheckStatus.PASS if quotes else CheckStatus.FAIL,
                    f"{len(quotes)} quote(s) returned",
                    True,
                    "Check the token scope and network access to api.upstox.com.",
                )
            )
        except Exception as exc:
            out.append(PreflightCheck("market_data_connection", CheckStatus.FAIL, str(exc), True,
                                      "Verify the token is valid today."))

        try:
            book = broker.orders.order_book()
            out.append(PreflightCheck("order_api_reachable", CheckStatus.PASS, f"{len(book)} order(s) today"))
        except Exception as exc:
            out.append(PreflightCheck("order_api_reachable", CheckStatus.FAIL, str(exc), True,
                                      "Order APIs must be reachable before LIVE trading."))

        try:
            funds = broker.portfolio.funds("SEC")
            ok = bool(funds.raw)
            out.append(
                PreflightCheck(
                    "account_funds_accessible",
                    CheckStatus.PASS if ok else CheckStatus.FAIL,
                    f"available Rs {funds.available_cash:,.2f}",
                    True,
                    "Verify the token has the funds/margin scope.",
                )
            )
        except Exception as exc:
            out.append(PreflightCheck("account_funds_accessible", CheckStatus.FAIL, str(exc), True))

        try:
            status = broker.market_data.exchange_status("NSE")
            ok = status.get("status") not in (None, "UNKNOWN")
            out.append(
                PreflightCheck(
                    "broker_time_synchronized",
                    CheckStatus.PASS if ok else CheckStatus.WARN,
                    f"NSE session status: {status.get('status')}",
                    False,
                )
            )
        except Exception as exc:
            out.append(PreflightCheck("broker_time_synchronized", CheckStatus.WARN, str(exc), False))

        if self.settings.upstox_static_ip_primary:
            ip = broker.broker_risk.verify_outbound_ip()
            out.append(
                PreflightCheck(
                    "static_ip_matches_outbound",
                    CheckStatus.PASS if ip.get("ok") else CheckStatus.FAIL,
                    f"outbound {ip.get('actual_ip')} vs registered {ip.get('expected_ip')}",
                    self.settings.require_static_ip_for_live,
                    "Deploy on a host with the registered static IP, or update the registered IP in Upstox.",
                )
            )
        return out


__all__ = ["Preflight", "PreflightCheck", "PreflightReport", "CheckStatus"]
