"""Application state container.

One object holds every long-lived component so routers stay thin and there is a
single wiring point for tests (``AppState`` can be constructed with fakes).
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..broker.upstox_broker import UpstoxBroker
from ..data.provider import MarketDataProvider
from ..execution.paper_engine import PaperTradingEngine
from ..execution.reconciliation import PositionReconciler, SlippageMonitor
from ..logging_setup import get_logger
from ..monitoring.health import HealthMonitor
from ..monitoring.notifications import NotificationCenter, get_notifications
from ..portfolio.engine import PortfolioEngine
from ..research.champion import ChampionRegistry
from ..risk.engine import RiskEngine
from ..risk.killswitch import KillSwitch, get_kill_switch
from ..risk.preflight import Preflight
from ..scheduling.calendar import TradingCalendar
from ..settings import OperatingMode, Settings, get_config_store, get_settings
from ..strategies.orb_vwap import ORBVWAPStrategy
from ..timeutil import now_ist
from ..data.quality import DataQualityEngine

log = get_logger(__name__, component="app_state")


@dataclass
class RuntimeCounters:
    started_at: dt.datetime = field(default_factory=now_ist)
    signals_generated: int = 0
    signals_rejected: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    trades_completed: int = 0
    last_scan_at: Optional[dt.datetime] = None
    last_cycle_at: Optional[dt.datetime] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": (now_ist() - self.started_at).total_seconds(),
            "started_at": self.started_at.isoformat(),
            "signals_generated": self.signals_generated,
            "signals_rejected": self.signals_rejected,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "trades_completed": self.trades_completed,
            "last_scan_at": self.last_scan_at.isoformat() if self.last_scan_at else None,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "last_error": self.last_error,
        }


class AppState:
    """Everything the API needs, built once at startup."""

    def __init__(self, settings: Optional[Settings] = None, *, start_services: bool = True) -> None:
        self.settings = settings or get_settings()
        self.config = get_config_store()
        self.mode: OperatingMode = self.settings.effective_mode()
        self.counters = RuntimeCounters()
        self._lock = threading.RLock()
        self.start_services = start_services

        # --- core components ------------------------------------------------
        self.broker = UpstoxBroker(self.settings, mode=self.mode)
        self.kill_switch: KillSwitch = get_kill_switch()
        self.risk_engine = RiskEngine(self.config)
        self.notifications: NotificationCenter = get_notifications()
        self.calendar = TradingCalendar(self.broker.market_data)
        self.data_quality = DataQualityEngine()
        self.reconciler = PositionReconciler()
        self.slippage = SlippageMonitor()
        self.preflight = Preflight(self.settings)

        self.provider = MarketDataProvider(self.settings, broker=self.broker)
        self.paper = PaperTradingEngine(
            initial_capital=float(
                (self.config.load("execution").get("paper") or {}).get("initial_capital", 500_000)
            ),
            config=self.config.load("execution"),
        )
        self.portfolio = PortfolioEngine(mode=self.mode.value)

        # --- strategy -------------------------------------------------------
        self.strategy_version: Optional[str] = None
        self.champion_config: Dict[str, Any] = {}
        self.strategy: Optional[ORBVWAPStrategy] = None
        self._load_champion()

        # --- monitoring -----------------------------------------------------
        self.health = HealthMonitor()
        self.health.install_default_checks(
            {
                "broker": self.broker,
                "market_feed": None,
                "order_stream": None,
                "risk_engine": self.risk_engine,
                "champion": {"version": self.strategy_version or "unset"},
                "kill_switch": self.kill_switch,
                "data_quality": self.data_quality,
                "reconciler": self.reconciler,
                "scheduler": None,
            }
        )

        # --- telemetry ------------------------------------------------------
        self.last_scan: Dict[str, Any] = {}
        self.last_backtest_id: Optional[str] = None
        self.scheduler: Optional[Any] = None

    # ------------------------------------------------------------------ boot
    def _load_champion(self) -> None:
        """Load the champion strategy from the database, creating the baseline
        from ``config/strategy.yaml`` on first run."""
        from ..data.db import session_scope

        try:
            with session_scope() as session:
                registry = ChampionRegistry(session)
                champion = registry.ensure_baseline_champion()
                self.strategy_version = champion.version
                self.champion_config = registry.config_for(champion.version)
                session.commit()
        except Exception as exc:
            log.warning(
                "could not load the champion from the database; using config/strategy.yaml",
                context={"error": str(exc)},
            )
            self.champion_config = self.config.load("strategy")
            self.strategy_version = str(
                ((self.champion_config.get("champion") or {}).get("version")) or "ORB_v1.0.0"
            )
        self.strategy = ORBVWAPStrategy(self.champion_config, version=self.strategy_version or "ORB_v1.0.0")

    def reload_champion(self) -> Dict[str, Any]:
        self._load_champion()
        return {
            "version": self.strategy_version,
            "config_hash": self.strategy.config_hash if self.strategy else None,
        }

    # ------------------------------------------------------------------ state
    @property
    def is_live(self) -> bool:
        return self.mode is OperatingMode.LIVE

    @property
    def live_blockers(self) -> List[str]:
        return self.settings.live_blocked_reasons

    def mode_summary(self) -> Dict[str, Any]:
        effective = self.settings.effective_mode()
        return {
            "mode": effective.value,
            "requested_mode": self.settings.operating_mode.value,
            "is_simulated": effective.is_simulated,
            "is_live": effective is OperatingMode.LIVE,
            "deployment_stage": self.settings.deployment_stage.name,
            "deployment_stage_value": self.settings.deployment_stage.value,
            "live_permitted": self.settings.live_permitted,
            "live_blocked_reasons": self.settings.live_blocked_reasons,
            "banner": (
                "SIMULATION - no real money is at risk"
                if effective.is_simulated
                else "LIVE TRADING - REAL MONEY IS AT RISK"
            ),
            "data_source": self.provider.data_source,
            "strategy_version": self.strategy_version,
        }

    def attach_scheduler(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.health.unregister("scheduler")
        self.health.install_default_checks(
            {
                "broker": self.broker,
                "market_feed": None,
                "order_stream": None,
                "risk_engine": self.risk_engine,
                "champion": {"version": self.strategy_version or "unset"},
                "kill_switch": self.kill_switch,
                "data_quality": self.data_quality,
                "reconciler": self.reconciler,
                "scheduler": scheduler,
            }
        )

    def refresh_market_placeholders(self) -> None:
        """Refresh the quote snapshot used by the dashboard.

        Uses the provider (live Upstox when a token exists, otherwise clearly
        labelled simulated data).
        """
        from ..data.db import session_scope

        with session_scope() as session:
            from sqlalchemy import select

            from ..data.schema import Instrument

            rows = (
                session.execute(
                    select(Instrument).where(Instrument.in_universe.is_(True)).limit(200)
                )
                .scalars()
                .all()
            )
            keys = [row.instrument_key for row in rows]
        if not keys:
            return
        try:
            quotes = self.provider.quotes(keys[:200])
        except Exception as exc:
            log.warning("could not refresh market quotes", context={"error": str(exc)})
            return

        self.market_snapshot = []
        for key, quote in quotes.items():
            ltp = quote.get("ltp") or quote.get("last_price")
            ohlc = quote.get("ohlc") or {}
            previous = ohlc.get("close")
            change_pct = ((float(ltp) - float(previous)) / float(previous)) if (ltp and previous) else None
            depth = quote.get("depth") or {}
            bid = (depth.get("buy") or [{}])[0].get("price")
            ask = (depth.get("sell") or [{}])[0].get("price")
            self.market_snapshot.append(
                {
                    "instrument_key": key,
                    "ltp": ltp,
                    "change_pct": change_pct,
                    "volume": quote.get("volume"),
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": ((float(ask) - float(bid)) / float(ltp)) if (bid and ask and ltp) else None,
                    "simulated": bool(quote.get("_simulated")),
                }
            )

    market_snapshot: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------- health
    def health_report(self) -> Dict[str, Any]:
        return self.health.run().to_dict()


_state: Optional[AppState] = None
_state_lock = threading.Lock()


def get_app_state() -> AppState:
    global _state
    with _state_lock:
        if _state is None:
            _state = AppState()
        return _state


def set_app_state(state: AppState) -> None:
    global _state
    with _state_lock:
        _state = state


def reset_app_state() -> None:
    global _state
    with _state_lock:
        _state = None


__all__ = ["AppState", "RuntimeCounters", "get_app_state", "set_app_state", "reset_app_state"]
