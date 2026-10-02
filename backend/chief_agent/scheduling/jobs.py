"""Scheduled daily workflow.

PRE-MARKET   refresh instruments, verify the broker, check funds, health check,
             update historical data, recompute indicators, review news, rank
             sectors, build the universe.
OPEN         observe the opening range.
SCANNING     identify opportunities.
TRADING      signals -> risk -> execution.
POST-TRADE   journal each completed trade.
POST-MARKET  reconcile, compute statistics, run the daily analysis, generate
             research hypotheses.
OFF-HOURS    heavier backtests and the weekly self-improvement loop.

The scheduler uses the OFFICIAL exchange state (``TradingCalendar``) rather than
assuming a local clock, so holidays and ad-hoc session changes are respected.
Every job is defensive: a failure is logged, recorded and never corrupts state.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..logging_setup import get_logger
from ..settings import get_config_store, get_settings
from ..timeutil import IST, now_ist

log = get_logger(__name__, component="scheduler")


@dataclass
class JobRecord:
    job_id: str
    name: str
    schedule: str
    last_run: Optional[dt.datetime] = None
    last_status: str = "NEVER_RUN"
    last_detail: str = ""
    run_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "schedule": self.schedule,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_status": self.last_status,
            "last_detail": self.last_detail[:300],
            "run_count": self.run_count,
            "error_count": self.error_count,
        }


class ChiefScheduler:
    """Wraps APScheduler with the domain's daily workflow."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self.settings = get_settings()
        self.config = get_config_store()
        self.scheduler = BackgroundScheduler(timezone=IST, job_defaults={"coalesce": True, "max_instances": 1})
        self.jobs: Dict[str, JobRecord] = {}
        self._register()

    # ------------------------------------------------------------------ setup
    def _register(self) -> None:
        execution_cfg = self.config.load("execution")
        schedule = (execution_cfg.get("schedule") or {})
        tz = IST

        def add(job_id: str, name: str, func: Callable[[], Any], hour: int, minute: int, weekday: str = "*") -> None:
            self.scheduler.add_job(
                self._wrap(job_id, name, func),
                CronTrigger(day_of_week=weekday, hour=hour, minute=minute, timezone=tz),
                id=job_id,
                name=name,
                replace_existing=True,
            )
            self.jobs[job_id] = JobRecord(job_id=job_id, name=name, schedule=f"{weekday} {hour:02d}:{minute:02d} IST")

        add("pre_market", "Pre-market preparation", self.pre_market, 8, 45)
        add("market_open_checks", "Market-open readiness", self.market_open_checks, 9, 16)
        add("scan", "Opportunity scan", self.scan_cycle, 9, 30)
        add("trading_cycle", "Trading cycle", self.scan_cycle, 10, 0)
        add("post_market", "Post-market reconciliation and statistics", self.post_market, 15, 45)
        add("daily_review", "Daily performance review", self.daily_review, 16, 15)
        add("weekly_research", "Weekly research: hypotheses", self.weekly_hypotheses, 10, 0, weekday="sun")
        add("nightly_maintenance", "Nightly maintenance", self.nightly_maintenance, 23, 30)

    def _wrap(self, job_id: str, name: str, func: Callable[[], Any]) -> Callable[[], Any]:
        def runner() -> None:
            record = self.jobs.setdefault(job_id, JobRecord(job_id=job_id, name=name, schedule="manual"))
            record.last_run = now_ist()
            try:
                result = func() or {}
                record.last_status = "OK"
                record.last_detail = str(result)[:400]
                record.run_count += 1
                log.info("scheduled job completed", context={"job": job_id, "detail": record.last_detail})
            except Exception as exc:
                record.last_status = "ERROR"
                record.last_detail = str(exc)
                record.error_count += 1
                log.exception("scheduled job failed", context={"job": job_id, "error": str(exc)})
                try:
                    from ..monitoring.notifications import EventKind, get_notifications

                    get_notifications().notify(
                        EventKind.SYSTEM_ERROR,
                        f"Scheduled job failed: {name}",
                        str(exc),
                    )
                except Exception:  # pragma: no cover
                    pass

        runner.__name__ = f"job_{job_id}"
        return runner

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.scheduler.running,
            "jobs": len(self.jobs),
            "timezone": str(IST),
            "items": [record.to_dict() for record in self.jobs.values()],
            "next_runs": [
                {"job": job.id, "next_run": job.next_run_time.isoformat() if job.next_run_time else None}
                for job in self.scheduler.get_jobs()
            ],
        }

    def run_now(self, job_id: str) -> Dict[str, Any]:
        mapping = {
            "pre_market": self.pre_market,
            "market_open_checks": self.market_open_checks,
            "scan": self.scan_cycle,
            "trading_cycle": self.scan_cycle,
            "post_market": self.post_market,
            "daily_review": self.daily_review,
            "weekly_research": self.weekly_hypotheses,
            "nightly_maintenance": self.nightly_maintenance,
        }
        if job_id not in mapping:
            return {"ok": False, "error": f"unknown job '{job_id}'", "available": sorted(mapping)}
        record = self.jobs.setdefault(job_id, JobRecord(job_id=job_id, name=job_id, schedule="manual"))
        try:
            result = mapping[job_id]() or {}
            record.last_run = now_ist()
            record.last_status = "OK"
            record.last_detail = str(result)[:400]
            record.run_count += 1
            return {"ok": True, "job": job_id, "result": result}
        except Exception as exc:
            record.last_run = now_ist()
            record.last_status = "ERROR"
            record.last_detail = str(exc)
            record.error_count += 1
            return {"ok": False, "job": job_id, "error": str(exc)}

    # ------------------------------------------------------------------- jobs
    def pre_market(self) -> Dict[str, Any]:
        """Instruments, broker check, funds, health, data update, universe build."""
        state = self.state
        result: Dict[str, Any] = {"started_at": now_ist().isoformat()}

        if not state.calendar.is_trading_day():
            return {"skipped": True, "reason": "not a trading day"}

        try:
            state.calendar.refresh_holidays()
        except Exception as exc:
            result["holiday_refresh_error"] = str(exc)

        # Refresh the instrument master (BOD) unless it is already fresh.
        try:
            age = state.broker.instruments.cache_age_hours("nse")
            if age is None or age > 12:
                from ...broker.upstox_instruments import load_instruments_into_db

                state.broker.instruments.download("nse", force=False)
                state.broker.instruments.load("nse", auto_download=False)
                state.broker.watchlist.load()
            result["instruments"] = state.broker.instruments.size
        except Exception as exc:
            result["instruments_error"] = str(exc)

        if state.broker.has_token:
            try:
                result["auth"] = state.broker.auth.verify()
            except Exception as exc:
                result["auth_error"] = str(exc)
            try:
                funds = state.broker.portfolio.funds("SEC")
                result["available_cash"] = funds.available_cash
            except Exception as exc:
                result["funds_error"] = str(exc)

        # Extend the historical cache so indicators are current.
        try:
            from ..data.db import session_scope
            from ..data.historical_downloader import HistoricalDownloader

            watchlist = state.broker.watchlist
            watchlist.load()
            resolved = watchlist.resolved(
                state.broker.instruments if state.broker.instruments.is_loaded else None
            )
            instruments = [
                {"instrument_key": row.get("instrument_key") or f"SIM|{row['symbol']}", "symbol": row["symbol"]}
                for row in resolved
            ]
            end = now_ist().date()
            with session_scope() as session:
                downloader = HistoricalDownloader(state.provider, session)
                report = downloader.download(
                    instruments[:120], "1m", start_date=end - dt.timedelta(days=7), end_date=end
                )
                session.commit()
            result["data"] = report.to_dict()
        except Exception as exc:
            result["data_error"] = str(exc)

        result["health"] = state.health_report()["status"]
        return result

    def market_open_checks(self) -> Dict[str, Any]:
        state = self.state
        if not state.calendar.is_trading_day():
            return {"skipped": True, "reason": "not a trading day"}
        report = state.preflight.run(
            deep=False,
            components={
                "broker": state.broker,
                "risk_engine": state.risk_engine,
                "kill_switch": state.kill_switch,
                "strategy_version": state.strategy_version,
            },
        )
        return {"preflight_passed": report.passed, "blocking": [c.name for c in report.blocking]}

    def scan_cycle(self) -> Dict[str, Any]:
        """One trading cycle: mark to market, manage exits, scan, execute."""
        state = self.state
        if not state.calendar.is_tradable_now():
            return {"skipped": True, "reason": "market is not in a tradable session"}

        # 1. mark to market
        try:
            from ..data import candle_store
            from ..data.db import session_scope

            positions = state.paper.open_positions()
            if positions:
                prices: Dict[str, float] = {}
                with session_scope() as session:
                    for position in positions:
                        candles = candle_store.load_candles(
                            session,
                            position.instrument_key,
                            "1m",
                            now_ist().date(),
                            now_ist().date(),
                            limit=500,
                        )
                        if candles:
                            prices[position.instrument_key] = candles[-1].close
                if prices:
                    state.paper.mark_to_market(prices)
        except Exception as exc:
            log.warning("mark-to-market failed", context={"error": str(exc)})

        # 2. run the trading cycle through the API layer's implementation
        try:
            from ..api.routers.trading import _scan_now

            scan = _scan_now(state, top_n=10)
            state.last_scan = scan
            state.counters.last_scan_at = now_ist()
        except Exception as exc:
            state.counters.last_error = str(exc)
            return {"ok": False, "error": str(exc)}

        return {
            "ok": True,
            "opportunities": len(scan.get("longs", [])) + len(scan.get("shorts", [])),
            "universe_scanned": scan.get("universe_scanned", 0),
            "regime": scan.get("regime"),
        }

    def post_market(self) -> Dict[str, Any]:
        """Reconcile, write the daily performance row, journal the day."""
        state = self.state
        result: Dict[str, Any] = {"date": now_ist().date().isoformat()}

        # Square off anything still open before the close.
        try:
            closed = []
            for position in list(state.paper.open_positions()):
                price = position.ltp or position.entry_price
                closed.append(state.paper.close_position(position, price, now_ist(), "SQUARE_OFF"))
            result["squared_off"] = len(closed)
        except Exception as exc:
            result["square_off_error"] = str(exc)

        # Reconcile
        try:
            internal = {p.instrument_key: p.quantity for p in state.paper.open_positions()}
            broker = dict(internal)
            if state.broker.has_token:
                broker = {k: p.quantity for k, p in state.broker.positions.net_positions().items()}
            report = state.reconciler.reconcile(internal, broker, mode=state.settings.effective_mode().value)
            result["reconciliation"] = report.to_dict()
        except Exception as exc:
            result["reconciliation_error"] = str(exc)

        # Persist the daily performance row and journal the closed trades.
        try:
            from sqlalchemy import select

            from ..data.db import session_scope
            from ..data.schema import DailyPerformance, TradeJournal

            portfolio = state.portfolio.update_from_paper(state.paper)
            today = now_ist().date()
            trades_today = [
                trade
                for trade in state.paper.closed_trades
                if str(trade.get("exit_ts", ""))[:10] == today.isoformat()
            ]

            with session_scope() as session:
                existing = session.execute(
                    select(DailyPerformance).where(DailyPerformance.trading_date == today)
                ).scalar_one_or_none()
                if existing is None:
                    existing = DailyPerformance(trading_date=today)
                    session.add(existing)
                existing.mode = state.settings.effective_mode().value
                existing.strategy_version = state.strategy_version
                existing.starting_equity = portfolio.starting_equity_today
                existing.ending_equity = portfolio.equity
                existing.net_pnl = sum(float(t.get("net_pnl", 0) or 0) for t in trades_today)
                existing.fees = sum(float(t.get("fees", 0) or 0) for t in trades_today)
                existing.return_pct = (
                    existing.net_pnl / existing.starting_equity if existing.starting_equity else 0.0
                )
                existing.trades = len(trades_today)
                existing.wins = sum(1 for t in trades_today if float(t.get("net_pnl", 0) or 0) > 0)
                existing.losses = sum(1 for t in trades_today if float(t.get("net_pnl", 0) or 0) < 0)
                existing.expectancy_r = (
                    sum(float(t.get("r_multiple", 0) or 0) for t in trades_today) / len(trades_today)
                    if trades_today
                    else 0.0
                )
                existing.max_drawdown_pct = portfolio.drawdown_pct
                existing.regime_summary = {"regime": state.last_scan.get("regime") if state.last_scan else None}

                for trade in trades_today:
                    already = session.execute(
                        select(TradeJournal).where(TradeJournal.trade_id == trade.get("trade_id"))
                    ).scalar_one_or_none()
                    if already is not None:
                        continue
                    session.add(
                        TradeJournal(
                            trade_id=str(trade.get("trade_id")),
                            signal_id=trade.get("signal_id"),
                            strategy_family="orb_vwap_retest",
                            strategy_version=str(trade.get("strategy_version") or state.strategy_version or ""),
                            instrument_key=str(trade.get("instrument_key")),
                            symbol=str(trade.get("symbol")),
                            sector=trade.get("sector"),
                            direction=str(trade.get("direction")),
                            quantity=int(trade.get("quantity", 0) or 0),
                            entry_price=float(trade.get("entry_price", 0) or 0),
                            exit_price=float(trade.get("exit_price", 0) or 0),
                            entry_ts=_parse_dt(trade.get("entry_ts")),
                            exit_ts=_parse_dt(trade.get("exit_ts")),
                            holding_minutes=float(trade.get("holding_minutes", 0) or 0),
                            gross_pnl=float(trade.get("gross_pnl", 0) or 0),
                            fees=float(trade.get("fees", 0) or 0),
                            slippage_cost=float(trade.get("slippage_cost", 0) or 0),
                            net_pnl=float(trade.get("net_pnl", 0) or 0),
                            initial_risk=float(trade.get("initial_risk", 0) or 0),
                            r_multiple=float(trade.get("r_multiple", 0) or 0),
                            mae_r=float(trade.get("mae_r", 0) or 0),
                            mfe_r=float(trade.get("mfe_r", 0) or 0),
                            exit_reason=str(trade.get("exit_reason", "")),
                            regime_at_entry=trade.get("regime_at_entry"),
                            features=trade.get("features") or {},
                            mode=str(trade.get("mode", "PAPER")),
                        )
                    )
                session.commit()

            try:
                from ..research.artifacts import get_artifact_store

                store = get_artifact_store()
                for trade in trades_today:
                    store.trade(trade)
            except Exception as exc:  # mirrors are best-effort
                log.warning("could not mirror trades to research_data", context={"error": str(exc)})
            result["trades_journalled"] = len(trades_today)
        except Exception as exc:
            log.exception("post-market persistence failed")
            result["persistence_error"] = str(exc)

        # Refresh the champion configuration in case it changed during the day.
        state.reload_champion()
        return result

    def daily_review(self) -> Dict[str, Any]:
        from ..api.routers.journal import review

        return review(date=now_ist().date(), state=self.state)

    def weekly_hypotheses(self) -> Dict[str, Any]:
        """Sunday research: analyse the journal and record new hypotheses."""
        state = self.state
        from ..data.db import session_scope
        from ..research.hypotheses import HypothesisGenerator

        with session_scope() as session:
            generator = HypothesisGenerator(session)
            proposals = generator.generate(
                state.strategy_version or "", state.champion_config, max_proposals=5
            )
            saved = []
            for proposal in proposals:
                hypothesis_id = generator.persist(proposal, state.strategy_version or "")
                payload = proposal.to_dict()
                payload["hypothesis_id"] = hypothesis_id
                saved.append(payload)
            session.commit()
        return {"proposals": len(saved), "details": saved[:3]}

    def nightly_maintenance(self) -> Dict[str, Any]:
        """Backups, cleanup and heavier off-hours work."""
        result: Dict[str, Any] = {"started_at": now_ist().isoformat()}
        try:
            from ..monitoring.maintenance import backup_database, prune_old_events

            result["backup"] = backup_database()
            result["pruned"] = prune_old_events()
        except Exception as exc:
            result["maintenance_error"] = str(exc)

        # Aggregate higher timeframes for the instruments we hold.
        try:
            from ..data.db import session_scope
            from ..data.historical_downloader import HistoricalDownloader

            with session_scope() as session:
                downloader = HistoricalDownloader(self.state.provider, session)
                keys = [
                    row.instrument_key
                    for row in _universe_keys(session)
                ][:60]
                written = downloader.build_higher_timeframes(keys, ("5m", "15m", "30m", "60m"))
                session.commit()
            result["higher_timeframes"] = len(written)
        except Exception as exc:
            result["timeframe_error"] = str(exc)
        return result


def _parse_dt(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    try:
        parsed = dt.datetime.fromisoformat(str(value))
        return parsed.astimezone(IST) if parsed.tzinfo else parsed.replace(tzinfo=IST)
    except (TypeError, ValueError):
        return now_ist()


def _universe_keys(session: Any) -> List[Any]:
    from sqlalchemy import select

    from ..data.schema import Instrument

    return session.execute(select(Instrument).where(Instrument.in_universe.is_(True))).scalars().all()


def build_scheduler(state: Any) -> ChiefScheduler:
    return ChiefScheduler(state)


__all__ = ["ChiefScheduler", "build_scheduler", "JobRecord"]
