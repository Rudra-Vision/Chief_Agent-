"""Portfolio engine.

Owns the authoritative view of equity, cash, open positions, open risk, gross
exposure and daily P&L. The risk engine reads from here; nothing else computes
these numbers independently, so there is exactly one source of truth.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..logging_setup import get_logger
from ..risk.engine import RiskContext
from ..timeutil import IST, now_ist

log = get_logger(__name__, component="portfolio")


@dataclass
class PortfolioState:
    mode: str = "PAPER"
    starting_equity_today: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
    fees_today: float = 0.0
    peak_equity: float = 0.0
    open_risk_pct: float = 0.0
    gross_exposure_pct: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    consecutive_rejections: int = 0
    weekly_return_pct: float = 0.0
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    strategy_returns: Dict[str, float] = field(default_factory=dict)
    as_of: Any = None

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "as_of": self.as_of.isoformat() if hasattr(self.as_of, "isoformat") else str(self.as_of),
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(self.positions_value, 2),
            "starting_equity_today": round(self.starting_equity_today, 2),
            "realized_pnl_today": round(self.realized_pnl_today, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "fees_today": round(self.fees_today, 2),
            "daily_return_pct": round(
                (self.realized_pnl_today + self.unrealized_pnl) / self.starting_equity_today, 6
            ) if self.starting_equity_today else 0.0,
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 6),
            "open_risk_pct": round(self.open_risk_pct, 6),
            "gross_exposure_pct": round(self.gross_exposure_pct, 6),
            "open_position_count": len(self.open_positions),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_rejections": self.consecutive_rejections,
            "weekly_return_pct": round(self.weekly_return_pct, 6),
            "open_positions": self.open_positions,
            "strategy_returns": {k: round(v, 6) for k, v in self.strategy_returns.items()},
        }


class PortfolioEngine:
    """Maintains portfolio state from paper positions or broker positions."""

    def __init__(self, mode: str = "PAPER") -> None:
        self.mode = mode
        self.state = PortfolioState(mode=mode, as_of=now_ist())
        self._day = now_ist().date()
        self._starting_equity_by_day: Dict[dt.date, float] = {}
        self._daily_returns: Dict[dt.date, float] = {}
        self._trade_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ from paper
    def update_from_paper(self, paper_engine: Any) -> PortfolioState:
        snapshot = paper_engine.snapshot()
        account = snapshot["account"]
        positions = paper_engine.open_positions()
        equity = account["equity"]

        self._roll_day(equity)
        if not self.state.peak_equity or equity > self.state.peak_equity:
            self.state.peak_equity = max(self.state.peak_equity, equity)

        self.state.equity = equity
        self.state.cash = account["cash"]
        self.state.positions_value = account.get("positions_value", 0.0)
        self.state.unrealized_pnl = account["unrealized_pnl"]
        self.state.fees_today = paper_engine.account.fees_paid
        self.state.open_positions = [p.to_dict() for p in positions]
        self.state.realized_pnl_today = self._realized_today(paper_engine.closed_trades)
        self.state.trades_today = len([t for t in paper_engine.closed_trades if self._is_today(t.get("exit_ts"))])
        self.state.consecutive_losses = self._consecutive_losses(paper_engine.closed_trades)
        self.state.open_risk_pct = self._open_risk(positions, equity)
        self.state.gross_exposure_pct = self._gross_exposure(positions, equity)
        self.state.weekly_return_pct = self._weekly_return(equity)
        self.state.strategy_returns = self._strategy_returns(paper_engine.closed_trades, equity)
        self.state.as_of = now_ist()
        return self.state

    # --------------------------------------------------------- from broker
    def update_from_broker(
        self,
        *,
        equity: float,
        available_cash: float,
        broker_positions: Sequence[Any],
        internal_positions: Optional[Sequence[Any]] = None,
    ) -> PortfolioState:
        positions = internal_positions if internal_positions is not None else broker_positions
        exposed = []
        unrealized = 0.0
        for position in positions:
            if hasattr(position, "to_dict"):
                payload = position.to_dict()
            else:
                payload = dict(position)
            exposed.append(payload)
            unrealized += float(payload.get("unrealized_pnl") or payload.get("pnl") or 0.0)

        self._roll_day(equity)
        self.state.equity = equity
        self.state.cash = available_cash
        self.state.unrealized_pnl = unrealized
        self.state.open_positions = exposed
        self.state.open_risk_pct = self._open_risk_dicts(exposed, equity)
        self.state.gross_exposure_pct = self._gross_exposure_dicts(exposed, equity)
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.weekly_return_pct = self._weekly_return(equity)
        self.state.as_of = now_ist()
        return self.state

    # --------------------------------------------------------------- helpers
    def _roll_day(self, equity: float) -> None:
        today = now_ist().date()
        if self._day != today:
            self._record_daily_return()
            self._day = today
            self.state.realized_pnl_today = 0.0
            self.state.trades_today = 0
        self._starting_equity_by_day.setdefault(today, equity)
        self.state.starting_equity_today = self._starting_equity_by_day[today]

    def _record_daily_return(self) -> None:
        start = self._starting_equity_by_day.get(self._day)
        if start:
            self._daily_returns[self._day] = (self.state.equity - start) / start

    @staticmethod
    def _is_today(value: Any) -> bool:
        if value is None:
            return False
        try:
            if isinstance(value, str):
                value = dt.datetime.fromisoformat(value)
            day = value.astimezone(IST).date() if value.tzinfo else value.date()
            return day == now_ist().date()
        except Exception:
            return False

    def _realized_today(self, trades: Sequence[Mapping[str, Any]]) -> float:
        return sum(float(t.get("net_pnl", 0) or 0) for t in trades if self._is_today(t.get("exit_ts")))

    @staticmethod
    def _consecutive_losses(trades: Sequence[Mapping[str, Any]]) -> int:
        count = 0
        for trade in reversed(list(trades)):
            pnl = float(trade.get("net_pnl", 0) or 0)
            if pnl < 0:
                count += 1
            elif pnl > 0:
                break
        return count

    @staticmethod
    def _open_risk(positions: Sequence[Any], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for position in positions:
            quantity = abs(position.quantity * position.remaining_fraction)
            risk_per_share = abs(position.entry_price - position.current_stop)
            total += quantity * risk_per_share
        return total / equity

    @staticmethod
    def _open_risk_dicts(positions: Sequence[Mapping[str, Any]], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = 0.0
        for position in positions:
            quantity = abs(float(position.get("quantity", 0) or 0))
            entry = float(position.get("entry_price") or position.get("average_price") or 0)
            stop = float(position.get("current_stop") or position.get("initial_stop") or entry)
            total += quantity * abs(entry - stop)
        return total / equity

    @staticmethod
    def _gross_exposure(positions: Sequence[Any], equity: float) -> float:
        if equity <= 0:
            return 0.0
        return sum(position.exposure for position in positions) / equity

    @staticmethod
    def _gross_exposure_dicts(positions: Sequence[Mapping[str, Any]], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total = sum(
            abs(float(p.get("quantity", 0) or 0)) * float(p.get("ltp") or p.get("average_price") or 0)
            for p in positions
        )
        return total / equity

    def _weekly_return(self, equity: float) -> float:
        week_ago = now_ist().date() - dt.timedelta(days=7)
        candidates = [v for day, v in self._starting_equity_by_day.items() if day >= week_ago]
        base = min(candidates) if candidates else equity
        if not base:
            return 0.0
        return (equity - base) / base

    @staticmethod
    def _strategy_returns(trades: Sequence[Mapping[str, Any]], equity: float) -> Dict[str, float]:
        if equity <= 0:
            return {}
        out: Dict[str, float] = {}
        for trade in trades:
            version = str(trade.get("strategy_version") or "unknown")
            out[version] = out.get(version, 0.0) + float(trade.get("net_pnl", 0) or 0) / equity
        return out

    # ------------------------------------------------------------- risk context
    def risk_context(
        self,
        *,
        strategy_version: str = "",
        suspended_strategies: Optional[Sequence[str]] = None,
        kill_switch_engaged: bool = False,
        data_safe_mode: bool = False,
        data_stale: bool = False,
        reconciliation_pending: bool = False,
        broker_healthy: bool = True,
        engine_healthy: bool = True,
        order_attempts_today: int = 0,
        consecutive_rejections: int = 0,
        paused_until: Optional[dt.datetime] = None,
    ) -> RiskContext:
        """Build the :class:`RiskContext` the risk engine evaluates against."""
        return RiskContext(
            account_equity=self.state.equity,
            available_cash=self.state.cash,
            starting_equity_today=self.state.starting_equity_today or self.state.equity,
            realised_pnl_today=self.state.realized_pnl_today,
            unrealised_pnl=self.state.unrealized_pnl,
            open_positions=list(self.state.open_positions),
            open_risk_pct=self.state.open_risk_pct,
            gross_exposure_pct=self.state.gross_exposure_pct,
            trades_today=self.state.trades_today,
            order_attempts_today=order_attempts_today,
            consecutive_losses=self.state.consecutive_losses,
            consecutive_rejections=consecutive_rejections,
            weekly_return_pct=self.state.weekly_return_pct,
            peak_equity=self.state.peak_equity,
            strategy_version=strategy_version,
            suspended_strategies=list(suspended_strategies or []),
            strategy_returns=dict(self.state.strategy_returns),
            kill_switch_engaged=kill_switch_engaged,
            data_safe_mode=data_safe_mode,
            data_stale=data_stale,
            reconciliation_pending=reconciliation_pending,
            broker_healthy=broker_healthy,
            engine_healthy=engine_healthy,
            paused_until=paused_until,
        )

    def snapshot(self) -> Dict[str, Any]:
        return self.state.to_dict()


__all__ = ["PortfolioEngine", "PortfolioState"]
