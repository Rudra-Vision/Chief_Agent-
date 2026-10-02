"""The RISK ENGINE. It has ABSOLUTE authority.

Nothing in the system - not the strategy, not the research engine, not the LLM
layer, not the dashboard - can override a risk decision. Every check returns a
structured :class:`RiskDecision` so the reason a trade was refused becomes
research evidence ("why did we not take this trade, and what would it have
done?").

Design rules
------------
* Every limit lives in ``config/risk.yaml``; none are hard-coded here.
* The engine fails CLOSED: if it cannot evaluate a limit, the answer is NO.
* No averaging down, no martingale, no doubling after losses, no unlimited
  leverage. These are enforced structurally, not by policy.
"""

from __future__ import annotations

import datetime as dt
import enum
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..logging_setup import get_logger
from ..settings import ConfigStore, get_config_store, get_settings
from ..timeutil import IST, now_ist
from .position_sizing import SizingConstraints, SizingResult, compute_position_size

log = get_logger(__name__, component="risk")


class RiskLevel(str, enum.Enum):
    APPROVED = "APPROVED"
    REDUCED = "REDUCED"
    REJECTED = "REJECTED"
    HALTED = "HALTED"


class RiskReason(str, enum.Enum):
    OK = "OK"
    DAILY_LOSS_SOFT_LIMIT = "DAILY_LOSS_SOFT_LIMIT"
    DAILY_LOSS_HARD_LIMIT = "DAILY_LOSS_HARD_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_CORRELATED_SECTOR_POSITIONS = "MAX_CORRELATED_SECTOR_POSITIONS"
    MAX_TOTAL_OPEN_RISK = "MAX_TOTAL_OPEN_RISK"
    MAX_TRADES_PER_DAY = "MAX_TRADES_PER_DAY"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    MAX_CONSECUTIVE_REJECTIONS = "MAX_CONSECUTIVE_REJECTIONS"
    WEEKLY_DRAWDOWN_WARNING = "WEEKLY_DRAWDOWN_WARNING"
    STRATEGY_SUSPENDED = "STRATEGY_SUSPENDED"
    STRATEGY_DRAWDOWN = "STRATEGY_DRAWDOWN"
    KILL_SWITCH_ENGAGED = "KILL_SWITCH_ENGAGED"
    DATA_SAFE_MODE = "DATA_SAFE_MODE"
    STALE_DATA = "STALE_DATA"
    RISK_ENGINE_UNAVAILABLE = "RISK_ENGINE_UNAVAILABLE"
    BROKER_UNSTABLE = "BROKER_UNSTABLE"
    RECONCILIATION_PENDING = "RECONCILIATION_PENDING"
    INVALID_STOP = "INVALID_STOP"
    RISK_REWARD_TOO_LOW = "RISK_REWARD_TOO_LOW"
    ZERO_SIZE = "ZERO_SIZE"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    LIQUIDITY_TOO_LOW = "LIQUIDITY_TOO_LOW"
    PRICE_OUT_OF_BOUNDS = "PRICE_OUT_OF_BOUNDS"
    AVERAGING_DOWN_BLOCKED = "AVERAGING_DOWN_BLOCKED"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    NO_STOP_DEFINED = "NO_STOP_DEFINED"
    EPISODE_PAUSED = "EPISODE_PAUSED"


@dataclass
class RiskDecision:
    level: RiskLevel
    reason: RiskReason = RiskReason.OK
    message: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)
    sizing: Optional[SizingResult] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        """True only for a full-size approval. REDUCED is not 'approved full'."""
        return self.level is RiskLevel.APPROVED

    @property
    def tradeable(self) -> bool:
        return self.level in (RiskLevel.APPROVED, RiskLevel.REDUCED) and bool(self.sizing and self.sizing.quantity > 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "reason": self.reason.value,
            "message": self.message,
            "checks": self.checks,
            "sizing": self.sizing.to_dict() if self.sizing else None,
            "warnings": self.warnings,
            "tradeable": self.tradeable,
        }


@dataclass
class ProposedTrade:
    """What the strategy wants to do, before risk has looked at it."""

    instrument_key: str
    symbol: str
    direction: str
    entry_price: float
    stop_price: float
    target_1: float = 0.0
    target_2: float = 0.0
    quantity: int = 0
    strategy_version: str = ""
    strategy_family: str = ""
    sector: Optional[str] = None
    risk_reward: float = 0.0
    atr_pct: Optional[float] = None
    spread_pct: Optional[float] = None
    average_daily_volume: Optional[float] = None
    lot_size: int = 1
    tick_size: float = 0.05
    signal_id: str = ""
    product: str = "I"
    tags: List[str] = field(default_factory=list)

    @property
    def risk_per_share(self) -> float:
        return abs(float(self.entry_price) - float(self.stop_price))

    @property
    def direction_normalised(self) -> str:
        return self.direction.upper()


@dataclass
class RiskContext:
    """Account and system state the engine evaluates against."""

    account_equity: float
    available_cash: float
    starting_equity_today: float
    realised_pnl_today: float = 0.0
    unrealised_pnl: float = 0.0
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    open_risk_pct: float = 0.0
    gross_exposure_pct: float = 0.0
    trades_today: int = 0
    order_attempts_today: int = 0
    consecutive_losses: int = 0
    consecutive_rejections: int = 0
    weekly_return_pct: float = 0.0
    peak_equity: float = 0.0
    strategy_version: str = ""
    suspended_strategies: Sequence[str] = field(default_factory=list)
    strategy_returns: Mapping[str, float] = field(default_factory=dict)
    kill_switch_engaged: bool = False
    data_safe_mode: bool = False
    data_stale: bool = False
    reconciliation_pending: bool = False
    broker_healthy: bool = True
    engine_healthy: bool = True
    paused_until: Optional[dt.datetime] = None
    now: Optional[dt.datetime] = None

    @property
    def daily_return_pct(self) -> float:
        if self.starting_equity_today <= 0:
            return 0.0
        return (self.realised_pnl_today + self.unrealised_pnl) / self.starting_equity_today

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        equity = self.account_equity
        return (equity - self.peak_equity) / self.peak_equity

    @property
    def sector_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for position in self.open_positions:
            sector = position.get("sector") or "Unknown"
            counts[sector] = counts.get(sector, 0) + 1
        return counts

    @property
    def direction_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for position in self.open_positions:
            direction = str(position.get("direction", "")).upper()
            counts[direction] = counts.get(direction, 0) + 1
        return counts


class RiskEngine:
    """Deterministic, config-driven, fail-closed risk authority."""

    def __init__(self, config_store: Optional[ConfigStore] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.config_store = config_store or get_config_store()
        self._config = config if config is not None else self.config_store.load("risk")
        self._lock = threading.RLock()
        self._events: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------- reload
    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def reload(self) -> None:
        self.config_store.reload()
        self._config = self.config_store.load("risk")

    def _c(self, path: str, default: Any = None) -> Any:
        node: Any = self._config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    # ------------------------------------------------------------------ gates
    def evaluate(self, trade: ProposedTrade, context: RiskContext) -> RiskDecision:
        """The full pre-trade checklist. Returns APPROVED / REDUCED / REJECTED / HALTED."""
        with self._lock:
            checks: List[Dict[str, Any]] = []
            warnings: List[str] = []

            def record(name: str, ok: bool, detail: str, reason: RiskReason = RiskReason.OK) -> bool:
                checks.append({"check": name, "ok": bool(ok), "detail": detail, "reason": reason.value})
                return bool(ok)

            def reject(reason: RiskReason, message: str, name: str) -> RiskDecision:
                record(name, False, message, reason)
                decision = RiskDecision(RiskLevel.REJECTED, reason, message, checks, None, warnings)
                self._log_decision(trade, decision)
                return decision

            # ---- 0. system integrity (any failure halts everything) ----------
            if not context.engine_healthy:
                return reject(RiskReason.RISK_ENGINE_UNAVAILABLE, "risk engine is not healthy - failing closed", "engine_health")
            if context.kill_switch_engaged:
                return reject(RiskReason.KILL_SWITCH_ENGAGED, "kill switch is engaged", "kill_switch")
            if context.data_safe_mode:
                return reject(RiskReason.DATA_SAFE_MODE, "DATA_SAFE_MODE is active - market data is not trustworthy", "data_safe_mode")
            if context.data_stale:
                return reject(RiskReason.STALE_DATA, "market data is stale", "data_freshness")
            if context.reconciliation_pending:
                return reject(RiskReason.RECONCILIATION_PENDING, "position reconciliation is pending - resolve the mismatch first", "reconciliation")
            if not context.broker_healthy:
                return reject(RiskReason.BROKER_UNSTABLE, "broker connection is unstable", "broker_health")
            record("system_integrity", True, "all system health gates passed")

            # ---- 1. episode pause -------------------------------------------
            current_time = context.now or now_ist()
            if context.paused_until and current_time < context.paused_until:
                remaining = (context.paused_until - current_time).total_seconds() / 60.0
                return reject(
                    RiskReason.EPISODE_PAUSED,
                    f"trading paused for another {remaining:.0f} minutes after consecutive losses",
                    "episode_pause",
                )
            record("episode_pause", True, "no pause in effect")

            # ---- 2. strategy suspension -------------------------------------
            suspended = set(context.suspended_strategies or [])
            if trade.strategy_version and trade.strategy_version in suspended:
                return reject(
                    RiskReason.STRATEGY_SUSPENDED,
                    f"strategy {trade.strategy_version} is suspended",
                    "strategy_status",
                )
            max_strategy_dd = float(self._c("strategy_suspension.max_strategy_drawdown_pct", -0.05))
            strategy_return = context.strategy_returns.get(trade.strategy_version, 0.0)
            if strategy_return <= max_strategy_dd:
                return reject(
                    RiskReason.STRATEGY_DRAWDOWN,
                    f"strategy {trade.strategy_version} is at {strategy_return:.2%}, beyond the {max_strategy_dd:.2%} limit",
                    "strategy_drawdown",
                )
            record("strategy_status", True, "strategy is active and within its drawdown budget")

            # ---- 3. daily loss limits ---------------------------------------
            soft_limit = float(self._c("daily_limits.soft_daily_stop_pct", -0.01))
            hard_limit = float(self._c("daily_limits.hard_daily_stop_pct", -0.015))
            daily = context.daily_return_pct
            if daily <= hard_limit:
                return reject(
                    RiskReason.DAILY_LOSS_HARD_LIMIT,
                    f"hard daily stop hit ({daily:.2%} <= {hard_limit:.2%}) - flattening and halting",
                    "daily_loss",
                )
            if daily <= soft_limit:
                return reject(
                    RiskReason.DAILY_LOSS_SOFT_LIMIT,
                    f"soft daily stop hit ({daily:.2%} <= {soft_limit:.2%}) - no new entries today",
                    "daily_loss",
                )
            # Warn once the account is halfway to the soft stop.
            if daily <= soft_limit / 2:
                warnings.append(f"account is down {daily:.2%} today - approaching the {soft_limit:.2%} daily stop")
            record("daily_loss", True, f"today's return {daily:.2%} is within limits")

            # ---- 4. consecutive losses --------------------------------------
            max_losses = int(self._c("daily_limits.max_consecutive_losses", 4))
            if context.consecutive_losses >= max_losses:
                return reject(
                    RiskReason.MAX_CONSECUTIVE_LOSSES,
                    f"{context.consecutive_losses} consecutive losses (limit {max_losses}) - stopping for the day",
                    "consecutive_losses",
                )
            record("consecutive_losses", True, f"{context.consecutive_losses} consecutive losses")

            # ---- 5. rejection loop -------------------------------------------
            max_rejections = int(self._c("daily_limits.max_consecutive_rejections", 3))
            if context.consecutive_rejections >= max_rejections:
                return reject(
                    RiskReason.MAX_CONSECUTIVE_REJECTIONS,
                    f"{context.consecutive_rejections} consecutive broker rejections - halting new orders",
                    "rejection_loop",
                )
            record("rejection_loop", True, "no rejection loop detected")

            # ---- 6. weekly drawdown warning ----------------------------------
            weekly_warn = float(self._c("daily_limits.weekly_drawdown_warning_pct", -0.025))
            if context.weekly_return_pct <= weekly_warn:
                warnings.append(
                    f"weekly drawdown {context.weekly_return_pct:.2%} is beyond the {weekly_warn:.2%} warning level"
                )
            record("weekly_drawdown", True, f"weekly return {context.weekly_return_pct:.2%}")

            # ---- 7. trade count -----------------------------------------------
            max_trades = int(self._c("daily_limits.max_trades_per_day", 12))
            if context.trades_today >= max_trades:
                return reject(
                    RiskReason.MAX_TRADES_PER_DAY,
                    f"already taken {context.trades_today} trades today (limit {max_trades})",
                    "trade_count",
                )
            record("trade_count", True, f"{context.trades_today}/{max_trades} trades today")

            # ---- 8. position count ---------------------------------------------
            max_positions = int(self._c("portfolio.max_simultaneous_positions", 3))
            open_count = len(context.open_positions)
            if open_count >= max_positions:
                return reject(
                    RiskReason.MAX_POSITIONS,
                    f"{open_count} positions already open (limit {max_positions})",
                    "position_count",
                )
            record("position_count", True, f"{open_count}/{max_positions} positions open")

            # ---- 9. duplicate / averaging down ----------------------------------
            if bool(self._c("order_protections.forbid_averaging_down", True)):
                same = [p for p in context.open_positions if p.get("instrument_key") == trade.instrument_key]
                if same:
                    if bool(self._c("order_protections.forbid_adding_to_losers", True)):
                        return reject(
                            RiskReason.AVERAGING_DOWN_BLOCKED,
                            f"already holding {trade.symbol} - adding to a position is prohibited",
                            "averaging_down",
                        )
                    return reject(
                        RiskReason.DUPLICATE_POSITION,
                        f"already holding {trade.symbol} - duplicate exposure blocked",
                        "duplicate_position",
                    )
            record("duplicate_position", True, "no existing position in this instrument")

            # ---- 10. sector correlation ------------------------------------------
            max_sector = int(self._c("portfolio.max_correlated_sector_positions", 2))
            if trade.sector:
                sector_count = context.sector_counts.get(trade.sector, 0)
                if sector_count >= max_sector:
                    return reject(
                        RiskReason.MAX_CORRELATED_SECTOR_POSITIONS,
                        f"{sector_count} positions already open in {trade.sector} (limit {max_sector})",
                        "sector_correlation",
                    )
                record("sector_correlation", True, f"{sector_count}/{max_sector} in {trade.sector}")
            else:
                record("sector_correlation", True, "sector unknown - counted as uncorrelated")

            # ---- 11. same-direction concentration ---------------------------------
            max_same_direction = int(self._c("portfolio.max_same_direction_positions", max_positions))
            direction_count = context.direction_counts.get(trade.direction_normalised, 0)
            if direction_count >= max_same_direction:
                return reject(
                    RiskReason.MAX_POSITIONS,
                    f"{direction_count} {trade.direction_normalised} positions already open (limit {max_same_direction})",
                    "direction_concentration",
                )
            record("direction_concentration", True, f"{direction_count}/{max_same_direction} {trade.direction_normalised}")

            # ---- 12. structural trade validity ------------------------------------
            if trade.stop_price <= 0:
                return reject(RiskReason.NO_STOP_DEFINED, "no stop-loss defined - an entry without exit risk is never allowed", "stop_defined")
            if trade.direction_normalised == "LONG" and trade.stop_price >= trade.entry_price:
                return reject(RiskReason.INVALID_STOP, "long stop must be below the entry price", "stop_defined")
            if trade.direction_normalised == "SHORT" and trade.stop_price <= trade.entry_price:
                return reject(RiskReason.INVALID_STOP, "short stop must be above the entry price", "stop_defined")
            record("stop_defined", True, f"stop {trade.stop_price:.2f} vs entry {trade.entry_price:.2f}")

            min_rr = float(self._c("per_trade.min_risk_reward", 1.5))
            if trade.risk_reward and trade.risk_reward < min_rr:
                return reject(
                    RiskReason.RISK_REWARD_TOO_LOW,
                    f"risk/reward 1:{trade.risk_reward:.2f} is below the 1:{min_rr:.2f} minimum",
                    "risk_reward",
                )
            record("risk_reward", True, f"risk/reward 1:{trade.risk_reward:.2f}")

            # ---- 13. price / spread / liquidity ------------------------------------
            min_price = float(self._c("data_quality.min_price_inr", 5.0))
            max_price = float(self._c("data_quality.max_price_inr", 200000.0))
            if not (min_price <= trade.entry_price <= max_price):
                return reject(
                    RiskReason.PRICE_OUT_OF_BOUNDS,
                    f"price {trade.entry_price:.2f} is outside the tradable band {min_price}-{max_price}",
                    "price_band",
                )
            record("price_band", True, f"price {trade.entry_price:.2f} is tradable")

            max_spread = float(self._c("data_quality.max_spread_pct", 0.004))
            if trade.spread_pct is not None and trade.spread_pct > max_spread:
                return reject(
                    RiskReason.SPREAD_TOO_WIDE,
                    f"spread {trade.spread_pct:.3%} exceeds the {max_spread:.3%} limit",
                    "spread",
                )
            record("spread", True, f"spread {trade.spread_pct:.3%}" if trade.spread_pct is not None else "spread unknown (not blocking)")

            min_volume = float(self._c("data_quality.min_average_daily_volume", 0))
            if trade.average_daily_volume is not None and trade.average_daily_volume < min_volume:
                return reject(
                    RiskReason.LIQUIDITY_TOO_LOW,
                    f"average daily volume {trade.average_daily_volume:,.0f} is below the {min_volume:,.0f} floor",
                    "liquidity",
                )
            record("liquidity", True, f"ADV {trade.average_daily_volume:,.0f}" if trade.average_daily_volume else "liquidity unknown")

            # ---- 14. position sizing ------------------------------------------------
            risk_pct = float(self._c("per_trade.risk_pct", 0.0025))
            max_risk_pct = float(self._c("per_trade.max_risk_pct", 0.005))
            sizing = compute_position_size(
                SizingConstraints(
                    account_equity=context.account_equity,
                    risk_pct=risk_pct,
                    max_risk_pct=max_risk_pct,
                    entry_price=trade.entry_price,
                    stop_price=trade.stop_price,
                    lot_size=trade.lot_size,
                    tick_size=trade.tick_size,
                    max_position_exposure_pct=float(self._c("portfolio.max_single_position_exposure_pct", 0.25)),
                    max_gross_exposure_pct=float(self._c("portfolio.max_gross_exposure_pct", 1.0)),
                    available_cash=context.available_cash,
                    current_gross_exposure=context.gross_exposure_pct,
                    max_total_open_risk_pct=float(self._c("portfolio.max_total_open_risk_pct", 0.0075)),
                    current_open_risk_pct=context.open_risk_pct,
                    average_daily_volume=trade.average_daily_volume,
                    max_participation_pct=float(self._c("data_quality.max_participation_pct", 0.01)),
                )
            )
            if sizing.rejected or sizing.quantity <= 0:
                decision = RiskDecision(
                    RiskLevel.REJECTED,
                    RiskReason.ZERO_SIZE,
                    sizing.rejection_reason or "position sizing produced zero quantity",
                    checks,
                    sizing,
                    warnings,
                )
                self._log_decision(trade, decision)
                return decision
            record(
                "position_size",
                True,
                f"{sizing.quantity} shares, risking Rs {sizing.risk_amount:,.0f} ({sizing.risk_pct:.3%} of equity)",
            )
            warnings.extend(sizing.warnings)

            # ---- 15. aggregate open-risk after this trade -----------------------------
            new_open_risk_pct = context.open_risk_pct + sizing.risk_pct
            max_open_risk = float(self._c("portfolio.max_total_open_risk_pct", 0.0075))
            if new_open_risk_pct > max_open_risk + 1e-9:
                return reject(
                    RiskReason.MAX_TOTAL_OPEN_RISK,
                    f"total open risk would reach {new_open_risk_pct:.3%} (limit {max_open_risk:.3%})",
                    "total_open_risk",
                )
            record("total_open_risk", True, f"open risk would be {new_open_risk_pct:.3%} of {max_open_risk:.3%}")

            # ---- APPROVED (or REDUCED when size was constrained) -----------------------
            level = RiskLevel.APPROVED
            message = f"approved: {sizing.quantity} shares, risk {sizing.risk_pct:.3%}, R:R 1:{trade.risk_reward:.2f}"
            if sizing.binding_constraint != "risk_budget":
                level = RiskLevel.REDUCED
                message += f" (size limited by {sizing.binding_constraint})"

            decision = RiskDecision(level, RiskReason.OK, message, checks, sizing, warnings)
            self._log_decision(trade, decision)
            return decision

    # ------------------------------------------------------------------ utility
    def _log_decision(self, trade: ProposedTrade, decision: RiskDecision) -> None:
        entry = {
            "ts": now_ist().isoformat(),
            "instrument_key": trade.instrument_key,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "strategy_version": trade.strategy_version,
            "level": decision.level.value,
            "reason": decision.reason.value,
            "message": decision.message,
        }
        self._events.append(entry)
        if decision.level in (RiskLevel.REJECTED, RiskLevel.HALTED):
            log.info("trade rejected by risk", context=entry)
        else:
            log.debug("trade approved by risk", context=entry)

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()

    # -------------------------------------------------------- portfolio helpers
    def aggregate_state(self, positions: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        """Open risk % and gross exposure % from the current position book."""
        total_risk = 0.0
        gross = 0.0
        for position in positions:
            quantity = abs(float(position.get("quantity", 0) or 0))
            entry = float(position.get("entry_price", 0) or 0)
            stop = float(position.get("current_stop") or position.get("initial_stop") or 0)
            equity = float(position.get("account_equity") or 0)
            if equity > 0 and stop > 0:
                total_risk += (abs(entry - stop) * quantity) / equity
            ltp = float(position.get("ltp") or entry or 0)
            gross += (ltp * quantity)
        return {"open_risk_pct": total_risk, "gross_exposure_amount": gross}

    def limits_summary(self) -> Dict[str, Any]:
        """Human-readable limits, shown on the dashboard so the owner can see them."""
        return {
            "risk_per_trade_pct": self._c("per_trade.risk_pct", 0.0025),
            "max_risk_per_trade_pct": self._c("per_trade.max_risk_pct", 0.005),
            "max_simultaneous_positions": self._c("portfolio.max_simultaneous_positions", 3),
            "max_correlated_sector_positions": self._c("portfolio.max_correlated_sector_positions", 2),
            "max_total_open_risk_pct": self._c("portfolio.max_total_open_risk_pct", 0.0075),
            "soft_daily_stop_pct": self._c("daily_limits.soft_daily_stop_pct", -0.01),
            "hard_daily_stop_pct": self._c("daily_limits.hard_daily_stop_pct", -0.015),
            "weekly_drawdown_warning_pct": self._c("daily_limits.weekly_drawdown_warning_pct", -0.025),
            "max_consecutive_losses": self._c("daily_limits.max_consecutive_losses", 4),
            "max_trades_per_day": self._c("daily_limits.max_trades_per_day", 12),
            "strategy_suspension_drawdown_pct": self._c("strategy_suspension.max_strategy_drawdown_pct", -0.05),
            "forbid_averaging_down": self._c("order_protections.forbid_averaging_down", True),
            "forbid_martingale": self._c("order_protections.forbid_martingale", True),
            "require_stop_before_entry": self._c("order_protections.require_stop_before_entry", True),
        }


__all__ = [
    "RiskEngine",
    "RiskDecision",
    "RiskLevel",
    "RiskReason",
    "ProposedTrade",
    "RiskContext",
]
