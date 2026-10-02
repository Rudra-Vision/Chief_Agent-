"""Backtest performance and risk metrics.

Everything the project brief asks for, computed from a trade ledger and an
equity curve:

net return, annualised return, win rate, loss rate, average win, average loss,
expectancy, profit factor, Sharpe, Sortino, Calmar, maximum drawdown, recovery
factor, volatility, trade count, average holding time, exposure, fees, slippage,
largest win, largest loss, max consecutive wins/losses, monthly returns, regime
performance, long vs short, sector performance, time-of-day performance, MAE, MFE.

Conventions
-----------
* Returns are computed from the equity curve, not by summing trade P&L.
* Annualisation uses 252 trading days.
* Sharpe/Sortino use the daily equity series (per-trade Sharpe is a different,
  noisier statistic and is reported separately as ``trade_sharpe``).
* Expectancy is reported in **R multiples** as well as rupees, because R is the
  unit the research engine optimises.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Small statistics helpers (no external dependency for simple cases)
# --------------------------------------------------------------------------- #
def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else 0.0


def _std(values: Sequence[float], ddof: int = 1) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) <= ddof:
        return 0.0
    return float(np.std(clean, ddof=ddof))


def _percentile(values: Sequence[float], q: float) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return 0.0
    return float(np.percentile(clean, q))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
@dataclass
class DrawdownStats:
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    peak_index: int = 0
    trough_index: int = 0
    peak_value: float = 0.0
    trough_value: float = 0.0
    recovery_index: Optional[int] = None
    duration_days: int = 0
    recovery_days: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "max_drawdown_amount": round(self.max_drawdown_amount, 2),
            "peak_value": round(self.peak_value, 2),
            "trough_value": round(self.trough_value, 2),
            "duration_days": self.duration_days,
            "recovery_days": self.recovery_days,
            "recovered": self.recovery_index is not None,
        }


def drawdown_series(equity: Sequence[float]) -> List[float]:
    """Running drawdown as a negative fraction of the running peak."""
    out: List[float] = []
    peak = 0.0
    for value in equity:
        peak = max(peak, float(value))
        out.append(((float(value) - peak) / peak) if peak > 0 else 0.0)
    return out


def drawdown_stats(equity: Sequence[float], dates: Optional[Sequence[dt.date]] = None) -> DrawdownStats:
    if equity is None or len(equity) == 0:
        return DrawdownStats()
    peak = float(equity[0])
    peak_index = 0
    stats = DrawdownStats()
    for index, value in enumerate(equity):
        value = float(value)
        if value > peak:
            peak = value
            peak_index = index
        if peak > 0:
            dd = (value - peak) / peak
            if dd < stats.max_drawdown_pct:
                stats.max_drawdown_pct = dd
                stats.max_drawdown_amount = peak - value
                stats.peak_index = peak_index
                stats.trough_index = index
                stats.peak_value = peak
                stats.trough_value = value

    if dates is not None and len(dates) > 0 and stats.trough_index > stats.peak_index and len(dates) > stats.trough_index:
        stats.duration_days = (dates[stats.trough_index] - dates[stats.peak_index]).days
        for index in range(stats.trough_index, len(equity)):
            if float(equity[index]) >= stats.peak_value:
                stats.recovery_index = index
                stats.recovery_days = (dates[index] - dates[stats.peak_index]).days if index < len(dates) else None
                break
    return stats


# --------------------------------------------------------------------------- #
# Trade ledger aggregation
# --------------------------------------------------------------------------- #
@dataclass
class TradeStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0            # rupees per trade
    expectancy_r: float = 0.0          # R multiples per trade
    payoff_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_holding_minutes: float = 0.0
    median_holding_minutes: float = 0.0
    average_mae_r: float = 0.0
    average_mfe_r: float = 0.0
    average_r: float = 0.0
    r_std: float = 0.0
    trade_sharpe: float = 0.0
    exposure_pct: float = 0.0
    turnover: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _consecutive(sequence: Sequence[bool], target: bool) -> int:
    best = 0
    current = 0
    for value in sequence:
        if value is target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def trade_statistics(
    trades: Sequence[Mapping[str, Any]],
    *,
    total_minutes_in_market: float = 0.0,
    total_available_minutes: float = 0.0,
) -> TradeStats:
    """Aggregate a trade ledger into :class:`TradeStats`."""
    stats = TradeStats()
    if trades is None or len(trades) == 0:
        return stats

    pnls = [float(t.get("net_pnl", 0) or 0) for t in trades]
    r_multiples = [float(t.get("r_multiple", 0) or 0) for t in trades]
    holding = [float(t.get("holding_minutes", 0) or 0) for t in trades]

    stats.trades = len(trades)
    stats.net_pnl = sum(pnls)
    stats.total_fees = sum(float(t.get("fees", 0) or 0) for t in trades)
    stats.total_slippage = sum(float(t.get("slippage_cost", 0) or 0) for t in trades)

    win_flags: List[bool] = []
    for pnl in pnls:
        if pnl > 1e-9:
            stats.wins += 1
            stats.gross_profit += pnl
            win_flags.append(True)
        elif pnl < -1e-9:
            stats.losses += 1
            stats.gross_loss += abs(pnl)
            win_flags.append(False)
        else:
            stats.scratches += 1
            win_flags.append(False)

    decided = stats.wins + stats.losses
    stats.win_rate = _safe_div(stats.wins, decided)
    stats.loss_rate = _safe_div(stats.losses, decided)
    stats.average_win = _mean([p for p in pnls if p > 0])
    stats.average_loss = _mean([p for p in pnls if p < 0])
    stats.largest_win = max(pnls) if pnls else 0.0
    stats.largest_loss = min(pnls) if pnls else 0.0
    stats.profit_factor = _safe_div(stats.gross_profit, stats.gross_loss)
    # Expectancy = P(win) x AvgWin - P(loss) x AvgLoss
    stats.expectancy = stats.win_rate * stats.average_win + stats.loss_rate * stats.average_loss
    stats.expectancy_r = _mean(r_multiples)
    stats.payoff_ratio = _safe_div(stats.average_win, abs(stats.average_loss) if stats.average_loss else 0.0)
    stats.max_consecutive_wins = _consecutive(win_flags, True)
    stats.max_consecutive_losses = _consecutive(win_flags, False)
    stats.average_holding_minutes = _mean(holding)
    stats.median_holding_minutes = float(np.median(holding)) if holding else 0.0
    stats.average_mae_r = _mean([float(t.get("mae_r", 0) or 0) for t in trades])
    stats.average_mfe_r = _mean([float(t.get("mfe_r", 0) or 0) for t in trades])
    stats.average_r = _mean(r_multiples)
    stats.r_std = _std(r_multiples)
    stats.trade_sharpe = _safe_div(stats.average_r, stats.r_std) * math.sqrt(max(1, len(r_multiples)))
    stats.exposure_pct = _safe_div(total_minutes_in_market, total_available_minutes)
    stats.turnover = sum(
        abs(float(t.get("entry_price", 0) or 0) * float(t.get("quantity", 0) or 0))
        + abs(float(t.get("exit_price", 0) or 0) * float(t.get("quantity", 0) or 0))
        for t in trades
    )
    return stats


# --------------------------------------------------------------------------- #
# Equity-curve statistics
# --------------------------------------------------------------------------- #
@dataclass
class CurveStats:
    initial_equity: float = 0.0
    final_equity: float = 0.0
    net_return_pct: float = 0.0
    annualised_return_pct: float = 0.0
    volatility_pct: float = 0.0
    downside_deviation_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    recovery_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    trading_days: int = 0
    best_day_pct: float = 0.0
    worst_day_pct: float = 0.0
    positive_days: int = 0
    negative_days: int = 0
    var_95_pct: float = 0.0
    cvar_95_pct: float = 0.0
    risk_free_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def curve_statistics(
    equity: Sequence[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> CurveStats:
    stats = CurveStats()
    if equity is None or len(equity) < 2:
        return stats

    values = np.asarray([float(v) for v in equity], dtype=float)
    stats.initial_equity = float(values[0])
    stats.final_equity = float(values[-1])
    stats.trading_days = len(values)
    stats.net_return_pct = _safe_div(values[-1] - values[0], values[0])

    daily_returns = np.diff(values) / np.where(values[:-1] == 0, np.nan, values[:-1])
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if daily_returns.size == 0:
        return stats

    periods = daily_returns.size
    years = max(periods / periods_per_year, 1e-9)
    if values[0] > 0 and values[-1] > 0:
        stats.annualised_return_pct = (values[-1] / values[0]) ** (1.0 / years) - 1.0

    stats.volatility_pct = float(np.std(daily_returns, ddof=1) * math.sqrt(periods_per_year)) if periods > 1 else 0.0
    downside = daily_returns[daily_returns < 0]
    stats.downside_deviation_pct = (
        float(np.std(downside, ddof=1) * math.sqrt(periods_per_year)) if downside.size > 1 else 0.0
    )

    per_period_rf = risk_free_rate / periods_per_year
    excess = daily_returns - per_period_rf
    sharpe_denom = float(np.std(daily_returns, ddof=1))
    stats.sharpe = _safe_div(float(np.mean(excess)) * math.sqrt(periods_per_year), sharpe_denom)
    stats.sortino = _safe_div(float(np.mean(excess)) * math.sqrt(periods_per_year), stats.downside_deviation_pct)

    dd = drawdown_stats(values)
    stats.max_drawdown_pct = dd.max_drawdown_pct
    stats.calmar = _safe_div(stats.annualised_return_pct, abs(dd.max_drawdown_pct))
    stats.recovery_factor = _safe_div(stats.net_return_pct, abs(dd.max_drawdown_pct))

    stats.best_day_pct = float(np.max(daily_returns))
    stats.worst_day_pct = float(np.min(daily_returns))
    stats.positive_days = int((daily_returns > 0).sum())
    stats.negative_days = int((daily_returns < 0).sum())
    stats.var_95_pct = float(np.percentile(daily_returns, 5))
    tail = daily_returns[daily_returns <= stats.var_95_pct]
    stats.cvar_95_pct = float(np.mean(tail)) if tail.size else stats.var_95_pct
    stats.risk_free_rate = risk_free_rate
    return stats


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #
def monthly_returns(
    equity_by_date: Mapping[dt.date, float],
) -> Dict[str, Dict[str, float]]:
    """``{"2025-01": {"return_pct": .., "start": .., "end": ..}}``."""
    by_month: Dict[str, List[Tuple[dt.date, float]]] = defaultdict(list)
    for day, value in sorted(equity_by_date.items()):
        by_month[f"{day.year:04d}-{day.month:02d}"].append((day, float(value)))

    out: Dict[str, Dict[str, float]] = {}
    for month, entries in by_month.items():
        first = entries[0][1]
        last = entries[-1][1]
        out[month] = {
            "start_equity": round(first, 2),
            "end_equity": round(last, 2),
            "return_pct": round(_safe_div(last - first, first), 6),
            "trading_days": len(entries),
        }
    return dict(sorted(out.items()))


def group_performance(
    trades: Sequence[Mapping[str, Any]],
    key: str,
    *,
    min_trades: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Win rate / expectancy / P&L grouped by a trade attribute."""
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        value = trade.get(key)
        if value is None or value == "":
            value = "UNKNOWN"
        groups[str(value)].append(trade)

    out: Dict[str, Dict[str, Any]] = {}
    for name, rows in groups.items():
        stats = trade_statistics(rows)
        if stats.trades < min_trades:
            continue
        net = stats.net_pnl
        out[name] = {
            "trades": stats.trades,
            "net_pnl": round(net, 2),
            "win_rate": round(stats.win_rate, 4),
            "expectancy": round(stats.expectancy, 2),
            "expectancy_r": round(stats.expectancy_r, 4),
            "profit_factor": round(stats.profit_factor, 4),
            "average_r": round(stats.average_r, 4),
            "max_consecutive_losses": stats.max_consecutive_losses,
        }
    total = sum(row["net_pnl"] for row in out.values()) or 1.0
    for row in out.values():
        row["pnl_share"] = round(row["net_pnl"] / total, 4)
    return dict(sorted(out.items(), key=lambda kv: kv[1]["net_pnl"], reverse=True))


def time_of_day_performance(trades: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Performance by the 30-minute bucket in which the trade was entered."""
    def bucket(ts: Any) -> str:
        try:
            if isinstance(ts, str):
                ts = dt.datetime.fromisoformat(ts)
            minutes = ts.hour * 60 + ts.minute
        except Exception:
            return "UNKNOWN"
        start = (minutes // 30) * 30
        return f"{start // 60:02d}:{start % 60:02d}"

    decorated = []
    for trade in trades:
        row = dict(trade)
        row["time_bucket"] = bucket(trade.get("entry_ts"))
        decorated.append(row)
    return group_performance(decorated, "time_bucket")


def concentration_analysis(trades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """How dependent is the result on one stock / day / regime?

    A strategy that makes all its money on a single day or a single name is not
    robust, and the promotion gate rejects it.
    """
    if trades is None or len(trades) == 0:
        return {"single_stock_share": 0.0, "single_day_share": 0.0, "single_regime_share": 0.0, "top3_trade_share": 0.0}

    by_symbol = group_performance(trades, "symbol")
    by_regime = group_performance(trades, "regime_at_entry")

    by_day: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        ts = trade.get("entry_ts")
        try:
            day = ts.date().isoformat() if isinstance(ts, dt.datetime) else str(ts)[:10]
        except Exception:
            day = "UNKNOWN"
        by_day[day].append(trade)
    day_rows = [{"trades": len(rows), "net_pnl": sum(float(r.get("net_pnl", 0) or 0) for r in rows)} for rows in by_day.values()]

    pnls = sorted((float(t.get("net_pnl", 0) or 0) for t in trades), reverse=True)
    total_positive = sum(p for p in pnls if p > 0) or 1.0
    total_abs = sum(abs(p) for p in pnls) or 1.0

    return {
        "single_stock_share": max((row["pnl_share"] for row in by_symbol.values()), default=0.0),
        "single_day_share": max((row["net_pnl"] for row in day_rows), default=0.0) / total_positive,
        "single_regime_share": max((row["pnl_share"] for row in by_regime.values()), default=0.0),
        "top3_trade_share": sum(pnls[:3]) / total_positive,
        "top3_trade_share_of_absolute": sum(abs(p) for p in pnls[:3]) / total_abs,
        "distinct_symbols": len(by_symbol),
        "distinct_days": len(day_rows),
        "distinct_regimes": len(by_regime),
    }


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #
@dataclass
class BacktestMetrics:
    curve: CurveStats = field(default_factory=CurveStats)
    trades: TradeStats = field(default_factory=TradeStats)
    drawdown: DrawdownStats = field(default_factory=DrawdownStats)
    monthly_returns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sector_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    time_of_day_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    side_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    exit_reason_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    concentration: Dict[str, Any] = field(default_factory=dict)
    objective_score: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_return_pct": round(self.curve.net_return_pct, 6),
            "annualised_return_pct": round(self.curve.annualised_return_pct, 6),
            "volatility_pct": round(self.curve.volatility_pct, 6),
            "sharpe": round(self.curve.sharpe, 4),
            "sortino": round(self.curve.sortino, 4),
            "calmar": round(self.curve.calmar, 4),
            "recovery_factor": round(self.curve.recovery_factor, 4),
            "max_drawdown_pct": round(self.curve.max_drawdown_pct, 6),
            "trading_days": self.curve.trading_days,
            "var_95_pct": round(self.curve.var_95_pct, 6),
            "cvar_95_pct": round(self.curve.cvar_95_pct, 6),
            "best_day_pct": round(self.curve.best_day_pct, 6),
            "worst_day_pct": round(self.curve.worst_day_pct, 6),
            "trades": self.trades.trades,
            "win_rate": round(self.trades.win_rate, 4),
            "loss_rate": round(self.trades.loss_rate, 4),
            "average_win": round(self.trades.average_win, 2),
            "average_loss": round(self.trades.average_loss, 2),
            "largest_win": round(self.trades.largest_win, 2),
            "largest_loss": round(self.trades.largest_loss, 2),
            "expectancy": round(self.trades.expectancy, 2),
            "expectancy_r": round(self.trades.expectancy_r, 4),
            "profit_factor": round(self.trades.profit_factor, 4),
            "payoff_ratio": round(self.trades.payoff_ratio, 4),
            "average_r": round(self.trades.average_r, 4),
            "trade_sharpe": round(self.trades.trade_sharpe, 4),
            "max_consecutive_wins": self.trades.max_consecutive_wins,
            "max_consecutive_losses": self.trades.max_consecutive_losses,
            "average_holding_minutes": round(self.trades.average_holding_minutes, 2),
            "median_holding_minutes": round(self.trades.median_holding_minutes, 2),
            "average_mae_r": round(self.trades.average_mae_r, 4),
            "average_mfe_r": round(self.trades.average_mfe_r, 4),
            "fees": round(self.trades.total_fees, 2),
            "slippage": round(self.trades.total_slippage, 2),
            "turnover": round(self.trades.turnover, 2),
            "exposure_pct": round(self.trades.exposure_pct, 4),
            "objective_score": round(self.objective_score, 4),
            "concentration": {k: round(v, 4) if isinstance(v, float) else v for k, v in self.concentration.items()},
            "extras": self.extras,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.update(
            {
                "monthly_returns": self.monthly_returns,
                "regime_performance": self.regime_performance,
                "sector_performance": self.sector_performance,
                "time_of_day_performance": self.time_of_day_performance,
                "side_performance": self.side_performance,
                "exit_reason_performance": self.exit_reason_performance,
                "drawdown": self.drawdown.to_dict(),
            }
        )
        return payload


def evaluate_backtest(
    trades: Sequence[Mapping[str, Any]],
    equity_curve: Sequence[float],
    equity_dates: Sequence[dt.date],
    *,
    total_minutes_in_market: float = 0.0,
    total_available_minutes: float = 0.0,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Build the complete metric set for one backtest."""
    equity_by_date: Dict[dt.date, float] = {}
    for day, value in zip(equity_dates, equity_curve):
        equity_by_date[day] = float(value)  # last observation of the day wins

    metrics = BacktestMetrics()
    metrics.curve = curve_statistics(equity_curve, risk_free_rate=risk_free_rate)
    metrics.trades = trade_statistics(
        trades,
        total_minutes_in_market=total_minutes_in_market,
        total_available_minutes=total_available_minutes,
    )
    metrics.drawdown = drawdown_stats(equity_curve, list(equity_dates))
    metrics.monthly_returns = monthly_returns(equity_by_date)
    metrics.regime_performance = group_performance(trades, "regime_at_entry")
    metrics.sector_performance = group_performance(trades, "sector")
    metrics.time_of_day_performance = time_of_day_performance(trades)
    metrics.side_performance = group_performance(trades, "direction")
    metrics.exit_reason_performance = group_performance(trades, "exit_reason")
    metrics.concentration = concentration_analysis(trades)
    return metrics


def strategy_objective_score(
    metrics: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
    normalisation: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> float:
    """The balanced, configurable research objective.

    Deliberately NOT "maximum profit":

        0.25 x Sortino + 0.20 x Expectancy + 0.15 x ProfitFactor
      + 0.15 x Return  + 0.10 x WalkForwardStability + 0.10 x RegimeRobustness
      - 0.20 x MaxDrawdown - 0.05 x TurnoverCost

    ``normalisation`` supplies ``(low, high)`` reference ranges per component so
    each term lands in roughly [0, 1] before weighting.
    """
    default_weights = {
        "sortino": 0.25,
        "expectancy": 0.20,
        "profit_factor": 0.15,
        "return": 0.15,
        "walk_forward_stability": 0.10,
        "regime_robustness": 0.10,
        "max_drawdown": -0.20,
        "turnover_cost": -0.05,
    }
    weights = dict(weights or default_weights)

    default_ranges: Dict[str, Tuple[float, float]] = {
        "sortino": (0.0, 3.0),
        "expectancy": (-0.2, 0.5),
        "profit_factor": (0.8, 2.5),
        "return": (-0.1, 0.4),
        "walk_forward_stability": (0.0, 1.0),
        "regime_robustness": (0.0, 1.0),
        "max_drawdown": (0.0, 0.30),
        "turnover_cost": (0.0, 5.0),
    }
    ranges = dict(default_ranges)
    if normalisation:
        ranges.update(normalisation)

    def normalise(component: str, value: float) -> float:
        low, high = ranges.get(component, (0.0, 1.0))
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (float(value) - low) / (high - low)))

    components = {
        "sortino": normalise("sortino", metrics.get("sortino", 0.0)),
        "expectancy": normalise("expectancy", metrics.get("expectancy_r", 0.0)),
        "profit_factor": normalise("profit_factor", metrics.get("profit_factor", 0.0)),
        "return": normalise("return", metrics.get("net_return_pct", 0.0)),
        "walk_forward_stability": normalise(
            "walk_forward_stability", metrics.get("walk_forward_profitable_fraction", 0.5)
        ),
        "regime_robustness": normalise("regime_robustness", metrics.get("regime_robustness", 0.5)),
        "max_drawdown": normalise("max_drawdown", abs(metrics.get("max_drawdown_pct", 0.0))),
        "turnover_cost": normalise("turnover_cost", metrics.get("turnover_per_day", 0.0)),
    }
    return float(sum(weights.get(name, 0.0) * value for name, value in components.items()))


__all__ = [
    "evaluate_backtest",
    "trade_statistics",
    "curve_statistics",
    "drawdown_stats",
    "drawdown_series",
    "monthly_returns",
    "group_performance",
    "time_of_day_performance",
    "concentration_analysis",
    "strategy_objective_score",
    "BacktestMetrics",
    "TradeStats",
    "CurveStats",
    "DrawdownStats",
]
