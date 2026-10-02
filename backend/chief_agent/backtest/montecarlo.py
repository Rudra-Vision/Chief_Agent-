"""Monte Carlo analysis.

Estimates the distributions of the outcomes that actually matter for survival:

    maximum drawdown        expected return        losing streak length
    probability of ruin     terminal equity variation

Reported as median / 5th / 95th percentiles (plus 25th and 75th).

Method: bootstrap resampling of the *realised trade sequence* (R multiples and
rupee P&L). Shuffling trades is legitimate here **only** because it estimates the
distribution of an aggregate statistic under the i.i.d. assumption - it is never
used to construct a training set. Trade *ordering* effects are captured by
re-sampling with replacement rather than permuting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__, component="montecarlo")


@dataclass
class MonteCarloSummary:
    n_simulations: int = 0
    method: str = "trade_sequence_resample"
    initial_capital: float = 0.0
    trades_per_simulation: int = 0

    median_return_pct: float = 0.0
    p5_return_pct: float = 0.0
    p25_return_pct: float = 0.0
    p75_return_pct: float = 0.0
    p95_return_pct: float = 0.0

    median_max_drawdown_pct: float = 0.0
    p5_max_drawdown_pct: float = 0.0
    p95_max_drawdown_pct: float = 0.0
    worst_max_drawdown_pct: float = 0.0

    median_max_losing_streak: float = 0.0
    p95_max_losing_streak: float = 0.0
    worst_max_losing_streak: int = 0

    ruin_probability: float = 0.0
    ruin_threshold_pct: float = 0.30
    probability_of_loss: float = 0.0
    probability_of_profit: float = 0.0
    median_terminal_equity: float = 0.0
    p5_terminal_equity: float = 0.0
    p95_terminal_equity: float = 0.0

    passed: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "method": self.method,
            "initial_capital": round(self.initial_capital, 2),
            "trades_per_simulation": self.trades_per_simulation,
            "return_pct": {
                "p5": round(self.p5_return_pct, 6),
                "p25": round(self.p25_return_pct, 6),
                "median": round(self.median_return_pct, 6),
                "p75": round(self.p75_return_pct, 6),
                "p95": round(self.p95_return_pct, 6),
            },
            "max_drawdown_pct": {
                "p5": round(self.p5_max_drawdown_pct, 6),
                "median": round(self.median_max_drawdown_pct, 6),
                "p95": round(self.p95_max_drawdown_pct, 6),
                "worst": round(self.worst_max_drawdown_pct, 6),
            },
            "max_losing_streak": {
                "median": round(self.median_max_losing_streak, 2),
                "p95": round(self.p95_max_losing_streak, 2),
                "worst": self.worst_max_losing_streak,
            },
            "ruin_probability": round(self.ruin_probability, 6),
            "ruin_threshold_pct": self.ruin_threshold_pct,
            "probability_of_loss": round(self.probability_of_loss, 6),
            "probability_of_profit": round(self.probability_of_profit, 6),
            "terminal_equity": {
                "p5": round(self.p5_terminal_equity, 2),
                "median": round(self.median_terminal_equity, 2),
                "p95": round(self.p95_terminal_equity, 2),
            },
            "passed": self.passed,
            "notes": self.notes,
        }


def _max_losing_streak(pnls: np.ndarray) -> int:
    best = 0
    current = 0
    for value in pnls:
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    return float(dd.min())


def run_monte_carlo(
    trade_pnls: Sequence[float],
    *,
    initial_capital: float,
    n_simulations: int = 2000,
    ruin_threshold_pct: float = 0.30,
    seed: int = 424242,
    max_drawdown_limit_pct: float = 0.25,
    max_ruin_probability: float = 0.02,
) -> MonteCarloSummary:
    """Bootstrap the realised trade sequence and summarise the outcome distribution."""
    summary = MonteCarloSummary(
        initial_capital=float(initial_capital),
        ruin_threshold_pct=float(ruin_threshold_pct),
    )
    clean = [float(p) for p in trade_pnls if p is not None and np.isfinite(float(p))]
    if len(clean) < 5:
        summary.notes.append(
            f"only {len(clean)} trade(s) available - Monte Carlo needs at least 5 to be meaningful"
        )
        return summary
    if initial_capital <= 0:
        summary.notes.append("initial capital must be positive")
        return summary

    summary.n_simulations = int(n_simulations)
    pnls = np.asarray(clean, dtype=float)
    rng = np.random.default_rng(seed)
    n_trades = pnls.size
    summary.trades_per_simulation = n_trades

    returns = np.empty(n_simulations, dtype=float)
    drawdowns = np.empty(n_simulations, dtype=float)
    streaks = np.empty(n_simulations, dtype=int)
    terminals = np.empty(n_simulations, dtype=float)
    ruin_floor = initial_capital * (1.0 - ruin_threshold_pct)
    ruined = 0

    for simulation in range(n_simulations):
        sample = rng.choice(pnls, size=n_trades, replace=True)
        equity = initial_capital + np.cumsum(sample)
        terminals[simulation] = equity[-1]
        returns[simulation] = (equity[-1] - initial_capital) / initial_capital
        drawdowns[simulation] = _max_drawdown(equity)
        streaks[simulation] = _max_losing_streak(sample)
        if equity.min() <= ruin_floor:
            ruined += 1

    summary.median_return_pct = float(np.median(returns))
    summary.p5_return_pct = float(np.percentile(returns, 5))
    summary.p25_return_pct = float(np.percentile(returns, 25))
    summary.p75_return_pct = float(np.percentile(returns, 75))
    summary.p95_return_pct = float(np.percentile(returns, 95))

    summary.median_max_drawdown_pct = float(np.median(drawdowns))
    summary.p5_max_drawdown_pct = float(np.percentile(drawdowns, 5))
    summary.p95_max_drawdown_pct = float(np.percentile(drawdowns, 95))
    summary.worst_max_drawdown_pct = float(drawdowns.min())

    summary.median_max_losing_streak = float(np.median(streaks))
    summary.p95_max_losing_streak = float(np.percentile(streaks, 95))
    summary.worst_max_losing_streak = int(streaks.max())

    summary.ruin_probability = ruined / n_simulations
    summary.probability_of_loss = float((returns < 0).mean())
    summary.probability_of_profit = float((returns > 0).mean())
    summary.median_terminal_equity = float(np.median(terminals))
    summary.p5_terminal_equity = float(np.percentile(terminals, 5))
    summary.p95_terminal_equity = float(np.percentile(terminals, 95))

    notes: List[str] = []
    if summary.ruin_probability > max_ruin_probability:
        notes.append(
            f"probability of a {ruin_threshold_pct:.0%} drawdown is {summary.ruin_probability:.2%} "
            f"(limit {max_ruin_probability:.2%})"
        )
    if abs(summary.p95_max_drawdown_pct) > max_drawdown_limit_pct:
        notes.append(
            f"95th-percentile drawdown {abs(summary.p95_max_drawdown_pct):.2%} exceeds the "
            f"{max_drawdown_limit_pct:.0%} limit"
        )
    if summary.p5_return_pct < 0:
        notes.append("the 5th-percentile outcome is a loss - this is a fat-tailed downside")
    summary.notes = notes
    summary.passed = (
        summary.ruin_probability <= max_ruin_probability
        and abs(summary.p95_max_drawdown_pct) <= max_drawdown_limit_pct
    )
    return summary


__all__ = ["run_monte_carlo", "MonteCarloSummary"]
