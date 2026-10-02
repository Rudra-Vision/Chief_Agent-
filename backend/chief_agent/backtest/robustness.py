"""Robustness testing.

A strategy that collapses when a parameter moves slightly, when costs double, or
when a few trades are removed is not robust and must be rejected - no matter how
good its headline backtest looked.

Scenarios implemented:

    parameter perturbation    (+/- x% on the one changed variable)
    higher slippage           (x2 by default)
    higher fees               (x1.5 by default)
    delayed entry             (enter N bars later)
    worse fills               (extra slippage in bps)
    missed trades             (randomly skip a share of entries)
    random trade removal      (bootstrap: drop a share of trades)
    different regimes         (per-regime performance must not collapse)
    different months          (per-month performance must not collapse)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__, component="robustness")


@dataclass
class Scenario:
    name: str
    category: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def expectancy_r(self) -> float:
        return float(self.metrics.get("expectancy_r", 0.0) or 0.0)

    @property
    def profit_factor(self) -> float:
        return float(self.metrics.get("profit_factor", 0.0) or 0.0)

    @property
    def net_return_pct(self) -> float:
        return float(self.metrics.get("net_return_pct", 0.0) or 0.0)

    @property
    def trades(self) -> int:
        return int(self.metrics.get("trades", 0) or 0)

    @property
    def profitable(self) -> bool:
        return self.trades > 0 and self.expectancy_r > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "trades": self.trades,
            "net_return_pct": round(self.net_return_pct, 6),
            "expectancy_r": round(self.expectancy_r, 4),
            "profit_factor": round(self.profit_factor, 4),
            "profitable": self.profitable,
            "error": self.error,
        }


@dataclass
class RobustnessSummary:
    scenarios: List[Scenario] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    passed_scenarios: int = 0
    total_scenarios: int = 0
    profitability_fraction: float = 0.0
    worst_degradation_pct: float = 0.0
    monthly_stability: float = 0.0
    regime_stability: float = 0.0
    passed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.baseline.items()
                if k in ("net_return_pct", "expectancy_r", "profit_factor", "max_drawdown_pct", "trades")
            },
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "profitability_fraction": round(self.profitability_fraction, 4),
            "worst_degradation_pct": round(self.worst_degradation_pct, 4),
            "monthly_stability": round(self.monthly_stability, 4),
            "regime_stability": round(self.regime_stability, 4),
            "passed": self.passed,
            "rejection_reasons": self.rejection_reasons,
            "notes": self.notes,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


def evaluate_trade_fragility(
    trades: Sequence[Mapping[str, Any]],
    *,
    initial_capital: float,
    removal_fraction: float = 0.20,
    n_runs: int = 20,
    seed: int = 777,
    max_degradation_pct: float = 0.40,
) -> Dict[str, Any]:
    """Random trade removal: how much of the result depends on a few trades?"""
    pnls = np.asarray([float(t.get("net_pnl", 0) or 0) for t in trades], dtype=float)
    if pnls.size < 10 or initial_capital <= 0:
        return {"runs": 0, "median_return_pct": 0.0, "worst_return_pct": 0.0, "profitable_fraction": 0.0}

    rng = np.random.default_rng(seed)
    keep = max(1, int(pnls.size * (1 - removal_fraction)))
    returns: List[float] = []
    for _ in range(n_runs):
        sample = rng.choice(pnls, size=keep, replace=False)
        returns.append(float(sample.sum()) / initial_capital)
    returns_array = np.asarray(returns)
    return {
        "runs": n_runs,
        "removal_fraction": removal_fraction,
        "median_return_pct": float(np.median(returns_array)),
        "worst_return_pct": float(returns_array.min()),
        "best_return_pct": float(returns_array.max()),
        "profitable_fraction": float((returns_array > 0).mean()),
    }


def month_stability(trades: Sequence[Mapping[str, Any]], min_months: int = 4) -> Dict[str, Any]:
    """Are the results spread across months, or does one month carry everything?"""
    by_month: Dict[str, float] = {}
    for trade in trades:
        ts = trade.get("entry_ts") or trade.get("exit_ts")
        key = str(ts)[:7]
        by_month[key] = by_month.get(key, 0.0) + float(trade.get("net_pnl", 0) or 0)
    if not by_month:
        return {"months": 0, "profitable_months": 0, "fraction": 0.0, "top_month_share": 0.0}
    profits = [v for v in by_month.values() if v > 0]
    gross = sum(profits) or 1.0
    return {
        "months": len(by_month),
        "profitable_months": len(profits),
        "fraction": len(profits) / len(by_month),
        "top_month_share": (max(by_month.values()) / gross) if gross else 0.0,
        "min_months_met": len(by_month) >= min_months,
        "by_month": {k: round(v, 2) for k, v in sorted(by_month.items())},
    }


def regime_stability(trades: Sequence[Mapping[str, Any]], min_regimes: int = 2) -> Dict[str, Any]:
    """Are the results spread across market regimes?"""
    by_regime: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for trade in trades:
        regime = str(trade.get("regime_at_entry") or "UNKNOWN")
        by_regime[regime] = by_regime.get(regime, 0.0) + float(trade.get("net_pnl", 0) or 0)
        counts[regime] = counts.get(regime, 0) + 1
    if not by_regime:
        return {"regimes": 0, "profitable_regimes": 0, "fraction": 0.0}
    profits = [v for v in by_regime.values() if v > 0]
    return {
        "regimes": len(by_regime),
        "profitable_regimes": len(profits),
        "fraction": len(profits) / len(by_regime),
        "min_regimes_met": len(by_regime) >= min_regimes,
        "by_regime": {k: round(v, 2) for k, v in sorted(by_regime.items(), key=lambda kv: -kv[1])},
        "counts": counts,
    }


class RobustnessRunner:
    """Runs a set of stress scenarios against a baseline."""

    def __init__(self, evaluate: Callable[[Dict[str, Any]], Mapping[str, Any]]) -> None:
        """``evaluate(overrides)`` runs one backtest with the supplied overrides and
        returns its metrics dict."""
        self.evaluate = evaluate

    def run(
        self,
        *,
        baseline_overrides: Optional[Dict[str, Any]] = None,
        perturb_variable: Optional[str] = None,
        perturb_values: Optional[Sequence[Any]] = None,
        slippage_multiplier: float = 2.0,
        fee_multiplier: float = 1.5,
        delayed_entry_bars: int = 1,
        worse_fill_bps: float = 10.0,
        missed_trade_fraction: float = 0.10,
        random_removal_fraction: float = 0.20,
        max_degradation_pct: float = 0.40,
        min_profitable_fraction: float = 0.70,
    ) -> RobustnessSummary:
        base_overrides = dict(baseline_overrides or {})
        summary = RobustnessSummary()

        def record(name: str, category: str, overrides: Dict[str, Any]) -> Optional[Scenario]:
            scenario = Scenario(name=name, category=category)
            try:
                merged = {**base_overrides, **overrides}
                scenario.metrics = dict(self.evaluate(merged))
            except Exception as exc:
                scenario.error = str(exc)
                log.warning("robustness scenario failed", context={"scenario": name, "error": str(exc)})
            summary.scenarios.append(scenario)
            return scenario

        baseline = record("baseline", "baseline", {})
        if baseline is not None:
            summary.baseline = dict(baseline.metrics)
        baseline_expectancy = baseline.expectancy_r if baseline else 0.0

        if perturb_variable and perturb_values:
            for value in perturb_values:
                record(f"param:{perturb_variable}={value}", "parameter_perturbation", {perturb_variable: value})

        record("slippage_x2", "cost_stress", {"slippage_multiplier": slippage_multiplier})
        record("fees_x1.5", "cost_stress", {"fee_multiplier": fee_multiplier})
        record(f"delayed_entry_{delayed_entry_bars}_bar", "timing", {"delay_entry_bars": delayed_entry_bars})
        record(f"missed_trades_{missed_trade_fraction:.0%}", "execution", {"missed_trade_pct": missed_trade_fraction})

        stress = [s for s in summary.scenarios if s.category != "baseline" and s.error is None]
        summary.total_scenarios = len(stress)
        summary.passed_scenarios = sum(1 for s in stress if s.profitable)
        summary.profitability_fraction = (summary.passed_scenarios / summary.total_scenarios) if stress else 0.0

        if baseline_expectancy > 0:
            degradations = [
                max(0.0, (baseline_expectancy - s.expectancy_r) / baseline_expectancy) for s in stress
            ]
            summary.worst_degradation_pct = max(degradations) if degradations else 0.0
        else:
            summary.worst_degradation_pct = 0.0

        reasons: List[str] = []
        if summary.total_scenarios and summary.profitability_fraction < min_profitable_fraction:
            reasons.append(
                f"only {summary.profitability_fraction:.0%} of stress scenarios stayed profitable "
                f"(need {min_profitable_fraction:.0%})"
            )
        if summary.worst_degradation_pct > max_degradation_pct:
            reasons.append(
                f"worst-case expectancy degradation {summary.worst_degradation_pct:.0%} exceeds "
                f"the {max_degradation_pct:.0%} tolerance"
            )
        failed_parameter = [
            s.name for s in stress if s.category == "parameter_perturbation" and not s.profitable
        ]
        if failed_parameter:
            reasons.append(
                "the strategy only works at very specific parameter values "
                f"({', '.join(failed_parameter[:4])})"
            )

        summary.rejection_reasons = reasons
        summary.passed = not reasons
        return summary

    @staticmethod
    def analyse_trade_set(
        trades: Sequence[Mapping[str, Any]],
        *,
        initial_capital: float,
        removal_fraction: float = 0.20,
        max_single_stock_share: float = 0.35,
        max_single_day_share: float = 0.30,
        max_top3_trade_share: float = 0.60,
    ) -> Dict[str, Any]:
        """Anti-overfitting checks computed directly from the trade ledger."""
        from .metrics import concentration_analysis

        concentration = concentration_analysis(trades)
        failure = evaluate_trade_fragility(
            trades, initial_capital=initial_capital, removal_fraction=removal_fraction
        )
        months = month_stability(trades)
        regimes = regime_stability(trades)

        reasons: List[str] = []
        if concentration.get("single_stock_share", 0) > max_single_stock_share:
            reasons.append(f"single stock contributes {concentration['single_stock_share']:.0%} of net profit")
        if concentration.get("single_day_share", 0) > max_single_day_share:
            reasons.append(f"single day contributes {concentration['single_day_share']:.0%} of net profit")
        if concentration.get("top3_trade_share", 0) > max_top3_trade_share:
            reasons.append(f"the three biggest trades are {concentration['top3_trade_share']:.0%} of net profit")
        if failure.get("profitable_fraction", 0) < 0.60:
            reasons.append(
                f"removing {removal_fraction:.0%} of trades at random makes the result unprofitable "
                f"in {1 - failure.get('profitable_fraction', 0):.0%} of runs"
            )
        if months.get("months", 0) >= 3 and months.get("fraction", 0) < 0.5:
            reasons.append(f"only {months.get('fraction', 0):.0%} of months were profitable")
        if regimes.get("regimes", 0) >= 2 and regimes.get("fraction", 0) < 0.4:
            reasons.append(f"only {regimes.get('fraction', 0):.0%} of regimes were profitable")

        return {
            "concentration": concentration,
            "trade_removal": failure,
            "months": months,
            "regimes": regimes,
            "passed": not reasons,
            "rejection_reasons": reasons,
        }


__all__ = [
    "RobustnessRunner",
    "RobustnessSummary",
    "Scenario",
    "evaluate_trade_fragility",
    "month_stability",
    "regime_stability",
]
