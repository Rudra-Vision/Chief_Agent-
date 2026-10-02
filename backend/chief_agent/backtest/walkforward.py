"""Walk-forward validation.

Time-series data is **never randomly shuffled**. This module implements:

* anchored windows (expanding train, fixed-length test steps),
* rolling windows,
* purging (drop training observations whose outcome window overlaps the test
  boundary) and an embargo gap between adjacent windows.

The result answers the question that matters: "does this hold up out of sample,
consistently, across windows?" - not "what was the best in-sample number?".
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__, component="walkforward")


@dataclass
class Window:
    index: int
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date
    purge_days: int = 0
    embargo_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
        }


@dataclass
class WindowResult:
    window: Window
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def test_expectancy_r(self) -> float:
        return float(self.test_metrics.get("expectancy_r", 0.0) or 0.0)

    @property
    def test_profit_factor(self) -> float:
        return float(self.test_metrics.get("profit_factor", 0.0) or 0.0)

    @property
    def test_return_pct(self) -> float:
        return float(self.test_metrics.get("net_return_pct", 0.0) or 0.0)

    @property
    def profitable(self) -> bool:
        return self.test_metrics.get("trades", 0) > 0 and self.test_return_pct > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "test_trades": self.test_metrics.get("trades", 0),
            "test_return_pct": round(self.test_return_pct, 6),
            "test_expectancy_r": round(self.test_expectancy_r, 4),
            "test_profit_factor": round(self.test_profit_factor, 4),
            "test_max_drawdown_pct": round(float(self.test_metrics.get("max_drawdown_pct", 0.0) or 0.0), 6),
            "test_sharpe": round(float(self.test_metrics.get("sharpe", 0.0) or 0.0), 4),
            "test_sortino": round(float(self.test_metrics.get("sortino", 0.0) or 0.0), 4),
            "profitable": self.profitable,
            "error": self.error,
        }


@dataclass
class WalkForwardSummary:
    mode: str
    windows: List[WindowResult] = field(default_factory=list)
    profitable_windows: int = 0
    total_windows: int = 0
    oos_return_pct: float = 0.0
    oos_expectancy_r: float = 0.0
    mean_test_profit_factor: float = 0.0
    consistency: float = 0.0
    worst_window_return_pct: float = 0.0
    best_window_return_pct: float = 0.0
    return_dispersion: float = 0.0
    degradation_vs_train: float = 0.0
    passed: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "total_windows": self.total_windows,
            "profitable_windows": self.profitable_windows,
            "profitable_fraction": round(self.consistency, 4),
            "oos_return_pct": round(self.oos_return_pct, 6),
            "oos_expectancy_r": round(self.oos_expectancy_r, 4),
            "mean_test_profit_factor": round(self.mean_test_profit_factor, 4),
            "worst_window_return_pct": round(self.worst_window_return_pct, 6),
            "best_window_return_pct": round(self.best_window_return_pct, 6),
            "return_dispersion": round(self.return_dispersion, 6),
            "degradation_vs_train": round(self.degradation_vs_train, 4),
            "passed": self.passed,
            "notes": self.notes,
            "windows": [w.to_dict() for w in self.windows],
        }


def build_windows(
    start: dt.date,
    end: dt.date,
    *,
    n_windows: int = 6,
    mode: str = "anchored",
    embargo_days: int = 5,
    purge_days: int = 3,
    min_train_days: int = 60,
    initial_train_fraction: float = 0.4,
) -> List[Window]:
    """Split ``[start, end]`` into walk-forward train/test windows.

    Anchored  : the training set grows; each test window is a fixed step forward.
    Rolling   : the training set stays a fixed length and slides.
    """
    n_windows = max(1, int(n_windows))
    total_days = (end - start).days
    if total_days < min_train_days * 2:
        raise ValueError(
            f"date range of {total_days} days is too short for walk-forward testing "
            f"(need at least {min_train_days * 2})"
        )

    initial_train = max(min_train_days, int(total_days * initial_train_fraction))
    remaining = total_days - initial_train
    step = max(1, remaining // n_windows)

    windows: List[Window] = []
    for index in range(n_windows):
        test_start = start + dt.timedelta(days=initial_train + index * step)
        test_end = min(end, test_start + dt.timedelta(days=step - 1))
        if test_start > end:
            break
        train_end = test_start - dt.timedelta(days=1 + embargo_days + purge_days)
        if mode == "rolling":
            train_start = max(start, train_end - dt.timedelta(days=initial_train))
        else:
            train_start = start
        if (train_end - train_start).days < min_train_days:
            continue
        windows.append(
            Window(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_days=purge_days,
                embargo_days=embargo_days,
            )
        )
    if not windows:
        raise ValueError("could not construct any valid walk-forward window from the supplied range")
    return windows


class WalkForwardRunner:
    """Runs a strategy across walk-forward windows."""

    def __init__(self, evaluate: Callable[[dt.date, dt.date], Mapping[str, Any]]) -> None:
        """``evaluate(train_start, test_end)`` must return a metrics dict for the
        period. The caller builds the backtester; this class only orchestrates.
        """
        self.evaluate = evaluate

    def run(
        self,
        start: dt.date,
        end: dt.date,
        *,
        n_windows: int = 6,
        mode: str = "anchored",
        embargo_days: int = 5,
        purge_days: int = 3,
        min_profitable_fraction: float = 0.60,
        max_degradation_pct: float = 0.50,
    ) -> WalkForwardSummary:
        windows = build_windows(
            start,
            end,
            n_windows=n_windows,
            mode=mode,
            embargo_days=embargo_days,
            purge_days=purge_days,
        )
        summary = WalkForwardSummary(mode=mode)
        test_returns: List[float] = []
        test_expectancies: List[float] = []
        profit_factors: List[float] = []
        degradations: List[float] = []

        for window in windows:
            result = WindowResult(window=window)
            try:
                result.train_metrics = dict(self.evaluate(window.train_start, window.train_end))
                result.test_metrics = dict(self.evaluate(window.test_start, window.test_end))
            except Exception as exc:
                result.error = str(exc)
                log.warning("walk-forward window failed", context={**window.to_dict(), "error": str(exc)})
                summary.windows.append(result)
                continue

            summary.windows.append(result)
            if result.test_metrics.get("trades", 0) > 0:
                test_returns.append(result.test_return_pct)
                test_expectancies.append(result.test_expectancy_r)
                profit_factors.append(result.test_profit_factor)
            train_expectancy = float(result.train_metrics.get("expectancy_r", 0.0) or 0.0)
            if train_expectancy > 0:
                degradations.append(
                    max(0.0, (train_expectancy - result.test_expectancy_r) / train_expectancy)
                )

        summary.total_windows = len(summary.windows)
        summary.profitable_windows = sum(1 for w in summary.windows if w.profitable)
        summary.consistency = (summary.profitable_windows / summary.total_windows) if summary.total_windows else 0.0
        summary.oos_return_pct = float(np.prod([1 + r for r in test_returns]) - 1) if test_returns else 0.0
        summary.oos_expectancy_r = float(np.mean(test_expectancies)) if test_expectancies else 0.0
        summary.mean_test_profit_factor = float(np.mean(profit_factors)) if profit_factors else 0.0
        summary.worst_window_return_pct = min(test_returns) if test_returns else 0.0
        summary.best_window_return_pct = max(test_returns) if test_returns else 0.0
        summary.return_dispersion = float(np.std(test_returns, ddof=1)) if len(test_returns) > 1 else 0.0
        summary.degradation_vs_train = float(np.mean(degradations)) if degradations else 0.0

        notes: List[str] = []
        if summary.total_windows < 3:
            notes.append(f"only {summary.total_windows} valid window(s) - evidence is weak")
        if summary.oos_expectancy_r <= 0:
            notes.append("out-of-sample expectancy is not positive")
        if summary.degradation_vs_train > max_degradation_pct:
            notes.append(
                f"performance degrades {summary.degradation_vs_train:.0%} out of sample "
                f"(tolerance {max_degradation_pct:.0%})"
            )
        summary.notes = notes
        summary.passed = (
            summary.consistency >= min_profitable_fraction
            and summary.total_windows >= 3
            and summary.degradation_vs_train <= max_degradation_pct
        )
        return summary


__all__ = ["WalkForwardRunner", "WalkForwardSummary", "Window", "WindowResult", "build_windows"]
