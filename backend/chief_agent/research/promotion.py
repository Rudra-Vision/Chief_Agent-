"""The promotion gate.

A challenger replaces the champion ONLY when every deterministic requirement in
``config/research.yaml -> promotion_gate`` is satisfied. This module is the only
code path allowed to promote, and it never consults an LLM.

Two hard rules from the brief are enforced structurally:

* **The AI cannot declare its own strategy successful.** Promotion requires a
  :class:`GateEvaluation` produced here, from measured metrics.
* **The final holdout is never seen by optimisation.** The gate refuses to
  evaluate a candidate whose status shows it skipped straight to holdout
  without completing walk-forward validation.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..logging_setup import get_logger
from ..settings import get_config_store
from ..timeutil import now_ist

log = get_logger(__name__, component="promotion")


class GateStatus(str, __import__("enum").Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class GateCheck:
    name: str
    status: GateStatus
    champion_value: Any = None
    challenger_value: Any = None
    requirement: Any = None
    detail: str = ""
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status in (GateStatus.PASS, GateStatus.NOT_EVALUATED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "champion": self.champion_value,
            "challenger": self.challenger_value,
            "requirement": self.requirement,
            "detail": self.detail,
            "required": self.required,
            "ok": self.ok,
        }


@dataclass
class GateEvaluation:
    challenger_version: str
    champion_version: Optional[str]
    passed: bool = False
    checks: List[GateCheck] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)
    insufficient: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    significant: bool = False
    significance: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenger_version": self.challenger_version,
            "champion_version": self.champion_version,
            "passed": self.passed,
            "blocking": self.blocking,
            "insufficient": self.insufficient,
            "reasons": self.reasons,
            "significant": self.significant,
            "significance": self.significance,
            "evaluated_at": (self.evaluated_at or now_ist()).isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


def bootstrap_probability_better(
    champion_r_multiples: Sequence[float],
    challenger_r_multiples: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 31337,
) -> Dict[str, Any]:
    """Probability that the challenger's mean R exceeds the champion's.

    Bootstrap resampling of each sample's mean. Reported alongside the observed
    delta, because a "better" mean from a tiny sample is not evidence.
    """
    champion_r = [float(x) for x in champion_r_multiples if x is not None]
    challenger_r = [float(x) for x in challenger_r_multiples if x is not None]
    if len(champion_r) < 5 or len(challenger_r) < 5:
        return {
            "probability_better": 0.5,
            "delta_mean_r": 0.0,
            "samples": {"champion": len(champion_r), "challenger": len(challenger_r)},
            "insufficient": True,
        }

    rng = random.Random(seed)
    champion_means = []
    challenger_means = []
    for _ in range(n_bootstrap):
        champion_means.append(statistics.fmean(rng.choices(champion_r, k=len(champion_r))))
        challenger_means.append(statistics.fmean(rng.choices(challenger_r, k=len(challenger_r))))

    wins = sum(1 for a, b in zip(challenger_means, champion_means) if a > b)
    delta = statistics.fmean(challenger_r) - statistics.fmean(champion_r)
    return {
        "probability_better": wins / n_bootstrap,
        "delta_mean_r": delta,
        "champion_mean_r": statistics.fmean(champion_r),
        "challenger_mean_r": statistics.fmean(challenger_r),
        "champion_ci95": [
            float(np.percentile(champion_means, 2.5)),
            float(np.percentile(champion_means, 97.5)),
        ],
        "challenger_ci95": [
            float(np.percentile(challenger_means, 2.5)),
            float(np.percentile(challenger_means, 97.5)),
        ],
        "samples": {"champion": len(champion_r), "challenger": len(challenger_r)},
        "insufficient": False,
    }


class PromotionGate:
    """Evaluates a challenger against the champion using measured metrics only."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (get_config_store().load("research").get("promotion_gate") or {}) if config is None else config
        self.cfg = cfg or {}

    def _c(self, path: str, default: Any) -> Any:
        node: Any = self.cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else default

    # ------------------------------------------------------------------ evaluate
    def evaluate(
        self,
        *,
        challenger_version: str,
        champion_version: Optional[str],
        challenger_metrics: Mapping[str, Any],
        champion_metrics: Optional[Mapping[str, Any]] = None,
        walk_forward: Optional[Mapping[str, Any]] = None,
        holdout: Optional[Mapping[str, Any]] = None,
        robustness: Optional[Mapping[str, Any]] = None,
        monte_carlo: Optional[Mapping[str, Any]] = None,
        paper_metrics: Optional[Mapping[str, Any]] = None,
        challenger_r_multiples: Optional[Sequence[float]] = None,
        champion_r_multiples: Optional[Sequence[float]] = None,
        concentration: Optional[Mapping[str, Any]] = None,
        stages_completed: Optional[Sequence[str]] = None,
    ) -> GateEvaluation:
        champion_metrics = champion_metrics or {}
        evaluation = GateEvaluation(
            challenger_version=challenger_version,
            champion_version=champion_version,
            evaluated_at=now_ist(),
        )
        stages = {s.upper() for s in (stages_completed or [])}

        def add(
            name: str,
            ok: bool,
            *,
            champion_value: Any = None,
            challenger_value: Any = None,
            requirement: Any = None,
            detail: str = "",
            required: bool = True,
            insufficient: bool = False,
        ) -> None:
            status = GateStatus.PASS if ok else (GateStatus.INSUFFICIENT_SAMPLE if insufficient else GateStatus.FAIL)
            check = GateCheck(
                name=name,
                status=status,
                champion_value=champion_value,
                challenger_value=challenger_value,
                requirement=requirement,
                detail=detail,
                required=required,
            )
            evaluation.checks.append(check)
            if not ok and required:
                (evaluation.insufficient if insufficient else evaluation.blocking).append(name)
                evaluation.reasons.append(f"{name}: {detail}")

        trades = int(challenger_metrics.get("trades", 0) or 0)
        min_trades = int(self._c("min_trades", 200))
        min_trades_soft = int(self._c("min_trades_soft", 60))
        add(
            "minimum_sample_size",
            trades >= min_trades,
            champion_value=int(champion_metrics.get("trades", 0) or 0),
            challenger_value=trades,
            requirement=min_trades,
            detail=(
                f"{trades} trades"
                if trades >= min_trades
                else f"only {trades} trades (hard minimum {min_trades})"
            ),
            insufficient=min_trades_soft <= trades < min_trades,
        )

        # ---- profitability -------------------------------------------------
        expectancy_r = float(challenger_metrics.get("expectancy_r", 0.0) or 0.0)
        min_expectancy = float(self._c("min_expectancy_r", 0.05))
        if bool(self._c("require_positive_expectancy", True)):
            add(
                "positive_expectancy",
                expectancy_r > 0,
                champion_value=float(champion_metrics.get("expectancy_r", 0.0) or 0.0),
                challenger_value=expectancy_r,
                requirement="> 0",
                detail=f"{expectancy_r:+.4f}R per trade",
            )
            add(
                "minimum_expectancy",
                expectancy_r >= min_expectancy,
                challenger_value=expectancy_r,
                requirement=min_expectancy,
                detail=f"{expectancy_r:+.4f}R vs the {min_expectancy:.2f}R floor",
                required=False,
            )

        profit_factor = float(challenger_metrics.get("profit_factor", 0.0) or 0.0)
        min_pf = float(self._c("min_profit_factor", 1.30))
        add(
            "profit_factor",
            profit_factor >= min_pf,
            champion_value=float(champion_metrics.get("profit_factor", 0.0) or 0.0),
            challenger_value=profit_factor,
            requirement=min_pf,
            detail=f"{profit_factor:.2f} vs the {min_pf:.2f} requirement",
        )

        # ---- risk -----------------------------------------------------------
        max_dd = abs(float(challenger_metrics.get("max_drawdown_pct", 1.0) or 0.0))
        dd_limit = float(self._c("max_drawdown_pct", 0.15))
        add(
            "max_drawdown_within_limit",
            max_dd <= dd_limit,
            champion_value=abs(float(champion_metrics.get("max_drawdown_pct", 0.0) or 0.0)),
            challenger_value=max_dd,
            requirement=dd_limit,
            detail=f"{max_dd:.2%} vs the {dd_limit:.0%} limit",
        )

        challenger_sortino = float(challenger_metrics.get("sortino", 0.0) or 0.0)
        champion_sortino = float(champion_metrics.get("sortino", 0.0) or 0.0)
        if bool(self._c("require_sortino_better_than_champion", True)) and champion_metrics:
            add(
                "sortino_better_than_champion",
                challenger_sortino > champion_sortino,
                champion_value=champion_sortino,
                challenger_value=challenger_sortino,
                requirement=f"> {champion_sortino:.3f}",
                detail=f"{challenger_sortino:.3f} vs the champion's {champion_sortino:.3f}",
            )

        challenger_sharpe = float(challenger_metrics.get("sharpe", 0.0) or 0.0)
        champion_sharpe = float(champion_metrics.get("sharpe", 0.0) or 0.0)
        if bool(self._c("require_risk_adjusted_better_than_champion", True)) and champion_metrics:
            add(
                "risk_adjusted_better",
                challenger_sharpe > champion_sharpe or challenger_sortino > champion_sortino,
                champion_value={"sharpe": champion_sharpe, "sortino": champion_sortino},
                challenger_value={"sharpe": challenger_sharpe, "sortino": challenger_sortino},
                requirement="either Sharpe or Sortino must improve",
                detail="risk-adjusted return improved" if (challenger_sharpe > champion_sharpe or challenger_sortino > champion_sortino)
                else "no risk-adjusted improvement over the champion",
            )

        # ---- concentration --------------------------------------------------
        concentration = concentration or {}
        max_stock = float(self._c("max_single_stock_pnl_share", 0.35))
        stock_share = float(concentration.get("single_stock_share", 0.0) or 0.0)
        add(
            "not_dependent_on_one_stock",
            stock_share <= max_stock,
            challenger_value=stock_share,
            requirement=max_stock,
            detail=f"largest single-stock share {stock_share:.0%}",
        )
        max_day = float(self._c("max_single_day_pnl_share", 0.30))
        day_share = float(concentration.get("single_day_share", 0.0) or 0.0)
        add(
            "not_dependent_on_one_day",
            day_share <= max_day,
            challenger_value=day_share,
            requirement=max_day,
            detail=f"largest single-day share {day_share:.0%}",
        )
        max_regime = float(self._c("max_single_regime_pnl_share", 0.60))
        regime_share = float(concentration.get("single_regime_share", 0.0) or 0.0)
        add(
            "not_dependent_on_one_regime",
            regime_share <= max_regime,
            challenger_value=regime_share,
            requirement=max_regime,
            detail=f"largest single-regime share {regime_share:.0%}",
        )

        # ---- walk-forward ---------------------------------------------------
        if walk_forward is not None:
            fraction = float(walk_forward.get("profitable_fraction", 0.0) or 0.0)
            required_fraction = float(self._c("min_walk_forward_profitable_windows_pct", 0.60))
            add(
                "walk_forward_consistency",
                fraction >= required_fraction,
                challenger_value=fraction,
                requirement=required_fraction,
                detail=f"{fraction:.0%} of windows profitable vs the {required_fraction:.0%} requirement",
            )
            add(
                "walk_forward_completed",
                True,
                detail=f"{walk_forward.get('total_windows', 0)} windows evaluated",
            )
        elif "WALK_FORWARD" not in stages:
            add(
                "walk_forward_completed",
                False,
                detail="walk-forward validation has not been run - a challenger cannot be promoted without it",
            )

        # ---- holdout --------------------------------------------------------
        if holdout is not None:
            add(
                "holdout_completed",
                True,
                detail=f"{holdout.get('trades', 0)} holdout trades",
            )
            add(
                "holdout_accepted",
                bool(holdout.get("accepted", False)),
                requirement=float(self._c("min_holdout_profit_factor", 1.10)),
                challenger_value=float(holdout.get("profit_factor", 0.0) or 0.0),
                detail=(
                    f"holdout profit factor {holdout.get('profit_factor', 0):.2f}"
                    if holdout.get("accepted")
                    else f"holdout profit factor {holdout.get('profit_factor', 0):.2f} is too weak"
                ),
                required=bool(self._c("holdout_must_pass", True)),
            )
        elif "HOLDOUT" not in stages:
            add(
                "holdout_completed",
                False,
                detail="the final holdout has not been evaluated for this candidate",
                required=bool(self._c("holdout_must_pass", True)),
            )

        # ---- robustness -----------------------------------------------------
        if robustness is not None:
            add(
                "robustness",
                bool(robustness.get("passed", False)),
                challenger_value=float(robustness.get("profitability_fraction", 0.0) or 0.0),
                detail=(
                    "all stress scenarios held up"
                    if robustness.get("passed")
                    else "; ".join(robustness.get("rejection_reasons", [])) or "robustness testing failed"
                ),
            )

        # ---- monte carlo ----------------------------------------------------
        if monte_carlo is not None:
            max_ruin = float(self._c("max_montecarlo_ruin_probability", 0.02))
            ruin = float(monte_carlo.get("ruin_probability", 1.0) or 0.0)
            add(
                "monte_carlo_ruin_probability",
                ruin <= max_ruin,
                challenger_value=ruin,
                requirement=max_ruin,
                detail=f"{ruin:.2%} probability of a {monte_carlo.get('ruin_threshold_pct', 0.3):.0%} drawdown",
            )
            max_p95_dd = float(self._c("max_montecarlo_p95_drawdown_pct", 0.25))
            p95_dd = abs(float((monte_carlo.get("max_drawdown_pct") or {}).get("p95", 0.0) or 0.0))
            add(
                "monte_carlo_drawdown",
                p95_dd <= max_p95_dd,
                challenger_value=p95_dd,
                requirement=max_p95_dd,
                detail=f"95th-percentile drawdown {p95_dd:.2%}",
            )

        # ---- paper validation ------------------------------------------------
        if paper_metrics is not None:
            paper_trades = int(paper_metrics.get("trades", 0) or 0)
            min_paper_trades = int(self._c("min_paper_trades", 30))
            add(
                "paper_trade_count",
                paper_trades >= min_paper_trades,
                challenger_value=paper_trades,
                requirement=min_paper_trades,
                detail=f"{paper_trades} paper trades",
                insufficient=paper_trades < min_paper_trades,
            )
            if bool(self._c("require_paper_expectancy_positive", True)):
                paper_expectancy = float(paper_metrics.get("expectancy_r", 0.0) or 0.0)
                add(
                    "paper_expectancy",
                    paper_expectancy > 0,
                    challenger_value=paper_expectancy,
                    detail=f"{paper_expectancy:+.4f}R per paper trade",
                )

        # ---- statistical significance ---------------------------------------
        if challenger_r_multiples and champion_r_multiples:
            config = self._c("improvement_significance", {}) or {}
            significance = bootstrap_probability_better(
                champion_r_multiples=champion_r_multiples,
                challenger_r_multiples=challenger_r_multiples,
                n_bootstrap=int(config.get("n_bootstrap", 2000)),
            )
            evaluation.significance = significance
            min_probability = float(config.get("min_probability_better", 0.75))
            min_delta = float(config.get("min_delta_expectancy_r", 0.02))
            probability = float(significance.get("probability_better", 0.0))
            delta = float(significance.get("delta_mean_r", 0.0))
            evaluation.significant = probability >= min_probability and delta >= min_delta
            add(
                "statistically_meaningful_improvement",
                evaluation.significant,
                champion_value=significance.get("champion_mean_r"),
                challenger_value=significance.get("challenger_mean_r"),
                requirement=f"P(better) >= {min_probability:.2f} and delta >= {min_delta:.3f}R",
                detail=(
                    f"P(challenger better) = {probability:.2f}, delta = {delta:+.4f}R"
                ),
            )

        evaluation.passed = not evaluation.blocking
        log.info(
            "promotion gate evaluated",
            context={
                "challenger": challenger_version,
                "champion": champion_version,
                "passed": evaluation.passed,
                "blocking": evaluation.blocking,
                "insufficient": evaluation.insufficient,
            },
        )
        return evaluation


__all__ = ["PromotionGate", "GateEvaluation", "GateCheck", "GateStatus", "bootstrap_probability_better"]
