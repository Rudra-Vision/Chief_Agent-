"""Experiment runner: one controlled variable, measured end to end.

Pipeline for every challenger:

    1. create challenger  (exactly ONE variable changed, config hashed)
    2. backtest           (train + validation on data the optimiser may see)
    3. walk-forward       (anchored windows, embargo + purge)
    4. robustness         (parameter perturbation, cost stress, missed trades)
    5. Monte Carlo        (drawdown / ruin / losing-streak distributions)
    6. holdout            (FINAL unseen data - only now is it unlocked)
    7. promotion gate     (deterministic; decides ELIGIBLE / REJECTED)
    8. paper validation   (runs in the live loop, not here)

Every step is persisted so the dashboard can show exactly where a candidate
stands, and every evaluated combination is written to the multiple-testing
ledger so the system can never quietly select the luckiest of thousands of runs.
"""

from __future__ import annotations

import copy
import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..backtest.engine import BacktestConfig, BacktestResult, Backtester
from ..backtest.montecarlo import run_monte_carlo
from ..backtest.robustness import RobustnessRunner
from ..backtest.runner import BacktestRequest, BacktestRunner, ensure_universe_instruments
from ..backtest.walkforward import WalkForwardRunner, build_windows
from ..costs.transaction_costs import CostModel
from ..data.provider import MarketDataProvider
from ..data.schema import Experiment, MultipleTestingLedger, StrategyVersion, TradeJournal, WalkForwardRun
from ..logging_setup import get_logger
from ..regime.engine import RegimeEngine
from ..settings import get_config_store
from ..strategies.base import deep_get
from ..strategies.orb_vwap import ORBVWAPStrategy
from ..timeutil import now_ist
from .champion import ChampionRegistry, StrategyVersionInfo
from .hypotheses import HypothesisGenerator, HypothesisProposal
from .promotion import GateEvaluation, PromotionGate

log = get_logger(__name__, component="experiments")


@dataclass
class ExperimentPlan:
    """Everything needed to run one controlled experiment."""

    hypothesis_id: Optional[str]
    variable: str
    old_value: Any
    new_value: Any
    statement: str
    universe: List[str]
    start_date: dt.date
    end_date: dt.date
    initial_capital: float = 500_000.0
    risk_per_trade_pct: float = 0.0025
    timeframe: str = "1m"
    n_windows: int = 6
    holdout_fraction: float = 0.10
    embargo_days: int = 5
    purge_days: int = 3
    run_robustness: bool = True
    run_monte_carlo: bool = True
    max_universe: int = 40
    bump: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


@dataclass
class ExperimentOutcome:
    experiment_id: str
    challenger_version: str
    champion_version: Optional[str]
    status: str = "CREATED"
    variable_changed: str = ""
    old_value: Any = None
    new_value: Any = None
    champion_metrics: Dict[str, Any] = field(default_factory=dict)
    challenger_metrics: Dict[str, Any] = field(default_factory=dict)
    delta_metrics: Dict[str, Any] = field(default_factory=dict)
    walk_forward: Optional[Dict[str, Any]] = None
    holdout: Optional[Dict[str, Any]] = None
    robustness: Optional[Dict[str, Any]] = None
    monte_carlo: Optional[Dict[str, Any]] = None
    significance: Optional[Dict[str, Any]] = None
    gate: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    promotion_reason: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Any = None
    completed_at: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "challenger_version": self.challenger_version,
            "champion_version": self.champion_version,
            "status": self.status,
            "variable_changed": self.variable_changed,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "champion_metrics": self.champion_metrics,
            "challenger_metrics": self.challenger_metrics,
            "delta_metrics": self.delta_metrics,
            "walk_forward": self.walk_forward,
            "holdout": self.holdout,
            "robustness": self.robustness,
            "monte_carlo": self.monte_carlo,
            "significance": self.significance,
            "gate": self.gate,
            "rejection_reason": self.rejection_reason,
            "promotion_reason": self.promotion_reason,
            "steps": self.steps,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ExperimentRunner:
    """Runs a challenger through the full validation pipeline."""

    def __init__(
        self,
        session: Session,
        provider: MarketDataProvider,
        *,
        champion_registry: Optional[ChampionRegistry] = None,
        promotion_gate: Optional[PromotionGate] = None,
        progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.session = session
        self.provider = provider
        self.registry = champion_registry or ChampionRegistry(session)
        self.gate = promotion_gate or PromotionGate()
        self.config_store = get_config_store()
        self.progress = progress
        self._counter = self._next_experiment_number()
        from .artifacts import get_artifact_store

        self.artifacts = get_artifact_store()

    # ------------------------------------------------------------------ utils
    def _next_experiment_number(self) -> int:
        rows = self.session.execute(select(Experiment)).scalars().all()
        return len(rows) + 1

    def _new_id(self) -> str:
        experiment_id = f"EXP-{self._counter:06d}"
        while self.session.execute(
            select(Experiment).where(Experiment.experiment_id == experiment_id)
        ).scalar_one_or_none() is not None:
            self._counter += 1
            experiment_id = f"EXP-{self._counter:06d}"
        self._counter += 1
        return experiment_id

    def _step(self, outcome: ExperimentOutcome, name: str, detail: str, **extra: Any) -> None:
        entry = {"ts": now_ist().isoformat(), "step": name, "detail": detail, **extra}
        outcome.steps.append(entry)
        log.info("experiment step", context={"experiment": outcome.experiment_id, **entry})
        if self.progress is not None:
            try:
                self.progress(name, entry)
            except Exception:  # pragma: no cover
                pass

    # -------------------------------------------------------------------- run
    def run(
        self,
        plan: ExperimentPlan,
        *,
        champion_metrics: Optional[Dict[str, Any]] = None,
        champion_r_multiples: Optional[Sequence[float]] = None,
    ) -> ExperimentOutcome:
        experiment_id = self._new_id()
        champion = self.registry.champion()
        champion_version = champion.version if champion else None

        outcome = ExperimentOutcome(
            experiment_id=experiment_id,
            challenger_version="",
            champion_version=champion_version,
            variable_changed=plan.variable,
            old_value=plan.old_value,
            new_value=plan.new_value,
            created_at=now_ist(),
        )
        self._step(outcome, "created", f"experiment {experiment_id} for {plan.variable}")

        # ---- 1. challenger ---------------------------------------------------
        try:
            info, challenger_config, old_value = self.registry.create_challenger(
                variable=plan.variable,
                new_value=plan.new_value,
                reason=plan.statement,
                hypothesis_id=plan.hypothesis_id,
                bump=plan.bump,
            )
        except Exception as exc:
            outcome.status = "REJECTED"
            outcome.rejection_reason = f"could not create the challenger: {exc}"
            self._step(outcome, "challenger_failed", str(exc))
            self._persist(plan, outcome)
            self.artifacts.experiment(outcome.to_dict())
            return outcome

        outcome.challenger_version = info.version
        outcome.old_value = old_value
        self.registry.transition(info.version, "TESTING", reason="experiment started")
        self._step(
            outcome,
            "challenger_created",
            f"{info.version} changes {plan.variable}: {old_value!r} -> {plan.new_value!r}",
            config_hash=info.config_hash,
        )

        # ---- split the timeline (holdout stays locked) -----------------------
        total_days = (plan.end_date - plan.start_date).days
        holdout_days = max(20, int(total_days * plan.holdout_fraction))
        holdout_start = plan.end_date - dt.timedelta(days=holdout_days)
        optimisation_end = holdout_start - dt.timedelta(days=plan.embargo_days + plan.purge_days)
        if (optimisation_end - plan.start_date).days < 90:
            outcome.status = "INSUFFICIENT_SAMPLE"
            outcome.rejection_reason = (
                "the date range is too short to reserve a holdout AND run walk-forward validation"
            )
            self._step(outcome, "split_failed", outcome.rejection_reason)
            self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason)
            self._persist(plan, outcome)
            self.artifacts.experiment(outcome.to_dict())
            return outcome

        self._step(
            outcome,
            "split",
            f"optimisation {plan.start_date}..{optimisation_end}, holdout {holdout_start}..{plan.end_date} "
            f"(the holdout is not touched until validation completes)",
        )

        # ---- 2. backtest the challenger ---------------------------------------
        try:
            challenger_result = self._backtest(
                challenger_config, plan, plan.start_date, optimisation_end, tag="challenger"
            )
        except Exception as exc:
            outcome.status = "FAILED"
            outcome.rejection_reason = f"challenger backtest failed: {exc}"
            self._step(outcome, "backtest_failed", str(exc))
            self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason)
            self._persist(plan, outcome)
            return outcome

        outcome.challenger_metrics = challenger_result.metrics.to_dict()
        self._step(
            outcome,
            "backtest",
            f"{outcome.challenger_metrics.get('trades')} trades, "
            f"expectancy {outcome.challenger_metrics.get('expectancy_r')}R, "
            f"PF {outcome.challenger_metrics.get('profit_factor')}",
        )

        # Champion metrics for comparison (measured over the SAME period).
        if champion_metrics is None:
            try:
                champion_config = self.registry.config_for(champion_version)
                champion_result = self._backtest(
                    champion_config, plan, plan.start_date, optimisation_end, tag="champion"
                )
                champion_metrics = champion_result.metrics.to_dict()
                if champion_r_multiples is None:
                    champion_r_multiples = [t.r_multiple for t in champion_result.trades]
            except Exception as exc:
                log.warning("champion baseline backtest failed", context={"error": str(exc)})
                champion_metrics = {}
        outcome.champion_metrics = dict(champion_metrics or {})
        outcome.delta_metrics = _delta_metrics(outcome.champion_metrics, outcome.challenger_metrics)
        self._step(
            outcome,
            "baseline",
            f"champion expectancy {outcome.champion_metrics.get('expectancy_r')}R vs "
            f"challenger {outcome.challenger_metrics.get('expectancy_r')}R "
            f"(delta {outcome.delta_metrics.get('expectancy_r', 0):+.4f}R)",
        )

        convergence = self._convergence_verdict(outcome)
        if convergence is not None:
            outcome.status = "REJECTED"
            outcome.rejection_reason = convergence
            self._step(outcome, "convergence", convergence)
            self.registry.transition(info.version, "REJECTED", reason=convergence)
            self._persist(plan, outcome)
            return outcome

        # ---- 3. walk-forward ---------------------------------------------------
        self.registry.transition(info.version, "WALK_FORWARD", reason="walk-forward validation started")
        try:
            walk_forward = self._walk_forward(challenger_config, plan, plan.start_date, optimisation_end)
            outcome.walk_forward = walk_forward.to_dict()
            self._step(
                outcome,
                "walk_forward",
                f"{walk_forward.profitable_windows}/{walk_forward.total_windows} windows profitable, "
                f"degradation {walk_forward.degradation_vs_train:.0%}",
            )
        except Exception as exc:
            outcome.status = "FAILED"
            outcome.rejection_reason = f"walk-forward validation failed: {exc}"
            self._step(outcome, "walk_forward_failed", str(exc))
            self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason)
            self._persist(plan, outcome)
            return outcome

        if not walk_forward.passed:
            outcome.status = "FAILED_GATE"
            outcome.rejection_reason = "walk-forward validation failed: " + "; ".join(walk_forward.notes)
            self._step(outcome, "walk_forward_rejected", outcome.rejection_reason)
            self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason)
            self._persist(plan, outcome)
            return outcome

        # ---- 4. robustness ------------------------------------------------------
        if plan.run_robustness:
            try:
                robustness = self._robustness(challenger_config, plan, plan.start_date, optimisation_end)
                outcome.robustness = robustness.to_dict()
                self._step(
                    outcome,
                    "robustness",
                    f"{robustness.passed_scenarios}/{robustness.total_scenarios} stress scenarios stayed profitable",
                )
                if not robustness.passed:
                    outcome.status = "FAILED_GATE"
                    outcome.rejection_reason = "robustness testing failed: " + "; ".join(robustness.rejection_reasons)
                    self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason)
                    self._persist(plan, outcome)
                    return outcome
            except Exception as exc:
                self._step(outcome, "robustness_failed", str(exc))

        # ---- 5. Monte Carlo ------------------------------------------------------
        if plan.run_monte_carlo:
            try:
                pnls = [t.net_pnl for t in challenger_result.trades]
                monte_carlo = run_monte_carlo(
                    pnls,
                    initial_capital=plan.initial_capital,
                    n_simulations=int(
                        deep_get(self.config_store.load("research"), "montecarlo.n_simulations", 2000)
                    ),
                    ruin_threshold_pct=float(
                        deep_get(self.config_store.load("research"), "montecarlo.ruin_threshold_pct", 0.30)
                    ),
                    max_drawdown_limit_pct=float(
                        deep_get(
                            self.config_store.load("research"),
                            "promotion_gate.max_montecarlo_p95_drawdown_pct",
                            0.25,
                        )
                    ),
                )
                outcome.monte_carlo = monte_carlo.to_dict()
                self._step(
                    outcome,
                    "monte_carlo",
                    f"p95 drawdown {monte_carlo.p95_max_drawdown_pct:.2%}, "
                    f"ruin probability {monte_carlo.ruin_probability:.2%}",
                )
            except Exception as exc:
                self._step(outcome, "monte_carlo_failed", str(exc))

        # ---- 6. HOLDOUT (final unseen data) --------------------------------------
        self.registry.transition(info.version, "HOLDOUT", reason="validated on test data; unlocking the holdout")
        try:
            holdout_result = self._backtest(
                challenger_config, plan, holdout_start, plan.end_date, tag="holdout"
            )
            holdout_metrics = holdout_result.metrics.to_dict()
            min_pf = float(
                deep_get(self.config_store.load("research"), "promotion_gate.min_holdout_profit_factor", 1.10)
            )
            holdout = {
                **holdout_metrics,
                "accepted": bool(
                    holdout_metrics.get("trades", 0) >= 10
                    and float(holdout_metrics.get("expectancy_r", 0) or 0) > 0
                    and float(holdout_metrics.get("profit_factor", 0) or 0) >= min_pf
                ),
                "period": {"start": holdout_start.isoformat(), "end": plan.end_date.isoformat()},
            }
            outcome.holdout = holdout
            self._step(
                outcome,
                "holdout",
                f"{holdout_metrics.get('trades')} holdout trades, expectancy {holdout_metrics.get('expectancy_r')}R, "
                f"PF {holdout_metrics.get('profit_factor')} -> {'ACCEPTED' if holdout['accepted'] else 'REJECTED'}",
            )
        except Exception as exc:
            outcome.status = "FAILED"
            outcome.rejection_reason = f"holdout evaluation failed: {exc}"
            self._step(outcome, "holdout_failed", str(exc))
            self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason)
            self._persist(plan, outcome)
            return outcome

        # ---- 7. promotion gate ---------------------------------------------------
        challenger_r = [t.r_multiple for t in challenger_result.trades]
        evaluation = self.gate.evaluate(
            challenger_version=info.version,
            champion_version=champion_version,
            challenger_metrics=outcome.challenger_metrics,
            champion_metrics=outcome.champion_metrics,
            walk_forward=outcome.walk_forward,
            holdout=outcome.holdout,
            robustness=outcome.robustness,
            monte_carlo=outcome.monte_carlo,
            challenger_r_multiples=challenger_r,
            champion_r_multiples=champion_r_multiples,
            concentration=challenger_result.metrics.concentration,
            stages_completed=["BACKTEST", "WALK_FORWARD", "HOLDOUT"],
        )
        outcome.gate = evaluation.to_dict()
        outcome.significance = evaluation.significance
        self._record_trials(plan, outcome, info)

        if evaluation.passed:
            outcome.status = "PAPER_VALIDATION"
            outcome.promotion_reason = (
                "passed every deterministic promotion requirement; "
                "the candidate must still complete paper validation before promotion"
            )
            self.registry.transition(info.version, "PAPER", reason=outcome.promotion_reason,
                                     evidence={"backtest": outcome.challenger_metrics,
                                               "walk_forward": outcome.walk_forward,
                                               "holdout": outcome.holdout})
            self._step(outcome, "gate_passed", "all promotion requirements satisfied -> paper validation")
        else:
            outcome.status = "INSUFFICIENT_SAMPLE" if evaluation.insufficient and not evaluation.blocking else "FAILED_GATE"
            outcome.rejection_reason = "; ".join(evaluation.reasons[:5]) or "promotion requirements not met"
            self.registry.transition(info.version, "REJECTED", reason=outcome.rejection_reason,
                                     evidence={"backtest": outcome.challenger_metrics,
                                               "walk_forward": outcome.walk_forward,
                                               "holdout": outcome.holdout})
            self._step(outcome, "gate_failed", outcome.rejection_reason)

        outcome.completed_at = now_ist()
        self._persist(plan, outcome)
        self.artifacts.experiment(outcome.to_dict())
        return outcome

    # ------------------------------------------------------------------ steps
    def _backtest(
        self,
        strategy_config: Dict[str, Any],
        plan: ExperimentPlan,
        start: dt.date,
        end: dt.date,
        *,
        tag: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        config = copy.deepcopy(strategy_config)
        if overrides:
            for path, value in overrides.items():
                _set_path(config, path, value)
        request = BacktestRequest(
            start_date=start,
            end_date=end,
            universe=plan.universe,
            initial_capital=plan.initial_capital,
            risk_per_trade_pct=plan.risk_per_trade_pct,
            timeframe=plan.timeframe,
            strategy_config=config,
            max_universe=plan.max_universe,
        )
        # Use a dedicated runner so experiment runs are persisted separately.
        runner = BacktestRunner(self.session, self.provider, max_instruments=plan.max_universe)
        result = runner.run(request)
        result.metrics.extras["experiment_tag"] = tag
        return result

    def _walk_forward(self, strategy_config: Dict[str, Any], plan: ExperimentPlan, start: dt.date, end: dt.date):
        def evaluate(window_start: dt.date, window_end: dt.date) -> Mapping[str, Any]:
            result = self._backtest(strategy_config, plan, window_start, window_end, tag="wf-window")
            return result.metrics.to_dict()

        runner = WalkForwardRunner(evaluate)
        summary = runner.run(
            start,
            end,
            n_windows=plan.n_windows,
            mode="anchored",
            embargo_days=plan.embargo_days,
            purge_days=plan.purge_days,
            min_profitable_fraction=float(
                deep_get(self.config_store.load("research"), "promotion_gate.min_walk_forward_profitable_windows_pct", 0.6)
            ),
        )
        self.session.add(
            WalkForwardRun(
                run_id=f"WF-{uuid.uuid4().hex[:10].upper()}",
                strategy_version=str(deep_get(strategy_config, "champion.version", "unknown")),
                experiment_id=None,
                mode=summary.mode,
                n_windows=summary.total_windows,
                embargo_days=plan.embargo_days,
                windows=[w.to_dict() for w in summary.windows],
                summary=summary.to_dict(),
            )
        )
        self.session.flush()
        return summary

    def _robustness(self, strategy_config: Dict[str, Any], plan: ExperimentPlan, start: dt.date, end: dt.date):
        def evaluate(overrides: Mapping[str, Any]) -> Mapping[str, Any]:
            result = self._backtest(strategy_config, plan, start, end, tag="robustness", overrides=dict(overrides))
            return result.metrics.to_dict()

        runner = RobustnessRunner(evaluate)
        base = (self.config_store.load("research").get("robustness") or {})
        return runner.run(
            perturb_variable=plan.variable,
            perturb_values=_perturb_values(plan.old_value, plan.new_value),
            slippage_multiplier=float(base.get("slippage_multiplier", 2.0)),
            fee_multiplier=float(base.get("fees_multiplier", 1.5)),
            delayed_entry_bars=int(base.get("delayed_entry_candles", 1)),
            missed_trade_fraction=float(base.get("missed_trade_pct", 0.10)),
            random_removal_fraction=float(base.get("random_trade_removal_pct", 0.20)),
            max_degradation_pct=float(base.get("max_metric_degradation_pct", 0.40)),
            min_profitable_fraction=float(base.get("perturbed_runs_must_be_profitable_pct", 0.70)),
        )

    # -------------------------------------------------------------- convergence
    def _convergence_verdict(self, outcome: ExperimentOutcome) -> Optional[str]:
        """Cheap checks that let us reject early without burning the holdout."""
        challenger = outcome.challenger_metrics
        champion = outcome.champion_metrics
        trades = int(challenger.get("trades", 0) or 0)
        if trades < 10:
            return f"the challenger produced only {trades} trades in the validation period"
        if float(challenger.get("expectancy_r", 0) or 0) <= 0:
            return (
                f"the challenger's expectancy is {challenger.get('expectancy_r')}R - "
                f"a candidate must be profitable before it is worth validating further"
            )
        if float(challenger.get("profit_factor", 0) or 0) < 1.0:
            return f"the challenger's profit factor is {challenger.get('profit_factor')}"
        if champion:
            champion_expectancy = float(champion.get("expectancy_r", 0) or 0)
            challenger_expectancy = float(challenger.get("expectancy_r", 0) or 0)
            if challenger_expectancy < champion_expectancy:
                return (
                    f"the challenger does not improve on the champion over the same period "
                    f"({challenger_expectancy:+.4f}R vs {champion_expectancy:+.4f}R)"
                )
        return None

    # ------------------------------------------------------------ persistence
    def _record_trials(self, plan: ExperimentPlan, outcome: ExperimentOutcome, info: StrategyVersionInfo) -> None:
        """Every evaluated combination goes into the multiple-testing ledger."""
        for tag, metrics in (("challenger", outcome.challenger_metrics), ("champion", outcome.champion_metrics)):
            if not metrics:
                continue
            self.session.add(
                MultipleTestingLedger(
                    trial_kind="experiment",
                    strategy_version=outcome.challenger_version if tag == "challenger" else (outcome.champion_version or ""),
                    variable=plan.variable,
                    value=str(plan.new_value if tag == "challenger" else plan.old_value),
                    metric_name="expectancy_r",
                    metric_value=float(metrics.get("expectancy_r", 0) or 0),
                    experiment_id=outcome.experiment_id,
                    notes=f"{tag} leg of {outcome.experiment_id}",
                )
            )
        self.session.flush()

    def _persist(self, plan: ExperimentPlan, outcome: ExperimentOutcome) -> Experiment:
        row = self.session.execute(
            select(Experiment).where(Experiment.experiment_id == outcome.experiment_id)
        ).scalar_one_or_none()
        holdout = outcome.holdout or {}
        walk_forward = outcome.walk_forward or {}
        metrics = outcome.challenger_metrics or {}
        if row is None:
            row = Experiment(
                experiment_id=outcome.experiment_id,
                hypothesis_id=plan.hypothesis_id,
                parent_strategy=outcome.champion_version or "unset",
                candidate_strategy=outcome.challenger_version,
                hypothesis=plan.statement,
                variable_changed=plan.variable,
                old_value=str(plan.old_value),
                new_value=str(plan.new_value),
                trial_number=self._counter,
            )
            self.session.add(row)

        row.training_period_start = plan.start_date
        row.training_period_end = plan.end_date
        row.validation_period_start = plan.start_date
        row.validation_period_end = plan.end_date
        if holdout:
            row.holdout_period_start = _parse_date((holdout.get("period") or {}).get("start"))
            row.holdout_period_end = _parse_date((holdout.get("period") or {}).get("end"))

        row.trade_count = int(metrics.get("trades", 0) or 0)
        row.champion_metrics = outcome.champion_metrics
        row.challenger_metrics = metrics
        row.delta_metrics = outcome.delta_metrics
        row.return_pct = float(metrics.get("net_return_pct", 0) or 0)
        row.expectancy_r = float(metrics.get("expectancy_r", 0) or 0)
        row.profit_factor = float(metrics.get("profit_factor", 0) or 0)
        row.sharpe = float(metrics.get("sharpe", 0) or 0)
        row.sortino = float(metrics.get("sortino", 0) or 0)
        row.max_drawdown_pct = float(metrics.get("max_drawdown_pct", 0) or 0)
        row.fees = float(metrics.get("fees", 0) or 0)
        row.slippage = float(metrics.get("slippage", 0) or 0)
        row.regime_stability = float((metrics.get("extras") or {}).get("regime_robustness", 0) or 0)
        row.walk_forward_summary = outcome.walk_forward
        row.holdout_summary = outcome.holdout
        row.robustness_summary = outcome.robustness
        row.monte_carlo_summary = outcome.monte_carlo
        row.significance = outcome.significance
        row.objective_score = float(metrics.get("objective_score", 0) or 0)
        row.status = outcome.status
        row.rejection_reason = outcome.rejection_reason
        row.promotion_reason = outcome.promotion_reason
        row.completed_at = outcome.completed_at or now_ist()
        self.session.flush()
        log.info(
            "experiment persisted",
            context={
                "experiment_id": outcome.experiment_id,
                "status": outcome.status,
                "challenger": outcome.challenger_version,
            },
        )
        return row

    # ------------------------------------------------------------- convenience
    def run_from_proposal(
        self,
        proposal: HypothesisProposal,
        *,
        universe: Sequence[str],
        start_date: dt.date,
        end_date: dt.date,
        **kwargs: Any,
    ) -> ExperimentOutcome:
        plan = ExperimentPlan(
            hypothesis_id=kwargs.pop("hypothesis_id", None),
            variable=proposal.variable,
            old_value=proposal.old_value,
            new_value=proposal.new_value,
            statement=proposal.statement,
            universe=list(universe),
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        return self.run(plan)


def _delta_metrics(champion: Mapping[str, Any], challenger: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "expectancy_r",
        "profit_factor",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "net_return_pct",
        "win_rate",
        "average_r",
        "trades",
        "objective_score",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        try:
            old = float(champion.get(key, 0) or 0)
            new = float(challenger.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        out[key] = round(new - old, 6)
        out[f"{key}_champion"] = round(old, 6)
        out[f"{key}_challenger"] = round(new, 6)
    return out


def _perturb_values(old_value: Any, new_value: Any) -> List[Any]:
    """+/-10% around the proposed value (or neighbouring choices for booleans/enums)."""
    if isinstance(new_value, bool):
        return [True, False]
    if isinstance(new_value, (int, float)) and not isinstance(new_value, bool):
        return sorted({round(float(new_value) * 0.9, 6), round(float(new_value) * 1.1, 6)})
    if isinstance(new_value, list):
        return [list(new_value)]
    return []


def _parse_date(value: Any) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def _set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


__all__ = ["ExperimentRunner", "ExperimentPlan", "ExperimentOutcome"]
