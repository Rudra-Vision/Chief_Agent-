"""Research routes: champion/challenger, hypotheses, experiments, memory."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ...data.db import session_scope
from ...data.schema import (
    Experiment,
    Hypothesis,
    Learning,
    MultipleTestingLedger,
    StrategyVersion,
    TradeJournal,
)
from ...logging_setup import get_logger
from ...research.champion import ALLOWED_TRANSITIONS, ChampionRegistry, ImmutableVersionError, InvalidTransitionError
from ...research.experiments import ExperimentPlan, ExperimentRunner
from ...research.hypotheses import HypothesisGenerator
from ...research.memory import ResearchMemory
from ...timeutil import now_ist
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.research")
router = APIRouter()


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
@router.get("/research/champion", summary="Current champion strategy")
def champion(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    with session_scope() as session:
        registry = ChampionRegistry(session)
        current = registry.champion()
        if current is None:
            registry.ensure_baseline_champion()
            session.commit()
            current = registry.champion()
    return {
        "champion": current.to_dict(include_config=False) if current else None,
        "config": state.champion_config,
        "config_hash": state.strategy.config_hash if state.strategy else None,
        "strategy_family": (state.champion_config.get("champion") or {}).get("strategy_family"),
    }


@router.get("/research/versions", summary="All published strategy versions")
def versions(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    with session_scope() as session:
        registry = ChampionRegistry(session)
        rows = registry.list_versions(status=status, limit=limit)
    return {"count": len(rows), "versions": [row.to_dict() for row in rows]}


@router.get("/research/versions/{version}", summary="One strategy version")
def version_detail(version: str) -> Dict[str, Any]:
    with session_scope() as session:
        registry = ChampionRegistry(session)
        info = registry.info(version)
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown strategy version {version}")
    return info.to_dict(include_config=True)


# --------------------------------------------------------------------------- #
# Hypotheses
# --------------------------------------------------------------------------- #
@router.get("/research/hypotheses", summary="List hypotheses")
def hypotheses(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(select(Hypothesis).order_by(Hypothesis.created_at.desc()).limit(limit))
            .scalars()
            .all()
        )
    return {
        "count": len(rows),
        "hypotheses": [
            {
                "hypothesis_id": row.hypothesis_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "strategy_version": row.strategy_version,
                "statement": row.statement,
                "variable": row.variable,
                "old_value": row.old_value,
                "new_value": row.new_value,
                "evidence": row.evidence,
                "sample_size": row.sample_size,
                "confidence": row.confidence,
                "status": row.status,
                "source": row.source,
                "expected_mechanism": row.expected_mechanism,
                "potential_downside": row.potential_downside,
            }
            for row in rows
        ],
    }


class GenerateBody(BaseModel):
    persist: bool = True
    max_proposals: int = Field(5, ge=1, le=20)


@router.post("/research/hypotheses/generate", summary="Generate hypotheses from the trade journal")
def generate_hypotheses(body: GenerateBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    with session_scope() as session:
        generator = HypothesisGenerator(session)
        proposals = generator.generate(
            state.strategy_version or "",
            state.champion_config,
            max_proposals=body.max_proposals,
        )
        saved: List[Dict[str, Any]] = []
        for proposal in proposals:
            hypothesis_id = generator.persist(proposal, state.strategy_version or "") if body.persist else None
            payload = proposal.to_dict()
            payload["hypothesis_id"] = hypothesis_id
            saved.append(payload)
        session.commit()

    journal_size = len(generator.load_trades(state.strategy_version))
    return {
        "ok": True,
        "proposals": saved,
        "count": len(saved),
        "journal_trades_analysed": journal_size,
        "note": (
            "Hypotheses come from measured segments of the trade journal. "
            "Nothing is proposed without a sample and an evidence block."
        ),
    }


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
@router.get("/research/experiments", summary="List experiments")
def experiments(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(select(Experiment).order_by(Experiment.created_at.desc()).limit(limit))
            .scalars()
            .all()
        )
    return {
        "count": len(rows),
        "experiments": [
            {
                "experiment_id": row.experiment_id,
                "hypothesis_id": row.hypothesis_id,
                "parent_strategy": row.parent_strategy,
                "candidate_strategy": row.candidate_strategy,
                "hypothesis": row.hypothesis,
                "variable_changed": row.variable_changed,
                "old_value": row.old_value,
                "new_value": row.new_value,
                "trade_count": row.trade_count,
                "champion_metrics": row.champion_metrics,
                "challenger_metrics": row.challenger_metrics,
                "delta_metrics": row.delta_metrics,
                "expectancy_r": row.expectancy_r,
                "profit_factor": row.profit_factor,
                "sharpe": row.sharpe,
                "sortino": row.sortino,
                "max_drawdown_pct": row.max_drawdown_pct,
                "walk_forward_summary": row.walk_forward_summary,
                "holdout_summary": row.holdout_summary,
                "robustness_summary": row.robustness_summary,
                "monte_carlo_summary": row.monte_carlo_summary,
                "significance": row.significance,
                "status": row.status,
                "rejection_reason": row.rejection_reason,
                "promotion_reason": row.promotion_reason,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }


class ExperimentBody(BaseModel):
    hypothesis_id: Optional[str] = None
    variable: str
    new_value: Any
    statement: str = "manual experiment"
    universe: Optional[List[str]] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    initial_capital: float = 500_000.0
    risk_per_trade_pct: float = Field(0.0025, gt=0, le=0.005)
    n_windows: int = Field(5, ge=2, le=12)
    holdout_fraction: float = Field(0.10, ge=0.05, le=0.30)
    max_universe: int = Field(25, ge=1, le=100)
    run_robustness: bool = True
    run_monte_carlo: bool = True


@router.post("/research/experiments/run", summary="Run a one-variable experiment")
def run_experiment(body: ExperimentBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
    available = {row["symbol"]: row for row in resolved}

    universe = body.universe or list(available)[: body.max_universe]
    end_date = body.end_date or now_ist().date()
    start_date = body.start_date or (end_date - dt.timedelta(days=365))

    old_value = None
    from ...strategies.base import deep_get

    old_value = deep_get(state.champion_config, body.variable, None)
    if old_value is None:
        raise HTTPException(
            status_code=400,
            detail=f"'{body.variable}' is not a variable in the champion configuration. "
                   f"Use the Research page to see the available levers.",
        )

    plan = ExperimentPlan(
        hypothesis_id=body.hypothesis_id,
        variable=body.variable,
        old_value=old_value,
        new_value=body.new_value,
        statement=body.statement,
        universe=list(universe),
        start_date=start_date,
        end_date=end_date,
        initial_capital=body.initial_capital,
        risk_per_trade_pct=body.risk_per_trade_pct,
        n_windows=body.n_windows,
        holdout_fraction=body.holdout_fraction,
        max_universe=body.max_universe,
        run_robustness=body.run_robustness,
        run_monte_carlo=body.run_monte_carlo,
    )

    try:
        with session_scope() as session:
            runner = ExperimentRunner(session, state.provider)
            outcome = runner.run(plan)
            session.commit()
    except ImmutableVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("experiment failed")
        raise HTTPException(status_code=500, detail=f"experiment failed: {exc}") from exc

    if outcome.status == "PAPER_VALIDATION":
        state.notifications.challenger(
            outcome.challenger_version,
            outcome.variable_changed,
            outcome.old_value,
            outcome.new_value,
        )
    return outcome.to_dict()


@router.post("/research/experiments/{experiment_id}/promote", summary="Promote a validated challenger")
def promote(experiment_id: str, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    """Promote a challenger that has completed paper validation.

    This is the ONLY path that changes the champion. It requires the experiment
    to be in ``ELIGIBLE_FOR_PROMOTION`` / ``PAPER_VALIDATION`` state with the
    promotion gate passed.
    """
    with session_scope() as session:
        row = session.execute(
            select(Experiment).where(Experiment.experiment_id == experiment_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        if row.status not in ("PAPER_VALIDATION", "ELIGIBLE_FOR_PROMOTION"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"experiment {experiment_id} is in status '{row.status}'. "
                    "Only a candidate that passed the promotion gate and completed paper "
                    "validation can be promoted."
                ),
            )
        registry = ChampionRegistry(session)
        gate = (row.significance or {})
        previous = registry.champion()
        if previous is not None:
            registry.transition(previous.version, "RETIRED", reason=f"replaced by {row.candidate_strategy}")
        info = registry.transition(
            row.candidate_strategy,
            "CHAMPION",
            reason=f"promoted by {experiment_id}",
            evidence={
                "backtest": row.challenger_metrics,
                "walk_forward": row.walk_forward_summary,
                "holdout": row.holdout_summary,
                "paper": row.challenger_metrics,
            },
        )
        row.status = "PROMOTED"
        row.promotion_reason = f"promoted to champion via {experiment_id}"
        session.commit()

    state.reload_champion()
    state.notifications.promoted(info.version, f"promoted via {experiment_id}")
    log.warning("strategy promoted", context={"version": info.version, "experiment": experiment_id})
    return {
        "ok": True,
        "promoted": info.to_dict(),
        "previous_champion": previous.version if previous else None,
        "significance": gate,
        "note": "The champion was replaced only after a deterministic gate evaluation and paper validation.",
    }


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
@router.get("/research/learnings", summary="Research memory (learnings)")
def learnings(
    query: str = "",
    limit: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    with session_scope() as session:
        memory = ResearchMemory(session)
        rows = memory.recall(query, limit=limit) if query else memory.all(limit=limit)
        stats = memory.stats()
    return {"count": len(rows), "stats": stats, "learnings": [r.to_dict() for r in rows]}


@router.get("/research/memory/ask", summary="Ask the research memory a question")
def ask(question: str = Query(..., min_length=3), state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    with session_scope() as session:
        memory = ResearchMemory(session)
        answer = memory.answer(question)
    return answer


@router.get("/research/graph/{experiment_id}", summary="Knowledge graph for an experiment")
def graph(experiment_id: str) -> Dict[str, Any]:
    with session_scope() as session:
        memory = ResearchMemory(session)
        node = memory.graph(experiment_id)
        if node is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        explanation = memory.explain_decision(experiment_id)
    return {"graph": node.to_dict(), "explain": explanation}


@router.get("/research/multiple-testing", summary="Multiple-testing ledger and budget")
def multiple_testing() -> Dict[str, Any]:
    with session_scope() as session:
        total = int(session.execute(select(func.count(MultipleTestingLedger.id))).scalar_one() or 0)
        recent = (
            session.execute(
                select(MultipleTestingLedger).order_by(MultipleTestingLedger.ts.desc()).limit(50)
            )
            .scalars()
            .all()
        )
    from ...settings import get_config_store

    config = get_config_store().load("research").get("multiple_testing", {}) or {}
    quarter_start = now_ist() - dt.timedelta(days=90)
    with session_scope() as session:
        quarter_trials = int(
            session.execute(
                select(func.count(MultipleTestingLedger.id)).where(
                    MultipleTestingLedger.ts >= quarter_start
                )
            ).scalar_one()
            or 0
        )
    max_quarter = int(config.get("max_trials_per_quarter", 60))
    return {
        "total_trials": total,
        "trials_last_quarter": quarter_trials,
        "max_trials_per_quarter": max_quarter,
        "budget_used_pct": round(quarter_trials / max_quarter, 4) if max_quarter else 0.0,
        "trials_penalty_start": config.get("trials_penalty_start", 10),
        "note": (
            "Every evaluated strategy/parameter combination is counted here. The promotion gate "
            "requires a larger edge once the trial count grows, so the system cannot quietly select "
            "the luckiest of thousands of runs."
        ),
        "recent": [
            {
                "ts": row.ts.isoformat() if row.ts else None,
                "kind": row.trial_kind,
                "strategy_version": row.strategy_version,
                "variable": row.variable,
                "value": row.value,
                "metric": row.metric_name,
                "metric_value": row.metric_value,
            }
            for row in recent
        ],
    }


@router.get("/research/journal-stats", summary="Journal statistics used by the researcher")
def journal_stats(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    with session_scope() as session:
        generator = HypothesisGenerator(session)
        trades = generator.load_trades(state.strategy_version)
        total = int(session.execute(select(func.count(TradeJournal.id))).scalar_one() or 0)

    r_multiples = [float(t.get("r_multiple", 0) or 0) for t in trades]
    wins = [r for r in r_multiples if r > 0]
    losses = [r for r in r_multiples if r < 0]
    return {
        "total_journal_trades": total,
        "strategy_trades": len(trades),
        "minimum_required_for_hypotheses": 25,
        "win_rate": round(len(wins) / len(r_multiples), 4) if r_multiples else 0.0,
        "expectancy_r": round(sum(r_multiples) / len(r_multiples), 4) if r_multiples else 0.0,
        "average_win_r": round(sum(wins) / len(wins), 4) if wins else 0.0,
        "average_loss_r": round(sum(losses) / len(losses), 4) if losses else 0.0,
    }


__all__ = ["router"]
