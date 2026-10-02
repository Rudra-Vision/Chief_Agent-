"""Persistent structured research memory.

The research system must NOT rely on LLM conversation history. Every finding is
stored as a row with its evidence, the associated hypothesis and experiment, the
market conditions it was measured under, the decision taken and the result.

The knowledge graph the brief asks for is implicit in the foreign keys:

    STRATEGY -> EXPERIMENT -> MARKET CONDITION -> TRADE SET -> RESULT -> LESSON

:meth:`ResearchMemory.recall` answers questions like "what have we learned about
gap-up mornings?" by matching tag and condition keys, and
:meth:`ResearchMemory.explain_decision` answers "why was strategy v1.8 rejected?"
from the stored experiment row.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from ..data.schema import Experiment, Hypothesis, Learning, MultipleTestingLedger, TradeJournal
from ..logging_setup import get_logger
from ..settings import get_config_store
from ..timeutil import now_ist

log = get_logger(__name__, component="memory")


@dataclass
class LearningRecord:
    learning_id: str
    finding: str
    evidence: Dict[str, Any]
    sample_size: int
    confidence: float
    strategy_version: str
    hypothesis_id: Optional[str] = None
    experiment_id: Optional[str] = None
    market_conditions: Optional[Dict[str, Any]] = None
    decision: Optional[str] = None
    result: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_id": self.learning_id,
            "finding": self.finding,
            "evidence": self.evidence,
            "sample_size": self.sample_size,
            "confidence": round(self.confidence, 4),
            "strategy_version": self.strategy_version,
            "hypothesis_id": self.hypothesis_id,
            "experiment_id": self.experiment_id,
            "market_conditions": self.market_conditions,
            "decision": self.decision,
            "result": self.result,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else self.created_at,
        }


@dataclass
class KnowledgeNode:
    """A node in the research knowledge graph."""

    kind: str
    identifier: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["KnowledgeNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.identifier,
            "label": self.label,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


class ResearchMemory:
    """Structured, queryable memory of everything the system has learned."""

    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None) -> None:
        self.session = session
        self.config = config or (get_config_store().load("research").get("memory") or {})
        self.retrieval_top_k = int(self.config.get("retrieval_top_k", 12))

    # ------------------------------------------------------------------ write
    def record(
        self,
        *,
        finding: str,
        evidence: Dict[str, Any],
        sample_size: int = 0,
        confidence: float = 0.0,
        strategy_version: str = "",
        hypothesis_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        decision: Optional[str] = None,
        result: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> str:
        existing = self.session.execute(select(func.count(Learning.id))).scalar_one()
        number = int(existing or 0) + 1
        learning_id = f"LEARNING-{number:05d}"
        while self.session.execute(
            select(Learning).where(Learning.learning_id == learning_id)
        ).scalar_one_or_none() is not None:
            number += 1
            learning_id = f"LEARNING-{number:05d}"

        row = Learning(
            learning_id=learning_id,
            finding=finding,
            evidence=evidence,
            sample_size=int(sample_size),
            confidence=float(confidence),
            strategy_version=strategy_version,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            market_conditions=market_conditions,
            decision=decision,
            result=result,
            tags=list(tags or []),
        )
        self.session.add(row)
        self.session.flush()
        log.info(
            "learning recorded",
            context={"learning_id": learning_id, "decision": decision, "result": result, "tags": tags},
        )
        return learning_id

    # ---------------------------------------------------------------- read
    def recall(
        self,
        query: str = "",
        *,
        tags: Optional[Sequence[str]] = None,
        strategy_version: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[LearningRecord]:
        """Retrieve relevant learnings. Text search is a simple case-insensitive
        contains match over the finding, tags and market conditions - deliberately
        deterministic and inspectable rather than an embedding black box."""
        limit = limit or self.retrieval_top_k
        stmt: Select = select(Learning)
        if strategy_version:
            stmt = stmt.where(Learning.strategy_version == strategy_version)
        if query:
            pattern = f"%{query.lower()}%"
            stmt = stmt.where(
                or_(
                    func.lower(Learning.finding).like(pattern),
                    func.lower(func.coalesce(Learning.decision, "")).like(pattern),
                    func.lower(func.coalesce(Learning.result, "")).like(pattern),
                    func.lower(func.coalesce(Learning.tags, "")).like(pattern),
                )
            )
        rows = self.session.execute(stmt.order_by(Learning.created_at.desc()).limit(limit * 3)).scalars().all()

        if tags:
            wanted = {t.lower() for t in tags}
            rows = [row for row in rows if wanted & {str(t).lower() for t in (row.tags or [])}]
        return [self._to_record(row) for row in rows[:limit]]

    @staticmethod
    def _to_record(row: Learning) -> LearningRecord:
        return LearningRecord(
            learning_id=row.learning_id,
            finding=row.finding,
            evidence=row.evidence or {},
            sample_size=row.sample_size,
            confidence=row.confidence,
            strategy_version=row.strategy_version,
            hypothesis_id=row.hypothesis_id,
            experiment_id=row.experiment_id,
            market_conditions=row.market_conditions,
            decision=row.decision,
            result=row.result,
            tags=list(row.tags or []),
            created_at=row.created_at,
        )

    def all(self, limit: int = 200) -> List[LearningRecord]:
        rows = self.session.execute(
            select(Learning).order_by(Learning.created_at.desc()).limit(limit)
        ).scalars().all()
        return [self._to_record(row) for row in rows]

    def stats(self) -> Dict[str, Any]:
        total = int(self.session.execute(select(func.count(Learning.id))).scalar_one() or 0)
        accepted = int(
            self.session.execute(
                select(func.count(Learning.id)).where(Learning.result.in_(["ACCEPTED", "PROMOTED"]))
            ).scalar_one()
            or 0
        )
        rejected = int(
            self.session.execute(select(func.count(Learning.id)).where(Learning.result == "REJECTED")).scalar_one()
            or 0
        )
        return {
            "total_learnings": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": round(accepted / total, 4) if total else 0.0,
            "trials_recorded": int(
                self.session.execute(select(func.count(MultipleTestingLedger.id))).scalar_one() or 0
            ),
        }

    # ---------------------------------------------------------- knowledge graph
    def graph(self, experiment_id: str) -> Optional[KnowledgeNode]:
        experiment = self.session.execute(
            select(Experiment).where(Experiment.experiment_id == experiment_id)
        ).scalar_one_or_none()
        if experiment is None:
            return None

        hypothesis = None
        if experiment.hypothesis_id:
            hypothesis = self.session.execute(
                select(Hypothesis).where(Hypothesis.hypothesis_id == experiment.hypothesis_id)
            ).scalar_one_or_none()

        learnings = self.session.execute(
            select(Learning).where(Learning.experiment_id == experiment_id)
        ).scalars().all()

        result_node = KnowledgeNode(
            kind="result",
            identifier=experiment_id,
            label=experiment.status,
            attributes={
                "expectancy_r": experiment.expectancy_r,
                "profit_factor": experiment.profit_factor,
                "sortino": experiment.sortino,
                "max_drawdown_pct": experiment.max_drawdown_pct,
                "trade_count": experiment.trade_count,
                "rejection_reason": experiment.rejection_reason,
                "promotion_reason": experiment.promotion_reason,
            },
        )
        condition_node = KnowledgeNode(
            kind="market_condition",
            identifier=f"{experiment_id}-conditions",
            label="Conditions measured over",
            attributes={
                "period": f"{experiment.training_period_start}..{experiment.training_period_end}",
                "holdout_period": f"{experiment.holdout_period_start}..{experiment.holdout_period_end}",
            },
        )
        lesson_children = [
            KnowledgeNode(
                kind="lesson",
                identifier=row.learning_id,
                label=row.finding[:160],
                attributes={"decision": row.decision, "result": row.result, "confidence": row.confidence},
            )
            for row in learnings
        ]

        experiment_node = KnowledgeNode(
            kind="experiment",
            identifier=experiment_id,
            label=experiment.hypothesis[:160],
            attributes={
                "variable": experiment.variable_changed,
                "old": experiment.old_value,
                "new": experiment.new_value,
                "status": experiment.status,
            },
            children=[condition_node, result_node] + lesson_children,
        )
        hypothesis_node = KnowledgeNode(
            kind="hypothesis",
            identifier=experiment.hypothesis_id or "unlinked",
            label=(hypothesis.statement[:160] if hypothesis else experiment.hypothesis[:160]),
            attributes={
                "evidence": (hypothesis.evidence if hypothesis else {}),
                "confidence": hypothesis.confidence if hypothesis else None,
            },
            children=[experiment_node],
        )
        return KnowledgeNode(
            kind="strategy",
            identifier=experiment.parent_strategy,
            label=f"Strategy {experiment.parent_strategy}",
            attributes={"candidate": experiment.candidate_strategy},
            children=[hypothesis_node],
        )

    def explain_decision(self, version: str) -> Dict[str, Any]:
        """Answer 'why was strategy X rejected / promoted?' from stored evidence."""
        experiments = self.session.execute(
            select(Experiment).where(
                or_(Experiment.candidate_strategy == version, Experiment.parent_strategy == version)
            )
        ).scalars().all()
        learnings = self.session.execute(
            select(Learning).where(Learning.strategy_version == version)
        ).scalars().all()

        if not experiments and not learnings:
            return {"version": version, "found": False, "message": "no recorded research for this version"}

        return {
            "version": version,
            "found": True,
            "experiments": [
                {
                    "experiment_id": row.experiment_id,
                    "variable": row.variable_changed,
                    "change": f"{row.old_value} -> {row.new_value}",
                    "status": row.status,
                    "expectancy_r": row.expectancy_r,
                    "profit_factor": row.profit_factor,
                    "holdout": row.holdout_summary,
                    "walk_forward": (row.walk_forward_summary or {}).get("profitable_fraction"),
                    "rejection_reason": row.rejection_reason,
                    "promotion_reason": row.promotion_reason,
                }
                for row in experiments
            ],
            "learnings": [self._to_record(row).to_dict() for row in learnings],
        }

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a natural-language question using stored structured memory only.

        This is deliberately a *retrieval* function, not a generative one: it can
        only return evidence that exists in the database. If nothing matches it
        says so rather than inventing an answer.
        """
        lowered = question.lower()
        results = self.recall(question, limit=8)

        # Condition-focused questions
        conditions = []
        for keyword in ("gap-up", "gap up", "gap-down", "gap down", "high volatility", "trend", "range", "vix"):
            if keyword in lowered:
                conditions.append(keyword.replace(" ", "_").replace("-", "_"))
        condition_matches = self.recall(" ".join(conditions), limit=8) if conditions else []

        experiments = self.session.execute(
            select(Experiment).order_by(Experiment.created_at.desc()).limit(20)
        ).scalars().all()
        variable_like = [
            {
                "experiment_id": row.experiment_id,
                "variable": row.variable_changed,
                "change": f"{row.old_value} -> {row.new_value}",
                "status": row.status,
                "result": row.rejection_reason or row.promotion_reason,
            }
            for row in experiments
            if row.variable_changed and row.variable_changed.lower() in lowered
        ]

        return {
            "question": question,
            "found": bool(results or condition_matches or variable_like),
            "matching_learnings": [r.to_dict() for r in results],
            "condition_learnings": [r.to_dict() for r in condition_matches],
            "matching_experiments": variable_like,
            "note": (
                "Answers are drawn only from stored, dated research records. "
                "If the system has not measured something, it reports nothing rather than guessing."
            ),
        }


__all__ = ["ResearchMemory", "LearningRecord", "KnowledgeNode"]
