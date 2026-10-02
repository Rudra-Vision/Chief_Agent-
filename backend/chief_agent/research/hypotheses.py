"""Hypothesis generation from the trade journal.

A hypothesis is a falsifiable statement with evidence, a sample size, a
confidence estimate, an affected-trade list, the single variable it proposes to
change, the expected mechanism and the potential downside.

Two sources:

1. **Deterministic analysis** (always available, no LLM required). It slices the
   journal on the variables listed in ``config/research.yaml ->
   hypothesis_templates`` and reports where expectancy is materially different
   from the baseline.
2. **Optional LLM** proposals, which are only *added to the queue* - they are
   still validated by the same deterministic engine before any experiment runs.

The rule "ONE primary variable per experiment" is enforced here: a hypothesis
that would change more than one variable is rejected at creation time.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data.schema import Hypothesis as HypothesisRow
from ..data.schema import Learning, MultipleTestingLedger, TradeJournal
from ..logging_setup import get_logger
from ..settings import get_config_store
from ..strategies.base import deep_get
from ..timeutil import now_ist

log = get_logger(__name__, component="hypotheses")

MIN_SAMPLE_FOR_HYPOTHESIS = 25


@dataclass
class Evidence:
    sample_size: int = 0
    baseline_expectancy_r: float = 0.0
    segment_expectancy_r: float = 0.0
    delta_expectancy_r: float = 0.0
    segment_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    segment_trades: int = 0
    baseline_trades: int = 0
    p_value: Optional[float] = None
    confidence: float = 0.0
    condition: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "baseline_expectancy_r": round(self.baseline_expectancy_r, 4),
            "segment_expectancy_r": round(self.segment_expectancy_r, 4),
            "delta_expectancy_r": round(self.delta_expectancy_r, 4),
            "segment_win_rate": round(self.segment_win_rate, 4),
            "baseline_win_rate": round(self.baseline_win_rate, 4),
            "segment_trades": self.segment_trades,
            "baseline_trades": self.baseline_trades,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "confidence": round(self.confidence, 4),
            "condition": self.condition,
            "detail": self.detail,
        }


@dataclass
class HypothesisProposal:
    statement: str
    variable: str
    old_value: Any
    new_value: Any
    evidence: Evidence
    rationale: str = ""
    expected_mechanism: str = ""
    potential_downside: str = ""
    source: str = "deterministic"
    strategy_version: str = ""
    affected_trade_ids: List[str] = field(default_factory=list)
    priority: float = 0.0
    template_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "variable": self.variable,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "evidence": self.evidence.to_dict(),
            "rationale": self.rationale,
            "expected_mechanism": self.expected_mechanism,
            "potential_downside": self.potential_downside,
            "source": self.source,
            "strategy_version": self.strategy_version,
            "affected_trade_count": len(self.affected_trade_ids),
            "affected_trade_ids": self.affected_trade_ids[:50],
            "priority": round(self.priority, 4),
            "template_id": self.template_id,
        }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _welch_p_value(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided Welch t-test p-value without a SciPy dependency at import time."""
    if len(a) < 5 or len(b) < 5:
        return None
    try:
        from scipy import stats  # type: ignore

        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        pass
    # Normal approximation fallback
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    denominator = math.sqrt(var_a / len(a) + var_b / len(b))
    if denominator <= 0:
        return None
    z = abs(mean_a - mean_b) / denominator
    return float(math.erfc(z / math.sqrt(2)))


def segments_from_journal(
    trades: Sequence[Mapping[str, Any]],
    extractor: Any,
) -> Dict[str, List[Mapping[str, Any]]]:
    """Group trades by an arbitrary key extracted from each row."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for trade in trades:
        try:
            key = extractor(trade)
        except Exception:
            continue
        if key is None:
            continue
        groups.setdefault(str(key), []).append(trade)
    return groups


# --------------------------------------------------------------------------- #
# Deterministic generator
# --------------------------------------------------------------------------- #
class HypothesisGenerator:
    """Turns journal statistics into concrete, testable proposals."""

    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None) -> None:
        self.session = session
        self.config = config or get_config_store().load("research")
        self.templates = self.config.get("hypothesis_templates", []) or []
        self.min_sample = max(
            MIN_SAMPLE_FOR_HYPOTHESIS,
            int(deep_get(self.config, "cadence.min_trades_between_experiments", 20)),
        )

    # ---------------------------------------------------------------- journal
    def load_trades(self, strategy_version: Optional[str] = None, limit: int = 20_000) -> List[Dict[str, Any]]:
        stmt = select(TradeJournal).where(TradeJournal.mode != "BACKTEST")
        if strategy_version:
            stmt = stmt.where(TradeJournal.strategy_version == strategy_version)
        rows = self.session.execute(stmt.order_by(TradeJournal.exit_ts.desc()).limit(limit)).scalars().all()
        return [
            {
                "trade_id": row.trade_id,
                "strategy_version": row.strategy_version,
                "symbol": row.symbol,
                "sector": row.sector,
                "direction": row.direction,
                "net_pnl": row.net_pnl,
                "r_multiple": row.r_multiple,
                "exit_reason": row.exit_reason,
                "regime_at_entry": row.regime_at_entry,
                "entry_ts": row.entry_ts,
                "exit_ts": row.exit_ts,
                "holding_minutes": row.holding_minutes,
                "sector_rank_at_entry": row.sector_rank_at_entry,
                "vix_at_entry": row.vix_at_entry,
                "features": row.features or {},
                "rvol": (row.features or {}).get("rvol"),
                "atr_daily_pct": (row.features or {}).get("atr_daily_pct"),
                "minutes_from_open": (row.features or {}).get("minutes_from_open"),
                "or_width_atr_fraction": (row.features or {}).get("or_width_atr_fraction"),
            }
            for row in rows
        ]

    # ------------------------------------------------------------- generation
    def generate(
        self,
        strategy_version: str,
        champion_config: Dict[str, Any],
        *,
        max_proposals: int = 5,
    ) -> List[HypothesisProposal]:
        trades = self.load_trades(strategy_version)
        if len(trades) < self.min_sample:
            log.info(
                "not enough trades to generate hypotheses",
                context={"have": len(trades), "need": self.min_sample},
            )
            return []

        r_multiples = [float(t.get("r_multiple", 0) or 0) for t in trades]
        baseline_expectancy = float(np.mean(r_multiples)) if r_multiples else 0.0
        baseline_win_rate = (
            sum(1 for r in r_multiples if r > 0) / len(r_multiples) if r_multiples else 0.0
        )

        proposals: List[HypothesisProposal] = []
        for template in self.templates:
            try:
                proposal = self._evaluate_template(
                    template,
                    trades,
                    champion_config,
                    strategy_version,
                    baseline_expectancy,
                    baseline_win_rate,
                )
            except Exception as exc:
                log.warning("hypothesis template failed", context={"template": template.get("id"), "error": str(exc)})
                continue
            if proposal is not None:
                proposals.append(proposal)

        # Lower-volume, higher-signal analyses that are not template driven.
        proposals.extend(self._structural_analyses(trades, champion_config, strategy_version, baseline_expectancy, baseline_win_rate))

        proposals.sort(key=lambda p: p.priority, reverse=True)
        unique: List[HypothesisProposal] = []
        seen_variables: set = set()
        for proposal in proposals:
            if proposal.variable in seen_variables:
                continue
            seen_variables.add(proposal.variable)
            unique.append(proposal)
        return unique[:max_proposals]

    def _evaluate_template(
        self,
        template: Mapping[str, Any],
        trades: Sequence[Mapping[str, Any]],
        champion_config: Dict[str, Any],
        strategy_version: str,
        baseline_expectancy: float,
        baseline_win_rate: float,
    ) -> Optional[HypothesisProposal]:
        variable = str(template.get("variable"))
        search_space: Sequence[Any] = template.get("search") or []
        if not variable or not search_space:
            return None

        old_value = deep_get(champion_config, variable, None)
        if old_value is None and not _exists(champion_config, variable):
            return None

        extractor = _extractor_for(variable)
        if extractor is None:
            return None

        groups = segments_from_journal(trades, extractor)
        if not groups:
            return None

        # Find the candidate value whose segment is most clearly different from
        # the rest - either much better (raise the bar) or much worse (tighten).
        best: Optional[Tuple[float, Evidence, Any, str]] = None
        for candidate in search_space:
            if candidate == old_value:
                continue
            # The segment "would be excluded by the candidate value"
            keep, drop = [], []
            for trade in trades:
                try:
                    value = extractor(trade)
                except Exception:
                    continue
                if value is None:
                    continue
                if _would_pass(variable, value, candidate):
                    keep.append(trade)
                else:
                    drop.append(trade)
            if len(drop) < self.min_sample // 2 or len(keep) < self.min_sample // 2:
                continue
            drop_r = [float(t.get("r_multiple", 0) or 0) for t in drop]
            keep_r = [float(t.get("r_multiple", 0) or 0) for t in keep]
            drop_expectancy = float(np.mean(drop_r))
            keep_expectancy = float(np.mean(keep_r))
            p_value = _welch_p_value(drop_r, keep_r)

            # We only care about the "excluded trades were worse" case.
            delta = keep_expectancy - drop_expectancy
            if delta <= 0:
                continue
            confidence = _confidence(len(drop), delta, p_value)
            if confidence < 0.55:
                continue
            evidence = Evidence(
                sample_size=len(drop),
                baseline_expectancy_r=baseline_expectancy,
                segment_expectancy_r=drop_expectancy,
                delta_expectancy_r=delta,
                segment_win_rate=sum(1 for r in drop_r if r > 0) / len(drop_r),
                baseline_win_rate=baseline_win_rate,
                segment_trades=len(drop),
                baseline_trades=len(keep),
                p_value=p_value,
                confidence=confidence,
                condition=f"{variable} {'passes' if False else 'fails'} {candidate}",
                detail=(
                    f"{len(drop)} trades would be excluded; their expectancy is "
                    f"{drop_expectancy:+.3f}R versus {keep_expectancy:+.3f}R for the rest"
                ),
            )
            priority = abs(delta) * math.log1p(len(drop)) * confidence
            if best is None or priority > best[0]:
                best = (priority, evidence, candidate, str(template.get("description") or variable))

        if best is None:
            return None
        priority, evidence, candidate, description = best
        return HypothesisProposal(
            statement=(
                f"Changing {description.lower()} from {old_value!r} to {candidate!r} may improve expectancy. "
                f"{evidence.detail}."
            ),
            variable=variable,
            old_value=old_value,
            new_value=candidate,
            evidence=evidence,
            rationale=(
                f"Trades that would be filtered out by the proposed value have an expectancy of "
                f"{evidence.segment_expectancy_r:+.3f}R vs {evidence.baseline_expectancy_r:+.3f}R overall "
                f"(p={evidence.p_value if evidence.p_value is not None else float('nan'):.3f})."
            ),
            expected_mechanism=_mechanism_for(variable, old_value, candidate),
            potential_downside=_downside_for(variable, old_value, candidate, evidence),
            source="deterministic",
            strategy_version=strategy_version,
            priority=priority,
            template_id=str(template.get("id", "")),
        )

    def _structural_analyses(
        self,
        trades: Sequence[Mapping[str, Any]],
        champion_config: Dict[str, Any],
        strategy_version: str,
        baseline_expectancy: float,
        baseline_win_rate: float,
    ) -> List[HypothesisProposal]:
        """Journal-driven checks that do not map onto a single config knob."""
        out: List[HypothesisProposal] = []

        # A) time-of-day: are early entries worse?
        with_minutes = [t for t in trades if t.get("minutes_from_open") is not None]
        if len(with_minutes) >= self.min_sample:
            early = [t for t in with_minutes if float(t["minutes_from_open"]) <= 30]
            rest = [t for t in with_minutes if float(t["minutes_from_open"]) > 30]
            if len(early) >= 12 and len(rest) >= 12:
                early_r = [float(t.get("r_multiple", 0) or 0) for t in early]
                rest_r = [float(t.get("r_multiple", 0) or 0) for t in rest]
                delta = float(np.mean(rest_r)) - float(np.mean(early_r))
                if delta > 0:
                    p_value = _welch_p_value(early_r, rest_r)
                    confidence = _confidence(len(early), delta, p_value)
                    if confidence >= 0.5:
                        current = deep_get(champion_config, "long.entry_window_start_minutes", 15)
                        proposed = 30 if int(current or 0) < 30 else max(30, int(current) + 15)
                        out.append(
                            HypothesisProposal(
                                statement=(
                                    f"Entries in the first 30 minutes after the open have a lower expectancy "
                                    f"({float(np.mean(early_r)):+.3f}R) than later entries ({float(np.mean(rest_r)):+.3f}R). "
                                    f"Delaying the entry window to {proposed} minutes may help."
                                ),
                                variable="long.entry_window_start_minutes",
                                old_value=current,
                                new_value=proposed,
                                evidence=Evidence(
                                    sample_size=len(early),
                                    baseline_expectancy_r=baseline_expectancy,
                                    segment_expectancy_r=float(np.mean(early_r)),
                                    delta_expectancy_r=delta,
                                    segment_win_rate=sum(1 for r in early_r if r > 0) / len(early_r),
                                    baseline_win_rate=baseline_win_rate,
                                    segment_trades=len(early),
                                    baseline_trades=len(rest),
                                    p_value=p_value,
                                    confidence=confidence,
                                    condition="entry within 30 minutes of the open",
                                ),
                                rationale="Very early breakouts more often fail - the opening range has not settled.",
                                expected_mechanism="Fewer false breakouts from pre-open order imbalances that fade.",
                                potential_downside="Fewer trades, and the best trend days can break out early and never look back.",
                                strategy_version=strategy_version,
                                priority=abs(delta) * math.log1p(len(early)) * confidence,
                                template_id="time_of_day",
                            )
                        )

        # B) exit efficiency: is the time stop cutting winners?
        time_stops = [t for t in trades if t.get("exit_reason") == "TIME_STOP"]
        targets = [t for t in trades if str(t.get("exit_reason", "")).startswith("TARGET")]
        if len(time_stops) >= 12 and len(targets) >= 12:
            ts_r = [float(t.get("r_multiple", 0) or 0) for t in time_stops]
            if float(np.mean(ts_r)) < 0:
                current = deep_get(champion_config, "exits.time_stop_minutes", 240)
                proposed = int(current or 240) + 60
                delta = -float(np.mean(ts_r))
                out.append(
                    HypothesisProposal(
                        statement=(
                            f"Trades closed by the {current}-minute time stop average {float(np.mean(ts_r)):+.3f}R. "
                            f"Allowing more time ({proposed} minutes) may reduce premature exits."
                        ),
                        variable="exits.time_stop_minutes",
                        old_value=current,
                        new_value=proposed,
                        evidence=Evidence(
                            sample_size=len(time_stops),
                            baseline_expectancy_r=baseline_expectancy,
                            segment_expectancy_r=float(np.mean(ts_r)),
                            delta_expectancy_r=delta,
                            segment_win_rate=sum(1 for r in ts_r if r > 0) / len(ts_r),
                            baseline_win_rate=baseline_win_rate,
                            segment_trades=len(time_stops),
                            baseline_trades=len(targets),
                            confidence=min(0.85, 0.4 + math.log1p(len(time_stops)) / 12),
                            condition=f"exit reason = TIME_STOP at {current} minutes",
                        ),
                        rationale="Time-stopped trades average a loss, suggesting the clock is cutting positions early.",
                        expected_mechanism="Positions get more time to reach their first target.",
                        potential_downside="Capital stays tied up longer; a slow loser can become a larger loser.",
                        strategy_version=strategy_version,
                        priority=abs(delta) * math.log1p(len(time_stops)) * 0.6,
                        template_id="time_stop_efficiency",
                    )
                )

        # C) regime dependence.
        regimes = segments_from_journal(trades, lambda t: t.get("regime_at_entry"))
        if len(regimes) >= 3:
            worst = min(regimes.items(), key=lambda kv: float(np.mean([float(t.get("r_multiple", 0) or 0) for t in kv[1]])))
            regime_name, rows = worst
            if len(rows) >= 15:
                regime_r = [float(t.get("r_multiple", 0) or 0) for t in rows]
                mean_r = float(np.mean(regime_r))
                if mean_r < 0:
                    current = deep_get(champion_config, "long.allowed_regimes", [])
                    if regime_name in (current or []):
                        proposed = [r for r in current if r != regime_name]
                        out.append(
                            HypothesisProposal(
                                statement=(
                                    f"Long trades in a {regime_name} regime average {mean_r:+.3f}R "
                                    f"across {len(rows)} trades. Removing {regime_name} from the allowed regimes "
                                    f"may improve expectancy."
                                ),
                                variable="long.allowed_regimes",
                                old_value=current,
                                new_value=proposed,
                                evidence=Evidence(
                                    sample_size=len(rows),
                                    baseline_expectancy_r=baseline_expectancy,
                                    segment_expectancy_r=mean_r,
                                    delta_expectancy_r=baseline_expectancy - mean_r,
                                    segment_win_rate=sum(1 for r in regime_r if r > 0) / len(regime_r),
                                    baseline_win_rate=baseline_win_rate,
                                    segment_trades=len(rows),
                                    baseline_trades=len(trades) - len(rows),
                                    confidence=min(0.9, 0.4 + math.log1p(len(rows)) / 10),
                                    condition=f"regime_at_entry = {regime_name}",
                                ),
                                rationale="This regime has negative expectancy for this strategy family.",
                                expected_mechanism="Avoid a systematically unfavourable market state.",
                                potential_downside="Much lower trade count, and regimes can be misclassified in real time.",
                                strategy_version=strategy_version,
                                priority=abs(mean_r) * math.log1p(len(rows)) * 0.7,
                                template_id="regime_filter",
                            )
                        )
        return out

    # -------------------------------------------------------------- persistence
    def persist(self, proposal: HypothesisProposal, strategy_version: str) -> str:
        """Store a proposal and return its public hypothesis id (``H-0048``)."""
        existing = self.session.execute(select(HypothesisRow)).scalars().all()
        next_number = len(existing) + 1
        hypothesis_id = f"H-{next_number:04d}"
        while self.session.execute(
            select(HypothesisRow).where(HypothesisRow.hypothesis_id == hypothesis_id)
        ).scalar_one_or_none() is not None:
            next_number += 1
            hypothesis_id = f"H-{next_number:04d}"

        row = HypothesisRow(
            hypothesis_id=hypothesis_id,
            strategy_version=strategy_version,
            statement=proposal.statement,
            variable=proposal.variable,
            old_value=str(proposal.old_value),
            new_value=str(proposal.new_value),
            rationale=proposal.rationale,
            expected_mechanism=proposal.expected_mechanism,
            potential_downside=proposal.potential_downside,
            evidence=proposal.evidence.to_dict(),
            sample_size=proposal.evidence.sample_size,
            confidence=proposal.evidence.confidence,
            affected_trade_ids=proposal.affected_trade_ids[:200],
            source=proposal.source,
            status="PROPOSED",
        )
        self.session.add(row)
        self.session.flush()
        log.info(
            "hypothesis recorded",
            context={
                "hypothesis_id": hypothesis_id,
                "variable": proposal.variable,
                "old": proposal.old_value,
                "new": proposal.new_value,
                "sample": proposal.evidence.sample_size,
                "confidence": round(proposal.evidence.confidence, 3),
            },
        )
        return hypothesis_id


# --------------------------------------------------------------------------- #
# Helpers mapping config variables -> journal extractors
# --------------------------------------------------------------------------- #
def _extractor_for(variable: str):
    mapping = {
        "long.min_rvol": lambda t: t.get("rvol"),
        "short.min_rvol": lambda t: t.get("rvol"),
        "long.entry_window_start_minutes": lambda t: t.get("minutes_from_open"),
        "short.entry_window_start_minutes": lambda t: t.get("minutes_from_open"),
        "long.entry_window_end_minutes": lambda t: t.get("minutes_from_open"),
        "long.max_spread_pct": lambda t: (t.get("features") or {}).get("spread_pct"),
        "opening_range.min_or_width_atr_fraction": lambda t: t.get("or_width_atr_fraction"),
        "opening_range.max_or_width_atr_fraction": lambda t: t.get("or_width_atr_fraction"),
        "exits.time_stop_minutes": lambda t: t.get("holding_minutes"),
        "long.require_sector_rank_at_least": lambda t: t.get("sector_rank_at_entry"),
        "no_trade.vix_block_above": lambda t: t.get("vix_at_entry"),
        "targets.r_multiple_t1": lambda t: t.get("mfe_r") or (t.get("features") or {}).get("mfe_r"),
        "stops.atr_multiplier": lambda t: t.get("atr_daily_pct"),
        "long.min_atr_pct": lambda t: t.get("atr_daily_pct"),
        "long.max_atr_pct": lambda t: t.get("atr_daily_pct"),
    }
    return mapping.get(variable)


def _would_pass(variable: str, value: Any, candidate: Any) -> bool:
    """Would a trade with this feature value still be allowed under the candidate?"""
    try:
        if variable.endswith(("min_rvol", "min_atr_pct", "require_sector_rank_at_least")):
            if variable.endswith("require_sector_rank_at_least"):
                return float(value) <= float(candidate)
            return float(value) >= float(candidate)
        if variable.endswith(("max_atr_pct", "max_spread_pct", "vix_block_above")):
            return float(value) <= float(candidate)
        if variable.endswith("entry_window_start_minutes"):
            return float(value) >= float(candidate)
        if variable.endswith("entry_window_end_minutes"):
            return float(value) <= float(candidate)
        if variable.endswith(("min_or_width_atr_fraction", "max_or_width_atr_fraction")):
            return float(value) >= float(candidate) if "min" in variable else float(value) <= float(candidate)
        if variable.endswith("time_stop_minutes"):
            return float(value) <= float(candidate)
        if variable.endswith("r_multiple_t1") or variable.endswith("atr_multiplier"):
            return float(value) <= float(candidate)
    except (TypeError, ValueError):
        return False
    return True


def _confidence(sample_size: int, delta: float, p_value: Optional[float]) -> float:
    """Blend statistical significance with sample size into a 0-1 confidence."""
    size_factor = 1.0 - math.exp(-sample_size / 60.0)
    effect_factor = min(1.0, abs(delta) / 0.25)
    if p_value is None:
        significance = 0.5
    else:
        significance = max(0.0, 1.0 - p_value) ** 0.5
    return float(min(0.97, 0.20 + 0.45 * significance + 0.20 * size_factor + 0.15 * effect_factor))


def _mechanism_for(variable: str, old: Any, new: Any) -> str:
    if "rvol" in variable:
        return (
            "Higher relative volume implies genuine participation; low-volume breakouts are usually "
            "liquidity gaps that revert once the initial flow is absorbed."
        )
    if "entry_window_start" in variable:
        return "Waiting for the opening auction noise to settle reduces false breakouts."
    if "entry_window_end" in variable:
        return "Late breakouts have less time to reach the target before the close."
    if "retest" in variable:
        return "A retest filters breakouts that were immediately rejected."
    if "vix" in variable:
        return "Very high volatility regimes have unstable spreads and wider slippage."
    if "spread" in variable:
        return "Wider spreads mean a larger immediate cost and worse fills."
    if "sector" in variable:
        return "Sector-confirmed moves have a supportive flow behind them."
    if "allowed_regimes" in variable:
        return "Removing a systematically unfavourable regime removes a source of negative expectancy."
    if "time_stop" in variable:
        return "Giving a trade more time can allow a slower but valid thesis to work."
    if "atr" in variable:
        return "Volatility scaling keeps risk per trade comparable across names."
    return "The measured segment differs materially in expectancy from the rest of the sample."


def _downside_for(variable: str, old: Any, new: Any, evidence: Evidence) -> str:
    base = f"Reduces the trade count by roughly {evidence.segment_trades} trades in this sample"
    if "rvol" in variable:
        return base + ", and on genuine news days the first breakout can run without ever showing high volume."
    if "entry_window" in variable:
        return base + ", and the strongest trend days are often the ones that break out immediately."
    if "allowed_regimes" in variable:
        return base + ", and real-time regime classification may differ from the ex-post label."
    if "time_stop" in variable:
        return "Positions occupy capital for longer, and a slow loser can become a larger loser."
    return base + "; the effect may not survive out of sample."


def _exists(config: Dict[str, Any], path: str) -> bool:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


__all__ = [
    "HypothesisGenerator",
    "HypothesisProposal",
    "Evidence",
    "segments_from_journal",
    "MIN_SAMPLE_FOR_HYPOTHESIS",
]
