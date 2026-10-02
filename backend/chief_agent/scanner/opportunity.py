"""Opportunity scanner and ranking.

Continuously scores every eligible instrument and produces the Top-N LONG and
Top-N SHORT opportunity tables the dashboard shows.

The score is **deterministic** (see
:meth:`ORBVWAPStrategy._score_components`); an LLM never generates or edits it.
This module is the orchestration around that score: universe selection,
liquidity gates, de-duplication, ranking and the explanation payload.

Every opportunity carries a plain-English explanation built from the gates that
actually passed, so the owner can see *why* a name is on the list.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..broker.upstox_market_data import Candle
from ..indicators.features import BarFeatures
from ..logging_setup import get_logger
from ..sectors.engine import SectorEngine, SectorSnapshot
from ..strategies.base import StrategyContext, TradeCandidate
from ..strategies.orb_vwap import ORBVWAPStrategy
from ..timeutil import IST, now_ist

log = get_logger(__name__, component="scanner")


@dataclass
class Opportunity:
    """One ranked candidate, ready for the dashboard."""

    rank: int
    instrument_key: str
    symbol: str
    direction: str
    score: float
    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    risk_reward: float
    risk_per_share: float
    sector: Optional[str] = None
    sector_rank: Optional[int] = None
    regime: Optional[str] = None
    vwap: Optional[float] = None
    vwap_state: str = ""
    rvol: Optional[float] = None
    atr_daily_pct: Optional[float] = None
    spread_pct: Optional[float] = None
    opportunity_score_components: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    plain_english: str = ""
    advanced_detail: Dict[str, Any] = field(default_factory=dict)
    strategy_version: str = ""
    generated_at: Any = None
    quantity_hint: int = 0
    risk_amount_hint: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "direction": self.direction,
            "score": round(self.score, 2),
            "entry_price": round(self.entry_price, 2),
            "stop_price": round(self.stop_price, 2),
            "target_1": round(self.target_1, 2),
            "target_2": round(self.target_2, 2),
            "risk_reward": round(self.risk_reward, 2),
            "risk_per_share": round(self.risk_per_share, 2),
            "sector": self.sector,
            "sector_rank": self.sector_rank,
            "regime": self.regime,
            "vwap": round(self.vwap, 2) if self.vwap else None,
            "vwap_state": self.vwap_state,
            "rvol": round(self.rvol, 2) if self.rvol is not None else None,
            "atr_daily_pct": round(self.atr_daily_pct, 5) if self.atr_daily_pct is not None else None,
            "spread_pct": round(self.spread_pct, 6) if self.spread_pct is not None else None,
            "score_components": {k: round(v, 2) for k, v in self.opportunity_score_components.items()},
            "reasons": self.reasons,
            "risks": self.risks,
            "plain_english": self.plain_english,
            "advanced": self.advanced_detail,
            "strategy_version": self.strategy_version,
            "generated_at": self.generated_at.isoformat() if hasattr(self.generated_at, "isoformat") else str(self.generated_at),
            "quantity_hint": self.quantity_hint,
            "risk_amount_hint": round(self.risk_amount_hint, 2),
        }


@dataclass
class ScanResult:
    ts: Any
    regime: Optional[str] = None
    regime_confidence: float = 0.0
    longs: List[Opportunity] = field(default_factory=list)
    shorts: List[Opportunity] = field(default_factory=list)
    evaluated: int = 0
    rejected_by_gate: Dict[str, int] = field(default_factory=dict)
    sector_snapshot: Optional[Dict[str, Any]] = None
    data_source: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)

    def to_dict(self, top_n: int = 10) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "evaluated": self.evaluated,
            "long_count": len(self.longs),
            "short_count": len(self.shorts),
            "longs": [o.to_dict() for o in self.longs[:top_n]],
            "shorts": [o.to_dict() for o in self.shorts[:top_n]],
            "rejected_by_gate": dict(sorted(self.rejected_by_gate.items(), key=lambda kv: -kv[1])[:12]),
            "sectors": self.sector_snapshot,
            "data_source": self.data_source,
            "notes": self.notes,
        }


def liquidity_gate(
    features: BarFeatures,
    *,
    min_average_daily_volume: float = 200_000,
    min_average_daily_turnover: float = 50_000_000,
    min_price: float = 20.0,
    max_price: float = 20_000.0,
    max_spread_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    """Cheap pre-filter so the expensive strategy evaluation only runs on genuinely
    tradable names."""
    if features.close < min_price:
        return False, f"price {features.close:.2f} below the Rs {min_price:.0f} floor"
    if features.close > max_price:
        return False, f"price {features.close:.2f} above the Rs {max_price:,.0f} ceiling"
    if features.average_daily_volume is not None and features.average_daily_volume < min_average_daily_volume:
        return False, f"average daily volume {features.average_daily_volume:,.0f} is below the floor"
    if features.turnover_inr is not None and features.turnover_inr < min_average_daily_turnover:
        return False, f"average daily turnover Rs {features.turnover_inr:,.0f} is below the floor"
    if max_spread_pct is not None and features.spread_pct is not None and features.spread_pct > max_spread_pct:
        return False, f"spread {features.spread_pct:.3%} is too wide"
    return True, ""


def build_plain_english(candidate: TradeCandidate, features: BarFeatures) -> str:
    """SIMPLE MODE explanation: plain English, no jargon, no predictions."""
    direction_word = "LONG" if candidate.direction == "LONG" else "SHORT"
    move_word = "broken above" if candidate.direction == "LONG" else "broken below"
    or_range = features.opening_range
    lines: List[str] = []
    if or_range is not None:
        level = or_range.or_high if candidate.direction == "LONG" else or_range.or_low
        lines.append(
            f"{candidate.symbol} has {move_word} its {or_range.duration_minutes}-minute opening range "
            f"level of Rs {level:,.2f}."
        )
    if features.above_vwap is not None:
        side = "above" if features.above_vwap else "below"
        lines.append(f"It is trading {side} VWAP (Rs {features.vwap:,.2f})." if features.vwap else f"It is {side} VWAP.")
    if features.rvol is not None:
        lines.append(f"Volume is {features.rvol:.1f}x its normal level for this time of day.")
    if candidate.sector:
        rank_text = f" (ranked #{candidate.sector_rank})" if candidate.sector_rank else ""
        lines.append(f"The {candidate.sector} sector{rank_text} is supportive.")
    if candidate.regime:
        lines.append(f"The overall market is in a {candidate.regime.replace('_', ' ').lower()} state.")
    lines.append(
        f"Risk to the stop is Rs {candidate.risk_per_share:,.2f} per share; the first target is "
        f"Rs {candidate.target_1:,.2f} (about {candidate.risk_reward:.1f} times the risk)."
    )
    lines.append(f"Setup quality score: {candidate.score:.0f} out of 100.")
    return " ".join(lines)


class OpportunityScanner:
    """Scans a universe and produces ranked opportunities."""

    def __init__(self, strategy: ORBVWAPStrategy, sector_engine: Optional[SectorEngine] = None) -> None:
        self.strategy = strategy
        self.sector_engine = sector_engine or SectorEngine()

    # ------------------------------------------------------------------- scan
    def scan(
        self,
        *,
        ts: Any,
        bar_index_by_instrument: Mapping[str, int],
        series_map: Mapping[str, Any],
        symbols: Mapping[str, str],
        sector_map: Mapping[str, str],
        regime: Optional[str] = None,
        regime_assessment: Optional[Any] = None,
        nifty_return_pct: Optional[float] = None,
        index_above_vwap: Optional[bool] = None,
        sector_snapshot: Optional[SectorSnapshot] = None,
        spread_by_instrument: Optional[Mapping[str, float]] = None,
        liquidity_config: Optional[Dict[str, Any]] = None,
        equity: float = 0.0,
        risk_pct: float = 0.0025,
        top_n: int = 10,
        data_source: str = "UNKNOWN",
    ) -> ScanResult:
        liquidity = liquidity_config or {}
        spread_map = dict(spread_by_instrument or {})
        result = ScanResult(
            ts=ts,
            regime=regime,
            regime_confidence=float(getattr(regime_assessment, "confidence", 0.0) or 0.0),
            sector_snapshot=sector_snapshot.to_dict() if sector_snapshot else None,
            data_source=data_source,
        )
        if not series_map:
            result.notes.append("no instrument data available for scanning")
            return result

        candidates: List[Tuple[TradeCandidate, BarFeatures]] = []

        for instrument_key, series in series_map.items():
            index = int(bar_index_by_instrument.get(instrument_key, -1))
            if index < 0 or index >= len(series.candles):
                continue
            if series.candles[index].ts != ts:
                continue

            sector = sector_map.get(instrument_key)
            sector_return = sector_rank = None
            if sector_snapshot is not None:
                state = sector_snapshot.state_of(sector)
                if state is not None:
                    sector_return, sector_rank = state.return_pct, state.rank

            features = series.features_at(
                index,
                index_return=nifty_return_pct,
                spread_pct=spread_map.get(instrument_key),
                regime=regime,
                sector_return=sector_return,
                sector_rank=sector_rank,
            )
            result.evaluated += 1

            ok, reason = liquidity_gate(features, **liquidity)
            if not ok:
                result.rejected_by_gate["liquidity"] = result.rejected_by_gate.get("liquidity", 0) + 1
                continue

            context = StrategyContext(
                ts=ts,
                features=features,
                recent_candles=series.candles[max(0, index - 5) : index + 1],
                nifty_return_pct=nifty_return_pct,
                sector_snapshot=sector_snapshot,
                sector=sector,
                extras={
                    "symbol": symbols.get(instrument_key, instrument_key),
                    "index_above_vwap": index_above_vwap,
                },
            )
            candidate = self.strategy.evaluate(context)

            for rejection in context.extras.get("rejections", []) or []:
                gate = rejection.get("gate", "unknown")
                result.rejected_by_gate[gate] = result.rejected_by_gate.get(gate, 0) + 1

            if candidate is None:
                continue
            candidates.append((candidate, features))

        longs: List[Opportunity] = []
        shorts: List[Opportunity] = []
        for rank_index, (candidate, features) in enumerate(
            sorted(candidates, key=lambda item: item[0].score, reverse=True), start=1
        ):
            opportunity = self._to_opportunity(candidate, features, rank_index, equity, risk_pct)
            (longs if candidate.direction == "LONG" else shorts).append(opportunity)

        result.longs = sorted(longs, key=lambda o: o.score, reverse=True)[:top_n]
        result.shorts = sorted(shorts, key=lambda o: o.score, reverse=True)[:top_n]
        for index, opportunity in enumerate(result.longs, start=1):
            opportunity.rank = index
        for index, opportunity in enumerate(result.shorts, start=1):
            opportunity.rank = index
        return result

    # ------------------------------------------------------------- conversion
    def _to_opportunity(
        self,
        candidate: TradeCandidate,
        features: BarFeatures,
        rank: int,
        equity: float,
        risk_pct: float,
    ) -> Opportunity:
        risk_per_share = abs(candidate.entry_price - candidate.stop_price)
        quantity_hint = 0
        risk_amount_hint = 0.0
        if equity > 0 and risk_per_share > 0:
            risk_amount_hint = min(risk_pct, 0.005) * equity
            quantity_hint = int(risk_amount_hint // risk_per_share)

        vwap_state = ""
        if features.above_vwap is not None:
            vwap_state = "above VWAP" if features.above_vwap else "below VWAP"

        return Opportunity(
            rank=rank,
            instrument_key=candidate.instrument_key,
            symbol=candidate.symbol,
            direction=candidate.direction,
            score=candidate.score,
            entry_price=candidate.entry_price,
            stop_price=candidate.stop_price,
            target_1=candidate.target_1,
            target_2=candidate.target_2,
            risk_reward=candidate.risk_reward,
            risk_per_share=risk_per_share,
            sector=candidate.sector,
            sector_rank=candidate.sector_rank,
            regime=candidate.regime,
            vwap=features.vwap,
            vwap_state=vwap_state,
            rvol=features.rvol,
            atr_daily_pct=features.atr_daily_pct,
            spread_pct=features.spread_pct,
            opportunity_score_components=candidate.score_components,
            reasons=list(candidate.reasons),
            risks=list(candidate.risks),
            plain_english=build_plain_english(candidate, features),
            advanced_detail={
                "or_high": features.opening_range.or_high if features.opening_range else None,
                "or_low": features.opening_range.or_low if features.opening_range else None,
                "or_width_atr_fraction": (
                    features.opening_range.or_width_atr_fraction if features.opening_range else None
                ),
                "atr": features.atr,
                "atr_percentile": features.atr_percentile,
                "adx": features.adx,
                "rsi": features.rsi,
                "ema_spread_pct": features.ema_spread_pct,
                "relative_strength": features.relative_strength,
                "gap_pct": features.gap_pct,
                "range_position": features.range_position,
                "minutes_from_open": features.minutes_from_open,
                "average_daily_volume": features.average_daily_volume,
                "sector_return": features.sector_return,
            },
            strategy_version=candidate.strategy_version,
            generated_at=candidate.ts,
            quantity_hint=quantity_hint,
            risk_amount_hint=risk_amount_hint,
        )


__all__ = ["OpportunityScanner", "Opportunity", "ScanResult", "liquidity_gate", "build_plain_english"]
