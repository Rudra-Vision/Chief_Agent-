"""Sector engine.

Maps every instrument to an NSE sector, tracks the major sector indices, and
ranks them strongest -> weakest for the session.

Measures per sector:
    sector_return          - move since the open
    sector_momentum        - recent slope of the sector index
    sector_vwap_state      - above / below session VWAP, and the distance
    sector_breadth         - share of member stocks advancing
    relative_strength      - sector return minus NIFTY return

For LONG trades the system prefers strong stock + strong sector + supportive
market; for SHORT trades, the reverse.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..broker.upstox_instruments import SECTOR_INDEX_KEYS, InstrumentMaster
from ..broker.upstox_market_data import Candle
from ..indicators import core as ind
from ..logging_setup import get_logger
from ..timeutil import IST

log = get_logger(__name__, component="sectors")


@dataclass
class SectorState:
    name: str
    index_key: Optional[str] = None
    return_pct: Optional[float] = None
    relative_strength: Optional[float] = None
    vwap: Optional[float] = None
    vwap_distance_pct: Optional[float] = None
    above_vwap: Optional[bool] = None
    momentum: Optional[float] = None
    breadth: Optional[float] = None
    member_count: int = 0
    advancers: int = 0
    decliners: int = 0
    rank: Optional[int] = None
    strength_score: float = 0.0
    data_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "index_key": self.index_key,
            "return_pct": _r(self.return_pct),
            "relative_strength": _r(self.relative_strength),
            "vwap_distance_pct": _r(self.vwap_distance_pct),
            "above_vwap": self.above_vwap,
            "momentum": _r(self.momentum),
            "breadth": _r(self.breadth),
            "member_count": self.member_count,
            "advancers": self.advancers,
            "decliners": self.decliners,
            "rank": self.rank,
            "strength_score": round(self.strength_score, 4),
            "data_available": self.data_available,
        }


def _r(value: Optional[float], digits: int = 6) -> Optional[float]:
    return round(value, digits) if isinstance(value, (int, float)) else value


@dataclass
class SectorSnapshot:
    ts: Any
    sectors: List[SectorState] = field(default_factory=list)
    ranking: List[str] = field(default_factory=list)
    weakest: List[str] = field(default_factory=list)
    dispersion: Optional[float] = None

    def rank_of(self, sector: Optional[str]) -> Optional[int]:
        if not sector:
            return None
        for state in self.sectors:
            if state.name == sector:
                return state.rank
        return None

    def state_of(self, sector: Optional[str]) -> Optional[SectorState]:
        if not sector:
            return None
        return next((s for s in self.sectors if s.name == sector), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "ranking": self.ranking,
            "weakest": self.weakest,
            "dispersion": _r(self.dispersion),
            "sectors": [s.to_dict() for s in self.sectors],
        }


class SectorEngine:
    """Builds sector state and rankings for a session."""

    def __init__(self, master: Optional[InstrumentMaster] = None) -> None:
        self.master = master

    # ------------------------------------------------------------- membership
    @staticmethod
    def sector_of(symbol_or_key: str, master: Optional[InstrumentMaster] = None) -> Optional[str]:
        if master is not None and master.is_loaded:
            record = master.get(symbol_or_key) or master.by_symbol(symbol_or_key)
            if record is not None:
                # InstrumentRecord carries no sector itself; the watchlist does.
                pass
        return None

    @staticmethod
    def index_key_for(sector: Optional[str]) -> Optional[str]:
        if not sector:
            return None
        return SECTOR_INDEX_KEYS.get(sector)

    # ------------------------------------------------------- per-session state
    def build_snapshot(
        self,
        ts: Any,
        *,
        sector_index_candles: Mapping[str, Sequence[Candle]],
        index_bar_index: Mapping[str, int],
        index_returns: Mapping[str, Optional[float]],
        member_returns: Mapping[str, Mapping[str, float]],
        nifty_return_pct: Optional[float] = None,
    ) -> SectorSnapshot:
        """Assemble the sector table at one instant.

        ``sector_index_candles`` maps a sector name to that sector index's candle
        series. ``member_returns`` maps a sector name to ``{symbol: return_pct}``
        measured over the same window.
        """
        states: List[SectorState] = []
        for sector_name in sorted(set(list(member_returns) + list(sector_index_candles))):
            index_key = self.index_key_for(sector_name)
            candles = sector_index_candles.get(sector_name)
            state = SectorState(name=sector_name, index_key=index_key)

            if candles:
                index = index_bar_index.get(sector_name, len(candles) - 1)
                subset = list(candles[: index + 1])
                closes = [c.close for c in subset]
                highs = [c.high for c in subset]
                lows = [c.low for c in subset]
                volumes = [c.volume for c in subset]
                timestamps = [c.ts for c in subset]
                if closes:
                    vwap_series = ind.vwap_session(timestamps, highs, lows, closes, volumes)
                    vwap_values = [v for v in vwap_series if v is not None]
                    current = float(closes[-1])
                    session_open = float(subset[0].open) if subset else current
                    if session_open:
                        state.return_pct = (current - session_open) / session_open
                    if vwap_values:
                        state.vwap = vwap_values[-1]
                        state.above_vwap = current > vwap_values[-1]
                        state.vwap_distance_pct = (current - vwap_values[-1]) / vwap_values[-1]
                    if len(closes) >= 6:
                        state.momentum = (closes[-1] - closes[-6]) / max(abs(closes[-6]), 1e-9)
                    state.data_available = True

            members = member_returns.get(sector_name) or {}
            if members:
                values = [float(v) for v in members.values() if v is not None]
                state.member_count = len(values)
                state.advancers = sum(1 for v in values if v > 0)
                state.decliners = sum(1 for v in values if v < 0)
                if values:
                    state.breadth = state.advancers / len(values)
                    if state.return_pct is None:
                        state.return_pct = sum(values) / len(values)

            # Sector return fallback: average of its members (used when no index
            # series is available so the ranking still works offline).
            index_return = index_returns.get(sector_name, state.return_pct)
            if index_return is not None and nifty_return_pct is not None:
                state.relative_strength = index_return - nifty_return_pct
            state.strength_score = _strength_score(state)
            states.append(state)

        states.sort(key=lambda s: s.strength_score, reverse=True)
        for position, state in enumerate(states, start=1):
            state.rank = position

        returns = [s.return_pct for s in states if s.return_pct is not None]
        dispersion = float(max(returns) - min(returns)) if len(returns) >= 2 else None

        return SectorSnapshot(
            ts=ts,
            sectors=states,
            ranking=[s.name for s in states],
            weakest=[s.name for s in reversed(states)],
            dispersion=dispersion,
        )

    # ------------------------------------------------------------- convenience
    def is_sector_supportive(
        self,
        snapshot: Optional[SectorSnapshot],
        sector: Optional[str],
        direction: str,
        *,
        min_rank_for_long: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        """Deterministic sector confirmation check for a candidate trade."""
        reasons: List[str] = []
        if snapshot is None or not sector:
            return True, ["sector data unavailable - check skipped"]
        state = snapshot.state_of(sector)
        if state is None:
            return True, ["sector not tracked - check skipped"]

        direction = direction.upper()
        if direction == "LONG":
            if state.above_vwap is False:
                reasons.append(f"{sector} sector index is below its VWAP")
            if min_rank_for_long is not None and state.rank is not None and state.rank > min_rank_for_long:
                reasons.append(f"{sector} is ranked #{state.rank} (needs top {min_rank_for_long})")
            if state.relative_strength is not None and state.relative_strength < 0:
                reasons.append(f"{sector} is underperforming NIFTY by {state.relative_strength:.2%}")
            return (not reasons), reasons

        # SHORT
        if state.above_vwap is True and state.relative_strength is not None and state.relative_strength > 0.005:
            reasons.append(f"{sector} sector is strong (above VWAP, outperforming)")
            return False, reasons
        return True, reasons

    # ------------------------------------------------------------- fast path
    def build_snapshot_fast(
        self,
        ts: Any,
        *,
        sector_series: Mapping[str, Any],
        sector_pointers: Mapping[str, int],
        nifty_return_pct: Optional[float] = None,
    ) -> Optional[SectorSnapshot]:
        """O(sectors) snapshot from pre-computed causal arrays.

        Used by the backtester where the full slicing implementation in
        :meth:`build_snapshot` would make the loop quadratic.
        """
        if not sector_series:
            return None
        states: List[SectorState] = []
        for name, series in sector_series.items():
            pointer = int(sector_pointers.get(name, 0))
            if pointer < 0 or pointer >= len(series.candles):
                continue
            state = SectorState(name=name, index_key=self.index_key_for(name))
            current = float(series.closes[pointer])
            session_open = float(series.session_open_arr[pointer])
            if session_open:
                state.return_pct = (current - session_open) / session_open
            vwap_value = series.vwap[pointer] if pointer < len(series.vwap) else None
            if vwap_value:
                state.vwap = float(vwap_value)
                state.above_vwap = current > vwap_value
                state.vwap_distance_pct = (current - vwap_value) / vwap_value
            if pointer >= 5:
                past = float(series.closes[pointer - 5])
                if past:
                    state.momentum = (current - past) / abs(past)
            if state.return_pct is not None and nifty_return_pct is not None:
                state.relative_strength = state.return_pct - nifty_return_pct
            state.strength_score = _strength_score(state)
            state.data_available = True
            states.append(state)

        if not states:
            return None
        states.sort(key=lambda s: s.strength_score, reverse=True)
        for position, state in enumerate(states, start=1):
            state.rank = position
        returns = [s.return_pct for s in states if s.return_pct is not None]
        dispersion = float(max(returns) - min(returns)) if len(returns) >= 2 else None
        return SectorSnapshot(
            ts=ts,
            sectors=states,
            ranking=[s.name for s in states],
            weakest=[s.name for s in reversed(states)],
            dispersion=dispersion,
        )

    def ranking_table(self, snapshot: Optional[SectorSnapshot]) -> List[Dict[str, Any]]:
        if snapshot is None:
            return []
        return [s.to_dict() for s in snapshot.sectors]


def _strength_score(state: SectorState) -> float:
    """Deterministic sector strength score in roughly [-1, 1]."""
    score = 0.0
    if state.return_pct is not None:
        score += max(-1.0, min(1.0, state.return_pct / 0.02)) * 0.40
    if state.relative_strength is not None:
        score += max(-1.0, min(1.0, state.relative_strength / 0.015)) * 0.25
    if state.above_vwap is not None:
        score += 0.15 if state.above_vwap else -0.15
    if state.vwap_distance_pct is not None:
        score += max(-1.0, min(1.0, state.vwap_distance_pct / 0.01)) * 0.10
    if state.breadth is not None:
        score += (state.breadth - 0.5) * 0.20
    if state.momentum is not None:
        score += max(-1.0, min(1.0, state.momentum / 0.01)) * 0.10
    return max(-1.0, min(1.0, score))


__all__ = ["SectorEngine", "SectorState", "SectorSnapshot", "SECTOR_INDEX_KEYS"]
