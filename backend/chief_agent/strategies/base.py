"""Strategy base types.

A strategy is a PURE, DETERMINISTIC function of observable state:

    evaluate(context) -> Optional[TradeCandidate]

It may not read the future, may not place orders, may not size positions and may
not consult an LLM. Everything a strategy returns is a *proposal*; the risk
engine and the execution engine decide whether anything happens.

Every strategy configuration is immutable once published and identified by a
version string (``ORB_v1.0.0``). The research engine creates challengers by
producing a NEW config, never by mutating the champion.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from ..indicators.features import BarFeatures
from ..logging_setup import get_logger
from ..timeutil import IST

log = get_logger(__name__, component="strategy")


@dataclass
class TradeCandidate:
    """A proposed trade, fully specified but not yet risk-checked or sized."""

    instrument_key: str
    symbol: str
    direction: str                       # LONG | SHORT
    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    strategy_family: str
    strategy_version: str
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    score_components: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    regime: Optional[str] = None
    sector: Optional[str] = None
    sector_rank: Optional[int] = None
    ts: Any = None
    tags: List[str] = field(default_factory=list)

    @property
    def risk_per_share(self) -> float:
        return abs(float(self.entry_price) - float(self.stop_price))

    @property
    def reward_per_share_t1(self) -> float:
        return abs(float(self.target_1) - float(self.entry_price))

    @property
    def risk_reward(self) -> float:
        risk = self.risk_per_share
        if risk <= 0:
            return 0.0
        return self.reward_per_share_t1 / risk

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "stop_price": round(self.stop_price, 4),
            "target_1": round(self.target_1, 4),
            "target_2": round(self.target_2, 4),
            "risk_per_share": round(self.risk_per_share, 4),
            "risk_reward": round(self.risk_reward, 3),
            "score": round(self.score, 2),
            "score_components": {k: round(v, 2) for k, v in self.score_components.items()},
            "strategy_family": self.strategy_family,
            "strategy_version": self.strategy_version,
            "regime": self.regime,
            "sector": self.sector,
            "sector_rank": self.sector_rank,
            "reasons": list(self.reasons),
            "risks": list(self.risks),
            "features": self.features,
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "tags": list(self.tags),
        }

    def clone_with(self, **overrides: Any) -> "TradeCandidate":
        data = copy.deepcopy(self.__dict__)
        data.update(overrides)
        return TradeCandidate(**data)


@dataclass
class StrategyContext:
    """Everything a strategy is allowed to see at one decision point."""

    ts: Any
    features: BarFeatures
    previous_features: Optional[BarFeatures] = None
    recent_candles: Sequence[Any] = field(default_factory=list)
    position_open: bool = False
    open_trade_direction: Optional[str] = None
    daily_trades: int = 0
    nifty_return_pct: Optional[float] = None
    vix_level: Optional[float] = None
    sector_snapshot: Optional[Any] = None
    sector: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def minutes_from_open(self) -> float:
        return float(self.features.minutes_from_open)


class Strategy(Protocol):
    family: str
    version: str

    def evaluate(self, context: StrategyContext) -> Optional[TradeCandidate]:
        ...


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def deep_get(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """``deep_get(cfg, "long.min_rvol")`` with a safe default."""
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of a strategy configuration - used to prove immutability."""
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class ImmutableConfig:
    """A frozen, hashed view of a strategy configuration."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = copy.deepcopy(config)
        self._frozen = json.dumps(self._config, sort_keys=True, default=str)
        self.hash = config_hash(self._config)

    def get(self, path: str, default: Any = None) -> Any:
        return deep_get(self._config, path, default)

    @property
    def raw(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImmutableConfig):
            return NotImplemented
        return self.hash == other.hash

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ImmutableConfig(hash={self.hash})"


def bump_version(version: str, level: str = "patch", family: str = "ORB") -> str:
    """``ORB_v1.0.0`` + minor -> ``ORB_v1.1.0``.

    patch = single-parameter retune, minor = new filter or rule,
    major = structural strategy change.
    """
    core = version.split("_v")[-1]
    parts = [int(p) for p in core.split(".")] + [0, 0, 0]
    major, minor, patch = parts[0], parts[1], parts[2]
    level = level.lower()
    if level == "major":
        major, minor, patch = major + 1, 0, 0
    elif level == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{family}_v{major}.{minor}.{patch}"


__all__ = [
    "TradeCandidate",
    "StrategyContext",
    "Strategy",
    "ImmutableConfig",
    "deep_get",
    "config_hash",
    "bump_version",
]
