"""Market regime engine.

Classifies each session (and, when the evidence changes, intraday) into one of:

    TREND_UP | TREND_DOWN | RANGE | HIGH_VOLATILITY | LOW_VOLATILITY |
    GAP_UP   | GAP_DOWN   | EVENT_DRIVEN | UNKNOWN

The classifier is fully deterministic: a weighted evidence score over NIFTY
trend, VWAP slope, market breadth, India VIX level and percentile, gap size,
sector dispersion and intraday range. There is no LLM in the decision path.

The regime is stored with every trade so performance can later be analysed **by
regime** - which is what makes "this strategy only works in trend days" a
falsifiable, measurable statement rather than an opinion.
"""

from __future__ import annotations

import datetime as dt
import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..broker.upstox_market_data import Candle
from ..indicators import core as ind
from ..logging_setup import get_logger
from ..timeutil import IST, minutes_from_open
from ..settings import get_config_store

log = get_logger(__name__, component="regime")


class Regime(str, enum.Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    GAP_UP = "GAP_UP"
    GAP_DOWN = "GAP_DOWN"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    UNKNOWN = "UNKNOWN"

    @property
    def is_bullish(self) -> bool:
        return self in (Regime.TREND_UP, Regime.GAP_UP)

    @property
    def is_bearish(self) -> bool:
        return self in (Regime.TREND_DOWN, Regime.GAP_DOWN)

    @property
    def is_neutral(self) -> bool:
        return self in (Regime.RANGE, Regime.LOW_VOLATILITY)

    @property
    def is_high_risk(self) -> bool:
        return self in (Regime.HIGH_VOLATILITY, Regime.EVENT_DRIVEN, Regime.UNKNOWN)

    def label(self) -> str:
        return {
            Regime.TREND_UP: "Trending up",
            Regime.TREND_DOWN: "Trending down",
            Regime.RANGE: "Range-bound",
            Regime.HIGH_VOLATILITY: "High volatility",
            Regime.LOW_VOLATILITY: "Low volatility",
            Regime.GAP_UP: "Gap up",
            Regime.GAP_DOWN: "Gap down",
            Regime.EVENT_DRIVEN: "Event driven",
            Regime.UNKNOWN: "Unknown",
        }[self]


@dataclass
class RegimeFeatures:
    """Raw measured inputs to the classifier. Every value is computed, not assumed."""

    nifty_return_pct: Optional[float] = None
    nifty_vwap_distance_pct: Optional[float] = None
    nifty_vwap_slope: Optional[float] = None
    nifty_above_vwap: Optional[bool] = None
    nifty_ema_fast_above_slow: Optional[bool] = None
    breadth_ratio: Optional[float] = None              # advancers / (adv + dec)
    advance_decline_ratio: Optional[float] = None
    vix_level: Optional[float] = None
    vix_percentile: Optional[float] = None
    gap_pct: Optional[float] = None
    intraday_range_pct: Optional[float] = None
    atr_percentile: Optional[float] = None
    sector_dispersion: Optional[float] = None
    volume_vs_average: Optional[float] = None
    synthetic_gap: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class RegimeAssessment:
    regime: Regime
    confidence: float
    scores: Dict[str, float]
    features: RegimeFeatures
    ts: Any
    trading_date: Optional[dt.date] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "label": self.regime.label(),
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "features": self.features.to_dict(),
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "trading_date": self.trading_date.isoformat() if self.trading_date else None,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Thresholds (configurable through config/strategy.yaml -> regime)
# --------------------------------------------------------------------------- #
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "gap_significant_pct": 0.006,
    "range_return_band_pct": 0.0035,
    "trend_return_pct": 0.006,
    "high_vix_level": 22.0,
    "low_vix_level": 12.0,
    "high_vix_percentile": 0.80,
    "low_vix_percentile": 0.25,
    "high_atr_percentile": 0.80,
    "low_atr_percentile": 0.25,
    "breadth_strong": 0.62,
    "breadth_weak": 0.38,
    "dispersion_high": 0.012,
    "range_position_extreme": 0.80,
    "min_confidence": 0.35,
}


class RegimeEngine:
    """Deterministic market-regime classifier."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        merged = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            merged.update({k: float(v) for k, v in thresholds.items() if k in DEFAULT_THRESHOLDS})
        else:
            configured = (get_config_store().load("strategy").get("regime") or {})
            merged.update({k: float(v) for k, v in configured.items() if k in DEFAULT_THRESHOLDS})
        self.t = merged

    # ------------------------------------------------------------- classifier
    def classify(self, features: RegimeFeatures, ts: Optional[Any] = None) -> RegimeAssessment:
        from ..timeutil import now_ist

        ts = ts or now_ist()
        notes: List[str] = []
        scores: Dict[str, float] = {r.value: 0.0 for r in Regime}

        # --- gap dominates at the open -------------------------------------
        gap = features.gap_pct
        if gap is not None and abs(gap) >= self.t["gap_significant_pct"]:
            if gap > 0:
                scores[Regime.GAP_UP.value] += 2.0
                notes.append(f"gap up {gap:+.2%}")
            else:
                scores[Regime.GAP_DOWN.value] += 2.0
                notes.append(f"gap down {gap:+.2%}")

        # --- NIFTY trend ----------------------------------------------------
        ret = features.nifty_return_pct
        if ret is not None:
            if ret >= self.t["trend_return_pct"]:
                scores[Regime.TREND_UP.value] += 2.0
                notes.append(f"NIFTY {ret:+.2%} on the day")
            elif ret <= -self.t["trend_return_pct"]:
                scores[Regime.TREND_DOWN.value] += 2.0
                notes.append(f"NIFTY {ret:+.2%} on the day")
            elif abs(ret) <= self.t["range_return_band_pct"]:
                scores[Regime.RANGE.value] += 1.2
                notes.append("NIFTY oscillating near unchanged")

        if features.nifty_above_vwap is not None:
            if features.nifty_above_vwap:
                scores[Regime.TREND_UP.value] += 0.8
            else:
                scores[Regime.TREND_DOWN.value] += 0.8

        if features.nifty_vwap_slope is not None:
            if features.nifty_vwap_slope > 0:
                scores[Regime.TREND_UP.value] += 0.6
            elif features.nifty_vwap_slope < 0:
                scores[Regime.TREND_DOWN.value] += 0.6

        if features.nifty_ema_fast_above_slow is not None:
            if features.nifty_ema_fast_above_slow:
                scores[Regime.TREND_UP.value] += 0.5
            else:
                scores[Regime.TREND_DOWN.value] += 0.5

        # --- breadth --------------------------------------------------------
        breadth = features.breadth_ratio
        if breadth is not None:
            if breadth >= self.t["breadth_strong"]:
                scores[Regime.TREND_UP.value] += 1.0
                notes.append(f"breadth {breadth:.0%} advancing")
            elif breadth <= self.t["breadth_weak"]:
                scores[Regime.TREND_DOWN.value] += 1.0
                notes.append(f"breadth {breadth:.0%} advancing")
            else:
                scores[Regime.RANGE.value] += 0.6

        # --- volatility ------------------------------------------------------
        vix = features.vix_level
        vix_pct = features.vix_percentile
        if vix is not None:
            if vix >= self.t["high_vix_level"]:
                scores[Regime.HIGH_VOLATILITY.value] += 1.5
                notes.append(f"India VIX {vix:.1f}")
            elif vix <= self.t["low_vix_level"]:
                scores[Regime.LOW_VOLATILITY.value] += 1.2
                notes.append(f"India VIX {vix:.1f}")
        if vix_pct is not None:
            if vix_pct >= self.t["high_vix_percentile"]:
                scores[Regime.HIGH_VOLATILITY.value] += 1.0
            elif vix_pct <= self.t["low_vix_percentile"]:
                scores[Regime.LOW_VOLATILITY.value] += 0.8

        atr_pct = features.atr_percentile
        if atr_pct is not None:
            if atr_pct >= self.t["high_atr_percentile"]:
                scores[Regime.HIGH_VOLATILITY.value] += 1.0
                notes.append("ATR in the top quintile")
            elif atr_pct <= self.t["low_atr_percentile"]:
                scores[Regime.LOW_VOLATILITY.value] += 0.8

        intraday_range = features.intraday_range_pct
        if intraday_range is not None and intraday_range >= 0.025:
            scores[Regime.HIGH_VOLATILITY.value] += 0.8

        # --- event / dispersion ---------------------------------------------
        dispersion = features.sector_dispersion
        if dispersion is not None and dispersion >= self.t["dispersion_high"]:
            scores[Regime.EVENT_DRIVEN.value] += 0.7
            notes.append("wide sector dispersion")

        volume_ratio = features.volume_vs_average
        if volume_ratio is not None and volume_ratio >= 1.8 and (vix_pct or 0) >= 0.7:
            scores[Regime.EVENT_DRIVEN.value] += 0.6

        # --- decide ----------------------------------------------------------
        best = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values())
        if total <= 0 or best[1] <= 0:
            return RegimeAssessment(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                scores=scores,
                features=features,
                ts=ts,
                trading_date=ts.date() if hasattr(ts, "date") else None,
                notes=["insufficient evidence to classify"],
            )

        confidence = min(1.0, best[1] / max(total, 1e-9) + 0.15 * min(1.0, total / 8.0))
        regime = Regime(best[0])
        if confidence < self.t["min_confidence"]:
            regime = Regime.RANGE if regime.is_high_risk else regime

        return RegimeAssessment(
            regime=regime,
            confidence=round(confidence, 4),
            scores=scores,
            features=features,
            ts=ts,
            trading_date=ts.date() if hasattr(ts, "date") else None,
            notes=notes,
        )


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def build_regime_features_from_index(
    index_candles: Sequence[Candle],
    index_index: int,
    *,
    vix_candles: Optional[Sequence[Candle]] = None,
    breadth_ratio: Optional[float] = None,
    sector_dispersion: Optional[float] = None,
    average_volume: Optional[float] = None,
    previous_close: Optional[float] = None,
    vix_history: Optional[Sequence[float]] = None,
) -> RegimeFeatures:
    """Compute regime features for the index bar at ``index_index`` (causal)."""
    if not index_candles or index_index <= 0:
        return RegimeFeatures(breadth_ratio=breadth_ratio, sector_dispersion=sector_dispersion)

    subset = list(index_candles[: index_index + 1])
    closes = [c.close for c in subset]
    highs = [c.high for c in subset]
    lows = [c.low for c in subset]
    volumes = [c.volume for c in subset]
    timestamps = [c.ts for c in subset]

    vwap_series = ind.vwap_session(timestamps, highs, lows, closes, volumes)
    ema_fast = ind.ema(closes, 9)
    ema_slow = ind.ema(closes, 21)
    atr_pct_series = ind.atr_pct(highs, lows, closes, 14)

    ts = timestamps[-1]
    day_indices = [
        i
        for i, t in enumerate(timestamps)
        if (t.astimezone(IST) if t.tzinfo else t).date() == (ts.astimezone(IST) if ts.tzinfo else ts).date()
    ]
    session_open = float(closes[day_indices[0]]) if day_indices else float(closes[0])
    session_open_price = float(subset[day_indices[0]].open) if day_indices else session_open
    current = float(closes[-1])

    day_high = max(float(highs[i]) for i in day_indices) if day_indices else current
    day_low = min(float(lows[i]) for i in day_indices) if day_indices else current

    gap_pct = None
    if previous_close:
        gap_pct = (session_open_price - float(previous_close)) / float(previous_close)

    vwap_slope = None
    vwap_values = [v for v in vwap_series if v is not None]
    if len(vwap_values) >= 6:
        vwap_slope = (vwap_values[-1] - vwap_values[-6]) / max(abs(vwap_values[-6]), 1e-9)

    vix_level = float(vix_candles[index_index].close) if vix_candles and index_index < len(vix_candles) else None
    vix_percentile = None
    if vix_level is not None and vix_history:
        history = [float(v) for v in vix_history if v is not None]
        if len(history) >= 20:
            vix_percentile = ind.percentile_rank(history, vix_level)

    atr_percentile = None
    current_atr_pct = atr_pct_series[-1]
    if current_atr_pct is not None:
        history_atr = [v for v in atr_pct_series[:-1] if v is not None]
        if len(history_atr) >= 30:
            atr_percentile = ind.percentile_rank(history_atr, current_atr_pct)

    volume_ratio = None
    if average_volume:
        day_volume = sum(float(volumes[i]) for i in day_indices)
        volume_ratio = day_volume / float(average_volume) if average_volume > 0 else None

    return RegimeFeatures(
        nifty_return_pct=(current - session_open) / session_open if session_open else None,
        nifty_vwap_distance_pct=(
            (current - vwap_values[-1]) / vwap_values[-1] if vwap_values and vwap_values[-1] else None
        ),
        nifty_vwap_slope=vwap_slope,
        nifty_above_vwap=(current > vwap_values[-1]) if vwap_values else None,
        nifty_ema_fast_above_slow=(
            ema_fast[-1] > ema_slow[-1] if (ema_fast[-1] is not None and ema_slow[-1] is not None) else None
        ),
        breadth_ratio=breadth_ratio,
        vix_level=vix_level,
        vix_percentile=vix_percentile,
        gap_pct=gap_pct,
        intraday_range_pct=((day_high - day_low) / session_open) if session_open else None,
        atr_percentile=atr_percentile,
        sector_dispersion=sector_dispersion,
        volume_vs_average=volume_ratio,
    )


def breadth_from_changes(changes: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Advance/decline statistics from a sequence of per-stock % changes."""
    values = [float(v) for v in changes if v is not None]
    if not values:
        return None, None
    advances = sum(1 for v in values if v > 0)
    declines = sum(1 for v in values if v < 0)
    breadth = advances / len(values)
    ratio = (advances / declines) if declines else float(advances) if advances else None
    return breadth, ratio


__all__ = [
    "Regime",
    "RegimeEngine",
    "RegimeFeatures",
    "RegimeAssessment",
    "build_regime_features_from_index",
    "breadth_from_changes",
    "DEFAULT_THRESHOLDS",
]


# --------------------------------------------------------------------------- #
# Fast, fully causal regime feature extraction for backtests
# --------------------------------------------------------------------------- #
class RegimeFeatureSeries:
    """Pre-computes regime features for every bar of an index series in O(n).

    The live path can keep using :func:`build_regime_features_from_index`; the
    backtester uses this so a year of one-minute bars costs one linear pass
    instead of an O(n^2) recomputation per timestamp.
    """

    def __init__(
        self,
        index_series: Any,
        *,
        vix_closes: Optional[Sequence[float]] = None,
        breadth_ratio: Optional[float] = None,
        sector_dispersion: Optional[float] = None,
        vix_window: int = 250,
        vix_min_periods: int = 20,
    ) -> None:
        self.series = index_series
        self.features: List[RegimeFeatures] = []

        n = len(index_series)
        if n == 0:
            return

        vwap = index_series.vwap
        ema_fast = index_series.ema_fast
        ema_slow = index_series.ema_slow
        atr_pct = index_series.atr_pct_series

        has_vix = vix_closes is not None and len(vix_closes) > 0
        vix_percentiles = _rolling_percentile_values(vix_closes, vix_window, vix_min_periods) if has_vix else None
        atr_percentiles = index_series.atr_percentile_arr

        for i in range(n):
            session_open = float(index_series.session_open_arr[i])
            session_high = float(index_series.session_high_run[i])
            session_low = float(index_series.session_low_run[i])
            close = float(index_series.closes[i])

            vwap_value = vwap[i] if i < len(vwap) else None
            vwap_slope = None
            if i >= 5 and i < len(vwap):
                previous = vwap[i - 5]
                if previous and vwap_value:
                    vwap_slope = (vwap_value - previous) / max(abs(previous), 1e-9)

            previous_close = index_series.previous_close_arr[i]
            previous_close = float(previous_close) if previous_close == previous_close else None
            session_open_price = float(index_series.opens[index_series.day_indices(i)[0]]) if index_series.day_indices(i) else close

            vix_level = None
            vix_pct = None
            if has_vix and i < len(vix_closes):
                vix_level = float(vix_closes[i])
                if vix_percentiles is not None and i < len(vix_percentiles):
                    value = vix_percentiles[i]
                    vix_pct = None if value != value else float(value)

            atr_pct_value = atr_pct[i] if i < len(atr_pct) else None
            atr_pct_pct = None
            if i < len(atr_percentiles):
                value = atr_percentiles[i]
                atr_pct_pct = None if value != value else float(value)

            self.features.append(
                RegimeFeatures(
                    nifty_return_pct=(close - session_open) / session_open if session_open else None,
                    nifty_vwap_distance_pct=((close - vwap_value) / vwap_value) if vwap_value else None,
                    nifty_vwap_slope=vwap_slope,
                    nifty_above_vwap=(close > vwap_value) if vwap_value else None,
                    nifty_ema_fast_above_slow=(
                        ema_fast[i] > ema_slow[i] if (ema_fast[i] is not None and ema_slow[i] is not None) else None
                    ),
                    breadth_ratio=breadth_ratio,
                    vix_level=vix_level,
                    vix_percentile=vix_pct,
                    gap_pct=((session_open_price - previous_close) / previous_close) if previous_close else None,
                    intraday_range_pct=((session_high - session_low) / session_open) if session_open else None,
                    atr_percentile=atr_pct_pct,
                    sector_dispersion=sector_dispersion,
                )
            )

    def __len__(self) -> int:
        return len(self.features)

    def at(self, index: int) -> RegimeFeatures:
        if not self.features:
            return RegimeFeatures()
        return self.features[min(max(0, index), len(self.features) - 1)]


def _rolling_percentile_values(
    values: Sequence[float], window: int, min_periods: int
) -> np.ndarray:
    values = list(values)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    history: List[float] = []
    for i in range(n):
        current = values[i]
        if current is None or current != current:
            continue
        if len(history) >= min_periods:
            arr = np.asarray(history[-window:], dtype=float)
            out[i] = float((arr <= float(current)).mean())
        history.append(float(current))
    return out


__all__.append("RegimeFeatureSeries")
