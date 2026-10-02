"""Feature engine.

Computes the causal feature set that the strategy, the regime engine and the
opportunity scorer consume. Raw candles are never mutated - features are computed
on demand and attached to signals/trades for later research.

Feature-set version: ``v1``. Bump this if the meaning of any field changes; the
research layer keys experiments off it.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np

from ..broker.upstox_market_data import Candle
from ..timeutil import IST, minutes_from_open
from . import core as ind
from .opening_range import OpeningRange, build_opening_range, opening_range_state

FEATURE_SET_VERSION = "v1"


@dataclass
class BarFeatures:
    """Everything known about one instrument at one bar, using only past data."""

    ts: Any
    instrument_key: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    vwap_distance_pct: Optional[float] = None
    above_vwap: Optional[bool] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    atr_daily: Optional[float] = None
    atr_daily_pct: Optional[float] = None
    atr_percentile: Optional[float] = None
    rvol: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_spread_pct: Optional[float] = None
    rsi: Optional[float] = None
    adx: Optional[float] = None
    gap_pct: Optional[float] = None
    previous_close: Optional[float] = None
    day_range_pct: Optional[float] = None
    range_position: Optional[float] = None
    minutes_from_open: float = 0.0
    candle_body_fraction: float = 0.0
    is_bullish: bool = False
    relative_strength: Optional[float] = None       # vs the market index, same window
    sector_return: Optional[float] = None
    sector_rank: Optional[int] = None
    spread_pct: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    average_daily_volume: Optional[float] = None
    turnover_inr: Optional[float] = None
    opening_range: Optional[OpeningRange] = None
    regime: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "instrument_key": self.instrument_key,
            "ltp": self.ltp,
            "vwap": self.vwap,
            "vwap_distance_pct": self.vwap_distance_pct,
            "above_vwap": self.above_vwap,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "atr_daily": self.atr_daily,
            "atr_daily_pct": self.atr_daily_pct,
            "atr_percentile": self.atr_percentile,
            "rvol": self.rvol,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "ema_spread_pct": self.ema_spread_pct,
            "rsi": self.rsi,
            "adx": self.adx,
            "gap_pct": self.gap_pct,
            "day_range_pct": self.day_range_pct,
            "range_position": self.range_position,
            "minutes_from_open": self.minutes_from_open,
            "candle_body_fraction": self.candle_body_fraction,
            "is_bullish": self.is_bullish,
            "relative_strength": self.relative_strength,
            "sector_return": self.sector_return,
            "sector_rank": self.sector_rank,
            "spread_pct": self.spread_pct,
            "bid": self.bid,
            "ask": self.ask,
            "average_daily_volume": self.average_daily_volume,
            "turnover_inr": self.turnover_inr,
            "regime": self.regime,
            "or_high": self.opening_range.or_high if self.opening_range else None,
            "or_low": self.opening_range.or_low if self.opening_range else None,
            "or_width": self.opening_range.or_width if self.opening_range else None,
            "or_width_atr_fraction": (
                self.opening_range.or_width_atr_fraction if self.opening_range else None
            ),
            **self.extra,
        }



def _median_bars_per_session(timestamps: Sequence[dt.datetime]) -> int:
    """Median number of bars per IST session, used to scale per-bar volatility."""
    counts: Dict[dt.date, int] = {}
    for ts in timestamps:
        local = ts.astimezone(IST) if ts.tzinfo else ts
        counts[local.date()] = counts.get(local.date(), 0) + 1
    if not counts:
        return 1
    values = sorted(counts.values())
    return values[len(values) // 2]


def _rolling_percentile(
    values: Sequence[Optional[float]],
    window: int = 250,
    min_periods: int = 30,
) -> np.ndarray:
    """Causal rolling percentile rank (rank of the current value among the
    previous ``window`` values, excluding itself). NaN while warming up."""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    history: List[float] = []
    for i in range(n):
        current = values[i]
        if current is None:
            continue
        if len(history) >= min_periods:
            arr = np.asarray(history[-window:], dtype=float)
            out[i] = float((arr <= current).mean())
        history.append(float(current))
    return out


class InstrumentSeries:
    """Pre-computed causal indicator arrays for one instrument.

    ``at(index)`` returns the :class:`BarFeatures` visible at that bar. Nothing in
    this class looks forward: every array element ``i`` is derived from data at
    ``<= i`` only, and ``opening_range_for(index)`` returns the range that was
    already fully formed before that bar opened.
    """

    def __init__(self, candles: Sequence[Candle], *, opening_range_minutes: int = 15) -> None:
        self.candles: List[Candle] = sorted(candles, key=lambda c: c.ts)
        self.instrument_key = self.candles[0].instrument_key if self.candles else ""
        self.opening_range_minutes = opening_range_minutes
        n = len(self.candles)

        self.timestamps = [c.ts for c in self.candles]
        self.opens = np.array([c.open for c in self.candles], dtype=float)
        self.highs = np.array([c.high for c in self.candles], dtype=float)
        self.lows = np.array([c.low for c in self.candles], dtype=float)
        self.closes = np.array([c.close for c in self.candles], dtype=float)
        self.volumes = np.array([c.volume for c in self.candles], dtype=float)

        self.vwap = ind.vwap_session(self.timestamps, self.highs, self.lows, self.closes, self.volumes)
        self.atr14 = ind.atr(self.highs, self.lows, self.closes, 14)
        # Per-bar ATR as a fraction of price ...
        self.atr_pct_series = ind.atr_pct(self.highs, self.lows, self.closes, 14)
        # ... and the same figure scaled to a full session, which is the value
        # trading filters are written against ("daily ATR"). Volatility scales
        # with the square root of time, so bars_per_session is the right factor.
        bars_per_session = float(max(1, _median_bars_per_session(self.timestamps)))
        self._atr_scale = math.sqrt(bars_per_session)
        self.atr_daily_pct_series = [
            (value * self._atr_scale) if value is not None else None for value in self.atr_pct_series
        ]
        # Session-scaled ATR in price terms. The opening-range filters and the
        # breakout margin are written against THIS, because "how wide is the
        # opening range" is only meaningful relative to a full day's range.
        self.atr_daily_series = [
            (value * self._atr_scale) if value is not None else None for value in self.atr14
        ]
        self.ema_fast = ind.ema(self.closes, 9)
        self.ema_slow = ind.ema(self.closes, 21)
        self.rsi14 = ind.rsi(self.closes, 14)
        self.adx14 = ind.adx(self.highs, self.lows, self.closes, 14)
        self.volume_delta = ind.cumulative_volume_delta(self.closes, self.volumes, self.highs, self.lows)

        # Daily aggregates (computed per session, shifted so a bar only ever sees
        # sessions that have already finished).
        self.previous_close_by_day: Dict[dt.date, Optional[float]] = {}
        self.daily_volume_by_day: Dict[dt.date, float] = {}
        self.average_daily_volume_by_day: Dict[dt.date, Optional[float]] = {}
        self._compute_daily_aggregates()

        self.rvol_series = ind.relative_volume(
            self.volumes, self.timestamps, lookback_days=20, same_time_window_minutes=60
        )

        self._or_cache: Dict[dt.date, OpeningRange] = {}
        self._atr_percentile_cache: Dict[int, Optional[float]] = {}

    # ------------------------------------------------------------ daily stats
    def _compute_daily_aggregates(self) -> None:
        by_day: Dict[dt.date, List[int]] = {}
        for index, candle in enumerate(self.candles):
            ts = candle.ts.astimezone(IST) if candle.ts.tzinfo else candle.ts
            by_day.setdefault(ts.date(), []).append(index)

        ordered_days = sorted(by_day)
        running_volumes: List[float] = []
        for position, day in enumerate(ordered_days):
            indices = by_day[day]
            self.previous_close_by_day[day] = (
                float(self.closes[by_day[ordered_days[position - 1]][-1]]) if position > 0 else None
            )
            day_volume = float(np.sum(self.volumes[indices]))
            self.daily_volume_by_day[day] = day_volume
            self.average_daily_volume_by_day[day] = (
                float(np.mean(running_volumes[-20:])) if len(running_volumes) >= 5 else None
            )
            running_volumes.append(day_volume)

        self._days_in_order = ordered_days
        self._day_indices: Dict[dt.date, List[int]] = by_day
        self._day_of_index = {index: day for day, indices in by_day.items() for index in indices}

        # ---- pre-computed causal arrays (O(n) once, O(1) per lookup) --------
        n = len(self.candles)
        self.session_open_arr = np.zeros(n, dtype=float)
        self.session_high_run = np.zeros(n, dtype=float)
        self.session_low_run = np.zeros(n, dtype=float)
        self.previous_close_arr = np.full(n, np.nan, dtype=float)
        self.average_volume_arr = np.full(n, np.nan, dtype=float)
        self.day_ordinal = np.zeros(n, dtype=np.int64)

        for ordinal, day in enumerate(ordered_days):
            indices = np.asarray(by_day[day], dtype=int)
            open_price = float(self.opens[indices[0]])
            self.session_open_arr[indices] = open_price
            self.session_high_run[indices] = np.maximum.accumulate(self.highs[indices])
            self.session_low_run[indices] = np.minimum.accumulate(self.lows[indices])
            self.day_ordinal[indices] = ordinal
            previous_close = self.previous_close_by_day[day]
            if previous_close:
                self.previous_close_arr[indices] = float(previous_close)
            average_volume = self.average_daily_volume_by_day[day]
            if average_volume:
                self.average_volume_arr[indices] = float(average_volume)

        # Average SESSION range (high - low) over the previous 20 sessions, causal.
        # This is the reference the opening-range width filter is measured against
        # ("is the opening range unusually narrow or wide for this instrument?").
        self.average_session_range_arr = np.full(n, np.nan, dtype=float)
        session_ranges: List[float] = []
        for day in ordered_days:
            indices = np.asarray(by_day[day], dtype=int)
            session_range = float(self.highs[indices].max() - self.lows[indices].min())
            if len(session_ranges) >= 5:
                self.average_session_range_arr[indices] = float(np.mean(session_ranges[-20:]))
            session_ranges.append(session_range)

        # ATR percentile over a rolling window - computed once, never per bar.
        self.atr_percentile_arr = _rolling_percentile(self.atr_pct_series, window=250, min_periods=30)

    # ------------------------------------------------------------- public API
    def __len__(self) -> int:
        return len(self.candles)

    def day_indices(self, index: int) -> List[int]:
        """Indices of every bar belonging to the same session as ``index``."""
        day = self._day_of_index.get(index)
        return self._day_indices.get(day, []) if day else []

    def opening_range_for(self, index: int) -> Optional[OpeningRange]:
        """The opening range that is **fully formed before** the bar at ``index``.

        Returns ``None`` while the opening window is still in progress, which is
        what makes it impossible for a strategy to act on a partial range.
        """
        day = self._day_of_index.get(index)
        if day is None:
            return None
        cached = self._or_cache.get(day)
        if cached is None:
            indices = self._day_indices.get(day, [])
            if not indices:
                return None
            first_index = indices[0]
            day_candles = [self.candles[i] for i in indices]
            cached = build_opening_range(
                day_candles,
                self.opening_range_minutes,
                trading_date=day,
                instrument_key=self.instrument_key,
                atr_value=self.atr_daily_series[first_index],
                range_reference=self.average_session_range_arr[first_index],
            )
            self._or_cache[day] = cached

        ts = self.candles[index].ts
        if cached.formed_at is None or ts < cached.formed_at:
            return None
        return cached if cached.complete else None

    def session_open_price(self, index: int) -> Optional[float]:
        indices = self.day_indices(index)
        if not indices:
            return None
        return float(self.opens[indices[0]])

    def return_since_open(self, index: int) -> Optional[float]:
        session_open = self.session_open_price(index)
        if not session_open:
            return None
        return (float(self.closes[index]) - session_open) / session_open

    def atr_percentile_at(self, index: int, lookback: int = 250) -> Optional[float]:
        """Percentile rank of today's ATR% within its own trailing window.

        Pre-computed as an array in ``_compute_daily_aggregates`` so repeated
        lookups during a backtest are O(1).
        """
        if not (0 <= index < len(self.atr_percentile_arr)):
            return None
        value = self.atr_percentile_arr[index]
        return None if value != value else float(value)

    def features_at(
        self,
        index: int,
        *,
        index_return: Optional[float] = None,
        spread_pct: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        regime: Optional[str] = None,
        sector_return: Optional[float] = None,
        sector_rank: Optional[int] = None,
    ) -> BarFeatures:
        """Feature vector for the bar at ``index`` (causal only)."""
        candle = self.candles[index]
        ts = candle.ts.astimezone(IST) if candle.ts.tzinfo else candle.ts
        day = ts.date()

        vwap = self.vwap[index]
        close = float(self.closes[index])
        vwap_distance = ((close - vwap) / vwap) if (vwap and vwap > 0) else None
        above_vwap = (close > vwap) if vwap else None

        previous_close = self.previous_close_arr[index]
        previous_close = float(previous_close) if previous_close == previous_close else None
        gap_pct = ((candle.open - previous_close) / previous_close) if previous_close else None

        day_high = float(self.session_high_run[index])
        day_low = float(self.session_low_run[index])
        day_span = day_high - day_low
        day_range_pct = (day_span / previous_close) if previous_close else None
        range_position = ((close - day_low) / day_span) if day_span > 0 else 0.5

        or_range = self.opening_range_for(index)

        # Relative strength: the stock's move since the open minus the index move.
        session_open = float(self.session_open_arr[index])
        stock_return = ((close - session_open) / session_open) if session_open > 0 else 0.0
        relative_strength = (stock_return - index_return) if index_return is not None else None

        average_volume_value = self.average_volume_arr[index]
        average_volume = float(average_volume_value) if average_volume_value == average_volume_value else None
        turnover = average_volume * close if average_volume else None

        return BarFeatures(
            ts=ts,
            instrument_key=self.instrument_key,
            ltp=close,
            open=float(candle.open),
            high=float(candle.high),
            low=float(candle.low),
            close=close,
            volume=float(candle.volume),
            vwap=vwap,
            vwap_distance_pct=vwap_distance,
            above_vwap=above_vwap,
            atr=self.atr14[index],
            atr_pct=self.atr_pct_series[index],
            atr_daily=self.atr_daily_series[index],
            atr_daily_pct=self.atr_daily_pct_series[index],
            atr_percentile=self.atr_percentile_at(index),
            rvol=self.rvol_series[index],
            ema_fast=self.ema_fast[index],
            ema_slow=self.ema_slow[index],
            ema_spread_pct=(
                (self.ema_fast[index] - self.ema_slow[index]) / self.ema_slow[index]
                if (self.ema_fast[index] and self.ema_slow[index])
                else None
            ),
            rsi=self.rsi14[index],
            adx=self.adx14[index],
            gap_pct=gap_pct,
            previous_close=previous_close,
            day_range_pct=day_range_pct,
            range_position=range_position,
            minutes_from_open=minutes_from_open(ts),
            candle_body_fraction=ind.candle_body_pct(candle.open, candle.high, candle.low, candle.close),
            is_bullish=candle.close >= candle.open,
            relative_strength=relative_strength,
            sector_return=sector_return,
            sector_rank=sector_rank,
            spread_pct=spread_pct,
            bid=bid,
            ask=ask,
            average_daily_volume=average_volume,
            turnover_inr=turnover,
            opening_range=or_range,
            regime=regime,
            extra={"volume_delta": float(self.volume_delta[index])},
        )

    def opening_range_signal(
        self,
        index: int,
        direction: str,
        *,
        margin_atr_fraction: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        or_range = self.opening_range_for(index)
        if or_range is None:
            return None
        return opening_range_state(
            or_range,
            self.candles[index],
            direction=direction,
            margin_atr_fraction=margin_atr_fraction,
            atr_value=self.atr_daily_series[index],
        )


class FeatureEngine:
    """Builds and caches :class:`InstrumentSeries` for a universe."""

    def __init__(self, opening_range_minutes: int = 15) -> None:
        self.opening_range_minutes = opening_range_minutes
        self._cache: Dict[str, InstrumentSeries] = {}

    def series(
        self,
        instrument_key: str,
        candles: Sequence[Candle],
        *,
        opening_range_minutes: Optional[int] = None,
    ) -> InstrumentSeries:
        key = f"{instrument_key}|{opening_range_minutes or self.opening_range_minutes}|{len(candles)}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        series = InstrumentSeries(
            candles,
            opening_range_minutes=opening_range_minutes or self.opening_range_minutes,
        )
        self._cache[key] = series
        return series

    def clear(self) -> None:
        self._cache.clear()


def index_return_at_open(
    index_series: Optional[InstrumentSeries],
    index_index: int,
) -> Optional[float]:
    """The index's move since its session open, at a matching bar."""
    if index_series is None or index_index >= len(index_series.candles):
        return None
    return index_series.return_since_open(index_index)


__all__ = [
    "BarFeatures",
    "InstrumentSeries",
    "FeatureEngine",
    "FEATURE_SET_VERSION",
    "index_return_at_open",
]
