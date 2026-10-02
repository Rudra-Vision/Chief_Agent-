"""Opening Range calculation.

The opening range (OR) is built from verified 1-minute candles between the
09:15 IST session open and ``open + duration_minutes``. Only COMPLETE candles
are consumed, and the range is considered "formed" strictly AFTER the last
constituent candle closes - so a strategy can never see a partially formed
opening range.

Derived quantities stored on every OR:
    or_high, or_low, or_midpoint, or_width, or_width_atr_fraction,
    or_high_ts, or_low_ts, or_volume, or_range_position
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..broker.upstox_market_data import Candle
from ..timeutil import IST, market_open, minutes_from_open


@dataclass
class OpeningRange:
    """The opening range for one instrument on one session."""

    instrument_key: str
    trading_date: dt.date
    duration_minutes: int
    or_high: float = 0.0
    or_low: float = 0.0
    or_midpoint: float = 0.0
    or_width: float = 0.0
    or_width_pct: float = 0.0
    or_volume: float = 0.0
    or_open: float = 0.0
    or_close: float = 0.0
    or_bar_count: int = 0
    or_high_ts: Optional[dt.datetime] = None
    or_low_ts: Optional[dt.datetime] = None
    formed_at: Optional[dt.datetime] = None
    atr_at_formation: Optional[float] = None
    or_width_atr_fraction: Optional[float] = None
    complete: bool = False
    reason_incomplete: str = ""
    candles: List[Candle] = field(default_factory=list)

    @property
    def range_height(self) -> float:
        return max(0.0, self.or_high - self.or_low)

    def contains(self, price: float) -> bool:
        return self.or_low <= price <= self.or_high

    def position_within(self, price: float) -> float:
        """0.0 at the OR low, 1.0 at the OR high (0.5 when width is zero)."""
        height = self.range_height
        if height <= 0:
            return 0.5
        return max(0.0, min(1.0, (price - self.or_low) / height))

    def breakout_level(self, direction: str, margin: float = 0.0) -> float:
        if direction.upper() == "LONG":
            return self.or_high + margin
        return self.or_low - margin

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("candles", None)
        for key in ("or_high_ts", "or_low_ts", "formed_at"):
            value = payload.get(key)
            payload[key] = value.isoformat() if value else None
        payload["trading_date"] = self.trading_date.isoformat()
        return payload


def build_opening_range(
    candles: Sequence[Candle],
    duration_minutes: int,
    *,
    trading_date: Optional[dt.date] = None,
    instrument_key: str = "",
    atr_value: Optional[float] = None,
    range_reference: Optional[float] = None,
    session_start: Optional[dt.datetime] = None,
) -> OpeningRange:
    """Build the opening range from 1-minute candles.

    ``candles`` may contain the whole session; only bars inside the opening
    window are used. Bars are matched by their START timestamp being within
    ``[open, open + duration)``, which is how Upstox reports candle start times.
    """
    if not candles:
        target_date = trading_date or dt.date.today()
        return OpeningRange(
            instrument_key=instrument_key,
            trading_date=target_date,
            duration_minutes=duration_minutes,
            complete=False,
            reason_incomplete="no candles supplied",
        )

    first_ts = candles[0].ts.astimezone(IST) if candles[0].ts.tzinfo else candles[0].ts
    target_date = trading_date or first_ts.date()
    open_ts = session_start or market_open(target_date)
    window_end = open_ts + dt.timedelta(minutes=duration_minutes)

    window: List[Candle] = []
    for candle in candles:
        ts = candle.ts.astimezone(IST) if candle.ts.tzinfo else candle.ts
        if ts.date() != target_date:
            continue
        if open_ts <= ts < window_end:
            window.append(candle)

    result = OpeningRange(
        instrument_key=instrument_key or (window[0].instrument_key if window else ""),
        trading_date=target_date,
        duration_minutes=duration_minutes,
        candles=sorted(window, key=lambda c: c.ts),
    )

    expected_bars = max(1, duration_minutes)
    result.or_bar_count = len(result.candles)
    if not result.candles:
        result.reason_incomplete = "no candles inside the opening window"
        return result
    if len(result.candles) < expected_bars:
        result.reason_incomplete = (
            f"incomplete opening range: {len(result.candles)}/{expected_bars} one-minute candles"
        )
        return result

    result.or_high = max(c.high for c in result.candles)
    result.or_low = min(c.low for c in result.candles)
    result.or_midpoint = (result.or_high + result.or_low) / 2.0
    result.or_width = result.or_high - result.or_low
    result.or_open = result.candles[0].open
    result.or_close = result.candles[-1].close
    result.or_volume = sum(c.volume for c in result.candles)
    result.or_high_ts = next((c.ts for c in result.candles if c.high == result.or_high), None)
    result.or_low_ts = next((c.ts for c in result.candles if c.low == result.or_low), None)
    result.formed_at = result.candles[-1].ts + dt.timedelta(minutes=1)
    last_close = result.candles[-1].close
    result.or_width_pct = (result.or_width / last_close) if last_close > 0 else 0.0
    result.atr_at_formation = atr_value
    # ``or_width_atr_fraction`` is the opening-range width expressed as a fraction
    # of a TYPICAL SESSION RANGE for this instrument. That is the only denominator
    # that makes "is this opening range unusually narrow or wide?" meaningful and
    # comparable across instruments. If no session-range history exists yet we fall
    # back to the session-scaled ATR.
    denominator = range_reference if (range_reference and range_reference > 0) else atr_value
    if denominator and denominator > 0:
        result.or_width_atr_fraction = result.or_width / denominator
    result.complete = True
    return result


def opening_range_series(
    candles: Sequence[Candle],
    duration_minutes: int,
    *,
    atr_values: Optional[Sequence[Optional[float]]] = None,
) -> Dict[dt.date, OpeningRange]:
    """Build one opening range per session present in ``candles``."""
    by_day: Dict[dt.date, List[Candle]] = {}
    for candle in candles:
        ts = candle.ts.astimezone(IST) if candle.ts.tzinfo else candle.ts
        by_day.setdefault(ts.date(), []).append(candle)

    out: Dict[dt.date, OpeningRange] = {}
    for day, day_candles in by_day.items():
        atr_value = None
        if atr_values is not None:
            for index, candle in enumerate(candles):
                if candle is day_candles[0]:
                    if index < len(atr_values):
                        atr_value = atr_values[index]
                    break
        out[day] = build_opening_range(
            day_candles,
            duration_minutes,
            trading_date=day,
            instrument_key=day_candles[0].instrument_key if day_candles else "",
            atr_value=atr_value,
        )
    return out


def opening_range_state(
    or_range: OpeningRange,
    candle: Candle,
    *,
    direction: str,
    margin_atr_fraction: float = 0.05,
    atr_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate the current bar against the opening range - the ORB trigger check.

    Returns a dict with ``broken``, ``broke_level``, ``close_confirmed``,
    ``distance`` and ``retest_ok``. Pure function of the bar and the OR.
    """
    margin = 0.0
    if atr_value and margin_atr_fraction:
        margin = atr_value * margin_atr_fraction

    close = float(candle.close)
    high = float(candle.high)
    low = float(candle.low)

    if direction.upper() == "LONG":
        level = or_range.or_high
        broke = high > level
        close_confirmed = close > (level + margin)
        distance = close - level
    else:
        level = or_range.or_low
        broke = low < level
        close_confirmed = close < (level - margin)
        distance = level - close

    return {
        "broken": bool(broke),
        "broke_level": level,
        "close_confirmed": bool(close_confirmed),
        "distance": float(distance),
        "distance_atr_fraction": (distance / atr_value) if atr_value else None,
        "or_position": or_range.position_within(close),
        "direction": direction.upper(),
    }


def retest_succeeded(
    post_breakout_candles: Sequence[Candle],
    level: float,
    direction: str,
    *,
    tolerance_pct: float = 0.002,
) -> Tuple[bool, Optional[dt.datetime]]:
    """Did price come back to the breakout level and hold?

    LONG : a candle low touches/pierces the level and the candle closes back above.
    SHORT: a candle high touches the level and the candle closes back below.
    """
    tolerance = level * tolerance_pct
    for candle in post_breakout_candles:
        if direction.upper() == "LONG":
            if candle.low <= level + tolerance and candle.close > level:
                return True, candle.ts
        else:
            if candle.high >= level - tolerance and candle.close < level:
                return True, candle.ts
    return False, None


__all__ = [
    "OpeningRange",
    "build_opening_range",
    "opening_range_series",
    "opening_range_state",
    "retest_succeeded",
]
