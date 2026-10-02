"""Technical indicators.

Implemented with explicit, auditable loops/series so there is no ambiguity about
warm-up periods, and so **no indicator ever uses future data**. Every function
here is causal: the value at index ``i`` depends only on inputs at ``<= i``.

Primary implementation is NumPy on plain sequences, because the backtester feeds
it candle-by-candle in an event loop; :func:`to_dataframe` bridges to pandas for
research/dashboard work.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..timeutil import IST, minutes_from_open


# --------------------------------------------------------------------------- #
# Basic series helpers
# --------------------------------------------------------------------------- #
def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0:
        return out
    running = 0.0
    for i, value in enumerate(values):
        running += float(value)
        if i >= period:
            running -= float(values[i - period])
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Exponential moving average seeded with an SMA (never a guessed seed)."""
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    multiplier = 2.0 / (period + 1.0)
    seed = sum(float(v) for v in values[:period]) / period
    out[period - 1] = seed
    previous = seed
    for i in range(period, len(values)):
        previous = (float(values[i]) - previous) * multiplier + previous
        out[i] = previous
    return out


def rolling_std(values: Sequence[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 1:
        return out
    for i in range(period - 1, len(values)):
        window = np.asarray(values[i - period + 1 : i + 1], dtype=float)
        out[i] = float(window.std(ddof=0))
    return out


def rolling_max(values: Sequence[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        start = max(0, i - period + 1)
        if i - start + 1 < period:
            continue
        out[i] = max(float(v) for v in values[start : i + 1])
    return out


def rolling_min(values: Sequence[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        start = max(0, i - period + 1)
        if i - start + 1 < period:
            continue
        out[i] = min(float(v) for v in values[start : i + 1])
    return out


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(len(highs)):
        high = float(highs[i])
        low = float(lows[i])
        if i == 0:
            out.append(high - low)
        else:
            previous_close = float(closes[i - 1])
            out.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return out


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Wilder's Average True Range (the standard smoothing used by NSE traders)."""
    tr = true_range(highs, lows, closes)
    out: List[Optional[float]] = [None] * len(tr)
    if len(tr) < period:
        return out
    first = sum(tr[:period]) / period
    out[period - 1] = first
    previous = first
    for i in range(period, len(tr)):
        previous = (previous * (period - 1) + tr[i]) / period
        out[i] = previous
    return out


def atr_pct(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """ATR expressed as a fraction of price - the volatility measure used for filters."""
    values = atr(highs, lows, closes, period)
    return [
        (value / float(closes[i])) if (value is not None and float(closes[i]) > 0) else None
        for i, value in enumerate(values)
    ]


def rsi(values: Sequence[float], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = float(values[i]) - float(values[i - 1])
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        delta = float(values[i]) - float(values[i - 1])
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> List[Optional[float]]:
    """Average Directional Index (trend strength, direction-agnostic)."""
    n = len(highs)
    out: List[Optional[float]] = [None] * n
    if n < period * 2 + 1:
        return out

    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up = float(highs[i]) - float(highs[i - 1])
        down = float(lows[i - 1]) - float(lows[i])
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    tr = true_range(highs, lows, closes)
    smoothed_tr = sum(tr[1 : period + 1])
    smoothed_plus = sum(plus_dm[1 : period + 1])
    smoothed_minus = sum(minus_dm[1 : period + 1])
    dx_values: List[Optional[float]] = [None] * n

    for i in range(period, n):
        if i > period:
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr[i]
            smoothed_plus = smoothed_plus - (smoothed_plus / period) + plus_dm[i]
            smoothed_minus = smoothed_minus - (smoothed_minus / period) + minus_dm[i]
        if smoothed_tr <= 0:
            continue
        plus_di = 100.0 * (smoothed_plus / smoothed_tr)
        minus_di = 100.0 * (smoothed_minus / smoothed_tr)
        denominator = plus_di + minus_di
        dx_values[i] = 0.0 if denominator == 0 else 100.0 * abs(plus_di - minus_di) / denominator

    valid = [value for value in dx_values[period : period * 2] if value is not None]
    if len(valid) < period:
        return out
    previous = sum(valid[:period]) / period
    out[period * 2 - 1] = previous
    for i in range(period * 2, n):
        if dx_values[i] is None:
            continue
        previous = (previous * (period - 1) + dx_values[i]) / period
        out[i] = previous
    return out


def vwap_session(
    timestamps: Sequence[dt.datetime],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> List[Optional[float]]:
    """Session VWAP, reset at the start of each IST trading day.

    Uses the standard typical price ``(H + L + C) / 3``. Returns ``None`` while
    cumulative volume is zero, so a missing-volume bar cannot fabricate a VWAP.
    """
    out: List[Optional[float]] = [None] * len(timestamps)
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    current_day: Optional[dt.date] = None

    for i, ts in enumerate(timestamps):
        local = ts.astimezone(IST) if ts.tzinfo else ts
        day = local.date()
        if current_day != day:
            current_day = day
            cumulative_pv = 0.0
            cumulative_volume = 0.0
        typical = (float(highs[i]) + float(lows[i]) + float(closes[i])) / 3.0
        volume = float(volumes[i])
        cumulative_pv += typical * volume
        cumulative_volume += volume
        out[i] = (cumulative_pv / cumulative_volume) if cumulative_volume > 0 else None
    return out


def anchored_vwap(
    timestamps: Sequence[dt.datetime],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    anchor_index: int = 0,
) -> List[Optional[float]]:
    """VWAP anchored at ``anchor_index`` (e.g. the session open) - no daily reset."""
    out: List[Optional[float]] = [None] * len(timestamps)
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    for i in range(len(timestamps)):
        if i < anchor_index:
            continue
        typical = (float(highs[i]) + float(lows[i]) + float(closes[i])) / 3.0
        volume = float(volumes[i])
        cumulative_pv += typical * volume
        cumulative_volume += volume
        out[i] = (cumulative_pv / cumulative_volume) if cumulative_volume > 0 else None
    return out


def relative_volume(
    volumes: Sequence[float],
    timestamps: Sequence[dt.datetime],
    *,
    lookback_days: int = 20,
    same_time_window_minutes: int = 60,
    min_prior_days: int = 5,
) -> List[Optional[float]]:
    """Relative volume versus the same intraday window on prior sessions.

    Compares the volume accumulated over the last ``same_time_window_minutes``
    with the average of the equivalent window on the previous ``lookback_days``
    sessions. Only *completed* prior sessions are used, so there is no
    look-ahead.

    Implemented with per-session cumulative sums and ``searchsorted`` lookups, so
    a year of one-minute bars costs one linear pass plus a small constant factor
    per bar (rather than a nested scan).
    """
    n = len(volumes)
    out: List[Optional[float]] = [None] * n
    if n == 0:
        return out

    # --- per-session state ---------------------------------------------------
    day_minutes: Dict[dt.date, np.ndarray] = {}
    day_cumulative: Dict[dt.date, np.ndarray] = {}
    day_of_index: List[dt.date] = []
    index_in_day: List[int] = []

    current_day: Optional[dt.date] = None
    minutes_buffer: List[float] = []
    volume_buffer: List[float] = []

    def flush() -> None:
        if current_day is None or not minutes_buffer:
            return
        minutes_array = np.asarray(minutes_buffer, dtype=float)
        cumulative = np.cumsum(np.asarray(volume_buffer, dtype=float))
        day_minutes[current_day] = minutes_array
        day_cumulative[current_day] = cumulative

    for index, ts in enumerate(timestamps):
        local = ts.astimezone(IST) if ts.tzinfo else ts
        day = local.date()
        if day != current_day:
            flush()
            current_day = day
            minutes_buffer = []
            volume_buffer = []
        minutes_buffer.append(minutes_from_open(local))
        volume_buffer.append(float(volumes[index]))
        day_of_index.append(day)
        index_in_day.append(len(minutes_buffer) - 1)
    flush()

    ordered_days = sorted(day_minutes)
    day_position = {day: position for position, day in enumerate(ordered_days)}

    for index in range(n):
        day = day_of_index[index]
        position = day_position.get(day)
        if position is None or position == 0:
            continue
        current_minute = minutes_from_open(
            timestamps[index].astimezone(IST) if timestamps[index].tzinfo else timestamps[index]
        )
        if current_minute < same_time_window_minutes:
            continue

        prior_days = ordered_days[max(0, position - lookback_days) : position]
        if len(prior_days) < min_prior_days:
            continue

        window_start = current_minute - same_time_window_minutes

        # This session's window volume (inclusive of the current bar)
        cumulative = day_cumulative[day]
        minutes_array = day_minutes[day]
        idx = index_in_day[index]
        start_index = int(np.searchsorted(minutes_array, window_start, side="left")) - 1
        window_volume = float(cumulative[idx] - (cumulative[start_index] if start_index >= 0 else 0.0))

        # Mean of the same window across prior sessions
        benchmarks: List[float] = []
        for prior_day in prior_days:
            prior_minutes = day_minutes[prior_day]
            prior_cumulative = day_cumulative[prior_day]
            lo = int(np.searchsorted(prior_minutes, window_start, side="left")) - 1
            hi = int(np.searchsorted(prior_minutes, current_minute, side="right")) - 1
            if hi <= lo:
                continue
            total = prior_cumulative[hi] - (prior_cumulative[lo] if lo >= 0 else 0.0)
            benchmarks.append(float(total))
        if not benchmarks:
            continue
        average = sum(benchmarks) / len(benchmarks)
        out[index] = (window_volume / average) if average > 0 else None

    return out


def cumulative_volume_delta(
    closes: Sequence[float],
    volumes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
) -> List[float]:
    """Simple buy/sell pressure proxy in [-1, 1] per bar."""
    out: List[float] = []
    for i in range(len(closes)):
        high = float(highs[i])
        low = float(lows[i])
        close = float(closes[i])
        span = high - low
        if span <= 0:
            out.append(0.0)
            continue
        out.append(((close - low) - (high - close)) / span)
    return out


def supertrend_like(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> List[Tuple[Optional[float], Optional[int]]]:
    """ATR-based trailing trend line returning ``(value, direction)``.

    ``direction`` is ``1`` for an uptrend and ``-1`` for a downtrend. Used only as
    a regime/feature input, never as a standalone entry rule.
    """
    n = len(closes)
    atr_values = atr(highs, lows, closes, period)
    out: List[Tuple[Optional[float], Optional[int]]] = [(None, None)] * n
    upper = lower = None
    direction = 1

    for i in range(n):
        if atr_values[i] is None:
            continue
        mid = (float(highs[i]) + float(lows[i])) / 2.0
        band = multiplier * atr_values[i]
        basic_upper = mid + band
        basic_lower = mid - band

        if upper is None:
            upper, lower = basic_upper, basic_lower
            out[i] = (lower, 1)
            continue

        upper = basic_upper if (basic_upper < upper or float(closes[i - 1]) > upper) else upper
        lower = basic_lower if (basic_lower > lower or float(closes[i - 1]) < lower) else lower

        if float(closes[i]) > upper:
            direction = 1
        elif float(closes[i]) < lower:
            direction = -1
        out[i] = (lower if direction == 1 else upper, direction)
    return out


# --------------------------------------------------------------------------- #
# Candle quality / structure helpers
# --------------------------------------------------------------------------- #
def candle_body_pct(open_price: float, high: float, low: float, close: float) -> float:
    """Body size as a fraction of the candle range (0 = doji, 1 = marubozu)."""
    span = float(high) - float(low)
    if span <= 0:
        return 0.0
    return abs(float(close) - float(open_price)) / span


def is_bullish_engulfing(previous: Tuple[float, float], current: Tuple[float, float]) -> bool:
    prev_open, prev_close = previous
    open_price, close = current
    return prev_close < prev_open and close > open_price and close >= prev_open and open_price <= prev_close


def swing_high(highs: Sequence[float], end_index: int, lookback: int) -> Optional[float]:
    start = max(0, end_index - lookback + 1)
    if end_index - start + 1 < lookback:
        return None
    return max(float(v) for v in highs[start : end_index + 1])


def swing_low(lows: Sequence[float], end_index: int, lookback: int) -> Optional[float]:
    start = max(0, end_index - lookback + 1)
    if end_index - start + 1 < lookback:
        return None
    return min(float(v) for v in lows[start : end_index + 1])


def round_to_tick(price: float, tick: float, mode: str = "nearest") -> float:
    """Round a price to a valid exchange tick (Indian equities: Re 0.05)."""
    tick = float(tick) or 0.05
    if tick <= 0:
        return float(price)
    steps = float(price) / tick
    if mode == "down":
        rounded = math.floor(steps)
    elif mode == "up":
        rounded = math.ceil(steps)
    else:
        rounded = round(steps)
    return round(rounded * tick, 4)


def percentile_rank(values: Sequence[float], value: float) -> float:
    """Fraction of historical values at or below ``value`` (0..1)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.5
    below = sum(1 for v in clean if v <= value)
    return below / len(clean)


__all__ = [
    "sma",
    "ema",
    "rolling_std",
    "rolling_max",
    "rolling_min",
    "true_range",
    "atr",
    "atr_pct",
    "rsi",
    "adx",
    "vwap_session",
    "anchored_vwap",
    "relative_volume",
    "cumulative_volume_delta",
    "supertrend_like",
    "candle_body_pct",
    "is_bullish_engulfing",
    "swing_high",
    "swing_low",
    "round_to_tick",
    "percentile_rank",
]
