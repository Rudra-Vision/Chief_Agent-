"""Canonical time handling for the whole system.

Rules enforced everywhere in this codebase:

* The canonical timezone is ``Asia/Kolkata`` (IST).
* Naive ``datetime`` objects are NEVER stored or compared. Every helper here
  returns timezone-aware datetimes, and :func:`ensure_ist` converts/attaches IST.
* Broker timestamps (epoch millis or ISO-8601 with offsets) are normalised once,
  at the edge, and never re-interpreted downstream.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional, Union
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = dt.timezone.utc

TimestampLike = Union[int, float, str, dt.datetime, dt.date]


def now_ist() -> dt.datetime:
    """Current time, timezone-aware, in Asia/Kolkata."""
    return dt.datetime.now(tz=IST)


def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def ensure_ist(value: TimestampLike) -> dt.datetime:
    """Normalise any supported timestamp representation to an IST-aware datetime.

    * naive datetime  -> assumed to already be wall-clock IST (documented behaviour
      of Upstox candle timestamps) and tagged with IST.
    * aware datetime  -> converted to IST.
    * int/float       -> treated as epoch (seconds if < 1e11 else milliseconds).
    * str             -> ISO-8601, optionally with offset; naive strings assumed IST.
    """
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=IST)
        return value.astimezone(IST)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=IST)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if abs(seconds) >= 1e11:  # milliseconds
            seconds /= 1000.0
        return dt.datetime.fromtimestamp(seconds, tz=IST)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(text)
        return ensure_ist(parsed)
    raise TypeError(f"unsupported timestamp type: {type(value)!r}")


def to_epoch_ms(value: TimestampLike) -> int:
    """Epoch milliseconds for a normalised timestamp (used by Upstox payloads)."""
    return int(ensure_ist(value).timestamp() * 1000)


def ist_date(value: TimestampLike) -> dt.date:
    return ensure_ist(value).date()


def trading_day(value: TimestampLike) -> dt.date:
    """The trading session date a timestamp belongs to (IST calendar date)."""
    return ist_date(value)


def market_open(d: dt.date) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)


def market_close(d: dt.date) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, 15, 30, tzinfo=IST)


def pre_open_start(d: dt.date) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, 9, 0, tzinfo=IST)


def minutes_from_open(value: TimestampLike) -> float:
    """Minutes elapsed since the 09:15 IST open of the timestamp's session."""
    ts = ensure_ist(value)
    return (ts - market_open(ts.date())).total_seconds() / 60.0


def is_weekend(d: dt.date) -> bool:
    return d.weekday() >= 5


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    """Inclusive forward date range."""
    if end < start:
        return
    cur = start
    one_day = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one_day


def iso(value: TimestampLike) -> str:
    """Canonical ISO-8601 string with IST offset, for storage and JSON output."""
    return ensure_ist(value).isoformat()


def parse_iso(text: str) -> dt.datetime:
    return ensure_ist(text)


def humanize_delta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def session_of(value: TimestampLike) -> str:
    """Coarse session label for a timestamp, based on official NSE windows."""
    ts = ensure_ist(value)
    d = ts.date()
    if ts < pre_open_start(d):
        return "PRE_MARKET_CLOSED"
    if ts < market_open(d):
        return "PRE_OPEN"
    if ts < market_close(d):
        return "NORMAL"
    return "CLOSED"


__all__ = [
    "IST",
    "UTC",
    "now_ist",
    "now_utc",
    "ensure_ist",
    "to_epoch_ms",
    "ist_date",
    "trading_day",
    "market_open",
    "market_close",
    "pre_open_start",
    "minutes_from_open",
    "is_weekend",
    "daterange",
    "iso",
    "parse_iso",
    "humanize_delta",
    "session_of",
]
