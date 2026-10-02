"""Candle storage and retrieval.

Rules
-----
* Source market data is **append-only**: existing candles are never overwritten
  or deleted. Re-ingesting an overlapping range only fills gaps.
* Derived (aggregated) candles are tagged ``is_derived=True`` and can always be
  regenerated from the 1-minute source, which is kept intact.
* Every read is bounded by an explicit date range so a research job cannot
  accidentally pull a decade of data into memory.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import and_, delete, func, select
from sqlalchemy.orm import Session

from ..broker.upstox_market_data import Candle
from ..logging_setup import get_logger
from ..timeutil import IST, ist_date
from ..data.schema import Candle as CandleRow

log = get_logger(__name__, component="candle_store")


@dataclass
class IngestResult:
    instrument_key: str
    timeframe: str
    received: int
    inserted: int
    duplicates: int
    invalid: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _validate(candle: Candle) -> Optional[str]:
    """Return a rejection reason, or None if the candle is usable."""
    if candle.high < candle.low:
        return "high < low"
    if candle.high < max(candle.open, candle.close) - 1e-9:
        return "high below body"
    if candle.low > min(candle.open, candle.close) + 1e-9:
        return "low above body"
    if candle.volume < 0:
        return "negative volume"
    if candle.close <= 0 or candle.open <= 0:
        return "non-positive price"
    return None


def upsert_candles(
    session: Session,
    instrument_key: str,
    timeframe: str,
    candles: Sequence[Candle],
    *,
    source: str = "upstox_v3",
    is_derived: bool = False,
    batch_size: int = 2000,
) -> IngestResult:
    """Insert candles that are not already present. Never modifies existing rows."""
    result = IngestResult(instrument_key=instrument_key, timeframe=timeframe, received=len(candles), inserted=0, duplicates=0, invalid=0)
    if not candles:
        return result

    valid: List[Candle] = []
    for candle in candles:
        reason = _validate(candle)
        if reason:
            result.invalid += 1
            continue
        valid.append(candle)
    if not valid:
        return result

    # Existing timestamps inside the requested span
    times = [c.ts for c in valid]
    existing = set(
        session.execute(
            select(CandleRow.ts).where(
                CandleRow.instrument_key == instrument_key,
                CandleRow.timeframe == timeframe,
                CandleRow.ts >= min(times),
                CandleRow.ts <= max(times),
            )
        ).scalars()
    )
    existing_keys = {_key(ts) for ts in existing}

    rows: List[CandleRow] = []
    for candle in valid:
        if _key(candle.ts) in existing_keys:
            result.duplicates += 1
            continue
        rows.append(
            CandleRow(
                instrument_key=instrument_key,
                timeframe=timeframe,
                ts=candle.ts,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                open_interest=candle.open_interest,
                source=source,
                is_derived=is_derived,
            )
        )

    for start in range(0, len(rows), batch_size):
        session.add_all(rows[start : start + batch_size])
        session.flush()
    result.inserted = len(rows)
    return result


def _key(ts: dt.datetime) -> str:
    """Timezone-safe key so IST-aware and naive-but-IST datetimes match."""
    aware = ts.astimezone(IST) if ts.tzinfo else ts.replace(tzinfo=IST)
    return aware.strftime("%Y-%m-%dT%H:%M")


def load_candles(
    session: Session,
    instrument_key: str,
    timeframe: str,
    start: dt.date,
    end: dt.date,
    *,
    limit: int = 2_000_000,
) -> List[Candle]:
    start_dt = dt.datetime(start.year, start.month, start.day, tzinfo=IST)
    end_dt = dt.datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=IST)
    rows = (
        session.execute(
            select(CandleRow)
            .where(
                CandleRow.instrument_key == instrument_key,
                CandleRow.timeframe == timeframe,
                CandleRow.ts >= start_dt,
                CandleRow.ts <= end_dt,
            )
            .order_by(CandleRow.ts)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        Candle(
            ts=row.ts.astimezone(IST) if row.ts.tzinfo else row.ts.replace(tzinfo=IST),
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            open_interest=row.open_interest,
            instrument_key=row.instrument_key,
            timeframe=row.timeframe,
        )
        for row in rows
    ]


def coverage(
    session: Session,
    instrument_key: str,
    timeframe: str = "1m",
) -> Dict[str, Any]:
    """What date range and how many candles do we already hold locally?"""
    row = session.execute(
        select(
            func.count(CandleRow.id),
            func.min(CandleRow.ts),
            func.max(CandleRow.ts),
        ).where(CandleRow.instrument_key == instrument_key, CandleRow.timeframe == timeframe)
    ).one()
    count, minimum, maximum = row
    days = set()
    if minimum and maximum:
        stmt = (
            select(func.distinct(func.strftime("%Y-%m-%d", CandleRow.ts)))
            .where(CandleRow.instrument_key == instrument_key, CandleRow.timeframe == timeframe)
        )
        try:
            days = {value for (value,) in session.execute(stmt)}
        except Exception:  # non-SQLite dialects
            days = set()
    return {
        "instrument_key": instrument_key,
        "timeframe": timeframe,
        "count": int(count or 0),
        "first": minimum.isoformat() if minimum else None,
        "last": maximum.isoformat() if maximum else None,
        "trading_days": len(days) if days else None,
    }


def missing_ranges(
    session: Session,
    instrument_key: str,
    timeframe: str,
    start: dt.date,
    end: dt.date,
    holidays: Optional[Iterable[dt.date]] = None,
) -> List[Tuple[dt.date, dt.date]]:
    """Business-day gaps in the local cache, so we never re-download data we hold."""
    holidays = set(holidays or ())
    try:
        stmt = select(func.distinct(func.strftime("%Y-%m-%d", CandleRow.ts))).where(
            CandleRow.instrument_key == instrument_key, CandleRow.timeframe == timeframe
        )
        have = {value for (value,) in session.execute(stmt)}
    except Exception:
        have = set()

    ranges: List[Tuple[dt.date, dt.date]] = []
    cursor = start
    gap_start: Optional[dt.date] = None
    one_day = dt.timedelta(days=1)
    while cursor <= end:
        is_session = cursor.weekday() < 5 and cursor not in holidays
        present = cursor.isoformat() in have
        if is_session and not present and gap_start is None:
            gap_start = cursor
        elif (not is_session or present) and gap_start is not None:
            ranges.append((gap_start, cursor - one_day))
            gap_start = None
        cursor += one_day
    if gap_start is not None:
        ranges.append((gap_start, end))
    return ranges


def delete_derived(session: Session, instrument_key: Optional[str] = None, timeframe: Optional[str] = None) -> int:
    """Remove DERIVED candles only. Source data is never deleted by this function."""
    stmt = delete(CandleRow).where(CandleRow.is_derived.is_(True))
    if instrument_key:
        stmt = stmt.where(CandleRow.instrument_key == instrument_key)
    if timeframe:
        stmt = stmt.where(CandleRow.timeframe == timeframe)
    result = session.execute(stmt)
    return int(result.rowcount or 0)


__all__ = ["upsert_candles", "load_candles", "coverage", "missing_ranges", "delete_derived", "IngestResult"]
