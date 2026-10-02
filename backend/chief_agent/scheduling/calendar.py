"""Trading calendar.

Uses the OFFICIAL exchange state wherever possible rather than assuming local
clock arithmetic:

    GET /v2/market/holidays            (public, no auth)
    GET /v2/market/holidays/{date}     (public, no auth)
    GET /v2/market/timings/{date}      (public, no auth)
    GET /v2/market/status/{exchange}   (public, no auth)

Documented session statuses handled here: NORMAL_OPEN, NORMAL_CLOSE,
PRE_OPEN_START, PRE_OPEN_M_END, PRE_OPEN_END, CLOSING_START, CLOSING_END and the
Closing Auction Session statuses CT S_CLOSE / CAS_LM_START / CAS_M_STOP / CAS_STOP.

Falls back to the static NSE defaults (09:15-15:30 IST, weekdays) only when the
API is unreachable, and says so explicitly in :meth:`TradingCalendar.status`.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..logging_setup import get_logger
from ..timeutil import IST, is_weekend, market_close, market_open, now_ist, pre_open_start

log = get_logger(__name__, component="calendar")

#: Documented market-status values (appendix/market-status).
SESSION_STATUSES = {
    "NORMAL_OPEN",
    "NORMAL_CLOSE",
    "PRE_OPEN_START",
    "PRE_OPEN_M_END",
    "PRE_OPEN_END",
    "CLOSING_START",
    "CLOSING_END",
}
CAS_STATUSES = {"CTS_CLOSE", "CAS_LM_START", "CAS_M_STOP", "CAS_STOP"}

OPEN_STATUSES = {"NORMAL_OPEN"}
TRADABLE_STATUSES = {"NORMAL_OPEN", "CLOSING_START"}


@dataclass
class SessionInfo:
    date: dt.date
    is_trading_day: bool
    is_holiday: bool = False
    holiday_description: str = ""
    open_time: Optional[dt.datetime] = None
    close_time: Optional[dt.datetime] = None
    status: str = "UNKNOWN"
    cas_status: Optional[str] = None
    source: str = "static_default"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "is_trading_day": self.is_trading_day,
            "is_holiday": self.is_holiday,
            "holiday_description": self.holiday_description,
            "open_time": self.open_time.isoformat() if self.open_time else None,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "status": self.status,
            "cas_status": self.cas_status,
            "source": self.source,
            "notes": self.notes,
        }


class TradingCalendar:
    """Calendar and session-state service."""

    def __init__(self, market_data: Optional[Any] = None, exchange: str = "NSE") -> None:
        self.market_data = market_data
        self.exchange = exchange
        self._holidays: Dict[dt.date, Dict[str, Any]] = {}
        self._holiday_year: Optional[int] = None
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_at: Optional[dt.datetime] = None
        self._lock = threading.RLock()
        self._api_available = False
        # When the market-information API is unreachable we back off instead of
        # retrying on every dashboard refresh.
        self._api_backoff_until: Optional[dt.datetime] = None
        self._timings_cache: Dict[dt.date, Optional[tuple]] = {}

    # -------------------------------------------------------------- holidays
    def refresh_holidays(self, force: bool = False) -> int:
        """Load the official holiday list for the current year (cached in memory)."""
        today = now_ist().date()
        with self._lock:
            if not force and self._holiday_year == today.year and self._holidays:
                return len(self._holidays)
        if self.market_data is None or not self._api_available_now():
            with self._lock:
                self._holiday_year = today.year
            return len(self._holidays)
        try:
            rows = self.market_data.market_holidays()
        except Exception as exc:
            self._note_api_failure()
            log.warning("could not load market holidays", context={"error": str(exc)})
            return 0
        parsed: Dict[dt.date, Dict[str, Any]] = {}
        for row in rows:
            try:
                day = dt.date.fromisoformat(str(row.get("date")))
            except (TypeError, ValueError):
                continue
            parsed[day] = row
        with self._lock:
            self._holidays = parsed
            self._holiday_year = today.year
            self._api_available = bool(parsed)
            if not parsed:
                self._note_api_failure()
        log.info("market holidays loaded", context={"count": len(parsed), "year": today.year})
        return len(parsed)

    def is_holiday(self, day: dt.date) -> Optional[Dict[str, Any]]:
        if not self._holidays or self._holiday_year != day.year:
            self.refresh_holidays()
        return self._holidays.get(day)

    # ---------------------------------------------------------------- session
    def session(self, day: Optional[dt.date] = None) -> SessionInfo:
        day = day or now_ist().date()
        info = SessionInfo(date=day, is_trading_day=True)

        if is_weekend(day):
            info.is_trading_day = False
            info.notes.append("weekend")
            return info

        holiday = self.is_holiday(day)
        if holiday:
            closed_exchanges = [str(x).upper() for x in (holiday.get("closed_exchanges") or [])]
            open_exchanges = holiday.get("open_exchanges") or []
            exchange_codes = {self.exchange.upper(), "NFO" if self.exchange.upper() == "NSE" else self.exchange.upper()}
            if any(code in closed_exchanges for code in exchange_codes):
                info.is_trading_day = False
                info.is_holiday = True
                info.holiday_description = str(holiday.get("description") or "holiday")
                info.notes.append(f"exchange closed: {info.holiday_description}")
                return info
            # Special session (e.g. Muhurat trading): use the published timings.
            for entry in open_exchanges:
                if str(entry.get("exchange", "")).upper() in exchange_codes:
                    start = entry.get("start_time")
                    end = entry.get("end_time")
                    if start and end:
                        from ..timeutil import ensure_ist

                        info.open_time = ensure_ist(start)
                        info.close_time = ensure_ist(end)
                        info.notes.append("special session with modified timings")
                    break
            info.source = "upstox_holidays_api"
            if info.open_time is None:
                info.open_time, info.close_time = market_open(day), market_close(day)
            return info

        # Published timings for the day (handles early closes and ad-hoc changes)
        timings = self._exchange_timings(day)
        if timings:
            info.open_time, info.close_time = timings
            info.source = "upstox_timings_api"
        else:
            info.open_time, info.close_time = market_open(day), market_close(day)
            info.source = "static_default"
            info.notes.append("exchange timings API unavailable - using the standard 09:15-15:30 IST session")
        return info

    def _api_available_now(self) -> bool:
        return self._api_backoff_until is None or now_ist() >= self._api_backoff_until

    def _note_api_failure(self) -> None:
        self._api_backoff_until = now_ist() + dt.timedelta(minutes=15)

    def _exchange_timings(self, day: dt.date) -> Optional[tuple]:
        if day in self._timings_cache:
            return self._timings_cache[day]
        if self.market_data is None or not self._api_available_now():
            return None
        try:
            rows = self.market_data.market_timings(day)
        except Exception:
            self._note_api_failure()
            return None
        if not rows:
            self._note_api_failure()
            self._timings_cache[day] = None
            return None
        wanted = {"NSE", "NFO"} if self.exchange.upper() == "NSE" else {self.exchange.upper()}
        for row in rows:
            if str(row.get("exchange", "")).upper() in wanted:
                start = row.get("start_time")
                end = row.get("end_time")
                if start and end:
                    from ..timeutil import ensure_ist

                    resolved = (ensure_ist(start), ensure_ist(end))
                    self._timings_cache[day] = resolved
                    return resolved
        self._timings_cache[day] = None
        return None

    # ----------------------------------------------------------------- status
    def exchange_status(self, force: bool = False, max_age_seconds: float = 30.0) -> Dict[str, Any]:
        """Official exchange session status (cached briefly)."""
        with self._lock:
            if (
                not force
                and self._status_cache is not None
                and self._status_at is not None
                and (now_ist() - self._status_at).total_seconds() < max_age_seconds
            ):
                return dict(self._status_cache)
        if self.market_data is None or not self._api_available_now():
            return {"exchange": self.exchange, "status": "UNKNOWN", "source": "unavailable"}
        try:
            status = self.market_data.exchange_status(self.exchange)
            status["source"] = "upstox_market_status_api"
        except Exception as exc:
            self._note_api_failure()
            status = {"exchange": self.exchange, "status": "UNKNOWN", "error": str(exc), "source": "unavailable"}
        with self._lock:
            self._status_cache = status
            self._status_at = now_ist()
        return dict(status)

    # ------------------------------------------------------------------ queries
    def is_trading_day(self, day: Optional[dt.date] = None) -> bool:
        return self.session(day).is_trading_day

    def is_open(self, when: Optional[dt.datetime] = None) -> bool:
        when = when or now_ist()
        info = self.session(when.date())
        if not info.is_trading_day or info.open_time is None or info.close_time is None:
            return False
        local = when.astimezone(IST)
        return info.open_time <= local <= info.close_time

    def is_tradable_now(self, when: Optional[dt.datetime] = None) -> bool:
        """True only when the official session says trading is actually allowed."""
        status = self.exchange_status().get("status", "UNKNOWN")
        if status in TRADABLE_STATUSES:
            return True
        if status == "UNKNOWN":
            # Fall back to the clock, but only inside the standard session.
            return self.is_open(when)
        return False

    def session_phase(self, when: Optional[dt.datetime] = None) -> str:
        when = (when or now_ist()).astimezone(IST)
        info = self.session(when.date())
        if not info.is_trading_day:
            return "CLOSED"
        if when < pre_open_start(when.date()):
            return "PRE_MARKET_CLOSED"
        if info.open_time and when < info.open_time:
            return "PRE_OPEN"
        if info.close_time and when <= info.close_time:
            return "NORMAL"
        return "CLOSING_AUCTION_OR_CLOSED"

    def next_trading_day(self, from_day: Optional[dt.date] = None, max_lookahead: int = 15) -> Optional[dt.date]:
        day = (from_day or now_ist().date()) + dt.timedelta(days=1)
        for _ in range(max_lookahead):
            if self.is_trading_day(day):
                return day
            day += dt.timedelta(days=1)
        return None

    def previous_trading_day(self, from_day: Optional[dt.date] = None, max_lookback: int = 15) -> Optional[dt.date]:
        day = (from_day or now_ist().date()) - dt.timedelta(days=1)
        for _ in range(max_lookback):
            if self.is_trading_day(day):
                return day
            day -= dt.timedelta(days=1)
        return None

    def trading_days(self, start: dt.date, end: dt.date) -> List[dt.date]:
        out: List[dt.date] = []
        cursor = start
        while cursor <= end:
            if self.is_trading_day(cursor):
                out.append(cursor)
            cursor += dt.timedelta(days=1)
        return out

    def status(self) -> Dict[str, Any]:
        today = now_ist().date()
        info = self.session(today)
        return {
            "today": today.isoformat(),
            "session": info.to_dict(),
            "phase": self.session_phase(),
            "open_now": self.is_open(),
            "tradable_now": self.is_tradable_now(),
            "exchange_status": self.exchange_status(),
            "holidays_loaded": len(self._holidays),
            "holiday_year": self._holiday_year,
            "api_available": self._api_available,
            "next_trading_day": (self.next_trading_day() or dt.date.today()).isoformat(),
        }


__all__ = ["TradingCalendar", "SessionInfo", "SESSION_STATUSES", "CAS_STATUSES", "TRADABLE_STATUSES"]
