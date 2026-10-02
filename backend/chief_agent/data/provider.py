"""Unified market-data provider.

Two interchangeable backends behind one interface:

* :class:`UpstoxDataBackend` - the real Upstox Historical Candle V3 API. Used
  whenever an access token is available.
* :class:`SimulatedDataBackend` - deterministic simulated sessions, so the whole
  system (backtests, scanner, dashboard, tests) runs with no token, no network
  and no cost. Every payload it produces is tagged ``SIMULATED`` and the UI
  shows a banner; simulated data is never presented as real market data.

The caller never needs to know which backend is active - but it CAN always ask,
via :meth:`MarketDataProvider.data_source`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..broker.upstox_market_data import Candle, TIMEFRAME_TO_UNIT, UpstoxMarketData
from ..logging_setup import get_logger
from ..settings import Settings, get_settings
from ..timeutil import IST, ist_date, now_ist
from . import synthetic

log = get_logger(__name__, component="data_provider")

DATA_SOURCE_LIVE = "UPSTOX"
DATA_SOURCE_SIMULATED = "SIMULATED"


@dataclass
class ProviderStatus:
    source: str
    live_available: bool
    reason: str = ""
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "live_available": self.live_available,
            "reason": self.reason,
            "last_error": self.last_error,
            "is_simulated": self.source == DATA_SOURCE_SIMULATED,
        }


class SimulatedDataBackend:
    """Deterministic simulated candles. Clearly labelled everywhere."""

    source = DATA_SOURCE_SIMULATED

    def __init__(self, seed_salt: str = "") -> None:
        self.seed_salt = seed_salt

    def history(
        self,
        instrument_key: str,
        timeframe: str,
        start: dt.date,
        end: dt.date,
        *,
        symbol: Optional[str] = None,
        interval_minutes: int = 1,
    ) -> List[Candle]:
        name = symbol or instrument_key.split("|")[-1] or instrument_key
        is_index = instrument_key.startswith("__index__") or instrument_key.startswith("NSE_INDEX|")
        if is_index:
            candles = synthetic.generate_index_history(name, start, end, interval_minutes=interval_minutes)
        else:
            candles = synthetic.generate_history(name, start, end, interval_minutes=interval_minutes)
        for candle in candles:
            candle.instrument_key = instrument_key
            candle.timeframe = timeframe
        if interval_minutes > 1:
            candles = synthetic.aggregate_candles(candles, interval_minutes, timeframe)
        return candles


class UpstoxDataBackend:
    """Live Upstox Historical Candle V3 backend."""

    source = DATA_SOURCE_LIVE

    def __init__(self, market_data: UpstoxMarketData) -> None:
        self.market_data = market_data
        self.last_error: Optional[str] = None

    def history(
        self,
        instrument_key: str,
        timeframe: str,
        start: dt.date,
        end: dt.date,
        *,
        symbol: Optional[str] = None,
        interval_minutes: int = 1,
    ) -> List[Candle]:
        try:
            candles = self.market_data.fetch_range(instrument_key, timeframe, start, end)
            self.last_error = None
            return candles
        except Exception as exc:
            self.last_error = str(exc)
            log.warning(
                "upstox history fetch failed",
                context={"instrument": instrument_key, "timeframe": timeframe, "error": str(exc)},
            )
            return []


class MarketDataProvider:
    """Chooses the backend and exposes one clean interface."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        broker: Optional[Any] = None,
        force_simulated: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.broker = broker
        self.force_simulated = force_simulated
        self._live: Optional[UpstoxDataBackend] = None
        self._simulated = SimulatedDataBackend(seed_salt="")

        if not force_simulated and broker is not None and getattr(broker, "has_token", False):
            self._live = UpstoxDataBackend(broker.market_data)
            self._active = self._live
            self._reason = "Upstox access token is available"
        else:
            self._active = self._simulated
            self._reason = (
                "forced simulated mode"
                if force_simulated
                else "no Upstox access token - running on clearly-labelled simulated data"
            )

    # ------------------------------------------------------------------ status
    @property
    def data_source(self) -> str:
        return self._active.source

    @property
    def is_simulated(self) -> bool:
        return self._active.source == DATA_SOURCE_SIMULATED

    @property
    def live_available(self) -> bool:
        return self._live is not None

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            source=self.data_source,
            live_available=self.live_available,
            reason=self._reason,
            last_error=getattr(self._live, "last_error", None) if self._live else None,
        )

    def switch_to_live(self, broker: Any) -> bool:
        if broker is None or not getattr(broker, "has_token", False):
            return False
        self._live = UpstoxDataBackend(broker.market_data)
        self._active = self._live
        self.broker = broker
        self._reason = "switched to Upstox live data"
        return True

    # ------------------------------------------------------------------- reads
    def history(
        self,
        instrument_key: str,
        timeframe: str,
        start: dt.date,
        end: dt.date,
        *,
        symbol: Optional[str] = None,
    ) -> List[Candle]:
        interval_minutes = _interval_minutes(timeframe)
        candles = self._active.history(
            instrument_key, timeframe, start, end, symbol=symbol, interval_minutes=interval_minutes
        )
        return candles

    def history_many(
        self,
        instruments: Sequence[Mapping[str, Any]],
        timeframe: str,
        start: dt.date,
        end: dt.date,
        *,
        progress: Optional[Any] = None,
    ) -> Dict[str, List[Candle]]:
        """Fetch many instruments. ``instruments`` items need ``instrument_key``
        and optionally ``symbol``. Falls back to simulated data per-instrument if
        a live fetch returns nothing, so a single bad symbol cannot break a run.
        """
        out: Dict[str, List[Candle]] = {}
        for index, item in enumerate(instruments, start=1):
            key = item.get("instrument_key") or item.get("symbol")
            if not key:
                continue
            candles = self.history(key, timeframe, start, end, symbol=item.get("symbol"))
            if not candles and self._live is not None:
                log.warning("live fetch empty; falling back to simulated for this instrument",
                            context={"instrument": key})
                candles = self._simulated.history(
                    key, timeframe, start, end, symbol=item.get("symbol"), interval_minutes=_interval_minutes(timeframe)
                )
                for candle in candles:
                    candle.instrument_key = key
            out[key] = candles
            if progress is not None:
                try:
                    progress(index, len(instruments), key, len(candles))
                except Exception:  # pragma: no cover
                    pass
        return out

    # ------------------------------------------------------------------ quotes
    def quotes(self, instrument_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Live quotes when available; otherwise the latest simulated bar per key."""
        if self._live is not None:
            try:
                return self._live.market_data.full_quote(instrument_keys)
            except Exception as exc:
                log.warning("quote fetch failed; using simulated snapshot", context={"error": str(exc)})

        today = now_ist().date()
        out: Dict[str, Dict[str, Any]] = {}
        for key in instrument_keys:
            session = synthetic.generate_session(key.split("|")[-1] or key, today)
            last = session.candles[-1]
            spread = max(0.05, last.close * 0.0004)
            out[key] = {
                "instrument_key": key,
                "last_price": last.close,
                "ltp": last.close,
                "ohlc": {"open": session.open, "high": session.high, "low": session.low, "close": last.close},
                "volume": sum(c.volume for c in session.candles),
                "depth": {
                    "buy": [{"price": round(last.close - spread / 2, 2), "quantity": 500}],
                    "sell": [{"price": round(last.close + spread / 2, 2), "quantity": 500}],
                },
                "_simulated": True,
            }
        return out


def _interval_minutes(timeframe: str) -> int:
    from ..broker.upstox_market_data import TIMEFRAME_MINUTES

    return TIMEFRAME_MINUTES.get(timeframe, 1)


__all__ = [
    "MarketDataProvider",
    "UpstoxDataBackend",
    "SimulatedDataBackend",
    "ProviderStatus",
    "DATA_SOURCE_LIVE",
    "DATA_SOURCE_SIMULATED",
]
