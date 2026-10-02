"""Market data: Historical Candle V3, Intraday Candle V3, Market Quote V3,
Market Information (holidays / timings / exchange status).

Verified against the current official documentation:

* Historical Candle V3
      GET /v3/historical-candle/:instrument_key/:unit/:interval/:to_date/:from_date
      units: minutes (1..300), hours (1..5), days, weeks, months
      minutes available from January 2022; max window 1 month for intervals 1-15
      and 1 quarter for intervals > 15. days available from January 2000 with a
      1 decade max window.
* Intraday Candle V3
      GET /v3/historical-candle/intraday/:instrument_key/:unit/:interval
* Market Quote V3
      GET /v3/market-quote/ltp        (<= 500 instrument keys)
      GET /v3/market-quote/ohlc
      GET /v3/market-quote/full
* Market Information (public, cached, no authentication)
      GET /v2/market/holidays
      GET /v2/market/holidays/:date
      GET /v2/market/timings/:date
      GET /v2/market/status/:exchange

The response shape for candles is
``[timestamp, open, high, low, close, volume, open_interest]`` where the
timestamp is the START time of the candle in IST.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from ..logging_setup import get_logger
from ..settings import CACHE_DIR, Settings, get_config_store, get_settings
from ..timeutil import IST, daterange, ensure_ist, ist_date, now_ist
from .upstox_client import ApiResponse, Endpoints, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="market_data")

# Supported timeframe labels -> (unit, interval) for the V3 API
TIMEFRAME_TO_UNIT: Dict[str, Tuple[str, int]] = {
    "1m": ("minutes", 1),
    "3m": ("minutes", 3),
    "5m": ("minutes", 5),
    "15m": ("minutes", 15),
    "30m": ("minutes", 30),
    "60m": ("hours", 1),
    "1h": ("hours", 1),
    "1d": ("days", 1),
    "1w": ("weeks", 1),
    "1M": ("months", 1),
}

TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 375,
    "1w": 375 * 5,
    "1M": 375 * 21,
}


@dataclass
class Candle:
    """A single OHLC candle with an IST-aware start timestamp."""

    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    open_interest: float = 0.0
    instrument_key: str = ""
    timeframe: str = "1m"

    def as_tuple(self) -> Tuple[dt.datetime, float, float, float, float, float, float]:
        return (self.ts, self.open, self.high, self.low, self.close, self.volume, self.open_interest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "open_interest": self.open_interest,
        }


def parse_candles(payload: Any, instrument_key: str = "", timeframe: str = "1m") -> List[Candle]:
    """Convert a raw Upstox candle payload into :class:`Candle` objects.

    Rows that are malformed or that violate OHLC sanity (high < low, etc.) are
    dropped here and reported later by the data-quality engine; they are never
    silently "fixed".
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    raw = data.get("candles")
    if not isinstance(raw, list):
        return []

    out: List[Candle] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            ts = ensure_ist(row[0])
            o, h, l, c = (float(row[i]) for i in range(1, 5))
        except (TypeError, ValueError, IndexError):
            continue
        volume = float(row[5]) if len(row) > 5 and row[5] is not None else 0.0
        oi = float(row[6]) if len(row) > 6 and row[6] is not None else 0.0
        if h < l or not all(_is_finite(v) for v in (o, h, l, c)):
            continue
        out.append(
            Candle(
                ts=ts,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=volume,
                open_interest=oi,
                instrument_key=instrument_key,
                timeframe=timeframe,
            )
        )
    out.sort(key=lambda candle: candle.ts)
    return out


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


class UpstoxMarketData:
    """Read-only market-data access. This class never places orders."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self._config = get_config_store().load("broker").get("historical", {})
        self._pause = float(self._config.get("request_pause_seconds", 0.25) or 0.25)

    # ------------------------------------------------------------------ candles
    def historical_candles(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        from_date: dt.date,
        to_date: dt.date,
        timeframe: str = "",
    ) -> List[Candle]:
        """One historical V3 request. The caller is responsible for chunking."""
        path = Endpoints.historical_v3(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            to_date=to_date.strftime("%Y-%m-%d"),
            from_date=from_date.strftime("%Y-%m-%d"),
        )
        response: ApiResponse = self.client.get(path, limit_class="standard")
        return parse_candles(response.payload, instrument_key, timeframe or f"{interval}{unit[:1]}")

    def intraday_candles(
        self, instrument_key: str, unit: str = "minutes", interval: int = 1, timeframe: str = "1m"
    ) -> List[Candle]:
        """Current trading day's candles."""
        path = Endpoints.intraday_v3(instrument_key, unit, interval)
        response = self.client.get(path, limit_class="standard")
        return parse_candles(response.payload, instrument_key, timeframe)

    # ------------------------------------------------------- chunked retrieval
    @staticmethod
    def max_window_days(unit: str, interval: int) -> int:
        """Documented maximum retrieval window, expressed in days."""
        if unit == "minutes":
            if interval <= 15:
                return 30          # "1 month"
            return 90              # "1 quarter"
        if unit == "hours":
            return 90              # "1 quarter"
        if unit == "days":
            return 3650            # "1 decade"
        return 3650

    def chunk_ranges(self, unit: str, interval: int, start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
        """Split a long range into legal request windows."""
        max_days = self.max_window_days(unit, interval)
        chunks: List[Tuple[dt.date, dt.date]] = []
        cursor = start
        step = dt.timedelta(days=max_days)
        while cursor <= end:
            chunk_end = min(cursor + step - dt.timedelta(days=1), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + dt.timedelta(days=1)
        return chunks

    def fetch_range(
        self,
        instrument_key: str,
        timeframe: str,
        start: dt.date,
        end: dt.date,
        progress: Optional[Any] = None,
    ) -> List[Candle]:
        """Fetch a (possibly long) historical range, chunked to legal windows."""
        if timeframe not in TIMEFRAME_TO_UNIT:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        unit, interval = TIMEFRAME_TO_UNIT[timeframe]
        available_from = _available_from(unit)
        if start < available_from:
            log.info(
                "clamping start date to documented availability",
                context={"requested": str(start), "available_from": str(available_from), "unit": unit},
            )
            start = available_from
        if end < start:
            return []

        collected: Dict[dt.datetime, Candle] = {}
        chunks = self.chunk_ranges(unit, interval, start, end)
        for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            try:
                candles = self.historical_candles(instrument_key, unit, interval, chunk_start, chunk_end, timeframe)
            except UpstoxError as exc:
                log.warning(
                    "historical chunk failed",
                    context={
                        "instrument": instrument_key,
                        "chunk": f"{chunk_start}..{chunk_end}",
                        "error": str(exc),
                    },
                )
                continue
            for candle in candles:
                collected[candle.ts] = candle
            if progress is not None:
                progress(index, len(chunks), len(collected))
            if len(chunks) > 1 and self._pause:
                import time

                time.sleep(self._pause)
        return sorted(collected.values(), key=lambda c: c.ts)

    # ------------------------------------------------------------------- quotes
    def ltp(self, instrument_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """LTP quotes V3 for up to 500 instrument keys."""
        return self._quote_call(Endpoints.LTP_V3, instrument_keys, "ltp")

    def ohlc(self, instrument_keys: Sequence[str], interval: str = "1d") -> Dict[str, Dict[str, Any]]:
        return self._quote_call(Endpoints.OHLC_V3, instrument_keys, "ohlc", extra={"interval": interval})

    def full_quote(self, instrument_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Full market quote V3 - includes OHLC, volume, market depth and OI."""
        return self._quote_call(Endpoints.FULL_QUOTE_V3, instrument_keys, "full")

    def _quote_call(
        self,
        path: str,
        instrument_keys: Sequence[str],
        kind: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        keys = [k for k in instrument_keys if k]
        if not keys:
            return {}
        if len(keys) > 500:
            raise ValueError("Upstox market-quote APIs accept at most 500 instrument keys per call")

        params = {"instrument_key": ",".join(keys)}
        if extra:
            params.update(extra)
        response = self.client.get(path, params=params, limit_class="standard")
        data = response.data if isinstance(response.data, dict) else {}

        normalised: Dict[str, Dict[str, Any]] = {}
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            # V3 LTP/OHLC keys are usually the instrument_key already; V2 nested
            # responses use "<SEGMENT>:<SYMBOL>". Normalise to instrument_key.
            instrument_key = entry.get("instrument_token") or entry.get("instrument_key") or key
            merged = dict(entry)
            merged["instrument_key"] = str(instrument_key)
            merged["_kind"] = kind
            normalised[str(instrument_key)] = merged
        return normalised

    @staticmethod
    def extract_ltp(quote: Dict[str, Any]) -> Optional[float]:
        for path in (("last_price",), ("ltp",), ("ohlc", "close")):
            node: Any = quote
            for part in path:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    node = None
                    break
            if isinstance(node, (int, float)) and node:
                return float(node)
        return None

    @staticmethod
    def extract_depth(quote: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Best bid/ask and their quantities, when the depth is present."""
        depth = quote.get("depth") or quote.get("market_depth")
        out: Dict[str, Optional[float]] = {"bid": None, "ask": None, "bid_qty": None, "ask_qty": None}
        if isinstance(depth, dict):
            buy = depth.get("buy") or []
            sell = depth.get("sell") or []
            if isinstance(buy, list) and buy and isinstance(buy[0], dict):
                out["bid"] = _num(buy[0].get("price"))
                out["bid_qty"] = _num(buy[0].get("quantity"))
            if isinstance(sell, list) and sell and isinstance(sell[0], dict):
                out["ask"] = _num(sell[0].get("price"))
                out["ask_qty"] = _num(sell[0].get("quantity"))
        if out["bid"] is None:
            out["bid"] = _num(quote.get("bid_price") or quote.get("bid"))
        if out["ask"] is None:
            out["ask"] = _num(quote.get("ask_price") or quote.get("ask"))
        return out

    @staticmethod
    def extract_ohlc(quote: Dict[str, Any]) -> Dict[str, Optional[float]]:
        ohlc = quote.get("ohlc") if isinstance(quote.get("ohlc"), dict) else quote
        return {
            "open": _num(ohlc.get("open")),
            "high": _num(ohlc.get("high")),
            "low": _num(ohlc.get("low")),
            "close": _num(ohlc.get("close")),
            "volume": _num(quote.get("volume")),
        }

    # -------------------------------------------------------- market information
    def market_holidays(self, date: Optional[dt.date] = None, authenticated: bool = False) -> List[Dict[str, Any]]:
        """Official holiday list. This endpoint is public and does not need a token."""
        try:
            if date is not None:
                response = self.client.get(
                    f"{Endpoints.MARKET_HOLIDAYS}/{date.strftime('%Y-%m-%d')}",
                    require_auth=authenticated,
                    limit_class="standard",
                    max_retries=0,
                )
            else:
                response = self.client.get(
                    Endpoints.MARKET_HOLIDAYS, require_auth=authenticated, limit_class="standard",
                    max_retries=0,
                )
        except UpstoxError as exc:
            log.warning("market holidays lookup failed", context={"error": str(exc)})
            return []
        data = response.data
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    def market_timings(self, date: dt.date) -> List[Dict[str, Any]]:
        path = Endpoints.MARKET_TIMINGS.format(date=date.strftime("%Y-%m-%d"))
        try:
            response = self.client.get(path, require_auth=False, limit_class="standard", max_retries=0)
        except UpstoxError as exc:
            log.warning("market timings lookup failed", context={"error": str(exc)})
            return []
        data = response.data
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def exchange_status(self, exchange: str = "NSE") -> Dict[str, Any]:
        """Official exchange session status (NORMAL_OPEN, PRE_OPEN_START, ...)."""
        path = Endpoints.MARKET_STATUS.format(exchange=exchange)
        try:
            response = self.client.get(path, require_auth=False, limit_class="standard", max_retries=0)
        except UpstoxError as exc:
            log.warning("exchange status lookup failed", context={"error": str(exc)})
            return {"exchange": exchange, "status": "UNKNOWN", "error": str(exc)}
        data = response.data if isinstance(response.data, dict) else {}
        return {
            "exchange": data.get("exchange", exchange),
            "status": data.get("status", "UNKNOWN"),
            "last_updated": data.get("last_updated"),
            "cas_eligible_status": data.get("cas_eligible_status"),
        }

    # ------------------------------------------------------------- ws authorize
    def authorize_market_feed(self) -> Optional[str]:
        """Get the single-use ``wss://`` URL for Market Data Feed V3."""
        return self._authorize(Endpoints.FEED_AUTHORIZE_V3)

    def authorize_portfolio_feed(self) -> Optional[str]:
        """Get the single-use ``wss://`` URL for the portfolio (order) stream."""
        return self._authorize(Endpoints.PORTFOLIO_FEED_AUTHORIZE)

    def _authorize(self, path: str) -> Optional[str]:
        try:
            response = self.client.get(path, limit_class="standard")
        except UpstoxError as exc:
            log.warning("feed authorization failed", context={"path": path, "error": str(exc)})
            return None
        data = response.data if isinstance(response.data, dict) else {}
        uri = data.get("authorized_redirect_uri") or data.get("authorizedRedirectUri")
        return str(uri) if uri else None

    # ------------------------------------------------------------------- helper
    def download_proto(self, target_dir: Optional[Any] = None) -> Optional[Any]:
        """Fetch the Market Data Feed V3 ``.proto`` file for reference/regeneration."""
        url = get_config_store().load("broker").get("websocket", {}).get(
            "proto_url", "https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto"
        )
        directory = target_dir or (CACHE_DIR / "proto")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "MarketDataFeed.proto"
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                path.write_text(response.text, encoding="utf-8")
            return path
        except Exception as exc:
            log.warning("could not download the V3 proto file", context={"error": str(exc)})
            return None


def _available_from(unit: str) -> dt.date:
    return dt.date(2000, 1, 1) if unit in ("days", "weeks", "months") else dt.date(2022, 1, 1)


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "UpstoxMarketData",
    "Candle",
    "parse_candles",
    "TIMEFRAME_TO_UNIT",
    "TIMEFRAME_MINUTES",
]
