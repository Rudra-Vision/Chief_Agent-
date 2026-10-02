"""Realtime streaming: Market Data Feed V3 and the Portfolio (order) stream.

Verified against the current official documentation
(https://upstox.com/developer/api-documentation/v3/get-market-data-feed):

* Authorize URL:  ``GET /v3/feed/market-data-feed/authorize`` returns
  ``data.authorized_redirect_uri`` - a single-use ``wss://`` URL.
* Feed endpoint:  ``WSS /feed/market-data-feed``, **protobuf encoded**; the
  schema is published as ``MarketDataFeed.proto`` at
  https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto
* Subscription messages MUST be sent as **binary** frames, not text, in the V3
  format::

      {"guid": "...", "method": "sub"|"change_mode"|"unsub",
       "data": {"mode": "ltpc"|"full"|"option_greeks"|"full_d30",
                "instrumentKeys": ["NSE_EQ|INE...", ...]}}

* Documented limits (normal plan): 2 connections per user; LTPC 5000 keys
  (2000 combined), Option Greeks 3000 (2000 combined), Full 2000 (1500 combined).

Because the payload is protobuf, we cannot decode it without the exact schema.
This module therefore:

  1. looks for a locally-compiled ``marketdatafeed_pb2`` module (generated from
     the official ``.proto`` by ``scripts/download_proto.py``);
  2. if it is missing, it *does not guess* - streaming is reported as
     ``DECODER_UNAVAILABLE`` and the caller falls back to the REST polling feed,
     which uses the same Market Quote V3 APIs and is always correct.

Automatic reconnection uses exponential backoff with jitter. Reconnection never
reconstructs uncertain ORDER state - that always goes through broker
reconciliation.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import CACHE_DIR, Settings, get_config_store, get_settings
from ..timeutil import ensure_ist, now_ist

log = get_logger(__name__, component="websocket")

MODE_LTPC = "ltpc"
MODE_FULL = "full"
MODE_OPTION_GREEKS = "option_greeks"
MODE_FULL_D30 = "full_d30"
VALID_MODES = {MODE_LTPC, MODE_FULL, MODE_OPTION_GREEKS, MODE_FULL_D30}
VALID_METHODS = {"sub", "change_mode", "unsub"}

PROTO_DIR = CACHE_DIR / "proto"
PROTO_MODULE_NAME = "marketdatafeed_pb2"


class FeedDecoderUnavailable(RuntimeError):
    """Raised when the protobuf schema has not been compiled locally."""


def load_feed_decoder():
    """Import the locally compiled protobuf module, if present.

    Returns the module or raises :class:`FeedDecoderUnavailable`. We deliberately
    do NOT hand-roll a decoder from guessed field numbers - inventing a wire
    format would silently corrupt market data.
    """
    import importlib
    import sys

    if str(PROTO_DIR) not in sys.path:
        sys.path.insert(0, str(PROTO_DIR))
    try:
        return importlib.import_module(PROTO_MODULE_NAME)
    except ImportError as exc:
        raise FeedDecoderUnavailable(
            "Market Data Feed V3 is protobuf-encoded and the schema is not compiled locally. "
            "Run `python scripts/download_proto.py` (needs grpcio-tools) to build "
            f"{PROTO_MODULE_NAME}, or use the REST market-data feed which needs no schema."
        ) from exc


def decoder_status() -> Dict[str, Any]:
    try:
        load_feed_decoder()
        return {"available": True, "module": PROTO_MODULE_NAME, "path": str(PROTO_DIR)}
    except FeedDecoderUnavailable as exc:
        return {"available": False, "reason": str(exc), "path": str(PROTO_DIR)}


@dataclass
class Tick:
    """A normalised market-data update."""

    instrument_key: str
    ts: Any
    ltp: Optional[float] = None
    last_traded_time: Optional[Any] = None
    last_traded_quantity: Optional[float] = None
    close_price: Optional[float] = None
    open_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    vwap: Optional[float] = None
    open_interest: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_qty: Optional[float] = None
    ask_qty: Optional[float] = None
    total_buy_qty: Optional[float] = None
    total_sell_qty: Optional[float] = None
    mode: str = MODE_LTPC
    source: str = "websocket"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def spread(self) -> Optional[float]:
        if self.bid is None or self.ask is None or self.ask < self.bid:
            return None
        return self.ask - self.bid

    @property
    def spread_pct(self) -> Optional[float]:
        spread = self.spread
        if spread is None or not self.ltp:
            return None
        return spread / self.ltp

    @property
    def mid(self) -> Optional[float]:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.ltp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "ts": self.ts.isoformat() if hasattr(self.ts, "isoformat") else str(self.ts),
            "ltp": self.ltp,
            "close_price": self.close_price,
            "open": self.open_price,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "vwap": self.vwap,
            "open_interest": self.open_interest,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "mode": self.mode,
        }


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def decode_feed_message(message: Any, decoder: Any = None, mode: str = MODE_LTPC) -> List[Tick]:
    """Decode one protobuf feed frame into normalised :class:`Tick` objects.

    Expected message shapes (per the published proto / docs):

    * ``type == "live_feed"``  -> ``feeds`` keyed by instrument_key, each with an
      ``ltpc`` block (ltp, ltt, ltq, cp) and, in ``full`` mode, ``oi``, ``vtt``,
      ``market_level``/depth, ``ohlc``.
    * ``type == "market_info"`` -> ``marketInfo.segmentStatus`` plus
      ``casMarketStatus`` / ``preOpenSessionStatus``. Returned as a synthetic
      tick with ``instrument_key = "__market_info__"``.
    """
    if decoder is None:
        decoder = load_feed_decoder()

    received = now_ist()
    out: List[Tick] = []

    msg_type = getattr(message, "type", "") or ""
    if msg_type == "market_info":
        info = getattr(message, "marketInfo", None)
        payload = _proto_to_dict(info) if info is not None else {}
        out.append(
            Tick(
                instrument_key="__market_info__",
                ts=received,
                mode="market_info",
                raw={"type": "market_info", "marketInfo": payload},
            )
        )
        return out

    feeds = getattr(message, "feeds", None)
    if feeds is None:
        return out

    current_ts = getattr(message, "currentTs", None)
    try:
        base_ts = ensure_ist(int(current_ts)) if current_ts else received
    except Exception:
        base_ts = received

    items = feeds.items() if isinstance(feeds, dict) else getattr(feeds, "items", lambda: [])()
    for instrument_key, feed in items:
        ltpc = getattr(feed, "ltpc", None)
        tick = Tick(instrument_key=str(instrument_key), ts=base_ts, mode=mode, source="websocket")
        if ltpc is not None:
            tick.ltp = _num(getattr(ltpc, "ltp", None))
            tick.last_traded_quantity = _num(getattr(ltpc, "ltq", None))
            tick.close_price = _num(getattr(ltpc, "cp", None))
            ltt = getattr(ltpc, "ltt", None)
            if ltt:
                try:
                    tick.last_traded_time = ensure_ist(int(ltt))
                except Exception:
                    tick.last_traded_time = None
            iep = _num(getattr(ltpc, "iep", None))
            if iep:
                tick.raw["indicative_equilibrium_price"] = iep

        oi = _num(getattr(feed, "oi", None))
        if oi is not None:
            tick.open_interest = oi
        volume = _num(getattr(feed, "vtt", None))
        if volume is not None:
            tick.volume = volume

        ohlc = getattr(feed, "ohlc", None)
        if ohlc is not None:
            tick.open_price = _num(getattr(ohlc, "open", None))
            tick.high = _num(getattr(ohlc, "high", None))
            tick.low = _num(getattr(ohlc, "low", None))
            close = _num(getattr(ohlc, "close", None))
            if close is not None:
                tick.close_price = close

        market_level = getattr(feed, "marketLevel", None) or getattr(feed, "market_level", None)
        if market_level is not None:
            bids = list(getattr(market_level, "bidAskQuotes", None) or getattr(market_level, "bid_ask_quotes", None) or [])
            if bids:
                first = bids[0]
                tick.bid = _num(getattr(first, "bidP", None) or getattr(first, "bid_p", None))
                tick.bid_qty = _num(getattr(first, "bidQ", None) or getattr(first, "bid_q", None))
                tick.ask = _num(getattr(first, "askP", None) or getattr(first, "ask_p", None))
                tick.ask_qty = _num(getattr(first, "askQ", None) or getattr(first, "ask_q", None))

        total_buy = _num(getattr(feed, "tbq", None))
        total_sell = _num(getattr(feed, "tsq", None))
        if total_buy is not None:
            tick.total_buy_qty = total_buy
        if total_sell is not None:
            tick.total_sell_qty = total_sell

        atp = _num(getattr(feed, "atp", None))
        if atp is not None:
            tick.vwap = atp

        out.append(tick)
    return out


def _proto_to_dict(message: Any) -> Dict[str, Any]:
    """Best-effort, read-only conversion of a protobuf message to a dict."""
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(message, preserving_proto_field_name=True)
    except Exception:
        return {"_repr": str(message)[:2000]}


class MarketDataFeedClient:
    """Async WebSocket client for Market Data Feed V3 with safe reconnection."""

    def __init__(self, market_data: Any, settings: Optional[Settings] = None) -> None:
        self.market_data = market_data          # UpstoxMarketData (for authorize URL)
        self.settings = settings or get_settings()
        cfg = (get_config_store().load("broker").get("websocket", {}) or {})
        self._cfg = cfg
        self._decoder: Any = None
        self._decoder_error: Optional[str] = None
        self._subscriptions: Dict[str, str] = {}     # instrument_key -> mode
        self._queue: "asyncio.Queue[Tick]" = asyncio.Queue(maxsize=20000)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._connected = False
        self._last_message_at: Optional[Any] = None
        self._reconnects = 0
        self._errors = 0
        self._on_market_info: Optional[Callable[[Dict[str, Any]], None]] = None

        try:
            self._decoder = load_feed_decoder()
        except FeedDecoderUnavailable as exc:
            self._decoder_error = str(exc)

    # ------------------------------------------------------------------ state
    @property
    def decoder_available(self) -> bool:
        return self._decoder is not None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_message_at(self) -> Optional[Any]:
        return self._last_message_at

    def status(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "decoder_available": self.decoder_available,
            "decoder_error": self._decoder_error,
            "subscriptions": len(self._subscriptions),
            "reconnects": self._reconnects,
            "errors": self._errors,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "queued_ticks": self._queue.qsize(),
        }

    def set_market_info_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._on_market_info = handler

    # ----------------------------------------------------------- subscription
    def subscribe(self, instrument_keys: Sequence[str], mode: str = MODE_LTPC) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        for key in instrument_keys:
            if key:
                self._subscriptions[key] = mode

    def unsubscribe(self, instrument_keys: Sequence[str]) -> None:
        for key in instrument_keys:
            self._subscriptions.pop(key, None)

    def _subscription_payloads(self) -> List[bytes]:
        """V3 subscription messages: JSON, sent as BINARY frames."""
        chunk_size = int(self._cfg.get("subscription_chunk_size", 100) or 100)
        by_mode: Dict[str, List[str]] = {}
        for key, mode in self._subscriptions.items():
            by_mode.setdefault(mode, []).append(key)

        payloads: List[bytes] = []
        for mode, keys in by_mode.items():
            for start in range(0, len(keys), chunk_size):
                chunk = keys[start : start + chunk_size]
                message = {
                    "guid": uuid.uuid4().hex,
                    "method": "sub",
                    "data": {"mode": mode, "instrumentKeys": chunk},
                }
                payloads.append(json.dumps(message).encode("utf-8"))
        return payloads

    # --------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="market-data-feed")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._connected = False

    async def ticks(self) -> AsyncIterator[Tick]:
        while not self._stop.is_set():
            try:
                tick = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield tick
            except asyncio.TimeoutError:
                continue

    def drain(self, max_items: int = 5000) -> List[Tick]:
        """Non-blocking drain, for the synchronous engine loop."""
        out: List[Tick] = []
        while len(out) < max_items:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    # -------------------------------------------------------------------- loop
    async def _run(self) -> None:
        if not self.decoder_available:
            log.warning("market data feed not started", context={"reason": self._decoder_error})
            return

        import websockets

        initial = float(self._cfg.get("reconnect_initial_backoff_seconds", 1) or 1)
        maximum = float(self._cfg.get("reconnect_max_backoff_seconds", 60) or 60)
        jitter = bool(self._cfg.get("reconnect_jitter", True))
        ping_interval = float(self._cfg.get("ping_interval_seconds", 25) or 25)

        backoff = initial
        while not self._stop.is_set():
            authorized_url = await asyncio.to_thread(self.market_data.authorize_market_feed)
            if not authorized_url:
                log.warning("no authorized websocket URL; retrying later")
                await self._sleep(backoff, jitter)
                backoff = min(maximum, backoff * 2)
                continue

            try:
                # The authorized URL is single-use; connect immediately.
                async with websockets.connect(
                    authorized_url,
                    ping_interval=ping_interval,
                    max_size=8 * 1024 * 1024,
                    open_timeout=15,
                ) as socket:
                    self._connected = True
                    self._reconnects += 1
                    backoff = initial
                    log.info("market data feed connected", context={"subscriptions": len(self._subscriptions)})

                    for payload in self._subscription_payloads():
                        await socket.send(payload)   # BINARY, per the V3 requirement

                    while not self._stop.is_set():
                        try:
                            frame = await asyncio.wait_for(socket.recv(), timeout=ping_interval * 3)
                        except asyncio.TimeoutError:
                            # Keepalive read timeout: the session is stale.
                            log.warning("market data feed read timeout; reconnecting")
                            break

                        if isinstance(frame, str):
                            await self._handle_text_frame(frame)
                            continue

                        try:
                            message = self._decoder.FeedResponse()
                            message.ParseFromString(frame)
                        except Exception as exc:
                            self._errors += 1
                            log.warning("protobuf decode failed", context={"error": str(exc)})
                            continue

                        self._last_message_at = now_ist()
                        for tick in decode_feed_message(message, self._decoder, self._dominant_mode()):
                            if tick.instrument_key == "__market_info__":
                                if self._on_market_info:
                                    try:
                                        self._on_market_info(tick.raw)
                                    except Exception:  # pragma: no cover - handler must not break the feed
                                        pass
                                continue
                            self._enqueue(tick)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._errors += 1
                self._connected = False
                log.warning("market data feed error", context={"error": str(exc), "backoff_s": round(backoff, 2)})
            finally:
                self._connected = False

            if self._stop.is_set():
                break
            await self._sleep(backoff, jitter)
            backoff = min(maximum, backoff * 2)

    async def _handle_text_frame(self, frame: str) -> None:
        """Upstox may send JSON control frames (errors / segment status)."""
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict) and payload.get("type") == "market_info":
            if self._on_market_info:
                try:
                    self._on_market_info(payload)
                except Exception:
                    pass
            return
        log.info("market data feed control frame", context={"payload": payload})

    def _dominant_mode(self) -> str:
        if not self._subscriptions:
            return MODE_LTPC
        counts: Dict[str, int] = {}
        for mode in self._subscriptions.values():
            counts[mode] = counts.get(mode, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _enqueue(self, tick: Tick) -> None:
        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            # Drop the oldest tick rather than the newest - freshness matters more.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(tick)
            except Exception:
                pass

    async def _sleep(self, seconds: float, jitter: bool) -> None:
        delay = seconds
        if jitter:
            delay = seconds * (0.5 + random.random())
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


@dataclass
class OrderStreamEvent:
    """A normalised portfolio-stream event (order / position / holding update)."""

    event_type: str
    ts: Any
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def order_id(self) -> Optional[str]:
        value = self.payload.get("order_id")
        return str(value) if value else None

    @property
    def tag(self) -> Optional[str]:
        value = self.payload.get("tag")
        return str(value) if value else None

    @property
    def status(self) -> Optional[str]:
        value = self.payload.get("status")
        return str(value) if value else None


class PortfolioStreamClient:
    """Async client for the portfolio (order update) stream.

    ``GET /v2/feed/portfolio-stream-feed/authorize`` returns a single-use
    ``wss://`` URL. Events are JSON. Reconnection uses the same backoff policy as
    the market feed; order state is ALWAYS confirmed against the REST order book
    before any decision is made.
    """

    def __init__(self, market_data: Any, settings: Optional[Settings] = None) -> None:
        self.market_data = market_data
        self.settings = settings or get_settings()
        self._queue: "asyncio.Queue[OrderStreamEvent]" = asyncio.Queue(maxsize=5000)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._connected = False
        self._reconnects = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def status(self) -> Dict[str, Any]:
        return {"connected": self._connected, "reconnects": self._reconnects, "queued_events": self._queue.qsize()}

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="portfolio-stream")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._connected = False

    def drain(self, max_items: int = 1000) -> List[OrderStreamEvent]:
        out: List[OrderStreamEvent] = []
        while len(out) < max_items:
            try:
                out.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    async def _run(self) -> None:
        import websockets

        backoff = 1.0
        while not self._stop.is_set():
            url = await asyncio.to_thread(self.market_data.authorize_portfolio_feed)
            if not url:
                await self._sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            try:
                async with websockets.connect(url, ping_interval=25, open_timeout=15) as socket:
                    self._connected = True
                    self._reconnects += 1
                    backoff = 1.0
                    log.info("portfolio stream connected")
                    while not self._stop.is_set():
                        try:
                            frame = await asyncio.wait_for(socket.recv(), timeout=90)
                        except asyncio.TimeoutError:
                            break
                        self._handle(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("portfolio stream error", context={"error": str(exc)})
            finally:
                self._connected = False
            if self._stop.is_set():
                break
            await self._sleep(backoff)
            backoff = min(60.0, backoff * 2)

    def _handle(self, frame: Any) -> None:
        if isinstance(frame, (bytes, bytearray)):
            try:
                frame = frame.decode("utf-8")
            except UnicodeDecodeError:
                return
        try:
            payload = json.loads(frame)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("type") or payload.get("update_type") or "unknown")
        rows = payload.get("data")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    self._enqueue(OrderStreamEvent(event_type=event_type, ts=now_ist(), payload=row))
        elif isinstance(rows, dict):
            self._enqueue(OrderStreamEvent(event_type=event_type, ts=now_ist(), payload=rows))

    def _enqueue(self, event: OrderStreamEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except Exception:
                pass

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds * (0.5 + random.random()))
        except asyncio.TimeoutError:
            pass


__all__ = [
    "MarketDataFeedClient",
    "PortfolioStreamClient",
    "Tick",
    "OrderStreamEvent",
    "decode_feed_message",
    "load_feed_decoder",
    "decoder_status",
    "FeedDecoderUnavailable",
    "MODE_LTPC",
    "MODE_FULL",
    "MODE_OPTION_GREEKS",
]
