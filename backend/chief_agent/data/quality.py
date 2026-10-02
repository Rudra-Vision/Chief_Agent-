"""Data quality engine.

Before trading, the system verifies that its market data is trustworthy:

    missing candles        duplicate candles      out-of-order ticks
    stale quotes           impossible OHLC values zero/abnormal volume
    large timestamp gaps   WebSocket disconnection exchange closure
    symbol mapping errors

If the data is unreliable the system must NOT trade: it enters
**DATA_SAFE_MODE**. That mode is an explicit, visible, loggable state - never a
silent degradation.
"""

from __future__ import annotations

import datetime as dt
import enum
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..broker.upstox_market_data import Candle
from ..logging_setup import get_logger
from ..settings import get_config_store
from ..timeutil import IST, is_weekend, minutes_from_open, now_ist

log = get_logger(__name__, component="data_quality")


class IncidentType(str, enum.Enum):
    MISSING_CANDLES = "MISSING_CANDLES"
    DUPLICATE_CANDLES = "DUPLICATE_CANDLES"
    OUT_OF_ORDER_TICKS = "OUT_OF_ORDER_TICKS"
    STALE_QUOTE = "STALE_QUOTE"
    IMPOSSIBLE_OHLC = "IMPOSSIBLE_OHLC"
    ZERO_VOLUME = "ZERO_VOLUME"
    ABNORMAL_VOLUME = "ABNORMAL_VOLUME"
    TIMESTAMP_GAP = "TIMESTAMP_GAP"
    WEBSOCKET_DISCONNECT = "WEBSOCKET_DISCONNECT"
    EXCHANGE_CLOSED = "EXCHANGE_CLOSED"
    SYMBOL_MAPPING_ERROR = "SYMBOL_MAPPING_ERROR"
    NEGATIVE_SPREAD = "NEGATIVE_SPREAD"
    PRICE_JUMP = "PRICE_JUMP"


SEVERITY = {
    IncidentType.MISSING_CANDLES: "WARNING",
    IncidentType.DUPLICATE_CANDLES: "WARNING",
    IncidentType.OUT_OF_ORDER_TICKS: "WARNING",
    IncidentType.STALE_QUOTE: "CRITICAL",
    IncidentType.IMPOSSIBLE_OHLC: "CRITICAL",
    IncidentType.ZERO_VOLUME: "WARNING",
    IncidentType.ABNORMAL_VOLUME: "WARNING",
    IncidentType.TIMESTAMP_GAP: "WARNING",
    IncidentType.WEBSOCKET_DISCONNECT: "CRITICAL",
    IncidentType.EXCHANGE_CLOSED: "INFO",
    IncidentType.SYMBOL_MAPPING_ERROR: "CRITICAL",
    IncidentType.NEGATIVE_SPREAD: "CRITICAL",
    IncidentType.PRICE_JUMP: "WARNING",
}

#: Incidents that make trading unsafe even if they seem minor in isolation.
BLOCKING_TYPES = {
    IncidentType.STALE_QUOTE,
    IncidentType.IMPOSSIBLE_OHLC,
    IncidentType.WEBSOCKET_DISCONNECT,
    IncidentType.SYMBOL_MAPPING_ERROR,
    IncidentType.NEGATIVE_SPREAD,
}


@dataclass
class Incident:
    incident_type: IncidentType
    instrument_key: Optional[str] = None
    detail: str = ""
    severity: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    ts: Optional[dt.datetime] = None

    def __post_init__(self) -> None:
        if self.severity is None:
            self.severity = SEVERITY.get(self.incident_type, "WARNING")
        if self.ts is None:
            self.ts = now_ist()

    @property
    def blocking(self) -> bool:
        return self.incident_type in BLOCKING_TYPES or self.severity == "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_type": self.incident_type.value,
            "instrument_key": self.instrument_key,
            "detail": self.detail,
            "severity": self.severity,
            "value": self.value,
            "threshold": self.threshold,
            "blocking": self.blocking,
            "ts": self.ts.isoformat() if self.ts else None,
        }


@dataclass
class QualityReport:
    checked_at: dt.datetime
    safe_to_trade: bool
    incidents: List[Incident] = field(default_factory=list)
    checks_run: int = 0
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_incidents(self) -> List[Incident]:
        return [i for i in self.incidents if i.blocking]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "safe_to_trade": self.safe_to_trade,
            "summary": self.summary,
            "checks_run": self.checks_run,
            "incident_count": len(self.incidents),
            "blocking_count": len(self.blocking_incidents),
            "incidents": [i.to_dict() for i in self.incidents[:100]],
            "metrics": self.metrics,
        }


class DataQualityEngine:
    """Validates candles and quotes; owns DATA_SAFE_MODE."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        base = (get_config_store().load("risk").get("data_quality") or {}) if config is None else config
        self.config = base or {}
        self._safe_mode = False
        self._last_report: Optional[QualityReport] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ limits
    def _c(self, path: str, default: Any) -> Any:
        node: Any = self.config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    # ------------------------------------------------------------ candle checks
    def check_candles(
        self,
        candles: Sequence[Candle],
        *,
        instrument_key: str = "",
        expected_minutes_per_session: int = 375,
        session_start_minutes: float = 0.0,
    ) -> List[Incident]:
        incidents: List[Incident] = []
        if not candles:
            incidents.append(
                Incident(
                    IncidentType.MISSING_CANDLES,
                    instrument_key,
                    "no candles available at all",
                    value=0,
                    threshold=1,
                )
            )
            return incidents

        by_day: Dict[dt.date, List[Candle]] = {}
        timestamps: List[dt.datetime] = []
        for candle in candles:
            local = candle.ts.astimezone(IST) if candle.ts.tzinfo else candle.ts
            by_day.setdefault(local.date(), []).append(candle)
            timestamps.append(local)

        # duplicates
        seen: Dict[str, int] = {}
        duplicates = 0
        for ts in timestamps:
            key = ts.strftime("%Y-%m-%dT%H:%M")
            seen[key] = seen.get(key, 0) + 1
        duplicates = sum(count - 1 for count in seen.values() if count > 1)
        if duplicates:
            ratio = duplicates / max(1, len(timestamps))
            if ratio > float(self._c("max_duplicate_candle_pct", 0.001)):
                incidents.append(
                    Incident(
                        IncidentType.DUPLICATE_CANDLES,
                        instrument_key,
                        f"{duplicates} duplicate timestamps ({ratio:.3%})",
                        value=ratio,
                        threshold=float(self._c("max_duplicate_candle_pct", 0.001)),
                    )
                )

        # out-of-order
        out_of_order = sum(1 for i in range(1, len(timestamps)) if timestamps[i] < timestamps[i - 1])
        if out_of_order > int(self._c("max_out_of_order_ticks", 5)):
            incidents.append(
                Incident(
                    IncidentType.OUT_OF_ORDER_TICKS,
                    instrument_key,
                    f"{out_of_order} out-of-order timestamps",
                    value=out_of_order,
                    threshold=float(self._c("max_out_of_order_ticks", 5)),
                )
            )

        # impossible OHLC
        impossible = 0
        for candle in candles:
            if (
                candle.high < candle.low
                or candle.high < max(candle.open, candle.close) - 1e-8
                or candle.low > min(candle.open, candle.close) + 1e-8
                or candle.close <= 0
                or candle.open <= 0
            ):
                impossible += 1
        if impossible:
            incidents.append(
                Incident(
                    IncidentType.IMPOSSIBLE_OHLC,
                    instrument_key,
                    f"{impossible} candles violate OHLC sanity (high < low, or high below the body)",
                    value=impossible,
                    threshold=0,
                )
            )

        # missing candles per session
        missing_total = 0
        expected_total = 0
        for day, day_candles in by_day.items():
            if is_weekend(day):
                continue
            present = len(day_candles)
            expected = expected_minutes_per_session
            expected_total += expected
            missing_total += max(0, expected - present)
        if expected_total > 0:
            missing_ratio = missing_total / expected_total
            if missing_ratio > float(self._c("max_missing_candle_pct", 0.02)):
                incidents.append(
                    Incident(
                        IncidentType.MISSING_CANDLES,
                        instrument_key,
                        f"{missing_total} of {expected_total} expected candles are missing ({missing_ratio:.2%})",
                        value=missing_ratio,
                        threshold=float(self._c("max_missing_candle_pct", 0.02)),
                    )
                )

        # zero / abnormal volume
        volumes = np.asarray([c.volume for c in candles], dtype=float)
        zero_volume = int((volumes <= 0).sum())
        if zero_volume:
            ratio = zero_volume / max(1, len(volumes))
            if ratio > 0.05:
                incidents.append(
                    Incident(
                        IncidentType.ZERO_VOLUME,
                        instrument_key,
                        f"{ratio:.1%} of candles have zero or negative volume",
                        value=ratio,
                        threshold=0.05,
                    )
                )
        positive = volumes[volumes > 0]
        if positive.size > 50:
            median = float(np.median(positive))
            spikes = int((positive > median * 25).sum())
            if spikes:
                incidents.append(
                    Incident(
                        IncidentType.ABNORMAL_VOLUME,
                        instrument_key,
                        f"{spikes} candles with volume > 25x the session median",
                        value=float(spikes),
                        threshold=0.0,
                    )
                )

        # timestamp gaps inside a session
        for day, day_candles in by_day.items():
            ordered = sorted(day_candles, key=lambda c: c.ts)
            for i in range(1, len(ordered)):
                previous = ordered[i - 1].ts.astimezone(IST) if ordered[i - 1].ts.tzinfo else ordered[i - 1].ts
                current = ordered[i].ts.astimezone(IST) if ordered[i].ts.tzinfo else ordered[i].ts
                gap = (current - previous).total_seconds() / 60.0
                if gap > 5:
                    incidents.append(
                        Incident(
                            IncidentType.TIMESTAMP_GAP,
                            instrument_key,
                            f"{gap:.0f}-minute gap at {current.isoformat()}",
                            value=gap,
                            threshold=5.0,
                        )
                    )
                    break

        # price jumps (a data error looks different from a real gap)
        closes = np.asarray([c.close for c in candles], dtype=float)
        if closes.size > 2:
            returns = np.diff(closes) / np.where(closes[:-1] == 0, np.nan, closes[:-1])
            returns = returns[np.isfinite(returns)]
            extreme = int((np.abs(returns) > 0.25).sum())
            if extreme:
                incidents.append(
                    Incident(
                        IncidentType.PRICE_JUMP,
                        instrument_key,
                        f"{extreme} bar-to-bar price jumps above 25% - likely a bad print or an unadjusted corporate action",
                        value=float(extreme),
                        threshold=0.0,
                    )
                )
        return incidents

    # ------------------------------------------------------------- quote checks
    def check_quote(
        self,
        quote: Mapping[str, Any],
        *,
        instrument_key: str = "",
        received_at: Optional[dt.datetime] = None,
    ) -> List[Incident]:
        incidents: List[Incident] = []
        received_at = received_at or now_ist()

        ltp = quote.get("ltp") or quote.get("last_price")
        if ltp is None:
            incidents.append(
                Incident(IncidentType.SYMBOL_MAPPING_ERROR, instrument_key, "quote has no last traded price")
            )
            return incidents
        try:
            ltp = float(ltp)
        except (TypeError, ValueError):
            incidents.append(
                Incident(IncidentType.SYMBOL_MAPPING_ERROR, instrument_key, f"unparseable last price: {ltp!r}")
            )
            return incidents

        if ltp <= 0:
            incidents.append(Incident(IncidentType.IMPOSSIBLE_OHLC, instrument_key, f"non-positive price {ltp}"))

        bid = quote.get("bid")
        ask = quote.get("ask")
        if bid is not None and ask is not None:
            try:
                if float(ask) < float(bid):
                    incidents.append(
                        Incident(
                            IncidentType.NEGATIVE_SPREAD,
                            instrument_key,
                            f"ask {ask} < bid {bid}",
                            value=float(ask) - float(bid),
                        )
                    )
            except (TypeError, ValueError):
                pass

        quote_ts = quote.get("timestamp") or quote.get("ltt") or quote.get("last_traded_time")
        if quote_ts is not None:
            try:
                from ..timeutil import ensure_ist

                age = (received_at - ensure_ist(quote_ts)).total_seconds()
                if age > float(self._c("stale_quote_seconds", 5.0)):
                    incidents.append(
                        Incident(
                            IncidentType.STALE_QUOTE,
                            instrument_key,
                            f"quote is {age:.1f}s old",
                            value=age,
                            threshold=float(self._c("stale_quote_seconds", 5.0)),
                        )
                    )
            except Exception:
                pass
        return incidents

    # ------------------------------------------------------------- safe mode
    def evaluate(
        self,
        *,
        candles_by_instrument: Optional[Mapping[str, Sequence[Candle]]] = None,
        quotes: Optional[Mapping[str, Mapping[str, Any]]] = None,
        websocket_connected: Optional[bool] = None,
        websocket_age_seconds: Optional[float] = None,
        exchange_status: Optional[str] = None,
        instrument_master_loaded: Optional[bool] = None,
        universe_size: Optional[int] = None,
        resolved_instruments: Optional[int] = None,
    ) -> QualityReport:
        """Run every applicable check and decide whether trading is safe."""
        incidents: List[Incident] = []
        checks = 0

        if candles_by_instrument:
            for key, candles in candles_by_instrument.items():
                incidents.extend(self.check_candles(candles, instrument_key=key))
                checks += 1

        if quotes:
            for key, quote in quotes.items():
                incidents.extend(self.check_quote(quote, instrument_key=key))
                checks += 1

        if websocket_connected is False and websocket_age_seconds is not None:
            if websocket_age_seconds > float(self._c("websocket_reconnect_grace_seconds", 30)):
                incidents.append(
                    Incident(
                        IncidentType.WEBSOCKET_DISCONNECT,
                        None,
                        f"market-data feed disconnected for {websocket_age_seconds:.0f}s",
                        value=websocket_age_seconds,
                        threshold=float(self._c("websocket_reconnect_grace_seconds", 30)),
                    )
                )
            checks += 1

        if exchange_status is not None and exchange_status.upper() in ("CLOSED", "INACTIVE"):
            incidents.append(
                Incident(IncidentType.EXCHANGE_CLOSED, None, f"exchange status is {exchange_status}")
            )
            checks += 1

        if instrument_master_loaded is False:
            incidents.append(
                Incident(
                    IncidentType.SYMBOL_MAPPING_ERROR,
                    None,
                    "the Upstox instrument master is not loaded, so instrument_key values cannot be trusted",
                )
            )
            checks += 1
        if universe_size is not None and resolved_instruments is not None:
            if universe_size > 0 and resolved_instruments == 0:
                incidents.append(
                    Incident(
                        IncidentType.SYMBOL_MAPPING_ERROR,
                        None,
                        "none of the watchlist symbols resolved to an instrument_key",
                    )
                )
            checks += 1

        blocking = [i for i in incidents if i.blocking]
        safe = not blocking
        with self._lock:
            self._safe_mode = not safe

        if blocking:
            summary = f"DATA_SAFE_MODE: {len(blocking)} blocking incident(s) - " + ", ".join(
                sorted({i.incident_type.value for i in blocking})
            )
        elif incidents:
            summary = f"{len(incidents)} non-blocking data note(s)"
        else:
            summary = "all market-data checks passed"

        report = QualityReport(
            checked_at=now_ist(),
            safe_to_trade=safe,
            incidents=incidents,
            checks_run=checks,
            summary=summary,
            metrics={
                "instruments_checked": len(candles_by_instrument or {}),
                "quotes_checked": len(quotes or {}),
                "blocking_count": len(blocking),
            },
        )
        with self._lock:
            self._last_report = report
        if blocking:
            log.error("data safe mode engaged", context={"summary": summary})
        return report

    # ------------------------------------------------------------------ state
    @property
    def data_safe_mode(self) -> bool:
        with self._lock:
            return self._safe_mode

    @property
    def last_report(self) -> Optional[QualityReport]:
        with self._lock:
            return self._last_report

    def clear_safe_mode(self) -> None:
        with self._lock:
            self._safe_mode = False

    def status(self) -> Dict[str, Any]:
        report = self.last_report
        return {
            "data_safe_mode": self.data_safe_mode,
            "last_report": report.to_dict() if report else None,
        }

    def persist_incidents(self, session: Any, instrument_key: Optional[str] = None) -> int:
        """Write the last report's incidents to the database for the journal."""
        from .schema import DataQualityIncident

        report = self.last_report
        if report is None:
            return 0
        count = 0
        for incident in report.incidents:
            session.add(
                DataQualityIncident(
                    ts=incident.ts,
                    instrument_key=incident.instrument_key or instrument_key,
                    incident_type=incident.incident_type.value,
                    severity=incident.severity or "WARNING",
                    details={
                        "detail": incident.detail,
                        "value": incident.value,
                        "threshold": incident.threshold,
                        "blocking": incident.blocking,
                    },
                    resolved=report.safe_to_trade,
                    resolved_at=report.checked_at if report.safe_to_trade else None,
                )
            )
            count += 1
        return count


__all__ = ["DataQualityEngine", "Incident", "IncidentType", "QualityReport", "SEVERITY"]
