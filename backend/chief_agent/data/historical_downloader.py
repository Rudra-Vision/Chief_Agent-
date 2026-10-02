"""Historical data ingestion.

Behaviour
---------
* **Cache first.** Data already in the local database is never re-requested, so
  the Upstox historical APIs are called exactly once per (instrument, timeframe,
  day). This respects both the rate limits and the user's API budget.
* Only *missing* business days are fetched (:func:`missing_ranges`).
* Failures are per-instrument: one bad symbol cannot abort a whole download.
* Everything is written through :func:`upsert_candles`, which never overwrites
  existing source rows.

The downloader records a :class:`SystemEvent` per run so the dashboard can show
what was fetched and when.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from sqlalchemy.orm import Session

from ..broker.upstox_market_data import TIMEFRAME_TO_UNIT
from ..logging_setup import get_logger
from ..timeutil import IST, ist_date, now_ist
from . import candle_store
from .provider import DATA_SOURCE_LIVE, DATA_SOURCE_SIMULATED, MarketDataProvider

log = get_logger(__name__, component="downloader")


@dataclass
class DownloadReport:
    started_at: dt.datetime
    finished_at: Optional[dt.datetime] = None
    data_source: str = "UNKNOWN"
    timeframe: str = "1m"
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    instruments_requested: int = 0
    instruments_fetched: int = 0
    instruments_skipped_cached: int = 0
    candles_inserted: int = 0
    candles_duplicate: int = 0
    candles_invalid: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or now_ist()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "data_source": self.data_source,
            "timeframe": self.timeframe,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "instruments_requested": self.instruments_requested,
            "instruments_fetched": self.instruments_fetched,
            "instruments_skipped_cached": self.instruments_skipped_cached,
            "candles_inserted": self.candles_inserted,
            "candles_duplicate": self.candles_duplicate,
            "candles_invalid": self.candles_invalid,
            "failure_count": len(self.failures),
            "failures": self.failures[:20],
            "skipped": self.skipped[:20],
        }


class HistoricalDownloader:
    """Fetches and caches historical candles."""

    def __init__(self, provider: MarketDataProvider, session: Session, holidays: Optional[Iterable[dt.date]] = None) -> None:
        self.provider = provider
        self.session = session
        self.holidays = set(holidays or ())

    # ------------------------------------------------------------------ public
    def download(
        self,
        instruments: Sequence[Mapping[str, Any]],
        timeframe: str = "1m",
        *,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        years: int = 1,
        force: bool = False,
        progress: Optional[Callable[[int, int, str, int], None]] = None,
        max_instruments: Optional[int] = None,
    ) -> DownloadReport:
        if timeframe not in TIMEFRAME_TO_UNIT:
            raise ValueError(f"unsupported timeframe {timeframe}; supported: {sorted(TIMEFRAME_TO_UNIT)}")

        end_date = end_date or now_ist().date()
        if start_date is None:
            start_date = end_date - dt.timedelta(days=int(365 * years))

        report = DownloadReport(
            started_at=now_ist(),
            data_source=self.provider.data_source,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        selected = list(instruments)
        if max_instruments:
            selected = selected[:max_instruments]
        report.instruments_requested = len(selected)

        for index, item in enumerate(selected, start=1):
            instrument_key = item.get("instrument_key") or item.get("symbol")
            symbol = item.get("symbol") or instrument_key
            if not instrument_key:
                continue

            try:
                if not force:
                    gaps = candle_store.missing_ranges(
                        self.session, instrument_key, timeframe, start_date, end_date, self.holidays
                    )
                    if not gaps:
                        report.instruments_skipped_cached += 1
                        report.skipped.append(symbol)
                        if progress:
                            progress(index, len(selected), symbol, 0)
                        continue
                    fetch_start = min(g[0] for g in gaps)
                    fetch_end = max(g[1] for g in gaps)
                else:
                    fetch_start, fetch_end = start_date, end_date

                candles = self.provider.history(
                    instrument_key, timeframe, fetch_start, fetch_end, symbol=item.get("symbol")
                )
                if not candles:
                    report.failures.append(
                        {"instrument_key": instrument_key, "symbol": symbol, "error": "no candles returned"}
                    )
                    if progress:
                        progress(index, len(selected), symbol, 0)
                    continue

                result = candle_store.upsert_candles(
                    self.session,
                    instrument_key,
                    timeframe,
                    candles,
                    source="upstox_v3" if report.data_source == DATA_SOURCE_LIVE else "synthetic",
                )
                self.session.flush()
                report.instruments_fetched += 1
                report.candles_inserted += result.inserted
                report.candles_duplicate += result.duplicates
                report.candles_invalid += result.invalid
                if progress:
                    progress(index, len(selected), symbol, result.inserted)
            except Exception as exc:  # never abort the whole download for one symbol
                log.warning("download failed for instrument", context={"symbol": symbol, "error": str(exc)})
                self.session.rollback()
                report.failures.append({"instrument_key": instrument_key, "symbol": symbol, "error": str(exc)})

        report.finished_at = now_ist()
        log.info("historical download finished", context=report.to_dict())
        self._record_event(report)
        return report

    # ------------------------------------------------------------- aggregation
    def build_higher_timeframes(
        self,
        instrument_keys: Sequence[str],
        timeframes: Sequence[str] = ("5m", "15m", "30m", "60m"),
        *,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
    ) -> Dict[str, int]:
        """Aggregate 1-minute source data into higher timeframes (derived rows).

        Only complete buckets are produced, so a partial bar can never leak into
        a strategy or a backtest.
        """
        from .synthetic import aggregate_candles

        written: Dict[str, int] = {}
        end_date = end_date or now_ist().date()
        start_date = start_date or (end_date - dt.timedelta(days=400))

        for instrument_key in instrument_keys:
            base = candle_store.load_candles(self.session, instrument_key, "1m", start_date, end_date)
            if not base:
                continue
            for timeframe in timeframes:
                minutes = _minutes_for(timeframe)
                if minutes <= 1:
                    continue
                aggregated = aggregate_candles(base, minutes, timeframe)
                result = candle_store.upsert_candles(
                    self.session, instrument_key, timeframe, aggregated, source="derived", is_derived=True
                )
                self.session.flush()
                written[f"{instrument_key}:{timeframe}"] = result.inserted
        return written

    # ------------------------------------------------------------------ events
    def _record_event(self, report: DownloadReport) -> None:
        try:
            from .schema import SystemEvent

            self.session.add(
                SystemEvent(
                    category="data",
                    level="INFO" if not report.failures else "WARNING",
                    component="historical_downloader",
                    message=(
                        f"Downloaded {report.candles_inserted} candles for "
                        f"{report.instruments_fetched}/{report.instruments_requested} instruments "
                        f"({report.data_source})"
                    ),
                    details=report.to_dict(),
                )
            )
            self.session.flush()
        except Exception as exc:  # pragma: no cover - logging must never break ingestion
            log.warning("could not record download event", context={"error": str(exc)})


def _minutes_for(timeframe: str) -> int:
    from ..broker.upstox_market_data import TIMEFRAME_MINUTES

    return TIMEFRAME_MINUTES.get(timeframe, 1)


__all__ = ["HistoricalDownloader", "DownloadReport"]
