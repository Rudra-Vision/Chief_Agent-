"""Data routes: instruments, universe, historical download, candles, quality."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from ...data import candle_store
from ...data.db import session_scope
from ...data.historical_downloader import HistoricalDownloader
from ...data.schema import Candle as CandleRow
from ...data.schema import DataQualityIncident, Instrument
from ...logging_setup import get_logger
from ...timeutil import now_ist
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.data")
router = APIRouter()


@router.get("/data/instruments", summary="Search the instrument master")
def instruments(
    query: str = "",
    limit: int = Query(50, ge=1, le=500),
    universe_only: bool = False,
) -> Dict[str, Any]:
    with session_scope() as session:
        stmt = select(Instrument)
        if universe_only:
            stmt = stmt.where(Instrument.in_universe.is_(True))
        if query:
            pattern = f"%{query.upper()}%"
            stmt = stmt.where(
                (func.upper(Instrument.trading_symbol).like(pattern))
                | (func.upper(func.coalesce(Instrument.name, "")).like(pattern))
                | (func.upper(Instrument.instrument_key).like(pattern))
            )
        rows = session.execute(stmt.limit(limit)).scalars().all()
        total = session.execute(select(func.count(Instrument.id))).scalar_one()
        in_universe = session.execute(
            select(func.count(Instrument.id)).where(Instrument.in_universe.is_(True))
        ).scalar_one()
    return {
        "total_instruments": int(total),
        "in_universe": int(in_universe),
        "results": [
            {
                "instrument_key": row.instrument_key,
                "symbol": row.trading_symbol,
                "name": row.name,
                "sector": row.sector,
                "tier": row.tier,
                "segment": row.segment,
                "instrument_type": row.instrument_type,
                "isin": row.isin,
                "tick_size": row.tick_size,
                "lot_size": row.lot_size,
                "in_universe": row.in_universe,
                "is_suspended": row.is_suspended,
            }
            for row in rows
        ],
    }


@router.post("/data/instruments/refresh", summary="Download and load the Upstox instrument master")
def refresh_instruments(force: bool = False, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    from ...broker.upstox_instruments import load_instruments_into_db

    master = state.broker.instruments
    try:
        path = master.download("nse", force=True)
        try:
            master.download("suspended", force=True)
        except Exception as exc:  # optional extra file
            log.warning("suspended instrument file unavailable", context={"error": str(exc)})
        count = master.load("nse", auto_download=False)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"could not download the instrument master from Upstox: {exc}. "
                "This needs outbound network access to assets.upstox.com."
            ),
        ) from exc

    watchlist = state.broker.watchlist
    watchlist.load()
    with session_scope() as session:
        result = load_instruments_into_db(session, master, watchlist)
        session.commit()
    return {
        "ok": True,
        "instruments_loaded": count,
        "cache_path": str(path),
        **result,
        "note": "instrument_key is the canonical identifier; exchange_token is stored as metadata only.",
    }


@router.get("/data/universe", summary="The trading universe (watchlist)")
def universe(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
    sectors: Dict[str, int] = {}
    for row in resolved:
        sector = row.get("sector") or "Unknown"
        sectors[sector] = sectors.get(sector, 0) + 1
    return {
        "count": len(resolved),
        "resolved": sum(1 for row in resolved if row.get("resolved")),
        "unresolved": sum(1 for row in resolved if not row.get("resolved")),
        "sectors": dict(sorted(sectors.items(), key=lambda kv: -kv[1])),
        "items": resolved,
        "note": (
            "Symbols are resolved to instrument_key values using the official Upstox BOD master. "
            "Run 'Refresh instruments' first if the master is not loaded."
        ),
    }


@router.get("/data/coverage", summary="How much history is cached locally")
def coverage(
    timeframe: str = "1m",
    limit: int = Query(100, ge=1, le=1000),
    state: AppState = Depends(get_app_state),
) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(
                select(
                    CandleRow.instrument_key,
                    func.count(CandleRow.id).label("count"),
                    func.min(CandleRow.ts).label("first"),
                    func.max(CandleRow.ts).label("last"),
                )
                .where(CandleRow.timeframe == timeframe)
                .group_by(CandleRow.instrument_key)
                .order_by(func.count(CandleRow.id).desc())
                .limit(limit)
            )
            .all()
        )
        total = session.execute(
            select(func.count(CandleRow.id)).where(CandleRow.timeframe == timeframe)
        ).scalar_one()
    return {
        "timeframe": timeframe,
        "total_candles": int(total),
        "instruments_with_data": len(rows),
        "items": [
            {
                "instrument_key": row.instrument_key,
                "count": int(row.count),
                "first": row.first.isoformat() if row.first else None,
                "last": row.last.isoformat() if row.last else None,
            }
            for row in rows
        ],
    }


class DownloadBody(BaseModel):
    symbols: Optional[List[str]] = None
    timeframe: str = "1m"
    years: int = 1
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    max_instruments: int = 60
    force: bool = False


@router.post("/data/download", summary="Download historical candles (cache-first)")
def download(body: DownloadBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)

    if body.symbols:
        wanted = {s.strip().upper() for s in body.symbols}
        resolved = [row for row in resolved if row["symbol"] in wanted]

    # Attach pending instrument keys where the master is not loaded, so the user
    # can still exercise the pipeline (clearly marked as simulated).
    prepared: List[Dict[str, Any]] = []
    for row in resolved:
        key = row.get("instrument_key") or f"SIM|{row['symbol']}"
        prepared.append({"instrument_key": key, "symbol": row["symbol"]})

    # The benchmarks are needed for regime classification and relative strength,
    # so they are downloaded alongside the universe (clearly named and kept in
    # their own "__index__" namespace).
    benchmarks = [
        {"instrument_key": "__index__NIFTY 50", "symbol": "NIFTY 50"},
        {"instrument_key": "__index__India VIX", "symbol": "India VIX"},
    ]
    if state.broker.instruments.is_loaded:
        for name, key in (
            ("NIFTY 50", "NSE_INDEX|Nifty 50"),
            ("India VIX", "NSE_INDEX|India VIX"),
        ):
            benchmarks.append({"instrument_key": key, "symbol": name})
    prepared = benchmarks + prepared

    end_date = body.end_date or now_ist().date()
    start_date = body.start_date or (end_date - dt.timedelta(days=int(365 * max(1, body.years))))

    with session_scope() as session:
        downloader = HistoricalDownloader(state.provider, session, holidays=set())
        report = downloader.download(
            prepared[: max(1, body.max_instruments) + len(benchmarks)],
            body.timeframe,
            start_date=start_date,
            end_date=end_date,
            force=body.force,
        )
        session.commit()
    return {"ok": True, "report": report.to_dict(), "data_source": state.provider.data_source}


@router.get("/data/candles", summary="Read cached candles for a chart")
def candles(
    instrument_key: str,
    timeframe: str = "1m",
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    limit: int = Query(2000, ge=10, le=20000),
) -> Dict[str, Any]:
    end = end or now_ist().date()
    start = start or (end - dt.timedelta(days=10))
    with session_scope() as session:
        rows = candle_store.load_candles(session, instrument_key, timeframe, start, end, limit=limit)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"no cached candles for {instrument_key} ({timeframe}) between {start} and {end}",
        )
    return {
        "instrument_key": instrument_key,
        "timeframe": timeframe,
        "count": len(rows),
        "candles": [
            {
                "ts": row.ts.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            }
            for row in rows
        ],
    }


@router.get("/data/quality", summary="Data quality status and recent incidents")
def quality(limit: int = 50, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(
                select(DataQualityIncident).order_by(DataQualityIncident.ts.desc()).limit(limit)
            )
            .scalars()
            .all()
        )
    return {
        **state.data_quality.status(),
        "recent_incidents": [
            {
                "ts": row.ts.isoformat(),
                "instrument_key": row.instrument_key,
                "type": row.incident_type,
                "severity": row.severity,
                "details": row.details,
                "resolved": row.resolved,
            }
            for row in rows
        ],
    }


@router.post("/data/quality/check", summary="Run the data-quality checks now")
def run_quality_check(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    from ...data import candle_store as store

    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
    keys = [(row.get("instrument_key") or f"SIM|{row['symbol']}") for row in resolved][:20]

    end = now_ist().date()
    start = end - dt.timedelta(days=5)
    candles_by_instrument: Dict[str, Any] = {}
    with session_scope() as session:
        for key in keys:
            rows = store.load_candles(session, key, "1m", start, end, limit=5000)
            if rows:
                candles_by_instrument[key] = rows

    report = state.data_quality.evaluate(
        candles_by_instrument=candles_by_instrument,
        exchange_status=state.calendar.exchange_status().get("status"),
        instrument_master_loaded=state.broker.instruments.is_loaded,
        universe_size=len(resolved),
        resolved_instruments=sum(1 for row in resolved if row.get("resolved")),
    )
    with session_scope() as session:
        state.data_quality.persist_incidents(session)
        session.commit()
    return report.to_dict()


@router.get("/data/source", summary="Which market-data source is active")
def data_source(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return {
        **state.provider.status().to_dict(),
        "note": (
            "SIMULATED data is deterministic and clearly labelled. It exists so the system can be "
            "developed, tested and demonstrated without a broker token. It is NOT real market data "
            "and results computed from it are not evidence of any trading edge."
        ),
    }


__all__ = ["router"]
