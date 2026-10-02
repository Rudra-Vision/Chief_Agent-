"""Trading routes: opportunity scanner, paper positions, orders, risk context."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from ...ai.explainer import explain_opportunity
from ...data import candle_store
from ...data.db import session_scope
from ...data.schema import OpportunitySnapshot, Signal
from ...execution.engine import ExecutionEngine, OrderIntent
from ...indicators.features import FeatureEngine
from ...logging_setup import get_logger
from ...regime.engine import RegimeEngine, RegimeFeatureSeries
from ...scanner.opportunity import OpportunityScanner
from ...timeutil import now_ist
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.trading")
router = APIRouter()


def _scan_now(state: AppState, top_n: int = 10) -> Dict[str, Any]:
    """Run the scanner against the latest available bar of the cached data."""
    from ...data import candle_store

    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
    if not resolved:
        return {"error": "the watchlist is empty", "longs": [], "shorts": []}

    # Scan the LATEST session for which we actually hold data. Assuming "today"
    # would silently produce an empty scan after hours, at weekends, on holidays,
    # or whenever the historical cache has not been extended yet.
    with session_scope() as session:
        from sqlalchemy import func, select as _select

        from ...data.schema import Candle as CandleRow

        latest = session.execute(
            _select(func.max(CandleRow.ts)).where(CandleRow.timeframe == "1m")
        ).scalar_one()
    as_of = latest.date() if latest else now_ist().date()
    end = as_of
    start = end - dt.timedelta(days=15)
    feature_engine = FeatureEngine(opening_range_minutes=15)
    scanner = OpportunityScanner(state.strategy or None)  # type: ignore[arg-type]
    if state.strategy is None:
        return {"error": "no champion strategy is loaded", "longs": [], "shorts": []}

    series_map: Dict[str, Any] = {}
    symbols: Dict[str, str] = {}
    sector_map: Dict[str, str] = {}
    latest_ts: Optional[dt.datetime] = None

    with session_scope() as session:
        for row in resolved[:60]:
            key = row.get("instrument_key") or f"SIM|{row['symbol']}"
            candles = candle_store.load_candles(session, key, "1m", start, end, limit=6000)
            if not candles:
                continue
            series_map[key] = feature_engine.series(key, candles)
            symbols[key] = row["symbol"]
            sector_map[key] = row.get("sector") or "Unknown"
            if latest_ts is None or candles[-1].ts > latest_ts:
                latest_ts = candles[-1].ts

    if not series_map or latest_ts is None:
        return {
            "error": (
                "no cached candles for the current universe - run a historical download first "
                "(Data -> Download history)."
            ),
            "longs": [],
            "shorts": [],
            "as_of": None,
        }

    # ------------------------------------------------------------------ scan
    # A single instant is not a scan. We walk every evaluation point through the
    # session (inside the entry window, at a fixed step) and keep the
    # opportunities found at the MOST RECENT instant that produced any, together
    # with the full session's gate census. Nothing is invented: if the strategy
    # never fired all session, the result is an empty list and a census showing why.
    entry_window_end = float(
        (state.champion_config.get("long") or {}).get("entry_window_end_minutes", 210)
    )
    entry_window_start = float(
        (state.champion_config.get("long") or {}).get("entry_window_start_minutes", 15)
    )
    step_minutes = 5

    # Liquidity pre-filter. The scanner runs it before the (much more expensive)
    # strategy evaluation. Kept permissive here because the watchlist is already
    # restricted to liquid large caps; the risk engine enforces the hard floors.
    liquidity_config = {
        "min_average_daily_volume": 0,
        "min_average_daily_turnover": 0,
        "min_price": 0.0,
        "max_price": 1e9,
    }

    from ...timeutil import minutes_from_open

    # every timestamp present in the data for the chosen session
    day_timestamps = sorted(
        {
            candle.ts
            for series in series_map.values()
            for candle in series.candles
            if candle.ts.date() == as_of
        }
    )
    evaluation_points = [
        ts
        for ts in day_timestamps
        if entry_window_start <= minutes_from_open(ts) <= entry_window_end
    ][::step_minutes] or day_timestamps[-1:]

    # ---- benchmark / index series (regime + relative strength) -----------
    index_series = None
    index_key = "__index__NIFTY 50"
    with session_scope() as session:
        index_candles = candle_store.load_candles(session, index_key, "1m", start, end, limit=20000)
    if not index_candles:
        # Fetch it on demand so the scanner works the first time it is opened.
        index_candles = state.provider.history(index_key, "1m", start, end, symbol="NIFTY 50")
        if index_candles:
            with session_scope() as session:
                candle_store.upsert_candles(
                    session, index_key, "1m", index_candles,
                    source="synthetic" if state.provider.is_simulated else "upstox_v3",
                )
                session.commit()
    if index_candles:
        index_series = feature_engine.series(index_key, index_candles)

    regime_engine = RegimeEngine()
    regime_series = (
        RegimeFeatureSeries(index_series, vix_closes=None) if index_series is not None else None
    )
    sector_engine = scanner.sector_engine

    gate_census: Dict[str, int] = {}
    last_with_opportunities: Optional[Dict[str, Any]] = None
    evaluations = 0
    regime_counts: Dict[str, int] = {}

    for common_ts in evaluation_points:
        bar_index: Dict[str, int] = {}
        for key, series in series_map.items():
            for index in range(len(series.candles) - 1, -1, -1):
                if series.candles[index].ts <= common_ts:
                    bar_index[key] = index
                    break

        index_pointer = None
        nifty_return = None
        index_above_vwap = None
        regime = None
        regime_assessment = None
        if index_series is not None:
            for index in range(len(index_series.candles) - 1, -1, -1):
                if index_series.candles[index].ts <= common_ts:
                    index_pointer = index
                    break
        if index_pointer is not None and regime_series is not None:
            nifty_return = index_series.return_since_open(index_pointer)
            vwap = index_series.vwap[index_pointer]
            index_above_vwap = (index_series.closes[index_pointer] > vwap) if vwap else None
            regime_assessment = regime_engine.classify(regime_series.at(index_pointer), ts=common_ts)
            regime = regime_assessment.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        sector_snapshot = sector_engine.build_snapshot_fast(
            common_ts, sector_series={}, sector_pointers={}, nifty_return_pct=nifty_return
        )

        result = scanner.scan(
            ts=common_ts,
            bar_index_by_instrument=bar_index,
            series_map=series_map,
            symbols=symbols,
            sector_map=sector_map,
            regime=regime,
            regime_assessment=regime_assessment,
            nifty_return_pct=nifty_return,
            index_above_vwap=index_above_vwap,
            sector_snapshot=sector_snapshot,
            liquidity_config=liquidity_config,
            equity=state.portfolio.state.equity or state.paper.equity(),
            risk_pct=0.0025,
            top_n=top_n,
            data_source=state.provider.data_source,
        )
        evaluations += 1
        for gate, count in result.rejected_by_gate.items():
            gate_census[gate] = gate_census.get(gate, 0) + count
        if result.longs or result.shorts:
            last_with_opportunities = result.to_dict(top_n=top_n)
            last_with_opportunities["as_of"] = common_ts.isoformat()

    payload = last_with_opportunities or {
        "ts": evaluation_points[-1].isoformat() if evaluation_points else None,
        "as_of": evaluation_points[-1].isoformat() if evaluation_points else None,
        "regime": (max(regime_counts.items(), key=lambda kv: kv[1])[0] if regime_counts else None),
        "regime_confidence": 0.0,
        "evaluated": 0,
        "long_count": 0,
        "short_count": 0,
        "longs": [],
        "shorts": [],
        "rejected_by_gate": {},
        "sectors": None,
        "data_source": state.provider.data_source,
        "notes": [],
    }
    payload["rejected_by_gate"] = dict(sorted(gate_census.items(), key=lambda kv: -kv[1])[:12])
    payload["evaluations"] = evaluations
    payload["session"] = as_of.isoformat()
    payload["regime_mix"] = regime_counts
    payload["as_of_note"] = (
        f"Scanned {evaluations} evaluation point(s) across the {as_of} session, every "
        f"{step_minutes} minutes between {entry_window_start:.0f} and {entry_window_end:.0f} minutes "
        "after the open. Opportunities shown are from the most recent point that produced any; "
        "the gate census counts every rejection across the whole session."
    )
    if not payload.get("longs") and not payload.get("shorts"):
        payload["notes"] = list(payload.get("notes") or []) + [
            "No qualifying setup was found in this session. The gate census above shows exactly "
            "which filters blocked candidates."
        ]
    payload["universe_scanned"] = len(series_map)
    payload["equity"] = round(state.portfolio.state.equity or state.paper.equity(), 2)
    return payload





@router.get("/trading/opportunities", summary="Ranked LONG and SHORT opportunities")
def opportunities(
    top_n: int = Query(10, ge=1, le=50),
    refresh: bool = Query(True, description="Run a fresh scan"),
    state: AppState = Depends(get_app_state),
) -> Dict[str, Any]:
    if refresh or not state.last_scan:
        try:
            state.last_scan = _scan_now(state, top_n=top_n)
            state.counters.last_scan_at = now_ist()
        except Exception as exc:
            log.exception("scan failed")
            state.counters.last_error = str(exc)
            raise HTTPException(status_code=500, detail=f"scan failed: {exc}") from exc
    payload = dict(state.last_scan)
    payload["data_source"] = state.provider.data_source
    payload["is_simulated"] = state.provider.is_simulated
    return payload


@router.get("/trading/opportunities/{instrument_key}/explain", summary="Explain one opportunity")
def explain(instrument_key: str, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    payload = state.last_scan or {}
    for row in list(payload.get("longs", [])) + list(payload.get("shorts", [])):
        if row.get("instrument_key") == instrument_key:
            explanation = explain_opportunity(row, account_equity=payload.get("equity"))
            return {"opportunity": row, "explanation": explanation.to_dict()}
    raise HTTPException(status_code=404, detail="that instrument is not in the current opportunity list")


@router.get("/trading/positions", summary="Open paper positions and exposure")
def positions(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    portfolio = state.portfolio.update_from_paper(state.paper)
    return {
        "mode": state.settings.effective_mode().value,
        **portfolio.to_dict(),
        "positions": [p.to_dict() for p in state.paper.open_positions()],
        "slippage": state.slippage.summary().to_dict(),
        "data_source": state.provider.data_source,
    }


@router.get("/trading/paper/account", summary="Paper account snapshot")
def paper_account(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    snapshot = state.paper.snapshot()
    snapshot["equity_recomputed"] = round(state.paper.equity(), 2)
    snapshot["mode"] = state.settings.effective_mode().value
    return snapshot


class ManualOrderBody(BaseModel):
    instrument_key: str
    direction: str = "LONG"
    quantity: int
    entry_price: float
    stop_price: float
    target_1: float = 0.0
    target_2: float = 0.0
    reason: str = "manual order from the dashboard"


@router.post("/trading/paper/order", summary="Place a manual PAPER order (risk-checked)")
def manual_order(body: ManualOrderBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    """Manually submit a paper order. It still goes through the FULL risk engine
    and the execution lifecycle - the dashboard cannot bypass risk."""
    if state.settings.effective_mode().value == "LIVE":
        raise HTTPException(
            status_code=403,
            detail="manual orders are disabled in LIVE mode. Use the strategy pipeline.",
        )
    direction = body.direction.upper()
    if direction not in ("LONG", "SHORT"):
        raise HTTPException(status_code=400, detail="direction must be LONG or SHORT")
    if body.stop_price <= 0:
        raise HTTPException(status_code=400, detail="a stop price is mandatory")

    target_1 = body.target_1 or (
        body.entry_price + 1.5 * abs(body.entry_price - body.stop_price)
        if direction == "LONG"
        else body.entry_price - 1.5 * abs(body.entry_price - body.stop_price)
    )
    target_2 = body.target_2 or (
        body.entry_price + 2.5 * abs(body.entry_price - body.stop_price)
        if direction == "LONG"
        else body.entry_price - 2.5 * abs(body.entry_price - body.stop_price)
    )

    portfolio = state.portfolio.update_from_paper(state.paper)
    context = state.portfolio.risk_context(
        strategy_version=state.strategy_version or "",
        suspended_strategies=[],
        kill_switch_engaged=state.kill_switch.engaged,
        data_safe_mode=state.data_quality.data_safe_mode,
        data_stale=False,
        reconciliation_pending=state.reconciler.pending,
        engine_healthy=True,
        order_attempts_today=state.counters.orders_submitted,
    )
    intent = OrderIntent(
        trade_id=f"MAN-{now_ist().strftime('%Y%m%d%H%M%S')}",
        signal_id=f"MAN-SIG-{now_ist().strftime('%Y%m%d%H%M%S')}",
        strategy_version=state.strategy_version or "manual",
        instrument_key=body.instrument_key,
        symbol=body.instrument_key.split("|")[-1],
        direction=direction,
        quantity=body.quantity,
        entry_price=body.entry_price,
        stop_price=body.stop_price,
        target_1=target_1,
        target_2=target_2,
        features={"manual": True},
    )
    engine = ExecutionEngine(
        mode=state.settings.effective_mode(),
        risk_engine=state.risk_engine,
        paper_engine=state.paper,
        orders_api=None,
        kill_switch=state.kill_switch,
        preflight_check=lambda: state.preflight.is_fresh() and (state.preflight.last_report() or None) is not None
        and bool(state.preflight.last_report() and state.preflight.last_report().passed),
        data_quality_check=lambda: not state.data_quality.data_safe_mode,
        reconciliation_check=lambda: not state.reconciler.pending,
    )
    result = engine.submit(intent, context)
    state.counters.orders_submitted += 1
    if result.ok:
        state.counters.orders_filled += 1
        state.notifications.trade_entered(
            intent.symbol, direction, result.filled_quantity, result.average_price
        )
    else:
        state.counters.signals_rejected += 1
    return {
        "ok": result.ok,
        "mode": state.settings.effective_mode().value,
        "result": result.to_dict(),
        "note": "Manual paper orders pass through the same risk engine as strategy signals.",
    }


@router.post("/trading/positions/{position_id}/close", summary="Close a paper position")
def close_position(position_id: str, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    position = state.paper.positions.get(position_id)
    if position is None or not position.is_open:
        raise HTTPException(status_code=404, detail="open position not found")
    price = position.ltp or position.entry_price
    record = state.paper.close_position(position, price, now_ist(), "MANUAL")
    state.counters.trades_completed += 1
    state.notifications.trade_exited(
        position.symbol, "MANUAL", record["net_pnl"], record["r_multiple"]
    )
    return {"ok": True, "trade": record}


@router.post("/trading/positions/close-all", summary="Close every paper position")
def close_all(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    closed: List[Dict[str, Any]] = []
    for position in list(state.paper.open_positions()):
        price = position.ltp or position.entry_price
        record = state.paper.close_position(position, price, now_ist(), "MANUAL")
        closed.append(record)
        state.counters.trades_completed += 1
    return {"ok": True, "closed": len(closed), "trades": closed}


@router.post("/trading/cycle", summary="Run one trading cycle now (scan -> risk -> paper execution)")
def run_cycle(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    """One full cycle: mark to market -> manage exits -> scan -> risk -> execute."""
    from ...execution.engine import ExecutionEngine

    scan = _scan_now(state, top_n=10)
    state.last_scan = scan
    state.counters.last_scan_at = now_ist()
    if scan.get("error"):
        return {"ok": False, "stage": "scan", "error": scan["error"]}

    portfolio = state.portfolio.update_from_paper(state.paper)
    executed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    engine = ExecutionEngine(
        mode=state.settings.effective_mode(),
        risk_engine=state.risk_engine,
        paper_engine=state.paper,
        orders_api=None,
        kill_switch=state.kill_switch,
        reconciliation_check=lambda: not state.reconciler.pending,
        data_quality_check=lambda: not state.data_quality.data_safe_mode,
    )

    for opportunity in (scan.get("longs", []) + scan.get("shorts", []))[:5]:
        if state.kill_switch.engaged:
            break
        context = state.portfolio.risk_context(
            strategy_version=opportunity.get("strategy_version", ""),
            kill_switch_engaged=state.kill_switch.engaged,
            data_safe_mode=state.data_quality.data_safe_mode,
            reconciliation_pending=state.reconciler.pending,
        )
        intent = OrderIntent(
            trade_id=f"CYC-{opportunity.get('instrument_key','')[-6:]}-{now_ist().strftime('%H%M%S')}",
            signal_id=f"SIG-{opportunity.get('instrument_key','')[-6:]}-{now_ist().strftime('%H%M%S')}",
            strategy_version=opportunity.get("strategy_version", state.strategy_version or ""),
            instrument_key=opportunity["instrument_key"],
            symbol=opportunity["symbol"],
            direction=opportunity["direction"],
            quantity=max(1, int(opportunity.get("quantity_hint", 1) or 1)),
            entry_price=float(opportunity["entry_price"]),
            stop_price=float(opportunity["stop_price"]),
            target_1=float(opportunity["target_1"]),
            target_2=float(opportunity["target_2"]),
            sector=opportunity.get("sector"),
            regime=opportunity.get("regime"),
            features={
                "score": opportunity.get("score"),
                "rvol": opportunity.get("rvol"),
                "atr_daily_pct": opportunity.get("atr_daily_pct"),
                "sector_rank": opportunity.get("sector_rank"),
            },
        )
        result = engine.submit(
            intent,
            context,
            spread_pct=opportunity.get("spread_pct"),
            average_daily_volume=None,
        )
        if result.ok:
            state.counters.orders_submitted += 1
            state.counters.orders_filled += 1
            executed.append({"symbol": intent.symbol, "direction": intent.direction, **result.to_dict()})
            state.notifications.trade_entered(
                intent.symbol, intent.direction, result.filled_quantity, result.average_price
            )
        else:
            state.counters.signals_rejected += 1
            rejected.append(
                {
                    "symbol": intent.symbol,
                    "direction": intent.direction,
                    "reason": result.message,
                    "risk": result.risk_decision.to_dict() if result.risk_decision else None,
                }
            )

    state.counters.last_cycle_at = now_ist()
    return {
        "ok": True,
        "scanned": scan.get("universe_scanned", 0),
        "opportunities": len(scan.get("longs", [])) + len(scan.get("shorts", [])),
        "executed": executed,
        "rejected": rejected,
        "portfolio": state.portfolio.snapshot(),
    }


@router.get("/trading/signals", summary="Recent signals")
def recent_signals(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(select(Signal).order_by(Signal.ts.desc()).limit(limit)).scalars().all()
        )
    return {
        "count": len(rows),
        "signals": [
            {
                "signal_id": row.signal_id,
                "ts": row.ts.isoformat(),
                "symbol": row.symbol,
                "direction": row.direction,
                "score": row.score,
                "status": row.status,
                "rejection_reason": row.rejection_reason,
                "strategy_version": row.strategy_version,
                "entry_price": row.entry_price,
                "stop_price": row.stop_price,
                "target_1": row.target_1,
                "risk_reward": row.risk_reward,
                "reasons": row.reasons,
            }
            for row in rows
        ],
    }


@router.get("/trading/slippage", summary="Execution slippage analysis")
def slippage(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    return {
        "summary": state.slippage.summary().to_dict(),
        "records": [
            record.to_dict() for record in state.slippage.records[-200:]
        ],
    }


__all__ = ["router"]
