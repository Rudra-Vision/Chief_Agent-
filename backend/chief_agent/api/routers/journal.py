"""Trade journal routes: trades, post-trade classification, daily review."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from ...ai.explainer import daily_summary, explain_trade
from ...data.db import session_scope
from ...data.schema import BacktestTrade, DailyPerformance, RiskEvent, TradeJournal
from ...logging_setup import get_logger
from ...timeutil import now_ist
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.journal")
router = APIRouter()


#: Post-trade process classification. A losing trade is NOT automatically a
#: mistake and a winning trade is NOT automatically a good decision.
GOOD_PROCESS_EXIT_REASONS = {"STOP_LOSS", "TARGET_1", "TARGET_2", "TIME_STOP", "SQUARE_OFF"}


def classify_trade(row: TradeJournal) -> str:
    """Classify a trade as good/bad PROCESS x good/bad OUTCOME."""
    followed_plan = row.exit_reason in GOOD_PROCESS_EXIT_REASONS
    # A trade that was entered without a stop, sized outside the risk budget, or
    # exited for a non-strategy reason is a process violation.
    if row.exit_reason in ("MANUAL", "MANUAL_EXIT", "OVERRIDE"):
        followed_plan = False
    if row.initial_risk <= 0:
        followed_plan = False

    good_outcome = row.net_pnl > 0
    if followed_plan and good_outcome:
        return "GOOD_TRADE_GOOD_OUTCOME"
    if followed_plan and not good_outcome:
        return "GOOD_TRADE_BAD_OUTCOME"
    if not followed_plan and good_outcome:
        return "BAD_TRADE_GOOD_OUTCOME"
    return "BAD_TRADE_BAD_OUTCOME"


@router.get("/journal/trades", summary="Trade journal")
def trades(
    limit: int = Query(100, ge=1, le=1000),
    strategy_version: Optional[str] = None,
    symbol: Optional[str] = None,
    outcome: Optional[str] = Query(None, pattern="^(win|loss|all)$"),
) -> Dict[str, Any]:
    with session_scope() as session:
        stmt = select(TradeJournal).where(TradeJournal.mode != "BACKTEST")
        if strategy_version:
            stmt = stmt.where(TradeJournal.strategy_version == strategy_version)
        if symbol:
            stmt = stmt.where(TradeJournal.symbol == symbol.upper())
        if outcome == "win":
            stmt = stmt.where(TradeJournal.net_pnl > 0)
        elif outcome == "loss":
            stmt = stmt.where(TradeJournal.net_pnl < 0)
        rows = (
            session.execute(stmt.order_by(TradeJournal.exit_ts.desc()).limit(limit)).scalars().all()
        )

    return {
        "count": len(rows),
        "trades": [
            {
                "trade_id": row.trade_id,
                "symbol": row.symbol,
                "sector": row.sector,
                "direction": row.direction,
                "quantity": row.quantity,
                "entry_ts": row.entry_ts.isoformat(),
                "exit_ts": row.exit_ts.isoformat(),
                "entry_price": row.entry_price,
                "exit_price": row.exit_price,
                "net_pnl": row.net_pnl,
                "fees": row.fees,
                "slippage_cost": row.slippage_cost,
                "r_multiple": row.r_multiple,
                "mae_r": row.mae_r,
                "mfe_r": row.mfe_r,
                "holding_minutes": row.holding_minutes,
                "exit_reason": row.exit_reason,
                "regime_at_entry": row.regime_at_entry,
                "strategy_version": row.strategy_version,
                "process_quality": row.process_quality or classify_trade(row),
                "mode": row.mode,
                "lesson": row.lesson,
            }
            for row in rows
        ],
    }


@router.get("/journal/trades/{trade_id}", summary="One trade with its full story")
def trade_detail(trade_id: str) -> Dict[str, Any]:
    with session_scope() as session:
        row = session.execute(
            select(TradeJournal).where(TradeJournal.trade_id == trade_id)
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="trade not found")

    payload = {
        "trade_id": row.trade_id,
        "symbol": row.symbol,
        "sector": row.sector,
        "direction": row.direction,
        "quantity": row.quantity,
        "entry_ts": row.entry_ts.isoformat(),
        "exit_ts": row.exit_ts.isoformat(),
        "entry_price": row.entry_price,
        "exit_price": row.exit_price,
        "net_pnl": row.net_pnl,
        "gross_pnl": row.gross_pnl,
        "fees": row.fees,
        "slippage_cost": row.slippage_cost,
        "r_multiple": row.r_multiple,
        "mae_r": row.mae_r,
        "mfe_r": row.mfe_r,
        "holding_minutes": row.holding_minutes,
        "exit_reason": row.exit_reason,
        "entry_reason": row.entry_reason,
        "regime_at_entry": row.regime_at_entry,
        "regime_at_exit": row.regime_at_exit,
        "vix_at_entry": row.vix_at_entry,
        "sector_rank_at_entry": row.sector_rank_at_entry,
        "strategy_version": row.strategy_version,
        "features": row.features,
        "execution_quality": row.execution_quality,
        "news_context": row.news_context,
        "process_quality": row.process_quality or classify_trade(row),
        "lesson": row.lesson,
        "mode": row.mode,
    }
    return {"trade": payload, "explanation": explain_trade(payload).to_dict()}


@router.get("/journal/daily", summary="Daily performance series")
def daily(limit: int = Query(90, ge=1, le=1000)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(
                select(DailyPerformance).order_by(DailyPerformance.trading_date.desc()).limit(limit)
            )
            .scalars()
            .all()
        )
    return {
        "count": len(rows),
        "days": [
            {
                "trading_date": row.trading_date.isoformat(),
                "mode": row.mode,
                "starting_equity": row.starting_equity,
                "ending_equity": row.ending_equity,
                "net_pnl": row.net_pnl,
                "return_pct": row.return_pct,
                "trades": row.trades,
                "wins": row.wins,
                "losses": row.losses,
                "expectancy_r": row.expectancy_r,
                "max_drawdown_pct": row.max_drawdown_pct,
                "regime_summary": row.regime_summary,
                "notes": row.notes,
            }
            for row in rows
        ],
    }


@router.get("/journal/review", summary="Daily review (today or a chosen date)")
def review(
    date: Optional[dt.date] = None,
    state: AppState = Depends(get_app_state),
) -> Dict[str, Any]:
    trading_date = date or now_ist().date()
    start = dt.datetime(trading_date.year, trading_date.month, trading_date.day, tzinfo=now_ist().tzinfo)
    end = start + dt.timedelta(days=1)

    with session_scope() as session:
        rows = (
            session.execute(
                select(TradeJournal).where(
                    TradeJournal.exit_ts >= start, TradeJournal.exit_ts < end
                )
            )
            .scalars()
            .all()
        )
        risk_events = (
            session.execute(select(RiskEvent).where(RiskEvent.ts >= start, RiskEvent.ts < end)).scalars().all()
        )
        performance = session.execute(
            select(DailyPerformance).where(DailyPerformance.trading_date == trading_date)
        ).scalar_one_or_none()

    trade_payloads = [
        {
            "trade_id": row.trade_id,
            "symbol": row.symbol,
            "direction": row.direction,
            "net_pnl": row.net_pnl,
            "r_multiple": row.r_multiple,
            "exit_reason": row.exit_reason,
            "regime_at_entry": row.regime_at_entry,
            "sector": row.sector,
        }
        for row in rows
    ]
    equity_start = performance.starting_equity if performance else (state.paper.equity() if not rows else 0.0)
    equity_end = performance.ending_equity if performance else (
        equity_start + sum(float(t.net_pnl or 0) for t in rows)
    )

    summary = daily_summary(
        trading_date=trading_date,
        trades=trade_payloads,
        equity_start=equity_start,
        equity_end=equity_end,
        regime=state.last_scan.get("regime") if state.last_scan else None,
        data_incidents=0,
        risk_events=len(risk_events),
    )

    process_counts: Dict[str, int] = {}
    for row in rows:
        quality = row.process_quality or classify_trade(row)
        process_counts[quality] = process_counts.get(quality, 0) + 1

    return {
        **summary,
        "process_quality_counts": process_counts,
        "risk_events": [
            {
                "ts": event.ts.isoformat(),
                "type": event.event_type,
                "severity": event.severity,
                "action": event.action_taken,
                "message": event.message,
            }
            for event in risk_events
        ],
        "trades": trade_payloads,
        "note": (
            "A losing trade is not automatically a mistake: GOOD_TRADE_BAD_OUTCOME means the "
            "process was followed and the outcome was simply negative. Only process violations "
            "count against the strategy."
        ),
    }


@router.get("/journal/performance", summary="Performance analytics summary")
def performance(limit_days: int = Query(180, ge=10, le=2000)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(
                select(DailyPerformance).order_by(DailyPerformance.trading_date.desc()).limit(limit_days)
            )
            .scalars()
            .all()
        )
        trade_stats = session.execute(
            select(
                func.count(TradeJournal.id),
                func.sum(TradeJournal.net_pnl),
                func.sum(TradeJournal.fees),
                func.avg(TradeJournal.r_multiple),
            )
        ).one()

    ordered = list(reversed(rows))
    equities = [row.ending_equity for row in ordered if row.ending_equity]
    from ...backtest.metrics import curve_statistics, drawdown_stats

    curve = curve_statistics(equities) if len(equities) > 1 else None
    drawdown = drawdown_stats(equities) if len(equities) > 1 else None
    return {
        "days": len(rows),
        "total_trades": int(trade_stats[0] or 0),
        "total_net_pnl": round(float(trade_stats[1] or 0), 2),
        "total_fees": round(float(trade_stats[2] or 0), 2),
        "average_r_multiple": round(float(trade_stats[3] or 0), 4),
        "curve": curve.to_dict() if curve else None,
        "drawdown": drawdown.to_dict() if drawdown else None,
        "equity_series": [
            {"date": row.trading_date.isoformat(), "equity": row.ending_equity} for row in ordered
        ],
    }


__all__ = ["router", "classify_trade"]
