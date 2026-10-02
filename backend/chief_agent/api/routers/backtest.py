"""Backtest routes: run a backtest, list runs, inspect results and trades."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ...backtest.runner import BacktestRequest, BacktestRunner
from ...backtest.montecarlo import run_monte_carlo
from ...backtest.robustness import RobustnessRunner
from ...costs.transaction_costs import CostModel
from ...data.db import session_scope
from ...data.schema import BacktestRun, BacktestTrade
from ...logging_setup import get_logger
from ...timeutil import now_ist
from ..state import AppState, get_app_state

log = get_logger(__name__, component="api.backtest")
router = APIRouter()


class BacktestBody(BaseModel):
    start_date: dt.date
    end_date: dt.date
    universe: Optional[List[str]] = None
    initial_capital: float = Field(500_000.0, gt=0, le=100_000_000)
    risk_per_trade_pct: float = Field(0.0025, gt=0, le=0.005)
    timeframe: str = "1m"
    strategy_version: Optional[str] = None
    slippage_multiplier: float = Field(1.0, ge=0.0, le=10.0)
    fee_multiplier: float = Field(1.0, ge=0.0, le=10.0)
    include_shorts: bool = True
    max_universe: int = Field(40, ge=1, le=150)
    run_robustness: bool = False
    run_monte_carlo: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
                "universe": ["RELIANCE", "INFY", "HDFCBANK"],
                "initial_capital": 500000,
                "risk_per_trade_pct": 0.0025,
            }
        }


@router.get("/backtest/defaults", summary="Default backtest parameters and universe")
def defaults(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
    end = now_ist().date()
    return {
        "start_date": (end - dt.timedelta(days=365)).isoformat(),
        "end_date": end.isoformat(),
        "universe": [row["symbol"] for row in resolved][:60],
        "universe_size": len(resolved),
        "initial_capital": 500_000,
        "risk_per_trade_pct": 0.0025,
        "strategy_version": state.strategy_version,
        "cost_model": CostModel().describe(),
        "data_source": state.provider.data_source,
        "is_simulated": state.provider.is_simulated,
    }


@router.post("/backtest/run", summary="Run a backtest")
def run_backtest(body: BacktestBody, state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    if body.end_date <= body.start_date:
        raise HTTPException(status_code=400, detail="end_date must be after start_date")

    watchlist = state.broker.watchlist
    watchlist.load()
    resolved = watchlist.resolved(state.broker.instruments if state.broker.instruments.is_loaded else None)
    available = {row["symbol"]: row for row in resolved}

    universe = body.universe or list(available)[: body.max_universe]
    missing = [symbol for symbol in universe if symbol not in available]
    if missing and body.universe:
        raise HTTPException(
            status_code=400,
            detail=f"these symbols are not in the watchlist: {', '.join(missing[:10])}",
        )

    strategy_config = None
    version = body.strategy_version or state.strategy_version
    if version:
        from ...research.champion import ChampionRegistry

        with session_scope() as session:
            registry = ChampionRegistry(session)
            strategy_config = registry.config_for(version)

    request = BacktestRequest(
        start_date=body.start_date,
        end_date=body.end_date,
        universe=list(universe),
        initial_capital=body.initial_capital,
        risk_per_trade_pct=body.risk_per_trade_pct,
        timeframe=body.timeframe,
        strategy_config=strategy_config,
        strategy_version=version or "ORB_v1.0.0",
        slippage_multiplier=body.slippage_multiplier,
        fee_multiplier=body.fee_multiplier,
        include_shorts=body.include_shorts,
        max_universe=body.max_universe,
    )

    try:
        with session_scope() as session:
            runner = BacktestRunner(session, state.provider, max_instruments=body.max_universe)
            result = runner.run(request)
            backtest_id = result.metrics.extras.get("backtest_id")
            session.commit()
    except Exception as exc:
        log.exception("backtest failed")
        raise HTTPException(status_code=500, detail=f"backtest failed: {exc}") from exc

    payload = result.to_dict(include_trades=False)
    payload["backtest_id"] = backtest_id
    payload["data_source"] = result.data_source
    payload["is_simulated"] = state.provider.is_simulated

    if body.run_monte_carlo:
        monte_carlo = run_monte_carlo(
            [t.net_pnl for t in result.trades],
            initial_capital=body.initial_capital,
            n_simulations=1000,
        )
        payload["monte_carlo"] = monte_carlo.to_dict()

    if body.run_robustness:
        payload["robustness_note"] = (
            "Robustness testing runs inside the research pipeline (Research -> Run experiment). "
            "It requires a challenger version to compare against."
        )

    return payload


@router.get("/backtest/runs", summary="List backtest runs")
def list_runs(limit: int = Query(25, ge=1, le=200)) -> Dict[str, Any]:
    with session_scope() as session:
        rows = (
            session.execute(select(BacktestRun).order_by(BacktestRun.started_at.desc()).limit(limit))
            .scalars()
            .all()
        )
        total = session.execute(select(func.count(BacktestRun.id))).scalar_one()
    return {
        "total": int(total),
        "runs": [
            {
                "backtest_id": row.backtest_id,
                "strategy_version": row.strategy_version,
                "start_date": row.start_date.isoformat(),
                "end_date": row.end_date.isoformat(),
                "trade_count": row.trade_count,
                "status": row.status,
                "metrics": row.metrics,
                "created_at": row.started_at.isoformat() if row.started_at else None,
            }
            for row in rows
        ],
    }


@router.get("/backtest/{backtest_id}", summary="Backtest detail with curves and trades")
def backtest_detail(backtest_id: str, trade_limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
    with session_scope() as session:
        run = session.execute(
            select(BacktestRun).where(BacktestRun.backtest_id == backtest_id)
        ).scalar_one_or_none()
        if run is None:
            raise HTTPException(status_code=404, detail="backtest not found")
        trades = (
            session.execute(
                select(BacktestTrade)
                .where(BacktestTrade.backtest_id == backtest_id)
                .order_by(BacktestTrade.trade_index)
                .limit(trade_limit)
            )
            .scalars()
            .all()
        )
        trade_total = session.execute(
            select(func.count(BacktestTrade.id)).where(BacktestTrade.backtest_id == backtest_id)
        ).scalar_one()

    return {
        "backtest_id": run.backtest_id,
        "strategy_version": run.strategy_version,
        "start_date": run.start_date.isoformat(),
        "end_date": run.end_date.isoformat(),
        "initial_capital": run.initial_capital,
        "risk_per_trade_pct": run.risk_per_trade_pct,
        "config": run.config_snapshot,
        "metrics": run.metrics,
        "equity_curve": run.equity_curve,
        "drawdown_curve": run.drawdown_curve,
        "monthly_returns": run.monthly_returns,
        "regime_performance": run.regime_performance,
        "sector_performance": run.sector_performance,
        "time_of_day_performance": run.time_of_day_performance,
        "side_performance": run.side_performance,
        "trade_count": int(trade_total),
        "trades": [
            {
                "trade_index": row.trade_index,
                "symbol": row.symbol,
                "sector": row.sector,
                "direction": row.direction,
                "quantity": row.quantity,
                "entry_ts": row.entry_ts.isoformat(),
                "exit_ts": row.exit_ts.isoformat(),
                "entry_price": row.entry_price,
                "exit_price": row.exit_price,
                "stop_price": row.stop_price,
                "target_price": row.target_price,
                "net_pnl": row.net_pnl,
                "fees": row.fees,
                "r_multiple": row.r_multiple,
                "mae_r": row.mae_r,
                "mfe_r": row.mfe_r,
                "holding_minutes": row.holding_minutes,
                "exit_reason": row.exit_reason,
                "regime": row.regime,
            }
            for row in trades
        ],
    }


@router.get("/backtest/{backtest_id}/equity", summary="Equity and drawdown curves only")
def equity_curve(backtest_id: str) -> Dict[str, Any]:
    with session_scope() as session:
        run = session.execute(
            select(BacktestRun).where(BacktestRun.backtest_id == backtest_id)
        ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="backtest not found")
    return {
        "backtest_id": backtest_id,
        "equity_curve": run.equity_curve,
        "drawdown_curve": run.drawdown_curve,
        "metrics": run.metrics,
    }


__all__ = ["router"]
