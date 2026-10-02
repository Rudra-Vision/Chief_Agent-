"""Backtest orchestration.

Ties together: data ingestion (cache-first) -> strategy config -> the
event-driven engine -> persistence of the run, its metrics and its trades.

Everything the API/dashboard needs to run a backtest lives here so the route
handlers stay thin.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy.orm import Session

from ..broker.upstox_instruments import BENCHMARK_KEYS
from ..costs.transaction_costs import CostModel
from ..data import candle_store
from ..data.provider import MarketDataProvider
from ..data.schema import BacktestRun, BacktestTrade, MultipleTestingLedger, StrategyVersion
from ..logging_setup import get_logger
from ..regime.engine import RegimeEngine
from ..settings import VAR_DIR, get_config_store, get_settings
from ..strategies.base import config_hash
from ..strategies.orb_vwap import ORBVWAPStrategy
from ..timeutil import now_ist
from .engine import BacktestConfig, BacktestResult, Backtester

log = get_logger(__name__, component="backtest.runner")


@dataclass
class BacktestRequest:
    start_date: dt.date
    end_date: dt.date
    universe: List[str]                       # symbols, e.g. ["RELIANCE", "INFY"]
    initial_capital: float = 500_000.0
    risk_per_trade_pct: float = 0.0025
    timeframe: str = "1m"
    strategy_config: Optional[Dict[str, Any]] = None
    strategy_version: str = "ORB_v1.0.0"
    slippage_multiplier: float = 1.0
    fee_multiplier: float = 1.0
    include_shorts: bool = True
    random_seed: int = 20240101
    max_universe: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }


def build_universe_map(session: Session, symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Resolve symbols to instrument_key / sector / tick using the instruments table."""
    from sqlalchemy import select

    from ..data.schema import Instrument

    wanted = {s.strip().upper() for s in symbols if s}
    if not wanted:
        return {}
    rows = (
        session.execute(
            select(Instrument).where(
                Instrument.instrument_type == "EQ",
                Instrument.segment == "NSE_EQ",
            )
        )
        .scalars()
        .all()
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        symbol = (row.trading_symbol or "").upper()
        if symbol in wanted:
            out[symbol] = {
                "instrument_key": row.instrument_key,
                "symbol": symbol,
                "sector": row.sector or "Unknown",
                "tick_size": row.tick_size or 0.05,
                "lot_size": row.lot_size or 1,
            }
    return out


def ensure_universe_instruments(session: Session, symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Instrument keys for backtesting.

    Real Upstox instrument keys are only used when the official BOD master has
    been downloaded. Otherwise a clearly-marked placeholder key is used
    (``SIM|<SYMBOL>``) so simulated runs never masquerade as real instruments.
    """
    resolved = build_universe_map(session, symbols)
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        symbol_upper = symbol.strip().upper()
        if symbol_upper in resolved:
            out[symbol_upper] = resolved[symbol_upper]
        else:
            out[symbol_upper] = {
                "instrument_key": f"SIM|{symbol_upper}",
                "symbol": symbol_upper,
                "sector": "Unknown",
                "tick_size": 0.05,
                "lot_size": 1,
                "unresolved": True,
            }
    return out


#: Sector indices used for the sector-confirmation layer during a backtest.
DEFAULT_SECTOR_INDICES = {
    "Financial Services": "Nifty Bank",
    "Information Technology": "Nifty IT",
    "Automobile": "Nifty Auto",
    "Energy": "Nifty Energy",
    "Healthcare": "Nifty Pharma",
    "Metals": "Nifty Metal",
    "Fast Moving Consumer Goods": "Nifty FMCG",
    "Realty": "Nifty Realty",
}


class BacktestRunner:
    """Runs a backtest end to end and persists the results."""

    def __init__(
        self,
        session: Session,
        provider: MarketDataProvider,
        *,
        download_if_missing: bool = True,
        max_instruments: int = 60,
    ) -> None:
        self.session = session
        self.provider = provider
        self.download_if_missing = download_if_missing
        self.max_instruments = max_instruments
        self.config_store = get_config_store()

    # --------------------------------------------------------------------- run
    def run(self, request: BacktestRequest, progress: Optional[Any] = None) -> BacktestResult:
        started = now_ist()
        backtest_id = f"BT-{started.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        strategy_config = request.strategy_config or self.config_store.load("strategy")
        if "champion" not in strategy_config:
            strategy_config = {**strategy_config}
        strategy = ORBVWAPStrategy(strategy_config, version=request.strategy_version)

        universe = ensure_universe_instruments(self.session, request.universe[: request.max_universe])
        sector_map = {meta["instrument_key"]: meta.get("sector") or "Unknown" for meta in universe.values()}
        symbols = {meta["instrument_key"]: meta["symbol"] for meta in universe.values()}

        # ---- ensure data is cached ----------------------------------------
        instruments = [
            {"instrument_key": meta["instrument_key"], "symbol": meta["symbol"]} for meta in universe.values()
        ]
        if self.download_if_missing:
            from ..data.historical_downloader import HistoricalDownloader

            downloader = HistoricalDownloader(self.provider, self.session)
            report = downloader.download(
                instruments,
                request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                progress=None,
            )
            log.info("backtest data prepared", context=report.to_dict())

        # ---- load candles ---------------------------------------------------
        candles_by_instrument: Dict[str, List[Any]] = {}
        for meta in universe.values():
            key = meta["instrument_key"]
            candles = candle_store.load_candles(self.session, key, request.timeframe, request.start_date, request.end_date)
            if not candles and self.download_if_missing:
                candles = self.provider.history(
                    key, request.timeframe, request.start_date, request.end_date, symbol=meta["symbol"]
                )
                if candles:
                    candle_store.upsert_candles(
                        self.session,
                        key,
                        request.timeframe,
                        candles,
                        source="synthetic" if self.provider.is_simulated else "upstox_v3",
                    )
                    self.session.flush()
            if candles:
                candles_by_instrument[key] = candles

        if not candles_by_instrument:
            raise RuntimeError(
                "No market data available for the requested universe/date range. "
                "Run a data download first, or widen the date range."
            )

        # ---- index and sector series ---------------------------------------
        index_key = "__index__NIFTY50"
        index_candles = self._index_series("NIFTY 50", request)
        vix_candles = self._index_series("India VIX", request)

        sector_index_candles: Dict[str, List[Any]] = {}
        used_sectors = {meta.get("sector") for meta in universe.values() if meta.get("sector")}
        for sector in sorted(used_sectors):
            index_name = DEFAULT_SECTOR_INDICES.get(sector)
            if not index_name:
                continue
            candles = self._index_series(index_name, request)
            if candles:
                sector_index_candles[sector] = candles

        # ---- run ------------------------------------------------------------
        config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            risk_per_trade_pct=request.risk_per_trade_pct,
            slippage_multiplier=request.slippage_multiplier,
            fee_multiplier=request.fee_multiplier,
            include_shorts=request.include_shorts,
            random_seed=request.random_seed,
            opening_range_minutes=int(
                (strategy_config.get("opening_range") or {}).get("duration_minutes", 15)
            ),
        )
        engine = Backtester(
            strategy=strategy,
            cost_model=CostModel(),
            regime_engine=RegimeEngine(),
            config=config,
        )
        result = engine.run(
            candles_by_instrument,
            index_candles=index_candles,
            vix_candles=vix_candles,
            symbols=symbols,
            sector_map=sector_map,
            sector_index_candles=sector_index_candles,
            data_source=self.provider.data_source,
            progress_callback=progress,
        )

        self.persist(backtest_id, request, result, strategy)
        result.metrics.extras["backtest_id"] = backtest_id
        return result

    # ---------------------------------------------------------------- helpers
    def _index_series(self, index_name: str, request: BacktestRequest) -> List[Any]:
        """Index candles from the cache, fetching from the provider if absent."""
        instrument_key = BENCHMARK_KEYS.get(index_name.upper())
        lookup_keys = [f"__index__{index_name}", instrument_key] if instrument_key else [f"__index__{index_name}"]
        for key in lookup_keys:
            if not key:
                continue
            candles = candle_store.load_candles(self.session, key, "1m", request.start_date, request.end_date)
            if candles:
                return candles

        # Fetch through the provider (simulated backend knows index names).
        key = f"__index__{index_name}"
        candles = self.provider.history(key, "1m", request.start_date, request.end_date, symbol=index_name)
        if candles:
            candle_store.upsert_candles(
                self.session, key, "1m", candles,
                source="synthetic" if self.provider.is_simulated else "upstox_v3",
            )
            self.session.flush()
        return candles

    # -------------------------------------------------------------- persist
    def persist(
        self,
        backtest_id: str,
        request: BacktestRequest,
        result: BacktestResult,
        strategy: ORBVWAPStrategy,
    ) -> None:
        metrics = result.metrics.to_dict()
        run = BacktestRun(
            backtest_id=backtest_id,
            strategy_version=request.strategy_version,
            start_date=request.start_date,
            end_date=request.end_date,
            universe=list(request.universe),
            initial_capital=request.initial_capital,
            risk_per_trade_pct=request.risk_per_trade_pct,
            slippage_bps=0.0,
            fee_multiplier=request.fee_multiplier,
            config_snapshot={"request": request.to_dict(), "strategy_config_hash": config_hash(strategy.config.raw)},
            metrics=metrics,
            equity_curve=[
                {"date": d.isoformat(), "equity": round(v, 2)} for d, v in zip(result.equity_dates, result.equity_curve)
            ],
            drawdown_curve=[
                {"date": d.isoformat(), "drawdown_pct": round(v, 6)}
                for d, v in zip(result.equity_dates, result.drawdown_curve)
            ],
            monthly_returns=result.metrics.monthly_returns,
            regime_performance=result.metrics.regime_performance,
            sector_performance=result.metrics.sector_performance,
            time_of_day_performance=result.metrics.time_of_day_performance,
            side_performance=result.metrics.side_performance,
            trade_count=len(result.trades),
            status="COMPLETED",
            finished_at=now_ist(),
            notes=f"data_source={result.data_source}",
        )
        self.session.add(run)

        for index, trade in enumerate(result.trades):
            payload = trade.to_dict()
            self.session.add(
                BacktestTrade(
                    backtest_id=backtest_id,
                    strategy_version=request.strategy_version,
                    trade_index=index,
                    instrument_key=trade.instrument_key,
                    symbol=trade.symbol,
                    sector=trade.sector,
                    direction=trade.direction,
                    quantity=trade.quantity,
                    entry_ts=trade.entry_ts,
                    exit_ts=trade.exit_ts,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    stop_price=trade.stop_price,
                    target_price=trade.target_price,
                    gross_pnl=trade.gross_pnl,
                    fees=trade.fees,
                    slippage_cost=trade.slippage_cost,
                    net_pnl=trade.net_pnl,
                    initial_risk=trade.initial_risk,
                    r_multiple=trade.r_multiple,
                    mae_r=trade.mae_r,
                    mfe_r=trade.mfe_r,
                    holding_minutes=trade.holding_minutes,
                    exit_reason=trade.exit_reason,
                    regime=trade.regime,
                    features=trade.features,
                )
            )

        # Multiple-testing ledger: every evaluated combination is counted.
        self.session.add(
            MultipleTestingLedger(
                trial_kind="backtest",
                strategy_version=request.strategy_version,
                variable="(full backtest)",
                value=None,
                metric_name="expectancy_r",
                metric_value=metrics.get("expectancy_r"),
                notes=f"backtest {backtest_id}",
            )
        )
        self.session.flush()
        log.info("backtest persisted", context={"backtest_id": backtest_id, "trades": len(result.trades)})


__all__ = [
    "BacktestRunner",
    "BacktestRequest",
    "build_universe_map",
    "ensure_universe_instruments",
    "DEFAULT_SECTOR_INDICES",
]
