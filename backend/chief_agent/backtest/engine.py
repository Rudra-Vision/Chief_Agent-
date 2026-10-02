"""Event-driven backtesting engine.

Anti-look-ahead guarantees (structural, not conventions):

1. A signal is produced only from **completed** candles up to and including bar
   ``i``. Partially formed bars are never evaluated.
2. Signals are queued and filled on the **next** bar of that instrument: entries
   at that bar's opening price plus modelled slippage. There is no same-bar fill.
3. When a bar's range contains both the stop and the target, the **stop is
   assumed to fill first** - the pessimistic and honest assumption.
4. Indicators come from :class:`InstrumentSeries`, whose value at index ``i``
   depends only on data at ``<= i``.
5. The opening range only becomes visible after its last constituent candle
   closes (``OpeningRange.formed_at``).
6. Costs are always applied; the engine refuses to run with costs disabled.

Realism modelled
----------------
* Slippage from the :class:`CostModel`, volatility-scaled, wider on stop exits.
* Gap-through stops fill at the bar open, not at the stop price.
* Trailing stops (ATR / R-multiple / swing / break-even after R).
* A hard square-off before the close.
* Risk limits mirrored from ``config/risk.yaml``: max positions, max trades per
  day, max consecutive losses per day, soft/hard daily stops, sector caps,
  aggregate open-risk cap - so backtest behaviour is comparable to live.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..broker.upstox_market_data import Candle
from ..costs.transaction_costs import CostModel, Product
from ..indicators.features import FeatureEngine, InstrumentSeries
from ..logging_setup import get_logger
from ..regime.engine import RegimeEngine, RegimeFeatureSeries, RegimeFeatures, build_regime_features_from_index
from ..risk.position_sizing import SizingConstraints, compute_position_size
from ..sectors.engine import SectorEngine, SectorSnapshot
from ..strategies.base import StrategyContext, TradeCandidate
from ..strategies.orb_vwap import ORBVWAPStrategy
from ..timeutil import IST, minutes_from_open
from .metrics import BacktestMetrics, evaluate_backtest, drawdown_series, strategy_objective_score

log = get_logger(__name__, component="backtest")

SESSION_MINUTES = 375


@dataclass
class BacktestConfig:
    """Everything that defines one backtest run."""

    start_date: dt.date
    end_date: dt.date
    initial_capital: float = 500_000.0
    risk_per_trade_pct: float = 0.0025
    max_risk_per_trade_pct: float = 0.005
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    apply_costs: bool = True
    product: str = Product.INTRADAY.value
    max_positions: int = 3
    max_trades_per_day: int = 12
    max_consecutive_losses_per_day: int = 4
    soft_daily_stop_pct: float = -0.010
    hard_daily_stop_pct: float = -0.015
    max_total_open_risk_pct: float = 0.0075
    max_correlated_sector_positions: int = 2
    max_gross_exposure_pct: float = 1.0
    square_off_minutes_before_close: int = 10
    delay_entry_bars: int = 0
    missed_trade_pct: float = 0.0
    random_seed: int = 20240101
    include_shorts: bool = True
    min_opportunity_score: float = 0.0
    opening_range_minutes: int = 15
    max_universe: int = 200
    record_rejections: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


@dataclass
class SimulatedTrade:
    trade_id: str
    instrument_key: str
    symbol: str
    sector: Optional[str]
    direction: str
    quantity: int
    entry_ts: dt.datetime
    exit_ts: dt.datetime
    entry_price: float
    exit_price: float
    stop_price: float            # the INITIAL stop (the basis of 1R)
    exit_stop_price: float       # the stop actually in force at exit (after trailing)
    target_price: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    initial_risk: float
    r_multiple: float
    mae_r: float
    mfe_r: float
    holding_minutes: float
    exit_reason: str
    regime: Optional[str]
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "sector": self.sector,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_ts": self.entry_ts.isoformat(),
            "exit_ts": self.exit_ts.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "stop_price": round(self.stop_price, 4),
            "exit_stop_price": round(self.exit_stop_price, 4),
            "target_price": round(self.target_price, 4),
            "gross_pnl": round(self.gross_pnl, 2),
            "fees": round(self.fees, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "net_pnl": round(self.net_pnl, 2),
            "initial_risk": round(self.initial_risk, 2),
            "r_multiple": round(self.r_multiple, 4),
            "mae_r": round(self.mae_r, 4),
            "mfe_r": round(self.mfe_r, 4),
            "holding_minutes": round(self.holding_minutes, 2),
            "exit_reason": self.exit_reason,
            "regime_at_entry": self.regime,
            "features": self.features,
        }


@dataclass
class OpenPosition:
    instrument_key: str
    symbol: str
    sector: Optional[str]
    direction: str
    quantity: int
    entry_price: float
    entry_ts: dt.datetime
    stop_price: float
    initial_stop: float
    target_1: float
    target_2: float
    initial_risk_per_share: float
    regime: Optional[str]
    entry_fees: float
    entry_slippage: float
    features: Dict[str, Any]
    strategy_version: str
    signal_id: str
    trade_id: str
    mae_price: float = 0.0
    mfe_price: float = 0.0
    remaining_fraction: float = 1.0
    partial_realized: float = 0.0
    partial_fees: float = 0.0
    partial_slippage: float = 0.0
    trailing_active: bool = False
    entry_bar_index: int = 0

    @property
    def risk_per_share(self) -> float:
        return max(self.initial_risk_per_share, 1e-9)


@dataclass
class PendingOrder:
    """A signal waiting for its next-bar fill."""

    candidate: TradeCandidate
    meta: Dict[str, Any]
    quantity: int
    created_ts: dt.datetime
    created_bar_index: int
    bars_remaining: int = 1


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: List[SimulatedTrade]
    equity_curve: List[float]
    equity_dates: List[dt.date]
    drawdown_curve: List[float]
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    data_source: str = "UNKNOWN"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self, include_trades: bool = True, max_trades: int = 500) -> Dict[str, Any]:
        payload = {
            "metrics": self.metrics.to_dict(),
            "equity_curve": [
                {"date": d.isoformat(), "equity": round(v, 2)} for d, v in zip(self.equity_dates, self.equity_curve)
            ],
            "drawdown_curve": [
                {"date": d.isoformat(), "drawdown_pct": round(v, 6)}
                for d, v in zip(self.equity_dates, self.drawdown_curve)
            ],
            "trade_count": len(self.trades),
            "config": self.config,
            "data_source": self.data_source,
            "warnings": self.warnings,
        }
        if include_trades:
            payload["trades"] = [t.to_dict() for t in self.trades[:max_trades]]
            payload["trades_truncated"] = len(self.trades) > max_trades
        return payload


class Backtester:
    """Event-driven, cost-aware, multi-instrument backtester."""

    def __init__(
        self,
        strategy: ORBVWAPStrategy,
        cost_model: Optional[CostModel] = None,
        regime_engine: Optional[RegimeEngine] = None,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.strategy = strategy
        self.cost_model = cost_model or CostModel()
        self.regime_engine = regime_engine or RegimeEngine()
        self.config = config or BacktestConfig(start_date=dt.date.today(), end_date=dt.date.today())
        self.feature_engine = FeatureEngine(opening_range_minutes=self.config.opening_range_minutes)
        self.sector_engine = SectorEngine()
        self._rng = np.random.default_rng(self.config.random_seed)

    # -------------------------------------------------------------------- run
    def run(
        self,
        candles_by_instrument: Mapping[str, Sequence[Candle]],
        *,
        index_candles: Optional[Sequence[Candle]] = None,
        vix_candles: Optional[Sequence[Candle]] = None,
        symbols: Optional[Mapping[str, str]] = None,
        sector_map: Optional[Mapping[str, str]] = None,
        sector_index_candles: Optional[Mapping[str, Sequence[Candle]]] = None,
        data_source: str = "UNKNOWN",
        progress_callback: Optional[Any] = None,
    ) -> BacktestResult:
        if not self.config.apply_costs:
            raise ValueError(
                "Backtesting without realistic costs is prohibited. "
                "Set apply_costs=True; charges come from config/execution.yaml -> costs."
            )

        symbols = symbols or {}
        sector_map = sector_map or {}
        sector_index_candles = sector_index_candles or {}

        # ---- causal per-instrument series ---------------------------------
        series_map: Dict[str, InstrumentSeries] = {}
        for instrument_key, candles in candles_by_instrument.items():
            trimmed = [c for c in candles if self.config.start_date <= c.ts.date() <= self.config.end_date]
            if len(trimmed) < 30:
                continue
            series = self.feature_engine.series(
                instrument_key, trimmed, opening_range_minutes=self.config.opening_range_minutes
            )
            # The dict key is authoritative. A candle source that leaves
            # instrument_key blank must never cause a signal to be attributed to
            # the wrong instrument (or silently dropped).
            series.instrument_key = instrument_key
            for candle in series.candles:
                if not candle.instrument_key:
                    candle.instrument_key = instrument_key
            series_map[instrument_key] = series
        if not series_map:
            raise ValueError("no instruments with usable candles inside the requested date range")

        index_series = self.feature_engine.series("__index__", list(index_candles)) if index_candles else None
        vix_series = self.feature_engine.series("__vix__", list(vix_candles)) if vix_candles else None
        sector_series = {
            name: self.feature_engine.series(f"__sector__{name}", list(candles))
            for name, candles in sector_index_candles.items()
        }
        # One linear pass instead of an O(n^2) recomputation per timestamp.
        regime_series = (
            RegimeFeatureSeries(index_series, vix_closes=vix_series.closes if vix_series else None)
            if index_series is not None
            else None
        )

        timeline = sorted({c.ts for series in series_map.values() for c in series.candles})
        timeline = [ts for ts in timeline if self.config.start_date <= ts.date() <= self.config.end_date]
        if not timeline:
            raise ValueError("empty event timeline")

        # ---- state ---------------------------------------------------------
        equity = float(self.config.initial_capital)
        open_positions: List[OpenPosition] = []
        pending: List[PendingOrder] = []
        completed: List[SimulatedTrade] = []
        rejections: List[Dict[str, Any]] = []
        equity_curve: List[float] = []
        equity_dates: List[dt.date] = []
        warnings: List[str] = []

        current_day: Optional[dt.date] = None
        day_start_equity = equity
        day_realized_pnl = 0.0
        trades_today = 0
        consecutive_losses_today = 0
        day_halted = False
        day_sector_counts: Dict[str, int] = {}
        total_minutes_in_market = 0.0
        total_available_minutes = 0

        pointers: Dict[str, int] = {key: 0 for key in series_map}
        bars_seen: Dict[str, int] = {key: 0 for key in series_map}
        index_pointer = 0
        vix_pointer = 0
        sector_pointers: Dict[str, int] = {name: 0 for name in sector_series}

        def advance_series_pointers(ts: dt.datetime) -> None:
            nonlocal index_pointer, vix_pointer
            if index_series is not None:
                while (
                    index_pointer + 1 < len(index_series.candles)
                    and index_series.candles[index_pointer + 1].ts <= ts
                ):
                    index_pointer += 1
            if vix_series is not None:
                while vix_pointer + 1 < len(vix_series.candles) and vix_series.candles[vix_pointer + 1].ts <= ts:
                    vix_pointer += 1
            for name, series in sector_series.items():
                pointer = sector_pointers[name]
                while pointer + 1 < len(series.candles) and series.candles[pointer + 1].ts <= ts:
                    pointer += 1
                sector_pointers[name] = pointer

        def close_position(position: OpenPosition, exit_price: float, exit_ts: dt.datetime, reason: str,
                           atr_pct: Optional[float]) -> SimulatedTrade:
            return self._close_position(position, exit_price, exit_ts, reason, atr_pct=atr_pct)

        def record_exit(trade: SimulatedTrade) -> None:
            nonlocal equity, day_realized_pnl, consecutive_losses_today
            completed.append(trade)
            equity += trade.net_pnl
            day_realized_pnl += trade.net_pnl
            if trade.net_pnl < 0:
                consecutive_losses_today += 1
            elif trade.net_pnl > 0:
                consecutive_losses_today = 0
            sector = trade.sector or "Unknown"
            day_sector_counts[sector] = max(0, day_sector_counts.get(sector, 1) - 1)

        for ts in timeline:
            advance_series_pointers(ts)
            for key, series in series_map.items():
                pointer = pointers[key]
                if pointer < len(series.candles) and series.candles[pointer].ts == ts:
                    pointer += 1
                elif pointer < len(series.candles) and series.candles[pointer].ts < ts:
                    # the instrument has a gap; advance past bars we missed
                    while pointer < len(series.candles) and series.candles[pointer].ts < ts:
                        pointer += 1
                    if pointer < len(series.candles) and series.candles[pointer].ts == ts:
                        pointer += 1
                pointers[key] = pointer
                bars_seen[key] = pointer

            local_ts = ts.astimezone(IST) if ts.tzinfo else ts
            day = local_ts.date()
            minutes_into_session = minutes_from_open(local_ts)

            # ---------------- session rollover -----------------------------
            if current_day != day:
                if current_day is not None:
                    equity_dates.append(current_day)
                    equity_curve.append(equity)
                for position in list(open_positions):
                    price = self._price_at(position.instrument_key, ts, series_map, bars_seen)
                    record_exit(close_position(position, price, ts, "SESSION_END", None))
                    open_positions.remove(position)
                pending.clear()
                current_day = day
                day_start_equity = equity
                day_realized_pnl = 0.0
                trades_today = 0
                consecutive_losses_today = 0
                day_halted = False
                day_sector_counts = {}

            # ---------------- 1. fill pending entries at this bar's open ----
            for order in list(pending):
                key = order.candidate.instrument_key
                series = series_map.get(key)
                if series is None:
                    pending.remove(order)
                    continue
                index = bars_seen.get(key, 0) - 1
                if index < 0 or index >= len(series.candles) or series.candles[index].ts != ts:
                    continue  # this instrument has no bar at this timestamp yet
                if order.bars_remaining > 0:
                    order.bars_remaining -= 1
                    if order.bars_remaining > 0:
                        continue
                if len(open_positions) >= self.config.max_positions:
                    pending.remove(order)
                    continue

                bar = series.candles[index]
                direction = order.candidate.direction
                entry_price = self._apply_slippage(
                    float(bar.open), direction, order.meta.get("atr_pct"), is_exit=False
                )
                shift = entry_price - order.candidate.entry_price
                stop = self._round_tick(order.candidate.stop_price + shift, direction, "stop")
                target_1 = self._round_tick(order.candidate.target_1 + shift, direction, "target")
                target_2 = self._round_tick(order.candidate.target_2 + shift, direction, "target")
                risk_per_share = abs(entry_price - stop)
                if risk_per_share <= 0:
                    pending.remove(order)
                    continue

                quantity = order.quantity
                notional = entry_price * quantity
                entry_cost = self.cost_model.compute(
                    price=entry_price,
                    quantity=quantity,
                    side="BUY" if direction == "LONG" else "SELL",
                    product=self.config.product,
                    atr_pct=order.meta.get("atr_pct"),
                    slippage_bps_override=0.0,
                    fee_multiplier=self.config.fee_multiplier,
                )
                slippage_cost = self.cost_model.slippage_amount(
                    entry_price, quantity, atr_pct=order.meta.get("atr_pct"),
                    multiplier=self.config.slippage_multiplier,
                )
                # Entry charges are NOT booked to equity here: they are carried on
                # the position and settled inside the trade's net P&L at exit, so
                #  final equity == initial capital + sum(trade net P&L)
                # holds exactly and no cost is double counted.

                open_positions.append(
                    OpenPosition(
                        instrument_key=key,
                        symbol=order.candidate.symbol,
                        sector=order.meta.get("sector"),
                        direction=direction,
                        quantity=quantity,
                        entry_price=entry_price,
                        entry_ts=bar.ts,
                        stop_price=stop,
                        initial_stop=stop,
                        target_1=target_1,
                        target_2=target_2,
                        initial_risk_per_share=risk_per_share,
                        regime=order.candidate.regime,
                        entry_fees=entry_cost.charges_only,
                        entry_slippage=slippage_cost,
                        features={
                            k: v for k, v in order.candidate.features.items()
                            if v is None or isinstance(v, (int, float, str, bool))
                        },
                        strategy_version=order.candidate.strategy_version,
                        signal_id=order.meta.get("signal_id", ""),
                        trade_id=order.meta.get("trade_id", ""),
                        mae_price=entry_price,
                        mfe_price=entry_price,
                        entry_bar_index=index,
                    )
                )
                trades_today += 1
                sector = order.meta.get("sector") or "Unknown"
                day_sector_counts[sector] = day_sector_counts.get(sector, 0) + 1
                pending.remove(order)

            # ---------------- 2. manage open positions ----------------------
            for position in list(open_positions):
                series = series_map.get(position.instrument_key)
                if series is None:
                    continue
                index = bars_seen.get(position.instrument_key, 0) - 1
                if index < 0 or index >= len(series.candles):
                    continue
                bar = series.candles[index]
                if bar.ts != ts:
                    continue

                atr_pct = series.atr_pct_series[index]
                exit_price, exit_reason = self._check_exit(position, bar, series, index, minutes_into_session)

                if exit_price is None:
                    self._update_excursions(position, bar)
                    self._update_trailing(position, series, index, bar)
                    total_minutes_in_market += 1
                    continue

                partial = self._partial_exit_quantity(position, exit_reason)
                if partial > 0:
                    # scale-out: bank part of the position, keep a runner
                    runner = position.remaining_fraction
                    partial_fraction = partial / max(1.0, position.quantity * runner)
                    trade = self._close_fraction(position, exit_price, ts, exit_reason, partial_fraction, atr_pct)
                    if trade is not None:
                        record_exit(trade)
                    position.remaining_fraction -= partial_fraction
                    if bool(self.strategy.config.get("targets.move_stop_to_breakeven_on_t1", True)):
                        position.stop_price = self._round_tick(position.entry_price, position.direction, "stop")
                    continue

                trade = close_position(position, exit_price, ts, exit_reason, atr_pct)
                record_exit(trade)
                open_positions.remove(position)

            # ---------------- 3. daily halt evaluation ----------------------
            if not day_halted and day_start_equity > 0:
                unrealized = sum(
                    self._unrealized(p, self._price_at(p.instrument_key, ts, series_map, bars_seen))
                    for p in open_positions
                )
                daily_return = (day_realized_pnl + unrealized) / day_start_equity
                if daily_return <= self.config.hard_daily_stop_pct:
                    day_halted = True
                    for position in list(open_positions):
                        price = self._price_at(position.instrument_key, ts, series_map, bars_seen)
                        record_exit(close_position(position, price, ts, "HARD_DAILY_STOP", None))
                        open_positions.remove(position)
                    pending.clear()
                elif (
                    daily_return <= self.config.soft_daily_stop_pct
                    or consecutive_losses_today >= self.config.max_consecutive_losses_per_day
                ):
                    day_halted = True
                    pending.clear()

            # ---------------- 4. generate new signals -----------------------
            if (
                not day_halted
                and len(open_positions) + len(pending) < self.config.max_positions
                and trades_today < self.config.max_trades_per_day
                and minutes_into_session <= (SESSION_MINUTES - self.config.square_off_minutes_before_close)
            ):
                for candidate, meta in self._scan(
                    series_map, bars_seen, ts, index_series, index_pointer, regime_series,
                    sector_series, sector_pointers, sector_map, symbols, open_positions, pending,
                    equity, day_sector_counts,
                ):
                    if len(open_positions) + len(pending) >= self.config.max_positions:
                        break
                    if candidate.score < self.config.min_opportunity_score:
                        continue
                    if self.config.missed_trade_pct and float(self._rng.random()) < self.config.missed_trade_pct:
                        if self.config.record_rejections:
                            rejections.append({**meta, "reason": "MISSED_TRADE_SIMULATION"})
                        continue

                    sizing = compute_position_size(
                        SizingConstraints(
                            account_equity=equity,
                            risk_pct=self.config.risk_per_trade_pct,
                            max_risk_pct=self.config.max_risk_per_trade_pct,
                            entry_price=candidate.entry_price,
                            stop_price=candidate.stop_price,
                            lot_size=int(meta.get("lot_size", 1)),
                            tick_size=float(meta.get("tick_size", 0.05)),
                            max_position_exposure_pct=0.25,
                            max_gross_exposure_pct=self.config.max_gross_exposure_pct,
                            available_cash=equity,
                            current_gross_exposure=self._gross_exposure(open_positions, equity),
                            max_total_open_risk_pct=self.config.max_total_open_risk_pct,
                            current_open_risk_pct=self._open_risk_pct(open_positions, equity),
                            average_daily_volume=meta.get("average_daily_volume"),
                            max_participation_pct=0.01,
                        )
                    )
                    if sizing.rejected or sizing.quantity <= 0:
                        if self.config.record_rejections:
                            rejections.append({**meta, "reason": "SIZING_REJECTED", "detail": sizing.rejection_reason})
                        continue

                    sector = meta.get("sector") or "Unknown"
                    if day_sector_counts.get(sector, 0) >= self.config.max_correlated_sector_positions:
                        if self.config.record_rejections:
                            rejections.append({**meta, "reason": "SECTOR_CONCENTRATION"})
                        continue

                    pending.append(
                        PendingOrder(
                            candidate=candidate,
                            meta=meta,
                            quantity=sizing.quantity,
                            created_ts=ts,
                            created_bar_index=bars_seen.get(candidate.instrument_key, 0),
                            bars_remaining=max(0, int(self.config.delay_entry_bars)),
                        )
                    )

            total_available_minutes += 1
            if progress_callback and total_available_minutes % 10000 == 0:
                try:
                    progress_callback(total_available_minutes, len(timeline), equity)
                except Exception:  # pragma: no cover
                    pass

        # ---- final square-off -----------------------------------------------
        for position in list(open_positions):
            series = series_map[position.instrument_key]
            index = min(max(0, bars_seen.get(position.instrument_key, 1) - 1), len(series.candles) - 1)
            last_bar = series.candles[index]
            record_exit(close_position(position, float(last_bar.close), last_bar.ts, "FINAL_CLOSE", None))
            open_positions.remove(position)

        if current_day is not None:
            equity_curve.append(equity)
            equity_dates.append(current_day)

        trades_payload = [t.to_dict() for t in completed]
        metrics = evaluate_backtest(
            trades_payload,
            equity_curve,
            equity_dates,
            total_minutes_in_market=total_minutes_in_market,
            total_available_minutes=max(1.0, total_available_minutes),
        )

        regime_rows = list(metrics.regime_performance.values())
        positive_regimes = sum(1 for row in regime_rows if row["net_pnl"] > 0)
        metrics.extras = {
            "regime_robustness": (positive_regimes / len(regime_rows)) if regime_rows else 0.5,
            "turnover_per_day": (
                metrics.trades.turnover / max(1, metrics.curve.trading_days) / max(1.0, self.config.initial_capital)
            ),
            "costs_enabled": True,
        }
        objective_input = metrics.to_dict()
        objective_input["regime_robustness"] = metrics.extras["regime_robustness"]
        objective_input["turnover_per_day"] = metrics.extras["turnover_per_day"]
        metrics.objective_score = strategy_objective_score(objective_input)

        if metrics.trades.trades == 0:
            warnings.append(
                "no trades were generated - check the universe, date range and strategy filters"
            )
        if metrics.curve.trading_days < 20:
            warnings.append(
                f"only {metrics.curve.trading_days} trading days were simulated; treat the statistics as indicative"
            )

        return BacktestResult(
            metrics=metrics,
            trades=completed,
            equity_curve=equity_curve,
            equity_dates=equity_dates,
            drawdown_curve=drawdown_series(equity_curve),
            rejections=rejections,
            config=self.config.to_dict(),
            data_source=data_source,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ exits
    def _check_exit(
        self,
        position: OpenPosition,
        bar: Candle,
        series: InstrumentSeries,
        bar_index: int,
        minutes_into_session: float,
    ) -> Tuple[Optional[float], str]:
        direction = position.direction
        square_off_at = SESSION_MINUTES - self.config.square_off_minutes_before_close

        time_stop = self.strategy.config.get("exits.time_stop_minutes", 0) or 0
        if time_stop:
            held = minutes_from_open(bar.ts) - minutes_from_open(position.entry_ts)
            if held >= float(time_stop):
                return float(bar.close), "TIME_STOP"

        if minutes_into_session >= square_off_at:
            return float(bar.close), "SQUARE_OFF"

        if bool(self.strategy.config.get("exits.exit_on_vwap_cross_against", False)):
            vwap = series.vwap[bar_index]
            if vwap:
                if direction == "LONG" and bar.close < vwap:
                    return float(bar.close), "VWAP_CROSS"
                if direction == "SHORT" and bar.close > vwap:
                    return float(bar.close), "VWAP_CROSS"

        if direction == "LONG":
            if bar.low <= position.stop_price:
                return min(float(bar.open), position.stop_price), "STOP_LOSS"
            if bar.high >= position.target_1:
                return float(position.target_1), "TARGET_1"
        else:
            if bar.high >= position.stop_price:
                return max(float(bar.open), position.stop_price), "STOP_LOSS"
            if bar.low <= position.target_1:
                return float(position.target_1), "TARGET_1"
        return None, ""

    def _partial_exit_quantity(self, position: OpenPosition, exit_reason: str) -> int:
        """Quantity to scale out at target 1 (0 disables partial exits)."""
        if exit_reason != "TARGET_1":
            return 0
        pct = float(self.strategy.config.get("targets.book_partial_at_t1_pct", 0) or 0)
        if pct <= 0 or position.remaining_fraction < 0.999:
            return 0
        return int(math.floor(position.quantity * pct / 100.0))

    def _close_fraction(
        self,
        position: OpenPosition,
        exit_price: float,
        exit_ts: dt.datetime,
        reason: str,
        fraction: float,
        atr_pct: Optional[float],
    ) -> Optional[SimulatedTrade]:
        quantity = int(math.floor(position.quantity * fraction))
        if quantity <= 0:
            return None
        clone = OpenPosition(
            **{
                **position.__dict__,
                "quantity": quantity,
                "remaining_fraction": 1.0,
                "entry_fees": position.entry_fees * (quantity / max(1, position.quantity)),
                "entry_slippage": position.entry_slippage * (quantity / max(1, position.quantity)),
            }
        )
        return self._close_position(clone, exit_price, exit_ts, reason, atr_pct=atr_pct)

    @staticmethod
    def _update_excursions(position: OpenPosition, bar: Candle) -> None:
        if position.direction == "LONG":
            position.mae_price = min(position.mae_price, float(bar.low))
            position.mfe_price = max(position.mfe_price, float(bar.high))
        else:
            position.mae_price = max(position.mae_price, float(bar.high))
            position.mfe_price = min(position.mfe_price, float(bar.low))

    def _update_trailing(
        self, position: OpenPosition, series: InstrumentSeries, bar_index: int, bar: Candle
    ) -> None:
        if not bool(self.strategy.config.get("trailing.enabled", False)):
            return
        model = str(self.strategy.config.get("trailing.model", "none"))
        if model == "none":
            return

        risk = position.risk_per_share
        direction = position.direction
        close = float(bar.close)
        r_multiple = (
            (close - position.entry_price) / risk if direction == "LONG" else (position.entry_price - close) / risk
        )

        be_r = float(self.strategy.config.get("trailing.breakeven_after_r", 0) or 0)
        if be_r and r_multiple >= be_r:
            if direction == "LONG":
                position.stop_price = max(position.stop_price, position.entry_price)
            else:
                position.stop_price = min(position.stop_price, position.entry_price)

        activation = float(self.strategy.config.get("trailing.trail_activation_r", 0) or 0)
        if r_multiple < activation:
            return
        position.trailing_active = True

        if model == "atr":
            atr_value = series.atr14[bar_index]
            if atr_value:
                multiplier = float(self.strategy.config.get("trailing.atr_multiplier", 2.0))
                candidate = (
                    close - multiplier * atr_value if direction == "LONG" else close + multiplier * atr_value
                )
                position.stop_price = (
                    max(position.stop_price, candidate) if direction == "LONG" else min(position.stop_price, candidate)
                )
        elif model == "r_multiple":
            step = float(self.strategy.config.get("trailing.r_multiple_step", 0.5)) or 0.5
            locked_r = math.floor(r_multiple / step) * step
            candidate = (
                position.entry_price + locked_r * risk
                if direction == "LONG"
                else position.entry_price - locked_r * risk
            )
            position.stop_price = (
                max(position.stop_price, candidate) if direction == "LONG" else min(position.stop_price, candidate)
            )
        elif model == "swing":
            lookback = int(self.strategy.config.get("trailing.swing_lookback", 5))
            start = max(0, bar_index - lookback + 1)
            window = series.candles[start : bar_index + 1]
            if window:
                if direction == "LONG":
                    position.stop_price = max(position.stop_price, min(c.low for c in window))
                else:
                    position.stop_price = min(position.stop_price, max(c.high for c in window))

        position.stop_price = self._round_tick(position.stop_price, direction, "stop")

    # -------------------------------------------------------------- closing
    def _close_position(
        self,
        position: OpenPosition,
        exit_price: float,
        exit_ts: dt.datetime,
        reason: str,
        *,
        atr_pct: Optional[float],
    ) -> SimulatedTrade:
        direction = position.direction
        is_stop = reason in ("STOP_LOSS", "HARD_DAILY_STOP")
        slippage_bps = self.cost_model.slippage_bps(is_stop_exit=is_stop, atr_pct=atr_pct)
        adjusted_exit = self._apply_slippage(exit_price, direction, atr_pct, is_exit=True, is_stop=is_stop)

        quantity = max(1, int(round(position.quantity * max(position.remaining_fraction, 0.0))))
        exit_cost = self.cost_model.compute(
            price=adjusted_exit,
            quantity=quantity,
            side="SELL" if direction == "LONG" else "BUY",
            product=self.config.product,
            is_stop_exit=is_stop,
            atr_pct=atr_pct,
            slippage_bps_override=0.0,
            fee_multiplier=self.config.fee_multiplier,
        )
        exit_slippage = self.cost_model.slippage_amount(
            exit_price, quantity, is_stop_exit=is_stop, atr_pct=atr_pct,
            multiplier=self.config.slippage_multiplier,
        )

        gross = (
            (adjusted_exit - position.entry_price) * quantity
            if direction == "LONG"
            else (position.entry_price - adjusted_exit) * quantity
        )
        fees = position.entry_fees * max(position.remaining_fraction, 0.0) + exit_cost.charges_only
        slippage = position.entry_slippage * max(position.remaining_fraction, 0.0) + exit_slippage
        net = gross - fees

        risk_amount = position.initial_risk_per_share * quantity
        mae_r = (
            (position.entry_price - position.mae_price) / position.risk_per_share
            if direction == "LONG"
            else (position.mae_price - position.entry_price) / position.risk_per_share
        )
        mfe_r = (
            (position.mfe_price - position.entry_price) / position.risk_per_share
            if direction == "LONG"
            else (position.entry_price - position.mfe_price) / position.risk_per_share
        )
        holding = (exit_ts - position.entry_ts).total_seconds() / 60.0

        features = dict(position.features)
        features["exit_stop_price"] = position.stop_price
        return SimulatedTrade(
            trade_id=position.trade_id or f"BT-{position.instrument_key}-{exit_ts.strftime('%Y%m%d%H%M')}",
            instrument_key=position.instrument_key,
            symbol=position.symbol,
            sector=position.sector,
            direction=direction,
            quantity=quantity,
            entry_ts=position.entry_ts,
            exit_ts=exit_ts,
            entry_price=position.entry_price,
            exit_price=adjusted_exit,
            stop_price=position.initial_stop,
            exit_stop_price=position.stop_price,
            target_price=position.target_1,
            gross_pnl=gross,
            fees=fees,
            slippage_cost=slippage,
            net_pnl=net,
            initial_risk=risk_amount,
            r_multiple=(net / risk_amount) if risk_amount > 0 else 0.0,
            mae_r=max(0.0, mae_r),
            mfe_r=max(0.0, mfe_r),
            holding_minutes=holding,
            exit_reason=reason,
            regime=position.regime,
            features=features,
        )

    # ------------------------------------------------------------------ scans
    def _scan(
        self,
        series_map: Dict[str, InstrumentSeries],
        bars_seen: Dict[str, int],
        ts: dt.datetime,
        index_series: Optional[InstrumentSeries],
        index_pointer: int,
        regime_series: Optional[RegimeFeatureSeries],
        sector_series: Dict[str, InstrumentSeries],
        sector_pointers: Dict[str, int],
        sector_map: Mapping[str, str],
        symbols: Mapping[str, str],
        open_positions: List[OpenPosition],
        pending: List[PendingOrder],
        equity: float,
        day_sector_counts: Mapping[str, int],
    ) -> List[Tuple[TradeCandidate, Dict[str, Any]]]:
        if index_series is None or index_pointer >= len(index_series.candles):
            return []

        nifty_return = index_series.return_since_open(index_pointer)
        regime_features = (
            regime_series.at(index_pointer) if regime_series is not None else RegimeFeatures()
        )
        assessment = self.regime_engine.classify(regime_features, ts=ts)
        regime = assessment.regime.value
        sector_snapshot = self.sector_engine.build_snapshot_fast(
            ts,
            sector_series=sector_series,
            sector_pointers=sector_pointers,
            nifty_return_pct=nifty_return,
        )

        held = {p.instrument_key for p in open_positions} | {o.candidate.instrument_key for o in pending}
        candidates: List[Tuple[TradeCandidate, Dict[str, Any]]] = []

        for instrument_key, series in series_map.items():
            index = bars_seen.get(instrument_key, 0) - 1
            if index < 0 or index >= len(series.candles):
                continue
            if series.candles[index].ts != ts or instrument_key in held:
                continue

            sector = sector_map.get(instrument_key)
            sector_return = sector_rank = None
            if sector_snapshot is not None:
                state = sector_snapshot.state_of(sector)
                if state is not None:
                    sector_return, sector_rank = state.return_pct, state.rank

            features = series.features_at(
                index, index_return=nifty_return, regime=regime,
                sector_return=sector_return, sector_rank=sector_rank,
            )
            context = StrategyContext(
                ts=ts,
                features=features,
                recent_candles=series.candles[max(0, index - 5) : index + 1],
                nifty_return_pct=nifty_return,
                vix_level=regime_features.vix_level,
                sector_snapshot=sector_snapshot,
                sector=sector,
                extras={
                    "symbol": symbols.get(instrument_key, instrument_key),
                    "index_above_vwap": regime_features.nifty_above_vwap,
                },
            )
            candidate = self.strategy.evaluate(context)
            if candidate is None:
                continue
            if candidate.direction == "SHORT" and not self.config.include_shorts:
                continue
            candidates.append(
                (
                    candidate,
                    {
                        "instrument_key": instrument_key,
                        "symbol": symbols.get(instrument_key, instrument_key),
                        "sector": sector,
                        "regime": regime,
                        "atr_pct": features.atr_daily_pct if features.atr_daily_pct is not None else features.atr_pct,
                        "spread_pct": features.spread_pct,
                        "average_daily_volume": features.average_daily_volume,
                        "lot_size": 1,
                        "tick_size": 0.05,
                        "signal_id": f"BT-{instrument_key}-{ts.strftime('%Y%m%d%H%M')}",
                        "trade_id": f"BTT-{instrument_key}-{ts.strftime('%Y%m%d%H%M')}",
                    },
                )
            )

        candidates.sort(key=lambda item: item[0].score, reverse=True)
        return candidates

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _price_at(
        instrument_key: str,
        ts: dt.datetime,
        series_map: Mapping[str, InstrumentSeries],
        bars_seen: Mapping[str, int],
    ) -> float:
        series = series_map.get(instrument_key)
        if series is None or not series.candles:
            return 0.0
        index = min(max(0, bars_seen.get(instrument_key, 1) - 1), len(series.candles) - 1)
        return float(series.candles[index].close)

    @staticmethod
    def _unrealized(position: OpenPosition, price: float) -> float:
        if price <= 0:
            return 0.0
        quantity = position.quantity * position.remaining_fraction
        if position.direction == "LONG":
            return (price - position.entry_price) * quantity
        return (position.entry_price - price) * quantity

    @staticmethod
    def _gross_exposure(open_positions: Sequence[OpenPosition], equity: float) -> float:
        if equity <= 0:
            return 0.0
        return sum(p.quantity * p.entry_price for p in open_positions) / equity

    @staticmethod
    def _open_risk_pct(open_positions: Sequence[OpenPosition], equity: float) -> float:
        if equity <= 0:
            return 0.0
        risk = sum(abs(p.entry_price - p.stop_price) * p.quantity * p.remaining_fraction for p in open_positions)
        return risk / equity

    def _apply_slippage(
        self,
        price: float,
        direction: str,
        atr_pct: Optional[float],
        *,
        is_exit: bool,
        is_stop: bool = False,
    ) -> float:
        """Move the fill against the trader by the modelled slippage.

        Buys fill higher, sells fill lower - always the pessimistic direction.
        """
        bps = self.cost_model.slippage_bps(is_stop_exit=is_stop, atr_pct=atr_pct) * self.config.slippage_multiplier
        fraction = bps / 10_000.0
        buying = (direction == "LONG" and not is_exit) or (direction == "SHORT" and is_exit)
        return round(price * (1 + fraction) if buying else price * (1 - fraction), 4)

    @staticmethod
    def _round_tick(price: float, direction: str, kind: str) -> float:
        tick = 0.05
        price = float(price)
        if kind == "stop":
            rounded = math.floor(price / tick) * tick if direction == "LONG" else math.ceil(price / tick) * tick
        else:
            rounded = math.ceil(price / tick) * tick if direction == "LONG" else math.floor(price / tick) * tick
        return round(rounded, 4)


__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "SimulatedTrade",
    "OpenPosition",
    "PendingOrder",
]
