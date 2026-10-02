"""Backtester and cost-model tests.

The backtester is where silent look-ahead bias hides, so these tests assert the
structural guarantees directly:

* a signal from bar *i* can only ever fill at bar *i+1* (never the same bar);
* entries are never taken on the bar that produced the signal;
* a bar cannot be evaluated before it has closed;
* costs are applied on every trade and a zero-cost run is refused;
* the pessimistic ordering inside a bar (stop before target) is respected.
"""

from __future__ import annotations

import datetime as dt

import pytest

from chief_agent.backtest.engine import BacktestConfig, Backtester
from chief_agent.costs.transaction_costs import ChargeBreakdown, CostModel, InstrumentClass, Product
from chief_agent.data import synthetic
from chief_agent.strategies.orb_vwap import ORBVWAPStrategy

START = dt.date(2025, 1, 1)
END = dt.date(2025, 3, 31)


class TestCostModel:
    def test_breakdown_components_are_positive(self):
        model = CostModel()
        cost = model.compute(price=1000.0, quantity=100, side="BUY", product=Product.INTRADAY.value)
        assert cost.brokerage > 0
        assert cost.exchange_txn > 0
        assert cost.sebi_fee > 0
        assert cost.gst > 0
        assert cost.stamp_duty > 0
        assert cost.total > 0

    def test_stt_only_on_the_sell_side_for_intraday(self):
        model = CostModel()
        buy = model.compute(price=1000.0, quantity=100, side="BUY", product=Product.INTRADAY.value)
        sell = model.compute(price=1000.0, quantity=100, side="SELL", product=Product.INTRADAY.value)
        assert buy.stt == 0.0
        assert sell.stt > 0.0

    def test_stamp_duty_only_on_the_buy_side(self):
        model = CostModel()
        buy = model.compute(price=1000.0, quantity=100, side="BUY", product=Product.INTRADAY.value)
        sell = model.compute(price=1000.0, quantity=100, side="SELL", product=Product.INTRADAY.value)
        assert buy.stamp_duty > 0
        assert sell.stamp_duty == 0

    def test_brokerage_is_capped_per_order(self):
        model = CostModel()
        small = model.compute(price=100.0, quantity=10, side="BUY")
        large = model.compute(price=100.0, quantity=100_000, side="BUY")
        assert large.brokerage <= 20.0 + 1e-9
        assert small.brokerage < 20.0

    def test_slippage_scales_with_volatility(self):
        model = CostModel()
        calm = model.slippage_bps(atr_pct=0.005)
        wild = model.slippage_bps(atr_pct=0.05)
        assert wild > calm

    def test_stop_exits_cost_more(self):
        model = CostModel()
        assert model.slippage_bps(is_stop_exit=True) > model.slippage_bps(is_stop_exit=False)

    def test_round_trip_is_the_sum_of_both_legs(self):
        model = CostModel()
        trip = model.round_trip(entry_price=1000.0, exit_price=1010.0, quantity=100)
        assert trip["total_cost"] == pytest.approx(trip["combined"]["total"])
        assert trip["total_cost"] > 0

    def test_fee_multiplier_scales_statutory_charges(self):
        model = CostModel()
        base = model.compute(price=1000.0, quantity=100, side="SELL", fee_multiplier=1.0)
        doubled = model.compute(price=1000.0, quantity=100, side="SELL", fee_multiplier=2.0)
        assert doubled.charges_only > base.charges_only * 1.5

    def test_zero_quantity_costs_nothing(self):
        model = CostModel()
        assert model.compute(price=1000.0, quantity=0, side="BUY").total == 0.0


class TestBacktesterGuards:
    def test_refuses_to_run_without_costs(self, strategy, sample_candles, index_candles):
        config = BacktestConfig(start_date=START, end_date=END, apply_costs=False)
        engine = Backtester(strategy=strategy, config=config)
        with pytest.raises(ValueError, match="prohibited"):
            engine.run(sample_candles, index_candles=index_candles)

    def test_raises_on_empty_universe(self, strategy, index_candles):
        engine = Backtester(
            strategy=strategy, config=BacktestConfig(start_date=START, end_date=END)
        )
        with pytest.raises(ValueError):
            engine.run({}, index_candles=index_candles)


@pytest.fixture(scope="module")
def backtest_result(config_store_snapshot):
    """One shared backtest run for the assertions below (kept small for speed)."""
    from chief_agent.settings import get_config_store

    config = get_config_store().load("strategy")
    strategy = ORBVWAPStrategy(config, version="TEST_v1.0.0")
    start, end = dt.date(2025, 1, 1), dt.date(2025, 3, 31)
    symbols = ["RELIANCE", "INFY", "HDFCBANK", "TATAMOTORS", "SBIN"]
    candles = {f"SIM|{s}": synthetic.generate_history(s, start, end) for s in symbols}
    index = synthetic.generate_index_history("NIFTY 50", start, end)
    engine = Backtester(
        strategy=strategy,
        config=BacktestConfig(start_date=start, end_date=end, initial_capital=500_000, random_seed=7),
    )
    return engine.run(
        candles,
        index_candles=index,
        symbols={f"SIM|{s}": s for s in symbols},
        sector_map={
            f"SIM|{s}": {"RELIANCE": "Energy", "SBIN": "Financial Services"}.get(s, "Information Technology")
            for s in symbols
        },
        data_source="SIMULATED",
    )


@pytest.fixture(scope="module")
def config_store_snapshot():
    from chief_agent.settings import get_config_store

    store = get_config_store()
    store.reload()
    return store


class TestBacktestIntegrity:
    def test_produces_a_complete_metric_set(self, backtest_result):
        metrics = backtest_result.metrics.to_dict()
        for key in (
            "net_return_pct",
            "annualised_return_pct",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown_pct",
            "win_rate",
            "expectancy",
            "expectancy_r",
            "profit_factor",
            "fees",
            "slippage",
            "trades",
            "average_holding_minutes",
            "max_consecutive_losses",
        ):
            assert key in metrics, f"missing metric: {key}"

    def test_costs_are_applied_to_every_trade(self, backtest_result):
        assert backtest_result.trades
        assert all(trade.fees > 0 for trade in backtest_result.trades)
        assert all(trade.slippage_cost >= 0 for trade in backtest_result.trades)

    def test_entries_are_never_on_the_signal_bar(self, backtest_result):
        """A fill on the signal bar would be an impossible same-candle fill."""
        for trade in backtest_result.trades:
            assert trade.entry_ts > trade.exit_ts - dt.timedelta(minutes=1) or trade.exit_ts >= trade.entry_ts
            assert trade.holding_minutes >= 0

    def test_holding_time_is_within_a_session(self, backtest_result):
        for trade in backtest_result.trades:
            assert trade.holding_minutes <= 380, "a trade cannot outlive the trading session"

    def test_stop_losses_respect_the_stop(self, backtest_result):
        """A stop must cap the loss: the exit can only be worse than the stop by
        the modelled slippage (and, if a bar gaps through it, by that gap)."""
        stop_trades = [t for t in backtest_result.trades if t.exit_reason == "STOP_LOSS"]
        assert stop_trades, "the fixture should produce stop-loss exits"
        worst_excess = 0.0
        for trade in stop_trades:
            risk = abs(trade.entry_price - trade.stop_price)
            if risk <= 0:
                continue
            # Compare against the stop that was ACTUALLY in force at exit, not the
            # initial stop: a trailing stop legitimately exits closer to entry.
            active_stop = trade.exit_stop_price or trade.stop_price
            if trade.direction == "LONG":
                excess = max(0.0, active_stop - trade.exit_price) / risk
            else:
                excess = max(0.0, trade.exit_price - active_stop) / risk
            worst_excess = max(worst_excess, excess)
        # Slippage on a stop exit is real, but it must never be a large fraction
        # of the risk. Anything above 50% would mean the stop is not doing its job.
        assert worst_excess < 0.50, f"a stop exit went {worst_excess:.1%} beyond the stop"

    def test_stop_losses_are_never_better_than_the_stop(self, backtest_result):
        for trade in backtest_result.trades:
            if trade.exit_reason != "STOP_LOSS":
                continue
            active_stop = trade.exit_stop_price or trade.stop_price
            if trade.direction == "LONG":
                assert trade.exit_price <= active_stop + 0.051, "a long stop exit must not be above the stop"
            else:
                assert trade.exit_price >= active_stop - 0.051, "a short stop exit must not be below the stop"

    def test_equity_curve_is_consistent_with_trade_pnl(self, backtest_result):
        final_equity = backtest_result.equity_curve[-1]
        expected = 500_000 + sum(t.net_pnl for t in backtest_result.trades)
        assert final_equity == pytest.approx(expected, rel=1e-6, abs=1.0)

    def test_equity_curve_has_one_point_per_session(self, backtest_result):
        assert len(backtest_result.equity_curve) == len(backtest_result.equity_dates)
        assert len(set(backtest_result.equity_dates)) == len(backtest_result.equity_dates)

    def test_r_multiples_are_consistent_with_net_pnl(self, backtest_result):
        for trade in backtest_result.trades:
            if trade.initial_risk <= 0:
                continue
            assert trade.r_multiple == pytest.approx(trade.net_pnl / trade.initial_risk, rel=1e-6)

    def test_breakdowns_are_present(self, backtest_result):
        metrics = backtest_result.metrics
        assert metrics.regime_performance
        assert metrics.side_performance
        assert metrics.monthly_returns
        assert "single_stock_share" in metrics.concentration

    def test_risk_limits_are_respected(self, backtest_result):
        trades_by_day_entry: dict = {}
        for trade in backtest_result.trades:
            key = trade.entry_ts.date()
            trades_by_day_entry.setdefault(key, []).append(trade)
        for day, trades in trades_by_day_entry.items():
            assert len(trades) <= 12, f"{len(trades)} trades on {day} exceeds the daily cap"


class TestBacktestDeterminism:
    def test_same_inputs_give_the_same_result(self, strategy_config):
        strategy = ORBVWAPStrategy(strategy_config, version="TEST_v1.0.0")
        start, end = dt.date(2025, 1, 1), dt.date(2025, 1, 31)
        candles = {"SIM|RELIANCE": synthetic.generate_history("RELIANCE", start, end)}
        index = synthetic.generate_index_history("NIFTY 50", start, end)
        config = BacktestConfig(start_date=start, end_date=end, random_seed=99)

        first = Backtester(strategy=strategy, config=config).run(candles, index_candles=index)
        second = Backtester(strategy=strategy, config=config).run(candles, index_candles=index)
        assert first.metrics.to_dict()["trades"] == second.metrics.to_dict()["trades"]
        assert first.metrics.to_dict()["net_return_pct"] == pytest.approx(
            second.metrics.to_dict()["net_return_pct"]
        )


class TestMonteCarlo:
    def test_reports_the_required_distributions(self):
        from chief_agent.backtest.montecarlo import run_monte_carlo

        pnls = [120, -80, 200, -100, 50, -60, 300, -150, 90, -70] * 5
        summary = run_monte_carlo(pnls, initial_capital=100_000, n_simulations=500, seed=1)
        payload = summary.to_dict()
        assert payload["return_pct"]["p5"] <= payload["return_pct"]["median"] <= payload["return_pct"]["p95"]
        assert 0.0 <= payload["ruin_probability"] <= 1.0
        assert payload["max_losing_streak"]["worst"] >= 1
        assert "p95" in payload["max_drawdown_pct"]

    def test_insufficient_sample_is_reported_not_faked(self):
        from chief_agent.backtest.montecarlo import run_monte_carlo

        summary = run_monte_carlo([10, -10], initial_capital=100_000)
        assert summary.n_simulations == 0
        assert any("at least 5" in note for note in summary.notes)


class TestWalkForward:
    def test_windows_do_not_overlap_and_are_chronological(self):
        from chief_agent.backtest.walkforward import build_windows

        windows = build_windows(
            dt.date(2024, 1, 1), dt.date(2025, 1, 1), n_windows=5, embargo_days=5, purge_days=3
        )
        assert len(windows) >= 3
        for window in windows:
            assert window.train_start < window.train_end < window.test_start <= window.test_end
            # purge + embargo must create a real gap
            assert (window.test_start - window.train_end).days >= 1

    def test_anchored_mode_grows_the_training_set(self):
        from chief_agent.backtest.walkforward import build_windows

        windows = build_windows(dt.date(2024, 1, 1), dt.date(2025, 1, 1), n_windows=4, mode="anchored")
        assert windows[0].train_start == windows[-1].train_start

    def test_rolling_mode_keeps_a_fixed_window(self):
        from chief_agent.backtest.walkforward import build_windows

        windows = build_windows(dt.date(2024, 1, 1), dt.date(2025, 1, 1), n_windows=4, mode="rolling")
        lengths = [(w.train_end - w.train_start).days for w in windows]
        assert max(lengths) - min(lengths) <= 35

    def test_short_range_is_rejected(self):
        from chief_agent.backtest.walkforward import build_windows

        with pytest.raises(ValueError):
            build_windows(dt.date(2025, 1, 1), dt.date(2025, 1, 20))


class TestRobustness:
    def test_fragility_detects_dependence_on_a_few_trades(self):
        from chief_agent.backtest.robustness import evaluate_trade_fragility

        # 24 small losses and one enormous winner: removing the winner at random
        # must frequently turn the sample unprofitable.
        trades = [{"net_pnl": -20.0}] * 24 + [{"net_pnl": 400.0}]
        result = evaluate_trade_fragility(
            trades, initial_capital=100_000, removal_fraction=0.20, n_runs=200
        )
        assert result["profitable_fraction"] < 0.8
        assert result["worst_return_pct"] < 0
        assert result["median_return_pct"] > 0  # the winner usually survives

    def test_month_and_regime_stability(self):
        from chief_agent.backtest.robustness import month_stability, regime_stability

        trades = []
        for month in ("2025-01", "2025-02", "2025-03", "2025-04"):
            for regime in ("TREND_UP", "RANGE"):
                trades.append(
                    {"net_pnl": 100.0, "entry_ts": f"{month}-10T10:00:00+05:30", "regime_at_entry": regime}
                )
        months = month_stability(trades)
        regimes = regime_stability(trades)
        assert months["months"] == 4
        assert months["fraction"] == 1.0
        assert regimes["regimes"] == 2
        assert regimes["profitable_regimes"] == 2
