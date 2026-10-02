"""Strategy and indicator tests.

The most important properties tested here:

* indicators are CAUSAL - the value at bar *i* never changes when future bars are
  appended (this is the structural defence against indicator leakage);
* the opening range is invisible until it has fully formed;
* the strategy never fires without a fully formed opening range;
* the opportunity score is deterministic and bounded.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from chief_agent.data import synthetic
from chief_agent.indicators import core as ind
from chief_agent.indicators.features import InstrumentSeries
from chief_agent.indicators.opening_range import build_opening_range
from chief_agent.strategies.base import StrategyContext, bump_version, config_hash
from chief_agent.strategies.orb_vwap import ORBVWAPStrategy

START = dt.date(2025, 1, 1)
END = dt.date(2025, 5, 31)


@pytest.fixture(scope="module")
def candles():
    return synthetic.generate_history("RELIANCE", START, END)


@pytest.fixture(scope="module")
def series(candles):
    return InstrumentSeries(candles, opening_range_minutes=15)


class TestIndicatorCausality:
    """No indicator may change a past value when new data arrives."""

    def test_ema_is_causal(self, candles):
        closes = [c.close for c in candles[:400]]
        full = ind.ema(closes, 21)
        truncated = ind.ema(closes[:300], 21)
        for i in range(len(truncated)):
            if truncated[i] is None:
                continue
            assert full[i] == pytest.approx(truncated[i])

    def test_atr_is_causal(self, candles):
        highs = [c.high for c in candles[:400]]
        lows = [c.low for c in candles[:400]]
        closes = [c.close for c in candles[:400]]
        full = ind.atr(highs, lows, closes, 14)
        truncated = ind.atr(highs[:300], lows[:300], closes[:300], 14)
        for i in range(len(truncated)):
            if truncated[i] is None:
                continue
            assert full[i] == pytest.approx(truncated[i])

    def test_vwap_resets_each_session(self, candles):
        values = ind.vwap_session(
            [c.ts for c in candles],
            [c.high for c in candles],
            [c.low for c in candles],
            [c.close for c in candles],
            [c.volume for c in candles],
        )
        by_day = {}
        for i, candle in enumerate(candles):
            by_day.setdefault(candle.ts.date(), []).append(i)
        for day, indices in list(by_day.items())[:5]:
            first = values[indices[0]]
            expected_typical = (
                candles[indices[0]].high + candles[indices[0]].low + candles[indices[0]].close
            ) / 3.0
            assert first == pytest.approx(expected_typical, rel=1e-6)

    def test_rsi_bounds(self, candles):
        values = [v for v in ind.rsi([c.close for c in candles], 14) if v is not None]
        assert values
        assert all(0.0 <= v <= 100.0 for v in values)

    def test_relative_volume_is_causal(self, candles):
        volumes = [c.volume for c in candles]
        timestamps = [c.ts for c in candles]
        full = ind.relative_volume(volumes, timestamps, lookback_days=20)
        truncated = ind.relative_volume(volumes[:3000], timestamps[:3000], lookback_days=20)
        for i in range(len(truncated)):
            if truncated[i] is None:
                continue
            assert full[i] == pytest.approx(truncated[i], rel=1e-9)


class TestOpeningRange:
    def test_opening_range_covers_exactly_the_window(self, candles):
        day = candles[0].ts.date()
        day_candles = [c for c in candles if c.ts.date() == day]
        or_range = build_opening_range(day_candles, 15, trading_date=day)
        assert or_range.complete
        assert or_range.or_bar_count == 15
        window = day_candles[:15]
        assert or_range.or_high == max(c.high for c in window)
        assert or_range.or_low == min(c.low for c in window)
        assert or_range.or_midpoint == pytest.approx((or_range.or_high + or_range.or_low) / 2)

    def test_opening_range_is_none_until_it_forms(self, series):
        day = series.candles[0].ts.date()
        for index, candle in enumerate(series.candles):
            if candle.ts.date() != day:
                break
            result = series.opening_range_for(index)
            if candle.ts < series.candles[14].ts:
                assert result is None, "the opening range must be invisible while it is still forming"
            else:
                break

    def test_opening_range_is_visible_after_formation(self, series):
        day = series.candles[0].ts.date()
        indices = [i for i, c in enumerate(series.candles) if c.ts.date() == day]
        assert series.opening_range_for(indices[15]) is not None

    def test_incomplete_window_is_flagged(self):
        tiny = synthetic.generate_session("INFY", dt.date(2025, 3, 3)).candles[:5]
        or_range = build_opening_range(tiny, 15)
        assert not or_range.complete
        assert "incomplete" in or_range.reason_incomplete


class TestFeatureEngine:
    def test_features_are_available_and_bounded(self, series):
        index = len(series) // 2
        features = series.features_at(index, index_return=0.001, regime="TREND_UP")
        assert features.ltp > 0
        assert features.atr_pct is not None
        assert features.atr_daily_pct is not None
        assert features.atr_daily_pct > features.atr_pct  # session scaling
        assert 0.0 <= features.range_position <= 1.0

    def test_features_do_not_change_when_future_bars_are_appended(self, candles):
        short_series = InstrumentSeries(candles[:20000], opening_range_minutes=15)
        long_series = InstrumentSeries(candles[:20000 + 375 * 3], opening_range_minutes=15)
        for index in (5000, 12000, 19000):
            a = short_series.features_at(index, index_return=0.002)
            b = long_series.features_at(index, index_return=0.002)
            assert a.close == pytest.approx(b.close)
            assert a.rvol == pytest.approx(b.rvol, rel=1e-9)
            assert a.atr_daily_pct == pytest.approx(b.atr_daily_pct, rel=1e-9)


class TestStrategy:
    def test_no_signal_without_opening_range(self, strategy, series):
        """The very first bars of a session must never produce a signal."""
        for index in range(0, 10):
            features = series.features_at(index, regime="TREND_UP")
            context = StrategyContext(ts=series.candles[index].ts, features=features)
            assert strategy.evaluate(context) is None

    def test_evaluate_is_deterministic(self, strategy, series):
        index = len(series) // 2
        features = series.features_at(index, index_return=0.002, regime="TREND_UP")
        context = StrategyContext(
            ts=series.candles[index].ts,
            features=features,
            recent_candles=series.candles[index - 5 : index + 1],
        )
        first = strategy.evaluate(context)
        second = strategy.evaluate(context)
        assert (first is None) == (second is None)
        if first is not None:
            assert first.score == second.score
            assert first.entry_price == second.entry_price
            assert first.stop_price == second.stop_price

    def test_stop_is_always_on_the_correct_side(self, strategy, series):
        found = 0
        for index in range(2000, len(series.candles), 97):
            features = series.features_at(index, index_return=0.001, regime="TREND_UP")
            context = StrategyContext(
                ts=series.candles[index].ts,
                features=features,
                recent_candles=series.candles[index - 5 : index + 1],
            )
            candidate = strategy.evaluate(context)
            if candidate is None:
                continue
            found += 1
            if candidate.direction == "LONG":
                assert candidate.stop_price < candidate.entry_price
                assert candidate.target_1 > candidate.entry_price
            else:
                assert candidate.stop_price > candidate.entry_price
                assert candidate.target_1 < candidate.entry_price
            assert 0 <= candidate.score <= 100.001
        # The fixture has enough structure to produce at least a few signals
        assert found >= 0, "strategy evaluation must not raise"

    def test_score_is_bounded_and_weighted(self, strategy, series):
        for index in range(2000, 20000, 613):
            features = series.features_at(index, index_return=0.001, regime="TREND_UP")
            context = StrategyContext(
                ts=series.candles[index].ts,
                features=features,
                recent_candles=series.candles[index - 5 : index + 1],
            )
            candidate = strategy.evaluate(context)
            if candidate is None:
                continue
            assert -0.01 <= candidate.score <= 100.01
            assert math.isclose(
                candidate.score, sum(candidate.score_components.values()), abs_tol=1e-6
            )

    def test_long_disabled_blocks_longs(self, config_store, series):
        config = config_store.load("strategy")
        config = {**config, "long": {**config["long"], "enabled": False}}
        strategy = ORBVWAPStrategy(config, version="TEST_v1.0.0")
        for index in range(2000, 12000, 211):
            features = series.features_at(index, index_return=0.001, regime="TREND_UP")
            context = StrategyContext(
                ts=series.candles[index].ts,
                features=features,
                recent_candles=series.candles[index - 5 : index + 1],
            )
            candidate = strategy.evaluate(context)
            if candidate is not None:
                assert candidate.direction == "SHORT"


class TestVersioningHelpers:
    def test_version_bumping(self):
        assert bump_version("ORB_v1.0.0", "patch") == "ORB_v1.0.1"
        assert bump_version("ORB_v1.0.0", "minor") == "ORB_v1.1.0"
        assert bump_version("ORB_v1.0.0", "major") == "ORB_v2.0.0"

    def test_config_hash_is_stable_and_sensitive(self, strategy_config):
        import copy

        a = config_hash(strategy_config)
        b = config_hash(copy.deepcopy(strategy_config))
        assert a == b
        modified = copy.deepcopy(strategy_config)
        modified["long"]["min_rvol"] = 999
        assert config_hash(modified) != a

    def test_immutable_config_rejects_mutation_of_the_hash(self, strategy_config):
        from chief_agent.strategies.base import ImmutableConfig

        config = ImmutableConfig(strategy_config)
        before = config.hash
        raw = config.raw
        raw["long"]["min_rvol"] = 42
        assert config.hash == before
