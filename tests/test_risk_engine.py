"""Risk engine tests.

The risk engine has absolute authority, so these are the most important tests in
the suite. They deliberately exercise the failure paths: every one of them must
result in NO TRADE.
"""

from __future__ import annotations

import datetime as dt

import pytest

from chief_agent.risk.engine import ProposedTrade, RiskContext, RiskEngine, RiskLevel, RiskReason
from chief_agent.risk.position_sizing import SizingConstraints, compute_position_size


def make_trade(**overrides):
    base = dict(
        instrument_key="SIM|RELIANCE",
        symbol="RELIANCE",
        direction="LONG",
        entry_price=100.0,
        stop_price=98.0,
        target_1=103.0,
        target_2=106.0,
        risk_reward=1.5,
        strategy_version="TEST_v1.0.0",
        sector="Energy",
        average_daily_volume=1_000_000,
        spread_pct=0.0005,
        lot_size=1,
        tick_size=0.05,
    )
    base.update(overrides)
    return ProposedTrade(**base)


def make_context(**overrides):
    base = dict(
        account_equity=500_000.0,
        available_cash=500_000.0,
        starting_equity_today=500_000.0,
        peak_equity=500_000.0,
    )
    base.update(overrides)
    return RiskContext(**base)


class TestSizing:
    def test_quantity_follows_risk_budget(self):
        result = compute_position_size(
            SizingConstraints(
                account_equity=500_000,
                risk_pct=0.0025,
                max_risk_pct=0.005,
                entry_price=100.0,
                stop_price=99.0,
                max_position_exposure_pct=1.0,
                max_gross_exposure_pct=1.0,
                max_total_open_risk_pct=1.0,
            )
        )
        # 0.25% of 500,000 = 1,250 risk capital; 1.00 risk per share -> 1,250 shares
        assert result.quantity == 1250
        assert result.risk_amount == pytest.approx(1250.0)

    def test_quantity_rounds_down_to_lot_size(self):
        result = compute_position_size(
            SizingConstraints(
                account_equity=500_000,
                risk_pct=0.0025,
                entry_price=100.0,
                stop_price=99.0,
                lot_size=100,
                max_position_exposure_pct=1.0,
                max_gross_exposure_pct=1.0,
                max_total_open_risk_pct=1.0,
            )
        )
        assert result.quantity % 100 == 0
        assert result.risk_amount <= 1250.0

    def test_risk_is_never_exceeded(self):
        for stop_distance in (0.5, 1.0, 5.0, 20.0):
            result = compute_position_size(
                SizingConstraints(
                    account_equity=100_000,
                    risk_pct=0.0025,
                    entry_price=100.0,
                    stop_price=100.0 - stop_distance,
                    max_position_exposure_pct=1.0,
                    max_gross_exposure_pct=1.0,
                    max_total_open_risk_pct=1.0,
                )
            )
            assert result.risk_amount <= 100_000 * 0.0025 + 1e-6

    def test_zero_stop_distance_is_rejected(self):
        result = compute_position_size(
            SizingConstraints(account_equity=100_000, risk_pct=0.005, entry_price=100.0, stop_price=100.0)
        )
        assert result.rejected
        assert "stop distance is zero" in result.rejection_reason

    def test_exposure_cap_binds(self):
        result = compute_position_size(
            SizingConstraints(
                account_equity=100_000,
                risk_pct=0.005,
                entry_price=100.0,
                stop_price=99.0,
                max_position_exposure_pct=0.10,   # 10,000 notional cap
                max_gross_exposure_pct=1.0,
                max_total_open_risk_pct=1.0,
            )
        )
        assert result.binding_constraint == "max_position_exposure"
        assert result.notional <= 10_000 + 1e-6

    def test_requested_risk_above_ceiling_is_capped(self):
        result = compute_position_size(
            SizingConstraints(
                account_equity=500_000,
                risk_pct=0.05,           # 5% requested
                max_risk_pct=0.005,      # 0.5% ceiling
                entry_price=100.0,
                stop_price=95.0,
                max_position_exposure_pct=1.0,
                max_gross_exposure_pct=1.0,
                max_total_open_risk_pct=1.0,
            )
        )
        assert result.risk_pct <= 0.005
        assert any("capped" in w for w in result.warnings)


class TestRiskEngineApproval:
    def test_clean_trade_is_approved(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(), make_context())
        assert decision.tradeable
        assert decision.sizing.quantity > 0
        assert decision.sizing.risk_pct <= 0.005

    def test_zero_quantity_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(), make_context(account_equity=50.0, available_cash=50.0))
        assert not decision.tradeable


class TestRiskEngineFailClosed:
    """Every failure mode must result in NO TRADE."""

    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({"kill_switch_engaged": True}, RiskReason.KILL_SWITCH_ENGAGED),
            ({"data_safe_mode": True}, RiskReason.DATA_SAFE_MODE),
            ({"data_stale": True}, RiskReason.STALE_DATA),
            ({"reconciliation_pending": True}, RiskReason.RECONCILIATION_PENDING),
            ({"broker_healthy": False}, RiskReason.BROKER_UNSTABLE),
            ({"engine_healthy": False}, RiskReason.RISK_ENGINE_UNAVAILABLE),
        ],
    )
    def test_system_failures_block_trading(self, config_store, overrides, expected):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(), make_context(**overrides))
        assert decision.level is RiskLevel.REJECTED
        assert decision.reason is expected
        assert not decision.tradeable

    def test_soft_daily_stop_blocks_new_entries(self, config_store):
        engine = RiskEngine(config_store)
        context = make_context(realised_pnl_today=-5000.0)  # -1.0%
        decision = engine.evaluate(make_trade(), context)
        assert decision.reason is RiskReason.DAILY_LOSS_SOFT_LIMIT

    def test_hard_daily_stop_halts(self, config_store):
        engine = RiskEngine(config_store)
        context = make_context(realised_pnl_today=-8000.0)  # -1.6%
        decision = engine.evaluate(make_trade(), context)
        assert decision.reason is RiskReason.DAILY_LOSS_HARD_LIMIT

    def test_max_consecutive_losses(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(), make_context(consecutive_losses=4))
        assert decision.reason is RiskReason.MAX_CONSECUTIVE_LOSSES

    def test_rejection_loop_halts(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(), make_context(consecutive_rejections=3))
        assert decision.reason is RiskReason.MAX_CONSECUTIVE_REJECTIONS

    def test_max_positions(self, config_store):
        engine = RiskEngine(config_store)
        positions = [
            {"instrument_key": f"SIM|X{i}", "sector": f"S{i}", "direction": "LONG", "quantity": 10}
            for i in range(3)
        ]
        decision = engine.evaluate(make_trade(), make_context(open_positions=positions))
        assert decision.reason is RiskReason.MAX_POSITIONS

    def test_sector_concentration(self, config_store):
        engine = RiskEngine(config_store)
        positions = [
            {"instrument_key": "SIM|A", "sector": "Energy", "direction": "LONG", "quantity": 10},
            {"instrument_key": "SIM|B", "sector": "Energy", "direction": "LONG", "quantity": 10},
        ]
        decision = engine.evaluate(make_trade(sector="Energy"), make_context(open_positions=positions))
        assert decision.reason is RiskReason.MAX_CORRELATED_SECTOR_POSITIONS

    def test_averaging_down_is_forbidden(self, config_store):
        engine = RiskEngine(config_store)
        positions = [
            {"instrument_key": "SIM|RELIANCE", "sector": "Energy", "direction": "LONG", "quantity": 10}
        ]
        decision = engine.evaluate(make_trade(), make_context(open_positions=positions))
        assert decision.reason is RiskReason.AVERAGING_DOWN_BLOCKED

    def test_no_stop_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(stop_price=0.0), make_context())
        assert decision.reason is RiskReason.NO_STOP_DEFINED

    def test_inverted_stop_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(stop_price=101.0), make_context())
        assert decision.reason is RiskReason.INVALID_STOP

    def test_wide_spread_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(spread_pct=0.05), make_context())
        assert decision.reason is RiskReason.SPREAD_TOO_WIDE

    def test_illiquid_name_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(average_daily_volume=1000), make_context())
        assert decision.reason is RiskReason.LIQUIDITY_TOO_LOW

    def test_suspended_strategy_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(
            make_trade(), make_context(suspended_strategies=["TEST_v1.0.0"])
        )
        assert decision.reason is RiskReason.STRATEGY_SUSPENDED

    def test_pause_after_losses_is_respected(self, config_store):
        engine = RiskEngine(config_store)
        future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        decision = engine.evaluate(make_trade(), make_context(paused_until=future))
        assert decision.reason is RiskReason.EPISODE_PAUSED

    def test_low_risk_reward_is_rejected(self, config_store):
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(risk_reward=0.5), make_context())
        assert decision.reason is RiskReason.RISK_REWARD_TOO_LOW


class TestRiskEngineImmutability:
    def test_limits_come_from_config_not_code(self, config_store):
        engine = RiskEngine(config_store)
        limits = engine.limits_summary()
        assert limits["risk_per_trade_pct"] == pytest.approx(0.0025)
        assert limits["max_simultaneous_positions"] == 3
        assert limits["max_total_open_risk_pct"] == pytest.approx(0.0075)
        assert limits["forbid_averaging_down"] is True
        assert limits["require_stop_before_entry"] is True

    def test_decisions_are_logged_for_research(self, config_store):
        engine = RiskEngine(config_store)
        engine.evaluate(make_trade(), make_context())
        engine.evaluate(make_trade(spread_pct=0.09), make_context())
        assert len(engine.events) == 2
        assert engine.events[1]["level"] == "REJECTED"


class TestKillSwitch:
    def test_engage_and_release(self):
        from chief_agent.risk.killswitch import KillSwitchTrigger, get_kill_switch

        switch = get_kill_switch()
        assert not switch.block_new_orders()
        switch.engage(KillSwitchTrigger.MANUAL_BUTTON, reason="test")
        assert switch.block_new_orders()
        switch.release()
        assert not switch.block_new_orders()

    def test_automatic_engagement(self):
        from chief_agent.risk.killswitch import KillSwitchTrigger, get_kill_switch

        switch = get_kill_switch()
        switch.engage_automatically(KillSwitchTrigger.DATA_FAILURE, reason="feed died")
        assert switch.engaged
        assert switch.state.trigger is KillSwitchTrigger.DATA_FAILURE

    def test_engagement_blocks_risk_approval(self, config_store):
        from chief_agent.risk.killswitch import KillSwitchTrigger, get_kill_switch

        switch = get_kill_switch()
        switch.engage(KillSwitchTrigger.MANUAL_BUTTON, reason="test")
        engine = RiskEngine(config_store)
        decision = engine.evaluate(make_trade(), make_context(kill_switch_engaged=True))
        assert decision.reason is RiskReason.KILL_SWITCH_ENGAGED
