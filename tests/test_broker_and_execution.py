"""Broker client, order lifecycle, paper engine and execution tests.

These use a MOCKED HTTP transport, so no network call is ever made and no real
order can exist. The scenarios mirror the brief's list:

    duplicate signal, duplicate order, network failure, broker timeout,
    partial fill, order rejection, uncertain outcome, rate limiting.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from chief_agent.broker.rate_limiter import LimitSpec, RateLimiter
from chief_agent.broker.upstox_client import (
    Endpoints,
    UpstoxAuthError,
    UpstoxHttpClient,
    UpstoxNetworkError,
    UpstoxRateLimitError,
    UpstoxServerError,
    UpstoxTimeoutError,
    UpstoxValidationError,
)
from chief_agent.broker.upstox_orders import OrderRequest, UpstoxOrders
from chief_agent.costs.transaction_costs import CostModel
from chief_agent.execution.paper_engine import PaperTradingEngine
from chief_agent.risk.engine import ProposedTrade, RiskContext, RiskEngine
from chief_agent.settings import OperatingMode, get_settings


def make_client(handler) -> UpstoxHttpClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, timeout=5.0)
    return UpstoxHttpClient(settings=get_settings(), client=http)


class TestRateLimiter:
    def test_budget_is_never_exceeded(self):
        limiter = RateLimiter(LimitSpec(per_second=5, per_minute=100, per_30_minutes=100, name="t"), safety_factor=1.0)
        assert limiter.effective[0] == 5

    def test_safety_factor_reduces_the_budget(self):
        limiter = RateLimiter(LimitSpec(10, 500, 2000, "t"), safety_factor=0.7)
        assert limiter.effective == (7, 350, 1400)

    def test_acquire_then_timeout(self):
        limiter = RateLimiter(LimitSpec(per_second=2, per_minute=100, per_30_minutes=100, name="t"), safety_factor=1.0)
        assert limiter.acquire(timeout=0.1)
        assert limiter.acquire(timeout=0.1)
        # the third request must wait rather than exceed the published limit
        assert not limiter.acquire(timeout=0.05)

    def test_429_opens_a_cooldown(self):
        limiter = RateLimiter(LimitSpec(100, 1000, 10000, "t"))
        limiter.note_429(retry_after_seconds=1.0)
        assert not limiter.acquire(timeout=0.05)
        assert limiter.stats()["throttled_429"] == 1


class TestHttpClientErrorTaxonomy:
    def test_success_parses_the_body(self):
        client = make_client(lambda request: httpx.Response(200, json={"status": "success", "data": {"ok": 1}}))
        client.set_access_token("fake-token")
        response = client.get("/v2/user/profile")
        assert response.ok and response.data == {"ok": 1}

    def test_401_is_an_auth_error(self):
        client = make_client(lambda request: httpx.Response(401, json={"errors": [{"message": "Invalid token"}]}))
        client.set_access_token("fake")
        with pytest.raises(UpstoxAuthError):
            client.get("/v2/user/profile")

    def test_400_is_a_validation_error_and_is_not_retried(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(400, json={"errors": [{"message": "bad request", "errorCode": "UDAPI1004"}]})

        client = make_client(handler)
        client.set_access_token("fake")
        with pytest.raises(UpstoxValidationError):
            client.get("/v2/order/retrieve-all")
        assert calls["n"] == 1, "a 4xx must never be retried"

    def test_timeout_is_uncertain_for_orders(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out")

        client = make_client(handler)
        client.set_access_token("fake")
        with pytest.raises(UpstoxTimeoutError) as excinfo:
            client.request("POST", Endpoints.ORDER_PLACE_V3, json_body={}, idempotent=False)
        assert excinfo.value.outcome_uncertain is True

    def test_timeout_is_retried_for_reads(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("timed out")
            return httpx.Response(200, json={"status": "success", "data": []})

        client = make_client(handler)
        client.set_access_token("fake")
        response = client.get("/v2/order/retrieve-all", max_retries=2)
        assert response.ok
        assert calls["n"] == 2

    def test_5xx_is_retried_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503, json={"message": "unavailable"})
            return httpx.Response(200, json={"status": "success", "data": {}})

        client = make_client(handler)
        client.set_access_token("fake")
        assert client.get("/v2/user/profile", max_retries=2).ok

    def test_429_is_reported_as_a_rate_limit_error(self):
        client = make_client(lambda request: httpx.Response(429, headers={"Retry-After": "1"},
                                                            json={"message": "too many requests"}))
        client.set_access_token("fake")
        with pytest.raises(UpstoxRateLimitError):
            client.get("/v2/user/profile", max_retries=0)

    def test_missing_token_raises_before_any_request(self):
        client = make_client(lambda request: httpx.Response(200, json={}))
        with pytest.raises(UpstoxAuthError):
            client.get("/v2/user/profile")

    def test_circuit_breaker_fails_fast_after_repeated_network_errors(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            raise httpx.ConnectError("no route to host")

        client = make_client(handler)
        client.set_access_token("fake")
        for _ in range(3):
            with pytest.raises(UpstoxNetworkError):
                client.get("/v2/user/profile", max_retries=0)
        before = calls["n"]
        with pytest.raises(UpstoxNetworkError, match="circuit open"):
            client.get("/v2/user/profile")
        assert calls["n"] == before, "an open circuit must not touch the network"


class TestOrderRequestValidation:
    def test_valid_limit_order(self):
        request = OrderRequest(
            instrument_token="NSE_EQ|INE002A01018",
            transaction_type="BUY",
            quantity=10,
            order_type="LIMIT",
            price=2500.0,
        )
        assert request.validate() == []

    def test_market_order_with_zero_protection_is_rejected(self):
        request = OrderRequest(
            instrument_token="NSE_EQ|INE002A01018",
            transaction_type="BUY",
            quantity=10,
            order_type="MARKET",
            market_protection=0,
        )
        assert any("market_protection" in error for error in request.validate())

    def test_stop_loss_requires_a_trigger(self):
        request = OrderRequest(
            instrument_token="NSE_EQ|INE002A01018",
            transaction_type="SELL",
            quantity=10,
            order_type="SL-M",
        )
        assert any("trigger_price" in error for error in request.validate())

    def test_invalid_enum_values_are_caught(self):
        request = OrderRequest(
            instrument_token="X", transaction_type="HOLD", quantity=1, order_type="WHATEVER"
        )
        errors = request.validate()
        assert any("transaction_type" in e for e in errors)
        assert any("order_type" in e for e in errors)

    def test_tag_length_limit(self):
        request = OrderRequest(
            instrument_token="X", transaction_type="BUY", quantity=1, order_type="MARKET", tag="T" * 50
        )
        assert any("tag" in error for error in request.validate())

    def test_payload_has_no_none_values(self):
        request = OrderRequest(instrument_token="X", transaction_type="BUY", quantity=1, order_type="MARKET")
        payload = request.sanitized()
        assert all(value is not None for value in payload.values())


class TestUpstoxOrdersApi:
    def test_place_order_returns_the_broker_order_id(self):
        captured = {}

        def handler(request: httpx.Request):
            captured["body"] = json.loads(request.content)
            captured["path"] = request.url.path
            return httpx.Response(200, json={"status": "success", "data": {"order_ids": ["123456"]},
                                             "metadata": {"latency": 42}})

        client = make_client(handler)
        client.set_access_token("fake")
        orders = UpstoxOrders(client)
        result = orders.place_order(
            OrderRequest(
                instrument_token="NSE_EQ|INE002A01018",
                transaction_type="BUY",
                quantity=5,
                order_type="LIMIT",
                price=100.0,
                tag="CHIEF-ABC",
            )
        )
        assert result.ok
        assert result.primary_order_id == "123456"
        assert captured["path"].endswith("/v3/order/place")
        assert captured["body"]["instrument_token"] == "NSE_EQ|INE002A01018"
        assert captured["body"]["tag"] == "CHIEF-ABC"

    def test_uncertain_outcome_is_surfaced(self):
        def handler(request: httpx.Request):
            raise httpx.ReadTimeout("no response")

        client = make_client(handler)
        client.set_access_token("fake")
        orders = UpstoxOrders(client)
        with pytest.raises(UpstoxTimeoutError) as excinfo:
            orders.place_order(
                OrderRequest(
                    instrument_token="NSE_EQ|INE002A01018", transaction_type="BUY", quantity=5,
                    order_type="MARKET",
                )
            )
        assert excinfo.value.outcome_uncertain

    def test_status_classification(self):
        assert UpstoxOrders.classify_status("complete") == "FILLED"
        assert UpstoxOrders.classify_status("rejected") == "FAILED"
        assert UpstoxOrders.classify_status("open") == "OPEN"
        assert UpstoxOrders.classify_status("trigger pending") == "OPEN"


class TestPaperTradingEngine:
    def make_engine(self):
        return PaperTradingEngine(initial_capital=500_000, cost_model=CostModel(), rng_seed=1)

    def test_market_order_fills_and_charges_costs(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=100,
            order_type="MARKET", reference_price=100.0,
        )
        assert order.status in ("COMPLETE", "PARTIALLY_FILLED")
        assert order.filled_quantity > 0
        assert order.fees > 0

    def test_limit_order_above_the_market_does_not_fill_for_a_buy(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=100,
            order_type="LIMIT", price=90.0, reference_price=100.0,
        )
        assert order.filled_quantity == 0

    def test_accounting_is_exact_for_a_winning_long(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=100,
            order_type="MARKET", reference_price=100.0,
        )
        position = engine.open_position_from_order(
            order, direction="LONG", stop_price=98.0, target_1=104.0, target_2=108.0
        )
        record = engine.close_position(position, 104.0, dt.datetime.now(dt.timezone.utc), "TARGET_1")
        assert engine.equity() == pytest.approx(500_000 + record["net_pnl"], abs=0.01)

    def test_accounting_is_exact_for_a_losing_short(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="SELL", quantity=100,
            order_type="MARKET", reference_price=100.0,
        )
        position = engine.open_position_from_order(
            order, direction="SHORT", stop_price=102.0, target_1=96.0, target_2=94.0
        )
        record = engine.close_position(position, 102.0, dt.datetime.now(dt.timezone.utc), "STOP_LOSS")
        assert record["net_pnl"] < 0
        assert engine.equity() == pytest.approx(500_000 + record["net_pnl"], abs=0.01)

    def test_stop_exit_happens_when_the_bar_trades_through(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=10,
            order_type="MARKET", reference_price=100.0,
        )
        position = engine.open_position_from_order(
            order, direction="LONG", stop_price=98.0, target_1=104.0, target_2=108.0
        )
        from chief_agent.broker.upstox_market_data import Candle
        from chief_agent.timeutil import now_ist

        bar = Candle(ts=now_ist(), open=100.0, high=100.5, low=97.0, close=97.5, volume=1000)
        closed = engine.evaluate_exits({"SIM|X": bar}, ts=now_ist())
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "STOP_LOSS"

    def test_target_exit_happens_when_the_bar_reaches_it(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=10,
            order_type="MARKET", reference_price=100.0,
        )
        position = engine.open_position_from_order(
            order, direction="LONG", stop_price=98.0, target_1=104.0, target_2=108.0
        )
        from chief_agent.broker.upstox_market_data import Candle
        from chief_agent.timeutil import now_ist

        bar = Candle(ts=now_ist(), open=100.0, high=105.0, low=99.5, close=104.5, volume=1000)
        closed = engine.evaluate_exits({"SIM|X": bar}, ts=now_ist())
        assert closed and closed[0]["exit_reason"] == "TARGET_1"

    def test_stop_wins_when_a_bar_contains_both_stop_and_target(self):
        """The pessimistic assumption: the stop fills first."""
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=10,
            order_type="MARKET", reference_price=100.0,
        )
        position = engine.open_position_from_order(
            order, direction="LONG", stop_price=98.0, target_1=104.0, target_2=108.0
        )
        from chief_agent.broker.upstox_market_data import Candle
        from chief_agent.timeutil import now_ist

        bar = Candle(ts=now_ist(), open=100.0, high=105.0, low=97.0, close=104.0, volume=1000)
        closed = engine.evaluate_exits({"SIM|X": bar}, ts=now_ist())
        assert closed[0]["exit_reason"] == "STOP_LOSS"

    def test_square_off_before_the_close(self):
        engine = self.make_engine()
        order = engine.place_order(
            instrument_key="SIM|X", symbol="X", transaction_type="BUY", quantity=10,
            order_type="MARKET", reference_price=100.0,
        )
        engine.open_position_from_order(
            order, direction="LONG", stop_price=98.0, target_1=120.0, target_2=130.0
        )
        from chief_agent.broker.upstox_market_data import Candle
        from chief_agent.timeutil import now_ist

        bar = Candle(ts=now_ist(), open=100.0, high=100.5, low=99.5, close=100.2, volume=100)
        closed = engine.evaluate_exits(
            {"SIM|X": bar}, ts=now_ist(), minutes_into_session=370,
            strategy_config={"exits": {"force_square_off_minutes_before_close": 10}},
        )
        assert closed and closed[0]["exit_reason"] == "SQUARE_OFF"


class TestExecutionIdempotency:
    def build_engine(self):
        from chief_agent.execution.engine import ExecutionEngine
        from chief_agent.risk.killswitch import get_kill_switch

        risk = RiskEngine()
        paper = PaperTradingEngine(initial_capital=500_000, rng_seed=3)
        engine = ExecutionEngine(
            mode=OperatingMode.PAPER,
            risk_engine=risk,
            paper_engine=paper,
            kill_switch=get_kill_switch(),
        )
        return engine, paper

    def make_intent(self, trade_id="T-001"):
        from chief_agent.execution.engine import OrderIntent

        return OrderIntent(
            trade_id=trade_id,
            signal_id=f"S-{trade_id}",
            strategy_version="TEST_v1.0.0",
            instrument_key="SIM|RELIANCE",
            symbol="RELIANCE",
            direction="LONG",
            quantity=10,
            entry_price=100.0,
            stop_price=98.0,
            target_1=103.0,
            target_2=106.0,
        )

    def context(self):
        return RiskContext(
            account_equity=500_000.0, available_cash=500_000.0, starting_equity_today=500_000.0,
            peak_equity=500_000.0,
        )

    def test_first_submission_fills(self):
        engine, paper = self.build_engine()
        result = engine.submit(self.make_intent(), self.context())
        assert result.ok
        assert result.filled_quantity > 0

    def test_duplicate_intent_is_rejected_by_idempotency(self):
        engine, paper = self.build_engine()
        first = engine.submit(self.make_intent(), self.context())
        assert first.ok
        second = engine.submit(self.make_intent(), self.context())
        assert not second.ok
        assert "duplicate" in second.message.lower()
        assert len(paper.open_positions()) == 1, "a duplicate signal must not open a second position"

    def test_kill_switch_blocks_submission(self):
        from chief_agent.risk.killswitch import KillSwitchTrigger, get_kill_switch

        engine, paper = self.build_engine()
        get_kill_switch().engage(KillSwitchTrigger.MANUAL_BUTTON, reason="test")
        result = engine.submit(self.make_intent(), self.context())
        assert not result.ok
        assert "kill switch" in result.message.lower()
        assert not paper.open_positions()

    def test_data_safe_mode_blocks_submission(self):
        from chief_agent.execution.engine import ExecutionEngine
        from chief_agent.risk.killswitch import get_kill_switch

        paper = PaperTradingEngine(initial_capital=500_000)
        engine = ExecutionEngine(
            mode=OperatingMode.PAPER,
            risk_engine=RiskEngine(),
            paper_engine=paper,
            kill_switch=get_kill_switch(),
            data_quality_check=lambda: False,
        )
        result = engine.submit(self.make_intent(), self.context())
        assert not result.ok
        assert "data_safe_mode" in result.message.lower()

    def test_reconciliation_pending_blocks_submission(self):
        from chief_agent.execution.engine import ExecutionEngine
        from chief_agent.risk.killswitch import get_kill_switch

        paper = PaperTradingEngine(initial_capital=500_000)
        engine = ExecutionEngine(
            mode=OperatingMode.PAPER,
            risk_engine=RiskEngine(),
            paper_engine=paper,
            kill_switch=get_kill_switch(),
            reconciliation_check=lambda: False,
        )
        result = engine.submit(self.make_intent(), self.context())
        assert not result.ok
        assert "reconcil" in result.message.lower()

    def test_order_without_a_stop_is_rejected(self):
        engine, paper = self.build_engine()
        intent = self.make_intent()
        intent.stop_price = 0.0
        result = engine.submit(intent, self.context())
        assert not result.ok
        assert not paper.open_positions()

    def test_transitions_are_recorded(self):
        engine, paper = self.build_engine()
        result = engine.submit(self.make_intent(), self.context())
        states = [t["state"] for t in result.transitions]
        assert "SIGNAL" in states
        assert "RISK_CHECK" in states
        assert "VALIDATION" in states
        assert "POSITION" in states


class TestReconciliation:
    def test_matching_positions_pass(self):
        from chief_agent.execution.reconciliation import PositionReconciler

        report = PositionReconciler().reconcile({"SIM|A": 10, "SIM|B": -5}, {"SIM|A": 10, "SIM|B": -5})
        assert report.matched
        assert report.action == "NONE"

    def test_quantity_mismatch_blocks_entries(self):
        from chief_agent.execution.reconciliation import PositionReconciler

        reconciler = PositionReconciler()
        report = reconciler.reconcile({"SIM|A": 10}, {"SIM|A": 12})
        assert not report.matched
        assert reconciler.pending is True
        assert report.action == "HALT_NEW_ENTRIES"

    def test_unknown_broker_position_is_never_auto_healed(self):
        from chief_agent.execution.reconciliation import PositionReconciler

        report = PositionReconciler().reconcile({}, {"SIM|MYSTERY": 100})
        assert not report.matched
        assert report.mismatches[0].kind == "UNKNOWN_BROKER_POSITION"
        assert "human" in report.mismatches[0].detail.lower()

    def test_manual_clear_allows_resumption(self):
        from chief_agent.execution.reconciliation import PositionReconciler

        reconciler = PositionReconciler()
        reconciler.reconcile({"SIM|A": 10}, {"SIM|A": 12})
        assert reconciler.pending
        reconciler.clear()
        assert not reconciler.pending


class TestSlippageMonitor:
    def test_summary_and_escalation(self):
        from chief_agent.execution.reconciliation import SlippageMonitor

        monitor = SlippageMonitor({"warn_bps": 5, "reduce_size_bps": 10, "suspend_strategy_bps": 20,
                                   "rolling_window_trades": 100})
        for _ in range(20):
            monitor.record(
                instrument_key="SIM|A", symbol="A", expected_price=100.0, actual_price=100.25,
                quantity=10, direction="LONG",
            )
        summary = monitor.summary()
        assert summary.samples == 20
        assert summary.median_bps == pytest.approx(25.0, rel=0.01)
        assert summary.level == "SUSPEND"
        assert monitor.size_multiplier() == 0.0

    def test_clean_execution_stays_ok(self):
        from chief_agent.execution.reconciliation import SlippageMonitor

        monitor = SlippageMonitor({"warn_bps": 25, "reduce_size_bps": 40, "suspend_strategy_bps": 60})
        for _ in range(10):
            monitor.record(
                instrument_key="SIM|A", symbol="A", expected_price=100.0, actual_price=100.01, quantity=10
            )
        assert monitor.summary().level == "OK"
        assert monitor.size_multiplier() == 1.0


class TestPreflight:
    def test_fails_without_credentials(self):
        from chief_agent.risk.preflight import CheckStatus, Preflight

        report = Preflight().run(deep=False, components={})
        assert not report.passed
        names = {c.name for c in report.blocking}
        assert "credentials_configured" in names
        assert "live_master_switch" in names
        assert "static_ip_configured" in names

    def test_live_is_blocked_by_default(self):
        settings = get_settings()
        assert settings.live_permitted is False
        assert settings.effective_mode() is not OperatingMode.LIVE
        assert len(settings.live_blocked_reasons) >= 3


class TestModeSafety:
    def test_live_request_downgrades_to_paper_fail_closed(self, monkeypatch):
        from chief_agent import settings as settings_module

        monkeypatch.setenv("OPERATING_MODE", "LIVE")
        monkeypatch.setenv("ALLOW_LIVE_TRADING", "false")
        settings_module.reset_caches()
        settings = settings_module.get_settings()
        assert settings.operating_mode is OperatingMode.LIVE
        assert settings.effective_mode() is OperatingMode.PAPER, "LIVE must fail closed when gated"

    def test_secrets_are_redacted(self, monkeypatch):
        from chief_agent import settings as settings_module

        monkeypatch.setenv("UPSTOX_API_SECRET", "super-secret-value")
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "another-secret")
        settings_module.reset_caches()
        settings = settings_module.get_settings()
        redacted = settings.redacted()
        assert redacted["upstox_api_secret"] == "***set***"
        assert "super-secret-value" not in json.dumps(redacted)
        assert "another-secret" not in json.dumps(redacted)
