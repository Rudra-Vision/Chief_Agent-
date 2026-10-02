"""Research pipeline and API tests.

These cover the rules that make the self-improvement loop trustworthy:

* a challenger changes exactly ONE variable - multi-variable proposals are refused;
* a published strategy version is immutable;
* status transitions follow the state machine (no skipping straight to CHAMPION);
* the promotion gate refuses an under-sampled or degrading candidate;
* the API never exposes a broker token, and LIVE cannot be enabled by accident.
"""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from chief_agent.research.champion import (
    ChampionRegistry,
    ImmutableVersionError,
    InvalidTransitionError,
)
from chief_agent.research.promotion import PromotionGate, bootstrap_probability_better
from chief_agent.strategies.base import deep_get


class TestChampionRegistry:
    def test_baseline_champion_is_created_once(self, db_session, strategy_config):
        registry = ChampionRegistry(db_session)
        first = registry.ensure_baseline_champion()
        second = registry.ensure_baseline_champion()
        assert first.version == second.version
        assert registry.list_versions(status="CHAMPION").__len__() == 1

    def test_published_version_is_immutable(self, db_session, strategy_config):
        registry = ChampionRegistry(db_session)
        registry.publish(
            strategy_config, version="TEST_v1.0.0", family="orb_vwap_retest", status="CHAMPION"
        )
        with pytest.raises(ImmutableVersionError):
            registry.publish(strategy_config, version="TEST_v1.0.0", family="orb_vwap_retest")

    def test_challenger_changes_exactly_one_variable(self, db_session, strategy_config):
        registry = ChampionRegistry(db_session)
        registry.ensure_baseline_champion()
        info, config, old_value = registry.create_challenger(
            variable="long.min_rvol",
            new_value=1.35,
            reason="test",
        )
        assert info.variable_changed == "long.min_rvol"
        assert str(old_value) == str(1.2)
        assert deep_get(config, "long.min_rvol") == 1.35
        # everything else identical
        parent = registry.config_for(info.parent_version)
        assert parent["long"]["min_rvol"] == 1.2
        for key in ("opening_range", "stops", "targets", "short"):
            assert config[key] == parent[key]

    def test_challenger_that_changes_nothing_is_refused(self, db_session):
        registry = ChampionRegistry(db_session)
        champion = registry.ensure_baseline_champion()
        current = deep_get(registry.config_for(champion.version), "long.min_rvol")
        with pytest.raises(ValueError):
            registry.create_challenger(variable="long.min_rvol", new_value=current, reason="no-op")

    def test_challenger_with_an_unknown_variable_is_refused(self, db_session):
        registry = ChampionRegistry(db_session)
        registry.ensure_baseline_champion()
        with pytest.raises(KeyError):
            registry.create_challenger(variable="long.not_a_real_setting", new_value=1, reason="typo")

    def test_minor_bump_for_a_new_rule(self, db_session):
        registry = ChampionRegistry(db_session)
        registry.ensure_baseline_champion()
        info, _, _ = registry.create_challenger(
            variable="long.require_retest", new_value=True, reason="add a rule"
        )
        assert info.version.endswith("1.1.0")

    def test_patch_bump_for_a_retune(self, db_session):
        registry = ChampionRegistry(db_session)
        registry.ensure_baseline_champion()
        info, _, _ = registry.create_challenger(
            variable="long.min_rvol", new_value=1.5, reason="retune"
        )
        assert info.version.endswith("1.0.1")

    def test_status_machine_forbids_shortcuts(self, db_session):
        registry = ChampionRegistry(db_session)
        registry.ensure_baseline_champion()
        info, _, _ = registry.create_challenger(
            variable="long.min_rvol", new_value=1.4, reason="test"
        )
        with pytest.raises(InvalidTransitionError):
            registry.transition(info.version, "CHAMPION", reason="skipping validation")

    def test_full_transition_path_is_allowed(self, db_session):
        registry = ChampionRegistry(db_session)
        registry.ensure_baseline_champion()
        info, _, _ = registry.create_challenger(
            variable="long.min_rvol", new_value=1.4, reason="test"
        )
        for status in ("TESTING", "WALK_FORWARD", "HOLDOUT", "PAPER", "ELIGIBLE"):
            registry.transition(info.version, status)
        registry.transition(info.version, "CHAMPION")
        assert registry.champion().version == info.version

    def test_promoting_a_champion_retires_the_previous_one(self, db_session):
        registry = ChampionRegistry(db_session)
        first = registry.ensure_baseline_champion()
        info, _, _ = registry.create_challenger(variable="long.min_rvol", new_value=1.4, reason="test")
        for status in ("TESTING", "WALK_FORWARD", "HOLDOUT", "PAPER", "ELIGIBLE"):
            registry.transition(info.version, status)
        registry.transition(first.version, "RETIRED")
        registry.transition(info.version, "CHAMPION")
        assert registry.champion().version == info.version
        assert registry.info(first.version).status == "RETIRED"


class TestPromotionGate:
    def good_metrics(self, **overrides):
        metrics = {
            "trades": 260,
            "expectancy_r": 0.12,
            "profit_factor": 1.45,
            "sortino": 1.8,
            "sharpe": 1.4,
            "max_drawdown_pct": -0.08,
            "net_return_pct": 0.18,
            "win_rate": 0.45,
        }
        metrics.update(overrides)
        return metrics

    def good_walk_forward(self):
        return {"profitable_fraction": 0.75, "total_windows": 6, "passed": True}

    def good_holdout(self):
        return {"trades": 45, "profit_factor": 1.35, "expectancy_r": 0.09, "accepted": True}

    def good_robustness(self):
        return {"passed": True, "profitability_fraction": 0.85, "rejection_reasons": []}

    def good_monte_carlo(self):
        return {
            "ruin_probability": 0.005,
            "ruin_threshold_pct": 0.30,
            "max_drawdown_pct": {"p95": -0.12},
        }

    def test_a_fully_qualified_candidate_passes(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(),
            champion_metrics=self.good_metrics(expectancy_r=0.08, sortino=1.2, sharpe=1.0),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            robustness=self.good_robustness(),
            monte_carlo=self.good_monte_carlo(),
            challenger_r_multiples=[0.2] * 120 + [-0.1] * 80,
            champion_r_multiples=[0.05] * 120 + [-0.1] * 80,
            concentration={"single_stock_share": 0.15, "single_day_share": 0.10, "single_regime_share": 0.4},
            stages_completed=["BACKTEST", "WALK_FORWARD", "HOLDOUT"],
        )
        assert evaluation.passed, evaluation.blocking
        assert evaluation.significant is True

    def test_insufficient_sample_is_flagged_not_silently_passed(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            # 100 trades clears the soft floor (60) but not the hard one (200)
            challenger_metrics=self.good_metrics(trades=100),
            champion_metrics=self.good_metrics(),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            concentration={},
        )
        assert not evaluation.passed
        assert "minimum_sample_size" in evaluation.insufficient

    def test_negative_expectancy_is_rejected(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(expectancy_r=-0.05),
            champion_metrics=self.good_metrics(),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            concentration={},
        )
        assert not evaluation.passed
        assert "positive_expectancy" in evaluation.blocking

    def test_excessive_drawdown_is_rejected(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(max_drawdown_pct=-0.35),
            champion_metrics=self.good_metrics(),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            concentration={},
        )
        assert "max_drawdown_within_limit" in evaluation.blocking

    def test_worse_sortino_than_champion_is_rejected(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(sortino=0.9, sharpe=0.7),
            champion_metrics=self.good_metrics(sortino=2.0, sharpe=1.9),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            concentration={},
        )
        assert "sortino_better_than_champion" in evaluation.blocking
        assert "risk_adjusted_better" in evaluation.blocking

    def test_failed_holdout_is_rejected(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(),
            champion_metrics=self.good_metrics(sortino=1.0, sharpe=0.8),
            walk_forward=self.good_walk_forward(),
            holdout={"trades": 40, "profit_factor": 0.9, "expectancy_r": -0.02, "accepted": False},
            concentration={},
        )
        assert "holdout_accepted" in evaluation.blocking

    def test_missing_walk_forward_blocks_promotion(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(),
            champion_metrics=self.good_metrics(sortino=1.0, sharpe=0.8),
            holdout=self.good_holdout(),
            concentration={},
        )
        assert "walk_forward_completed" in evaluation.blocking

    def test_dependence_on_one_stock_is_rejected(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(),
            champion_metrics=self.good_metrics(sortino=1.0, sharpe=0.8),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            concentration={"single_stock_share": 0.8, "single_day_share": 0.1, "single_regime_share": 0.2},
        )
        assert "not_dependent_on_one_stock" in evaluation.blocking

    def test_excessive_monte_carlo_ruin_is_rejected(self):
        gate = PromotionGate()
        evaluation = gate.evaluate(
            challenger_version="ORB_v1.1.0",
            champion_version="ORB_v1.0.0",
            challenger_metrics=self.good_metrics(),
            champion_metrics=self.good_metrics(sortino=1.0, sharpe=0.8),
            walk_forward=self.good_walk_forward(),
            holdout=self.good_holdout(),
            monte_carlo={"ruin_probability": 0.2, "ruin_threshold_pct": 0.3, "max_drawdown_pct": {"p95": -0.5}},
            concentration={},
        )
        assert "monte_carlo_ruin_probability" in evaluation.blocking
        assert "monte_carlo_drawdown" in evaluation.blocking

    def test_bootstrap_significance(self):
        strong = bootstrap_probability_better(
            champion_r_multiples=[-0.1] * 100, challenger_r_multiples=[0.3] * 100, n_bootstrap=300
        )
        assert strong["probability_better"] > 0.95
        weak = bootstrap_probability_better(
            champion_r_multiples=[0.0] * 100, challenger_r_multiples=[0.0] * 100, n_bootstrap=200
        )
        assert weak["delta_mean_r"] == pytest.approx(0.0)

    def test_insufficient_bootstrap_sample_is_reported(self):
        result = bootstrap_probability_better([0.1, 0.2], [0.3])
        assert result["insufficient"] is True


class TestHypothesisGenerator:
    def _seed_journal(self, db_session, trades=120):
        """Write a synthetic journal directly so the generator has data to mine."""
        import random

        from chief_agent.data.schema import TradeJournal
        from chief_agent.timeutil import now_ist

        rng = random.Random(11)
        for i in range(trades):
            low_volume = i % 2 == 0
            # Low-volume trades are deliberately given negative expectancy so the
            # generator should find and report that segment.
            r_multiple = (rng.uniform(-1.2, 0.4) if low_volume else rng.uniform(-0.8, 1.6))
            entry = now_ist() - dt.timedelta(days=i % 60, minutes=i % 300)
            db_session.add(
                TradeJournal(
                    trade_id=f"TEST-{i:04d}",
                    strategy_family="orb_vwap_retest",
                    strategy_version="ORB_v1.0.0",
                    instrument_key="SIM|RELIANCE",
                    symbol="RELIANCE",
                    direction="LONG" if i % 3 else "SHORT",
                    quantity=10,
                    entry_price=100.0,
                    exit_price=100.0 + r_multiple,
                    entry_ts=entry,
                    exit_ts=entry + dt.timedelta(minutes=45),
                    holding_minutes=45,
                    net_pnl=r_multiple * 100.0,
                    initial_risk=100.0,
                    r_multiple=r_multiple,
                    exit_reason="STOP_LOSS" if r_multiple < 0 else "TARGET_1",
                    regime_at_entry="TREND_UP" if i % 3 else "RANGE",
                    sector="Energy",
                    features={"rvol": 1.05 if low_volume else 1.8},
                    mode="PAPER",
                )
            )
        db_session.flush()

    def test_generates_evidence_backed_proposals(self, db_session, strategy_config):
        from chief_agent.research.hypotheses import HypothesisGenerator

        self._seed_journal(db_session)
        generator = HypothesisGenerator(db_session)
        proposals = generator.generate("ORB_v1.0.0", strategy_config, max_proposals=5)
        assert proposals, "the generator should find the low-volume segment"
        for proposal in proposals:
            assert proposal.evidence.sample_size > 0
            assert proposal.evidence.confidence > 0
            assert proposal.old_value is not None
            assert proposal.new_value != proposal.old_value
            assert proposal.variable

    def test_refuses_to_propose_without_enough_data(self, db_session, strategy_config):
        from chief_agent.research.hypotheses import HypothesisGenerator

        self._seed_journal(db_session, trades=5)
        generator = HypothesisGenerator(db_session)
        assert generator.generate("ORB_v1.0.0", strategy_config) == []

    def test_persisted_hypothesis_has_a_stable_id(self, db_session, strategy_config):
        from chief_agent.research.hypotheses import HypothesisGenerator

        self._seed_journal(db_session)
        generator = HypothesisGenerator(db_session)
        proposals = generator.generate("ORB_v1.0.0", strategy_config, max_proposals=1)
        if proposals:
            hypothesis_id = generator.persist(proposals[0], "ORB_v1.0.0")
            assert hypothesis_id.startswith("H-")


class TestResearchMemory:
    def test_records_and_recalls(self, db_session):
        from chief_agent.research.memory import ResearchMemory

        memory = ResearchMemory(db_session)
        learning_id = memory.record(
            finding="ORB breakouts on gap-up mornings have lower expectancy",
            evidence={"trades": 120, "expectancy_r": -0.05},
            sample_size=120,
            confidence=0.7,
            strategy_version="ORB_v1.0.0",
            decision="TEST_GAP_FILTER",
            result="REJECTED",
            tags=["gap_up", "orb"],
        )
        assert learning_id.startswith("LEARNING-")
        found = memory.recall("gap")
        assert found and found[0].finding.startswith("ORB breakouts")

    def test_answer_reports_nothing_when_it_has_no_evidence(self, db_session):
        from chief_agent.research.memory import ResearchMemory

        answer = ResearchMemory(db_session).answer("what about lithium futures on Mars?")
        assert answer["found"] is False


class TestApi:
    def test_health_endpoint(self, fastapi_client):
        response = fastapi_client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in ("HEALTHY", "DEGRADED", "STOPPED", "UNKNOWN")
        assert any(check["name"] == "database" for check in payload["checks"])

    def test_status_endpoint_reports_the_mode_clearly(self, fastapi_client):
        payload = fastapi_client.get("/api/system/status").json()
        assert payload["mode"]["mode"] in ("SANDBOX", "PAPER", "LIVE")
        assert payload["mode"]["banner"]
        assert payload["mode"]["is_live"] is False
        assert payload["data"]["is_simulated"] in (True, False)

    def test_no_secret_is_ever_returned(self, fastapi_client, monkeypatch):
        from chief_agent import settings as settings_module

        monkeypatch.setenv("UPSTOX_API_SECRET", "leaky-secret-value")
        monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "leaky-token-value")
        settings_module.reset_caches()

        for path in ("/api/system/status", "/api/system/config", "/api/broker/status"):
            body = fastapi_client.get(path).text
            assert "leaky-secret-value" not in body
            assert "leaky-token-value" not in body

    def test_kill_switch_round_trip(self, fastapi_client):
        assert fastapi_client.get("/api/broker/kill-switch").json()["engaged"] is False
        engaged = fastapi_client.post(
            "/api/broker/kill-switch/engage", json={"reason": "test"}
        )
        assert engaged.status_code == 200
        assert engaged.json()["state"]["engaged"] is True
        released = fastapi_client.post("/api/broker/kill-switch/release")
        assert released.status_code == 200
        assert fastapi_client.get("/api/broker/kill-switch").json()["engaged"] is False

    def test_live_is_blocked_by_default(self, fastapi_client):
        payload = fastapi_client.get("/api/system/status").json()
        assert payload["mode"]["live_permitted"] is False
        assert payload["mode"]["live_blocked_reasons"]

    def test_preflight_reports_the_blocking_checks(self, fastapi_client):
        payload = fastapi_client.get("/api/broker/preflight?deep=false").json()
        assert payload["passed"] is False
        assert "credentials_configured" in payload["blocking"]

    def test_risk_limits_cannot_be_relaxed_through_the_api(self, fastapi_client):
        response = fastapi_client.post(
            "/api/system/config",
            json={"domain": "risk", "patch": {"per_trade": {"risk_pct": 0.5}}, "reason": "test"},
        )
        assert response.status_code == 403
        assert "risk limits cannot be relaxed" in response.json()["detail"].lower()

    def test_backtest_defaults_are_available(self, fastapi_client):
        payload = fastapi_client.get("/api/backtest/defaults").json()
        assert "universe" in payload
        assert payload["risk_per_trade_pct"] == pytest.approx(0.0025)

    def test_opportunities_endpoint_answers_without_data(self, fastapi_client):
        payload = fastapi_client.get("/api/trading/opportunities?refresh=true").json()
        assert "longs" in payload and "shorts" in payload

    def test_unknown_api_route_is_a_404_not_the_spa(self, fastapi_client):
        response = fastapi_client.get("/api/definitely-not-a-route")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")

    def test_dashboard_root_serves_html(self, fastapi_client):
        response = fastapi_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_security_headers_are_present(self, fastapi_client):
        response = fastapi_client.get("/api/system/status")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"

    def test_manual_order_requires_a_stop(self, fastapi_client):
        response = fastapi_client.post(
            "/api/trading/paper/order",
            json={
                "instrument_key": "SIM|RELIANCE",
                "direction": "LONG",
                "quantity": 1,
                "entry_price": 100.0,
                "stop_price": 0.0,
            },
        )
        assert response.status_code == 400


class TestDataQualityEngine:
    def test_clean_data_passes(self, sample_candles):
        from chief_agent.data.quality import DataQualityEngine

        report = DataQualityEngine().evaluate(candles_by_instrument=sample_candles)
        assert report.safe_to_trade

    def test_duplicate_candles_are_detected(self, sample_candles):
        from chief_agent.data.quality import DataQualityEngine

        instrument = next(iter(sample_candles))
        doubled = list(sample_candles[instrument]) + list(sample_candles[instrument][:500])
        report = DataQualityEngine().evaluate(candles_by_instrument={instrument: doubled})
        types = {i.incident_type.value for i in report.incidents}
        assert "DUPLICATE_CANDLES" in types

    def test_impossible_ohlc_blocks_trading(self, sample_candles):
        from chief_agent.broker.upstox_market_data import Candle
        from chief_agent.data.quality import DataQualityEngine
        from chief_agent.timeutil import now_ist

        broken = list(sample_candles[next(iter(sample_candles))])
        broken.append(Candle(ts=now_ist(), open=100, high=90, low=110, close=95, volume=100))
        report = DataQualityEngine().evaluate(candles_by_instrument={"SIM|BROKEN": broken})
        assert not report.safe_to_trade
        assert any(i.incident_type.value == "IMPOSSIBLE_OHLC" for i in report.incidents)

    def test_stale_quote_blocks_trading(self):
        from chief_agent.data.quality import DataQualityEngine
        from chief_agent.timeutil import now_ist

        old = now_ist() - dt.timedelta(seconds=120)
        report = DataQualityEngine().evaluate(
            quotes={"SIM|A": {"ltp": 100.0, "timestamp": old.isoformat()}}
        )
        assert not report.safe_to_trade
        assert any(i.incident_type.value == "STALE_QUOTE" for i in report.incidents)

    def test_negative_spread_blocks_trading(self):
        from chief_agent.data.quality import DataQualityEngine

        report = DataQualityEngine().evaluate(quotes={"SIM|A": {"ltp": 100.0, "bid": 101.0, "ask": 99.0}})
        assert not report.safe_to_trade

    def test_websocket_disconnect_beyond_grace_blocks(self):
        from chief_agent.data.quality import DataQualityEngine

        engine = DataQualityEngine()
        assert engine.evaluate(websocket_connected=False, websocket_age_seconds=120).safe_to_trade is False
        assert engine.evaluate(websocket_connected=False, websocket_age_seconds=5).safe_to_trade is True

    def test_safe_mode_latches_and_clears(self, sample_candles):
        from chief_agent.data.quality import DataQualityEngine

        engine = DataQualityEngine()
        engine.evaluate(quotes={"SIM|A": {"ltp": 100.0, "bid": 101.0, "ask": 99.0}})
        assert engine.data_safe_mode is True
        engine.evaluate(candles_by_instrument=sample_candles)
        assert engine.data_safe_mode is False
