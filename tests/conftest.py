"""Shared pytest fixtures.

Every test runs against an isolated, temporary SQLite database and a simulated
market-data provider, so the suite needs no network, no broker token and no
external services.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Point every piece of state at a throwaway location for each test."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("OPERATING_MODE", "PAPER")
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "false")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "")
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "")
    monkeypatch.setenv("UPSTOX_SANDBOX_TOKEN", "")
    monkeypatch.setenv("UPSTOX_API_KEY", "")
    monkeypatch.setenv("UPSTOX_API_SECRET", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    from chief_agent import settings as settings_module

    settings_module.reset_caches()

    from chief_agent.data import db as db_module

    db_module.reset_engine()

    from chief_agent.risk import killswitch as killswitch_module

    killswitch_module.reset_kill_switch()

    from chief_agent.monitoring import notifications as notifications_module

    notifications_module.reset_notifications()

    from chief_agent.broker.rate_limiter import reset_rate_limiters

    reset_rate_limiters()

    from chief_agent.api import security as security_module

    security_module.reset_auth()

    yield

    settings_module.reset_caches()
    db_module.reset_engine()
    killswitch_module.reset_kill_switch()
    notifications_module.reset_notifications()
    security_module.reset_auth()


@pytest.fixture
def db_session():
    """A session against the isolated database, with the schema created."""
    from chief_agent.data.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        yield session


@pytest.fixture
def provider():
    """A market-data provider forced into simulated mode."""
    from chief_agent.data.provider import MarketDataProvider

    return MarketDataProvider(force_simulated=True)


@pytest.fixture
def config_store():
    from chief_agent.settings import get_config_store

    store = get_config_store()
    store.reload()
    return store


@pytest.fixture
def strategy_config(config_store):
    return config_store.load("strategy")


@pytest.fixture
def strategy(strategy_config):
    from chief_agent.strategies.orb_vwap import ORBVWAPStrategy

    return ORBVWAPStrategy(strategy_config, version="TEST_v1.0.0")


@pytest.fixture
def sample_symbols():
    return ["RELIANCE", "INFY", "HDFCBANK", "TATAMOTORS", "SBIN"]


@pytest.fixture
def sample_candles(sample_symbols):
    """Two months of simulated 1-minute candles for a handful of symbols."""
    from chief_agent.data import synthetic

    start = dt.date(2025, 1, 1)
    end = dt.date(2025, 2, 28)
    return {
        f"SIM|{symbol}": synthetic.generate_history(symbol, start, end) for symbol in sample_symbols
    }


@pytest.fixture
def index_candles():
    from chief_agent.data import synthetic

    return synthetic.generate_index_history("NIFTY 50", dt.date(2025, 1, 1), dt.date(2025, 2, 28))


@pytest.fixture
def fastapi_client():
    """A TestClient with the app's lifespan run (so init_db happens)."""
    from fastapi.testclient import TestClient

    from chief_agent.api.app import create_app
    from chief_agent.api.state import AppState, set_app_state
    from chief_agent.data.db import init_db

    init_db()
    state = AppState(start_services=False)
    set_app_state(state)
    app = create_app(state, start_services=False)
    with TestClient(app) as client:
        yield client
    from chief_agent.api.state import reset_app_state

    reset_app_state()
