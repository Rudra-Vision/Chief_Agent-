"""SQLAlchemy 2.0 schema for Chief Agent.

Design notes
------------
* Source market data (candles/quotes) is stored separately from derived
  indicators/features, and is **never overwritten** - upserts only fill gaps.
* ``instrument_key`` is the canonical identifier (``exchange_token`` may be
  reused by the exchange after expiry, so it is stored only as metadata).
* Every trading artefact links back to a ``strategy_version`` so that any P&L
  number can be attributed to an immutable, auditable strategy definition.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..timeutil import now_ist


class Base(DeclarativeBase):
    """Base class with a shared ``type_annotation_map`` for JSON columns."""

    type_annotation_map = {dict: JSON, list: JSON}


def _now() -> dt.datetime:
    return now_ist()


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
class Instrument(Base):
    """BOD instrument master row (refreshed daily from the official JSON file)."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    exchange_token: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    segment: Mapped[str] = mapped_column(String(24), index=True, default="NSE_EQ")
    exchange: Mapped[str] = mapped_column(String(16), default="NSE")
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQ", index=True)
    trading_symbol: Mapped[str] = mapped_column(String(64), index=True)
    short_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    isin: Mapped[Optional[str]] = mapped_column(String(24), index=True, nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    tier: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    freeze_quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tick_size: Mapped[float] = mapped_column(Float, default=0.05)
    security_type: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    cas_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    expiry: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    underlying_symbol: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    strike_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    in_universe: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_suspended: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(24), default="upstox_bod")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (Index("ix_instruments_universe_symbol", "in_universe", "trading_symbol"),)


class Candle(Base):
    """Raw OHLCV candle. Source data is append-only; existing rows are never changed."""

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    timeframe: Mapped[str] = mapped_column(String(8), index=True)          # 1m, 5m, 1d ...
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    open_interest: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(24), default="upstox_v3")
    is_derived: Mapped[bool] = mapped_column(Boolean, default=False)       # aggregated up from 1m
    quality_flag: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("instrument_key", "timeframe", "ts", name="uq_candle_key_tf_ts"),
        Index("ix_candles_lookup", "instrument_key", "timeframe", "ts"),
    )


class Quote(Base):
    """Snapshot of a live quote (top-of-book). Retained for slippage/quality analysis."""

    __tablename__ = "quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    ltp: Mapped[float] = mapped_column(Float)
    close_prev: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vwap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bid_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ask_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    open_interest: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_buy_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_sell_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(24), default="upstox_ws")

    __table_args__ = (Index("ix_quotes_lookup", "instrument_key", "ts"),)


class MarketRegimeSnapshot(Base):
    """Regime classification captured per session (and intraday when it changes)."""

    __tablename__ = "market_regime_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    trading_date: Mapped[dt.date] = mapped_column(Date, index=True)
    regime: Mapped[str] = mapped_column(String(24), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    nifty_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vix_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    breadth_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gap_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class DataQualityIncident(Base):
    """Anything that made market data untrustworthy. Data quality gates trading."""

    __tablename__ = "data_quality_incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    instrument_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    incident_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="WARNING")   # INFO/WARNING/CRITICAL
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------- #
# Strategy versioning
# --------------------------------------------------------------------------- #
class StrategyVersion(Base):
    """An IMMUTABLE published strategy definition (champion or challenger)."""

    __tablename__ = "strategy_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(48), unique=True, index=True)     # ORB_v1.0.0
    family: Mapped[str] = mapped_column(String(48), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True, default="CANDIDATE")
    # status: CANDIDATE | TESTING | WALK_FORWARD | HOLDOUT | PAPER | ELIGIBLE |
    #         CHAMPION | REJECTED | SUSPENDED | RETIRED
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    config_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    parent_version: Mapped[Optional[str]] = mapped_column(String(48), nullable=True, index=True)
    reason_for_change: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    variable_changed: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    old_value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    git_commit: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    published_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    backtest_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    walk_forward_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    holdout_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    paper_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_strategy_version_status", "status", "family"),)


# --------------------------------------------------------------------------- #
# Signals, orders, fills, positions, trades
# --------------------------------------------------------------------------- #
class Signal(Base):
    """A candidate trade produced by the scanner/strategy. Deterministic output."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    trading_date: Mapped[dt.date] = mapped_column(Date, index=True)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(8))                     # LONG / SHORT
    strategy_family: Mapped[str] = mapped_column(String(48), index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    regime: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    score_components: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_price: Mapped[float] = mapped_column(Float, default=0.0)
    target_1: Mapped[float] = mapped_column(Float, default=0.0)
    target_2: Mapped[float] = mapped_column(Float, default=0.0)
    risk_reward: Mapped[float] = mapped_column(Float, default=0.0)
    quantity_proposed: Mapped[int] = mapped_column(Integer, default=0)
    reasons: Mapped[List[str]] = mapped_column(JSON, default=list)
    risks: Mapped[List[str]] = mapped_column(JSON, default=list)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(24), index=True, default="NEW")
    # NEW | REJECTED_RISK | REJECTED_FILTER | APPROVED | EXECUTED | EXPIRED | CANCELLED
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Order(Base):
    """Internal order record, one row per broker order leg."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)      # our deterministic id
    trade_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    signal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True, default="")
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    tag: Mapped[Optional[str]] = mapped_column(String(48), nullable=True, index=True)
    leg: Mapped[str] = mapped_column(String(16), default="ENTRY")     # ENTRY | STOP | TARGET | EXIT
    mode: Mapped[str] = mapped_column(String(12), index=True)          # SANDBOX | PAPER | LIVE
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    transaction_type: Mapped[str] = mapped_column(String(8))           # BUY | SELL
    order_type: Mapped[str] = mapped_column(String(12))                # MARKET | LIMIT | SL | SL-M
    product: Mapped[str] = mapped_column(String(8), default="I")
    validity: Mapped[str] = mapped_column(String(8), default="DAY")
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    pending_quantity: Mapped[int] = mapped_column(Integer, default=0)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    trigger_price: Mapped[float] = mapped_column(Float, default=0.0)
    average_fill_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), index=True, default="CREATED")
    status_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_amo: Mapped[bool] = mapped_column(Boolean, default=False)
    slice_order: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    reconcile_attempts: Mapped[int] = mapped_column(Integer, default=0)
    outcome_uncertain: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    submitted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_request: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    raw_response: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


class Fill(Base):
    """An execution (partial or complete) reported by the broker or simulator."""

    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fill_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trade_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    transaction_type: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    slippage_bps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    fees_breakdown: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    mode: Mapped[str] = mapped_column(String(12), default="PAPER")
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Position(Base):
    """An open (or closed) position tracked by the portfolio engine."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    trade_id: Mapped[str] = mapped_column(String(64), index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True, default="")
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    direction: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    initial_stop: Mapped[float] = mapped_column(Float, default=0.0)
    current_stop: Mapped[float] = mapped_column(Float, default=0.0)
    target_1: Mapped[float] = mapped_column(Float, default=0.0)
    target_2: Mapped[float] = mapped_column(Float, default=0.0)
    initial_risk_per_share: Mapped[float] = mapped_column(Float, default=0.0)
    ltp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    mae: Mapped[float] = mapped_column(Float, default=0.0)               # max adverse excursion (price)
    mfe: Mapped[float] = mapped_column(Float, default=0.0)               # max favourable excursion (price)
    r_multiple: Mapped[float] = mapped_column(Float, default=0.0)
    product: Mapped[str] = mapped_column(String(8), default="I")
    mode: Mapped[str] = mapped_column(String(12), default="PAPER")
    status: Mapped[str] = mapped_column(String(16), index=True, default="OPEN")  # OPEN | CLOSED
    broker_side_protection: Mapped[bool] = mapped_column(Boolean, default=False)
    stop_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    target_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    closed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_synced_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class TradeJournal(Base):
    """The research-grade record of a completed round-trip trade."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    signal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    strategy_family: Mapped[str] = mapped_column(String(48), index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    direction: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    entry_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    exit_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    holding_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_cost: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    initial_risk: Mapped[float] = mapped_column(Float, default=0.0)
    r_multiple: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    mae_price: Mapped[float] = mapped_column(Float, default=0.0)
    mfe_price: Mapped[float] = mapped_column(Float, default=0.0)
    mae_r: Mapped[float] = mapped_column(Float, default=0.0)
    mfe_r: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(32), index=True)   # STOP|TARGET|TIME|EOD|REGIME|MANUAL|SQUARE_OFF
    entry_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    regime_at_entry: Mapped[Optional[str]] = mapped_column(String(24), nullable=True, index=True)
    regime_at_exit: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    vix_at_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sector_rank_at_entry: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Post-trade classification (a losing trade is NOT automatically a mistake)
    process_quality: Mapped[Optional[str]] = mapped_column(String(24), nullable=True, index=True)
    #   GOOD_TRADE_GOOD_OUTCOME | GOOD_TRADE_BAD_OUTCOME |
    #   BAD_TRADE_GOOD_OUTCOME  | BAD_TRADE_BAD_OUTCOME
    process_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String(12), index=True, default="PAPER")
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    execution_quality: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    news_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    lesson: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TradeFeature(Base):
    """Wide feature vector captured at entry, one row per trade.

    Stored separately from ``trades`` so the feature schema can evolve without
    rewriting the journal.
    """

    __tablename__ = "trade_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), index=True)
    signal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    feature_set_version: Mapped[str] = mapped_column(String(16), default="v1")
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- #
# Research: hypotheses, experiments, learnings, backtests
# --------------------------------------------------------------------------- #
class Hypothesis(Base):
    """A falsifiable, testable statement produced by the research engine."""

    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)   # H-0048
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    statement: Mapped[str] = mapped_column(Text)
    variable: Mapped[str] = mapped_column(String(96), index=True)
    old_value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_mechanism: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    potential_downside: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    affected_trade_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(24), default="deterministic")  # deterministic | llm | manual
    status: Mapped[str] = mapped_column(String(24), index=True, default="PROPOSED")
    # PROPOSED | TESTING | ACCEPTED | REJECTED | INCONCLUSIVE | PROMOTED


class Experiment(Base):
    """One controlled change of ONE variable, fully measured."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)   # EXP-000184
    hypothesis_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    parent_strategy: Mapped[str] = mapped_column(String(48), index=True)
    candidate_strategy: Mapped[str] = mapped_column(String(48), index=True)
    hypothesis: Mapped[Text] = mapped_column(Text)
    variable_changed: Mapped[str] = mapped_column(String(96))
    old_value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)

    training_period_start: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    training_period_end: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    validation_period_start: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    validation_period_end: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    holdout_period_start: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    holdout_period_end: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)

    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    champion_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    challenger_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    delta_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expectancy_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sortino: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    regime_stability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    walk_forward_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    holdout_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    robustness_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    monte_carlo_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    significance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    objective_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(String(32), index=True, default="CREATED")
    # CREATED | RUNNING | FAILED_GATE | HOLDOUT | PAPER_VALIDATION | ELIGIBLE_FOR_PROMOTION |
    # PROMOTED | REJECTED | ABANDONED | INSUFFICIENT_SAMPLE
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    promotion_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # multiple-testing counter


class BacktestRun(Base):
    """A single backtest execution and its full metric set."""

    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    finished_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)
    universe: Mapped[List[str]] = mapped_column(JSON, default=list)
    initial_capital: Mapped[float] = mapped_column(Float, default=500000.0)
    risk_per_trade_pct: Mapped[float] = mapped_column(Float, default=0.0025)
    slippage_bps: Mapped[float] = mapped_column(Float, default=4.0)
    fee_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    config_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    equity_curve: Mapped[Optional[List[Any]]] = mapped_column(JSON, nullable=True)
    drawdown_curve: Mapped[Optional[List[Any]]] = mapped_column(JSON, nullable=True)
    monthly_returns: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    regime_performance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    sector_performance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    time_of_day_performance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    side_performance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="COMPLETED")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class WalkForwardRun(Base):
    """A walk-forward study over one candidate."""

    __tablename__ = "walk_forward_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    mode: Mapped[str] = mapped_column(String(16), default="anchored")     # anchored | rolling
    n_windows: Mapped[int] = mapped_column(Integer, default=6)
    embargo_days: Mapped[int] = mapped_column(Integer, default=5)
    windows: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    summary: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="COMPLETED")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class RobustnessRun(Base):
    """Parameter perturbation / cost stress / missed-trade stress results."""

    __tablename__ = "robustness_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    scenarios: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    summary: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class MonteCarloRun(Base):
    __tablename__ = "monte_carlo_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    n_simulations: Mapped[int] = mapped_column(Integer, default=2000)
    method: Mapped[str] = mapped_column(String(32), default="trade_sequence_resample")
    summary: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Learning(Base):
    """Persistent structured research memory (never rely on chat history)."""

    __tablename__ = "learnings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learning_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)   # LEARNING-00129
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    finding: Mapped[str] = mapped_column(Text)
    evidence: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True, default="")
    hypothesis_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    market_conditions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    decision: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    result: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)


class MultipleTestingLedger(Base):
    """Every strategy/parameter combination ever evaluated, for trial counting."""

    __tablename__ = "multiple_testing_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    trial_kind: Mapped[str] = mapped_column(String(32), index=True)   # backtest | experiment | sweep
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    variable: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    value: Mapped[Optional[str]] = mapped_column(String(160), nullable=True)
    metric_name: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    metric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# --------------------------------------------------------------------------- #
# Performance, news, events, config
# --------------------------------------------------------------------------- #
class DailyPerformance(Base):
    __tablename__ = "daily_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trading_date: Mapped[dt.date] = mapped_column(Date, unique=True, index=True)
    mode: Mapped[str] = mapped_column(String(12), default="PAPER")
    strategy_version: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    starting_equity: Mapped[float] = mapped_column(Float, default=0.0)
    ending_equity: Mapped[float] = mapped_column(Float, default=0.0)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    expectancy_r: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    regime_summary: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    data_quality_incidents: Mapped[int] = mapped_column(Integer, default=0)
    risk_events: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class NewsArticle(Base):
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    instrument_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    heading: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    article_link: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    sentiment: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)      # positive|negative|neutral|uncertain
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    classifier: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)     # lexicon | llm
    is_adverse_for_open_position: Mapped[bool] = mapped_column(Boolean, default=False)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    level: Mapped[str] = mapped_column(String(16), index=True, default="INFO")
    component: Mapped[str] = mapped_column(String(48), index=True)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="WARNING")
    action_taken: Mapped[str] = mapped_column(String(48), default="NONE")
    signal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trade_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    instrument_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class KillSwitchEvent(Base):
    __tablename__ = "kill_switch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    action: Mapped[str] = mapped_column(String(16))            # ENGAGED | RELEASED
    source: Mapped[str] = mapped_column(String(24))            # manual | automatic | broker
    reason: Mapped[str] = mapped_column(Text)
    broker_kill_switch_used: Mapped[bool] = mapped_column(Boolean, default=False)
    segments: Mapped[List[str]] = mapped_column(JSON, default=list)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class ReconciliationEvent(Base):
    __tablename__ = "reconciliation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    matched: Mapped[bool] = mapped_column(Boolean, default=True)
    internal_positions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    broker_positions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    mismatches: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    action_taken: Mapped[str] = mapped_column(String(48), default="NONE")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    mode: Mapped[str] = mapped_column(String(12), default="PAPER")
    equity: Mapped[float] = mapped_column(Float, default=0.0)
    cash: Mapped[float] = mapped_column(Float, default=0.0)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)
    open_risk_pct: Mapped[float] = mapped_column(Float, default=0.0)
    gross_exposure_pct: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_today: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class OpportunitySnapshot(Base):
    """The ranked opportunity table as it looked at a point in time."""

    __tablename__ = "opportunity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    trading_date: Mapped[dt.date] = mapped_column(Date, index=True)
    direction: Mapped[str] = mapped_column(String(8), index=True)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_price: Mapped[float] = mapped_column(Float, default=0.0)
    target_1: Mapped[float] = mapped_column(Float, default=0.0)
    risk_reward: Mapped[float] = mapped_column(Float, default=0.0)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    regime: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    components: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    reasons: Mapped[List[str]] = mapped_column(JSON, default=list)


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="INFO", index=True)
    title: Mapped[str] = mapped_column(String(160))
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    read: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    delivered_channels: Mapped[List[str]] = mapped_column(JSON, default=list)


class ConfigRecord(Base):
    """Audit log of configuration changes (who/what/when)."""

    __tablename__ = "configuration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    domain: Mapped[str] = mapped_column(String(32), index=True)
    key: Mapped[str] = mapped_column(String(96), index=True)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actor: Mapped[str] = mapped_column(String(48), default="system")
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PaperAccount(Base):
    """Persistent simulated account used by the paper-trading engine."""

    __tablename__ = "paper_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(32), unique=True, default="paper-default")
    strategy_version: Mapped[str] = mapped_column(String(48), default="")
    initial_capital: Mapped[float] = mapped_column(Float, default=500000.0)
    cash: Mapped[float] = mapped_column(Float, default=500000.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, default=500000.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class BacktestTrade(Base):
    """Individual simulated trade from a backtest (kept separate from live journal)."""

    __tablename__ = "backtest_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[str] = mapped_column(String(48), index=True)
    strategy_version: Mapped[str] = mapped_column(String(48), index=True)
    trade_index: Mapped[int] = mapped_column(Integer, default=0)
    instrument_key: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    direction: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    entry_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    exit_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    stop_price: Mapped[float] = mapped_column(Float, default=0.0)
    target_price: Mapped[float] = mapped_column(Float, default=0.0)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_cost: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    initial_risk: Mapped[float] = mapped_column(Float, default=0.0)
    r_multiple: Mapped[float] = mapped_column(Float, default=0.0)
    mae_r: Mapped[float] = mapped_column(Float, default=0.0)
    mfe_r: Mapped[float] = mapped_column(Float, default=0.0)
    holding_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(24))
    regime: Mapped[Optional[str]] = mapped_column(String(24), nullable=True, index=True)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("ix_backtest_trades_lookup", "backtest_id", "trade_index"),
    )


# Convenience export used by the metrics/reporting layer
TABLES = [
    Instrument,
    Candle,
    Quote,
    MarketRegimeSnapshot,
    DataQualityIncident,
    StrategyVersion,
    Signal,
    Order,
    Fill,
    Position,
    TradeJournal,
    TradeFeature,
    Hypothesis,
    Experiment,
    BacktestRun,
    BacktestTrade,
    WalkForwardRun,
    RobustnessRun,
    MonteCarloRun,
    Learning,
    MultipleTestingLedger,
    DailyPerformance,
    NewsArticle,
    SystemEvent,
    RiskEvent,
    KillSwitchEvent,
    ReconciliationEvent,
    PortfolioSnapshot,
    OpportunitySnapshot,
    Notification,
    ConfigRecord,
    PaperAccount,
]

__all__ = ["Base", "TABLES"] + [cls.__name__ for cls in TABLES]
