# Chief Agent — Architecture

## Design principles

1. **Fail closed.** Every uncertain state resolves to "do nothing".
2. **Deterministic trading, probabilistic research.** Signals are rules. Only the research
   layer may propose changes, and only a deterministic engine may accept them.
3. **One source of truth per concern.** `PortfolioEngine` owns equity and exposure; the
   risk engine owns limits; `CandleStore` owns market data. Nothing recomputes these.
4. **Immutable artefacts.** Published strategy versions, generated market data and executed
   orders are append-only.
5. **Interpretable before intelligent.** The baseline is a small number of named, auditable
   gates. ML is a later, evidence-gated addition — never a starting point.

## Service map

```
                       ┌──────────────────────────────┐
   Upstox REST  ◄──────┤ broker/                      │
   Upstox WS    ◄──────┤  upstox_client (rate-limited,│
                       │   circuit-broken, classified)│
                       │  auth · market_data · orders │
                       │  portfolio · news · options  │
                       │  risk (kill switch, static   │
                       │  IP) · websocket             │
                       └───────────────┬──────────────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                               ▼
┌───────────────┐            ┌───────────────────┐          ┌──────────────────┐
│ data/         │            │ indicators/       │          │ regime/ sectors/ │
│ candle_store  │───────────►│ InstrumentSeries  │─────────►│ deterministic    │
│ provider      │  candles   │ (causal features) │          │ context          │
│ downloader    │            └─────────┬─────────┘          └────────┬─────────┘
│ quality       │                      │                             │
└───────┬───────┘                      ▼                             │
        │                    ┌───────────────────┐                   │
        │                    │ strategies/       │◄──────────────────┘
        │                    │ ORBVWAPStrategy   │
        │                    └─────────┬─────────┘
        │                              │ TradeCandidate
        │                    ┌─────────▼─────────┐
        │                    │ scanner/          │  ranked opportunities + gate census
        │                    └─────────┬─────────┘
        │                              ▼
        │                    ┌───────────────────┐      ┌──────────────┐
        └───────────────────►│ risk/             │─────►│ execution/   │
                             │ RiskEngine        │      │ lifecycle    │
                             │  (ABSOLUTE        │      │ paper engine │
                             │   AUTHORITY)      │      │ reconciler   │
                             └─────────┬─────────┘      │ slippage     │
                                       │                └──────┬───────┘
                                       ▼                       ▼
                             ┌───────────────────┐    ┌──────────────────┐
                             │ portfolio/        │    │ journal          │
                             │ equity, exposure, │    │ schema.Trade     │
                             │ open risk         │    └────────┬─────────┘
                             └───────────────────┘             │
                                                               ▼
                       ┌────────────────────────────────────────────────────┐
                       │ research/                                          │
                       │  hypotheses → experiments → promotion → champion   │
                       │  backed by backtest / walk-forward / robustness /  │
                       │  Monte Carlo / holdout / paper validation          │
                       └────────────────────────────────────────────────────┘
```

## Layer contracts

| Layer | May | May NOT |
|---|---|---|
| `broker/*` | Make HTTP calls, parse payloads, classify errors | Make trading decisions, hold strategy state |
| `indicators/*` | Compute causal values from past bars | Look forward, mutate raw candles |
| `strategies/*` | Propose a `TradeCandidate` | Size, place, approve, or consult an LLM |
| `scanner/*` | Rank, explain, record rejections | Bypass the risk engine |
| `risk/*` | Approve, reduce, reject, halt | Be overridden by anything |
| `execution/*` | Submit, acknowledge, fill, reconcile | Invent an approval, resend an uncertain order |
| `research/*` | Propose challengers, run experiments | Promote by itself, touch the holdout early |
| `ai/*` | Explain, summarise, propose | Trade, size, change config, declare success |

## Anti-look-ahead guarantees

These are structural, not conventions:

1. `InstrumentSeries` pre-computes every array; element *i* depends only on data at `<= i`.
2. `opening_range_for(i)` returns `None` until `formed_at <= candle_ts(i)`.
3. The backtester emits signals from the close of bar *i* and fills them at the **open of
   bar *i+1*** through an explicit pending-order queue.
4. Inside a bar, the **stop is checked before the target**.
5. `relative_volume` only compares against *completed* prior sessions.
6. Higher timeframes are aggregated from verified 1-minute data and incomplete buckets are
   dropped.

The test suite asserts (1), (2), (3) and (4) directly.

## Error taxonomy for orders

| Failure | Classification | Action |
|---|---|---|
| 400 / 422 / validation | `UpstoxValidationError` | Never retry; surface the reason |
| 401 / 403 | `UpstoxAuthError` | Re-authenticate |
| 429 | `UpstoxRateLimitError` | Cool down, then retry |
| 5xx on a read | `UpstoxServerError` | Retry with backoff |
| 5xx on an order | `UpstoxServerError` (`outcome_uncertain`) | **Reconcile**, never resend |
| Timeout on a read | `UpstoxTimeoutError` | Retry |
| Timeout on an order | `UpstoxTimeoutError` (`outcome_uncertain`) | **Reconcile**, raise an alert |
| Repeated network failure | circuit breaker opens | Fail fast for the cooldown |

## Data model highlights

- `instrument_key` is the canonical identifier everywhere; `exchange_token` is metadata
  because the exchange may reuse it after expiry.
- `candles` is append-only and unique on `(instrument_key, timeframe, ts)`.
- `strategy_versions.config_hash` makes immutability verifiable.
- `trades.features` (JSON) keeps the full entry feature vector so research can slice on
  anything without a migration.
- `multiple_testing_ledger` records every evaluated combination.

## Concurrency model

- FastAPI runs sync endpoints in a thread pool; the `UpstoxHttpClient` is thread-safe and
  rate-limited with a lock.
- The APScheduler runs the daily workflow in background threads with
  `max_instances=1, coalesce=True`.
- The kill switch, notification centre and rate limiters are process-wide singletons with
  locks. In a multi-process deployment they would need Redis; the default deployment is a
  single process, which is stated rather than assumed.
