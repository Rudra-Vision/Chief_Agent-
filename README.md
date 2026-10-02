# Chief Agent

**A self-improving algorithmic trading system for the Indian stock market, built for an Upstox account.**

Chief Agent does two jobs at once:

1. **A trading engine** — scans NSE liquid equities, ranks opportunities, sizes them by risk, paper-trades them, and records every decision.
2. **An autonomous quantitative research lab** — turns your own trade journal into testable hypotheses, runs controlled one-variable experiments, validates them on unseen data, and only then proposes a change to the live strategy.

The single most important design decision in this project:

> **The system cannot decide that its own strategy is good.**
> The AI can *propose*. A deterministic validation engine *decides*.

---

## Table of contents

- [What this system does](#what-this-system-does)
- [What this system does NOT do](#what-this-system-does-not-do)
- [Quick start (5 minutes)](#quick-start-5-minutes)
- [The dashboard](#the-dashboard)
- [How the self-improvement loop works](#how-the-self-improvement-loop-works)
- [Safety: what stops a bad trade](#safety-what-stops-a-bad-trade)
- [Upstox setup](#upstox-setup)
- [Configuration](#configuration)
- [Operating modes and the live-capital ramp](#operating-modes-and-the-live-capital-ramp)
- [How to run a backtest](#how-to-run-a-backtest)
- [How to paper trade](#how-to-paper-trade)
- [Emergency stop](#emergency-stop)
- [Deployment to a VPS](#deployment-to-a-vps)
- [Docker](#docker)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Honest limitations](#honest-limitations)

---

## What this system does

| Area | What you get |
|---|---|
| **Market data** | Upstox Historical Candle V3 + Intraday V3 + Market Quote V3, cached locally. One year of 1-minute data downloads in minutes and is never re-requested. |
| **Universe** | NSE liquid large caps (NIFTY 50 / NEXT 50 scale). Symbol → `instrument_key` resolution from the official Upstox BOD master. No invented identifiers. |
| **Strategy** | Regime-filtered, sector-confirmed **opening-range breakout** with VWAP + relative-volume confirmation, structural stop, ATR/R trailing, scale-out at first target. |
| **Regime engine** | Deterministic 9-way session classification (TREND_UP, RANGE, HIGH_VOLATILITY, GAP_UP, …) stored with every trade. |
| **Scanner** | Walks every 5 minutes of the session, scores every eligible name 0–100 on a transparent weighted formula, and shows the **gate census** — exactly which filter blocked each candidate. |
| **Risk engine** | Absolute authority. 0.25% risk per trade, max 3 positions, 0.75% total open risk, −1% soft / −1.5% hard daily stops, sector caps, no averaging down. Fails **closed**. |
| **Paper trading** | Full order lifecycle with latency, slippage, spread, partial fills, rejections and **real Indian transaction charges**. No broker order is ever sent in PAPER mode. |
| **Backtester** | Event-driven, cost-aware, structurally free of look-ahead: signals fill on the *next* bar, stops fill before targets inside a bar, the opening range is invisible until it closes. |
| **Validation** | Walk-forward (anchored/rolling, purged + embargoed), robustness stress tests, Monte Carlo (drawdown / ruin / losing streaks), a reserved final holdout, and a bootstrap significance test. |
| **Research memory** | Every finding stored as a dated record with its evidence, sample size, decision and result. Ask it questions; it returns only what it actually measured. |
| **Dashboard** | React + TypeScript, SIMPLE and ADVANCED views, a giant STOP TRADING button, and a visible SIMULATION / LIVE banner at all times. |

---

## What this system does NOT do

Being explicit about this is part of the design.

- **It does not promise profit.** There is no target return, no "1% per day", no probability claim. The objective is a robust positive risk-adjusted expectancy, and the honest answer may be "the baseline does not have an edge yet".
- **It does not let an AI rewrite live trading code.** The champion strategy is immutable and versioned. The research layer can only create *challengers*, and only a deterministic gate can promote them.
- **It does not place a real order until you deliberately unlock it.** LIVE mode is blocked by a master switch, a credential check, a static-IP check, a deployment-stage check, a dashboard password, and a full preflight checklist.
- **It does not trade on an LLM's opinion.** An LLM is optional and can never place, size, or approve a trade. News sentiment can add context or veto an entry; it can never be the reason to take one.
- **It does not invent market data.** With no Upstox token the system runs on clearly labelled *simulated* data, and every screen says so. Simulated results are never evidence of an edge.
- **It does not bypass broker or exchange limits.** The published Upstox rate limits are enforced *client-side* at a 70% safety factor, and a 429 opens a cooling period.

---

## Quick start (5 minutes)

Requires Python 3.11+ (3.12 recommended) and, for the full dashboard, Node 20+.

```bash
git clone <your-repo> Chief_Agent-
cd Chief_Agent-

# 1. Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Configuration (optional — the system runs with safe defaults)
cp .env.example .env
#    then edit .env if you want to connect Upstox

# 3. Build the dashboard (optional: a built-in status page is used if you skip this)
cd frontend && npm install && npm run build && cd ..

# 4. Create the database
.venv/bin/python scripts/migrate.py --init

# 5. Start
./scripts/run_server.sh
```

Open **http://localhost:8000**.

You will see a blue **SIMULATION — no real money is at risk** banner, because no Upstox token is configured. That is the correct, safe starting state. The system already works end to end on simulated data:

1. Go to **Data & universe** → **Download history**. This generates and caches a year of simulated 1-minute candles.
2. Go to **Backtest** → pick a date range and universe → **RUN BACKTEST**.
3. Go to **Opportunities** → **Rescan** to see ranked setups with plain-English explanations.
4. Go to **Self-improvement** → **Generate hypotheses** (needs journaled trades first) or run the weekly research job.

When you are ready to use real data, see [Upstox setup](#upstox-setup).

---

## The dashboard

| Page | What it is for |
|---|---|
| **Overview** | Mode, equity, today's P&L, drawdown, open risk, kill-switch state, per-subsystem health, open positions, notifications. |
| **Opportunities** | Top ranked LONG and SHORT setups with score, entry, stop, targets, R:R, sector, regime, VWAP state, relative volume — plus a "Why?" button that gives a plain-English explanation and a full quantitative breakdown. |
| **Positions** | Open paper positions with live P&L, R multiple, MAE/MFE, the order book, and execution slippage analysis. |
| **Backtest** | Date range, universe picker, capital and risk, slippage/fee multipliers → equity curve, drawdown curve, R-distribution histogram, monthly returns, regime/sector/side/time-of-day breakdowns, Monte Carlo. |
| **Self-improvement** | Champion, published versions, hypotheses with evidence, experiments with the full validation pipeline and step log, promotion, research memory, multiple-testing budget. |
| **Trade journal** | Every trade with its entry context, MAE/MFE, execution quality and process classification, plus a daily review and performance analytics. |
| **Data & universe** | Which data source is active, instrument master refresh, historical download, cached coverage, data-quality incidents, watchlist and sector distribution. |
| **Settings & safety** | Upstox connection, the LIVE preflight checklist, every enforced risk limit, the kill switch, the news context, and the effective configuration with secrets redacted. |

The **SIMPLE / ADVANCED** toggle appears on the Opportunities page and changes how a signal is explained: plain English versus every weight, input and gate.

---

## How the self-improvement loop works

```
        ┌─────────────────────────────────────────────────────────┐
        │  CHAMPION  (one immutable, versioned strategy config)   │
        └───────────────────────┬─────────────────────────────────┘
                                │ trades are journaled
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │  TRADE JOURNAL  →  DAILY ANALYSIS                       │
        │  every decision recorded with its market context        │
        └───────────────────────┬─────────────────────────────────┘
                                │ weekly
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │  HYPOTHESIS  H-0048                                     │
        │  "raising min RVOL from 1.20 to 1.35 may remove         │
        │   low-quality ORB breakouts"                            │
        │  evidence: 65 trades, expectancy -0.28R vs +0.11R       │
        └───────────────────────┬─────────────────────────────────┘
                                │ ONE variable changed → new version
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │  CHALLENGER  ORB_v1.0.1                                 │
        └───────────────────────┬─────────────────────────────────┘
                                ▼
   backtest ─► walk-forward ─► robustness ─► Monte Carlo ─► HOLDOUT
   (train)     (anchored,      (parameter    (ruin,         (never seen
               purged,          perturbation,  drawdown,      by optimisation
               embargoed)       cost stress)   streaks)       until now)
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │  DETERMINISTIC PROMOTION GATE                           │
        │  ≥200 trades · expectancy > 0 · PF ≥ 1.30 · Sortino >   │
        │  champion · DD ≤ 15% · ≥60% WF windows profitable ·      │
        │  holdout passes · not one stock/day/regime · MC ruin ≤   │
        │  2% · bootstrap P(better) ≥ 0.75 · paper validation      │
        └───────────────────────┬─────────────────────────────────┘
                     FAIL ──────┴────── PASS
                       │                 │
               REJECTED with        PAPER VALIDATION
               a recorded reason         │
                                         ▼
                              ELIGIBLE FOR PROMOTION
                                         │
                                   you click Promote
                                         ▼
                                  NEW CHAMPION
```

Rules that make this trustworthy:

- **One primary variable per experiment.** A challenger whose configuration diff touches more than one leaf is refused at creation time.
- **Published versions are immutable.** They are content-hashed; a re-publish of the same version string raises.
- **The holdout is locked** until walk-forward validation completes.
- **Every combination is counted.** The multiple-testing ledger records every evaluated strategy/parameter pair, and the required edge grows with the trial count so the system cannot quietly keep the luckiest of thousands of runs.
- **A single losing trade changes nothing.** Research runs on a weekly cadence, not after every loss.

---

## Safety: what stops a bad trade

The system fails **closed**. If it cannot prove it is safe to trade, it does not trade.

| Condition | What happens |
|---|---|
| Risk engine unavailable / crashed | No new orders |
| Kill switch engaged | No new orders, immediately |
| Market data stale, duplicated, impossible, or missing | Enter **DATA_SAFE_MODE** — no new orders |
| Position reconciliation mismatch | Entries halted, alert raised, reconciliation mode entered |
| Broker unreachable or unstable | No new orders |
| Daily loss −1.0% | No new entries for the day |
| Daily loss −1.5% | Flatten and halt |
| 4 consecutive losses | Stop trading for the day |
| 3 consecutive broker rejections | Halt and alert |
| Total open risk would exceed 0.75% | Trade rejected |
| Stop distance zero, or no stop defined | Trade rejected — never entered |
| Risk/reward below 1:1.5 | Trade rejected |
| Spread too wide, liquidity too low | Trade rejected |
| Averaging down / duplicating a position | Structurally blocked |
| Order outcome uncertain (timeout, 5xx) | `requires_reconciliation`; **never** blindly resent |
| LIVE preflight fails | Kill switch engages automatically |

---

## Upstox setup

### 1. Create an Upstox developer app

1. Go to <https://account.upstox.com/developer/apps>.
2. Create an app. Note the **API key** (client_id) and **API secret** (client_secret).
3. Register a **redirect URI** that exactly matches what you will put in `.env`. For local use:
   `http://127.0.0.1:8000/broker/upstox/callback`

### 2. Fill in `.env`

```env
UPSTOX_API_KEY=your_api_key
UPSTOX_API_SECRET=your_api_secret
UPSTOX_REDIRECT_URI=http://127.0.0.1:8000/broker/upstox/callback
```

Then restart the app. **Never commit `.env`** — it is already in `.gitignore`.

### 3. Log in

Open the dashboard → **Settings & safety** → **Connect Upstox**. You will be sent to
Upstox, log in, and be redirected back. The access token is stored **server-side only**
and is never returned to the browser.

Upstox access tokens are daily. Re-authenticate each trading morning (the preflight
checklist will tell you if today's token is missing).

### 4. Sandbox (optional)

Create a sandbox app in the same developer portal and paste its token into
`UPSTOX_SANDBOX_TOKEN`. Sandbox tokens last 30 days and support **place / modify /
cancel order** only, against `https://sandbox.upstox.com`.

### 5. Static IP (required for live order placement)

API order placement must originate from a registered static IP. Set
`UPSTOX_STATIC_IP_PRIMARY` and register it with Upstox (the dashboard can do this via
**Settings & safety**, or `PUT /v2/user/ip`). Note the documented platform rules:

- a static IP can only be changed **once per calendar week**;
- a successful update **invalidates existing access tokens**, so you must log in again;
- the preflight checklist compares your host's outbound IP with the registered one.

### Nothing about the Upstox API is invented here

Every endpoint used by this project was taken from the current official documentation and
is collected in one auditable place: **`config/broker.yaml`**. That file is the single
source of truth for paths, rate limits, protocol details and platform rules.

---

## Configuration

Everything tunable lives in `config/*.yaml`. Nothing important is a magic number.

| File | Controls |
|---|---|
| `config/risk.yaml` | **Absolute limits.** Risk per trade, position caps, daily stops, suspension rules, fail-safe behaviours. |
| `config/strategy.yaml` | The champion strategy: opening range, entry gates, stop models, trailing, targets, opportunity-score weights. |
| `config/universe.yaml` | Universe source, liquidity floors, universe size caps. |
| `config/universe_nifty.csv` | The watchlist itself — **symbols only**; `instrument_key` values are resolved from the official master. |
| `config/broker.yaml` | Every Upstox endpoint, rate limit and platform rule. |
| `config/execution.yaml` | Order types, idempotency, paper-simulation realism, the transaction-cost model, the trading schedule. |
| `config/research.yaml` | Data splits, walk-forward settings, the promotion gate, robustness thresholds, Monte Carlo, the objective weights, and the deterministic hypothesis templates. |

Risk limits **cannot be relaxed through the API** — only tightened. Loosening them means
editing `config/risk.yaml` deliberately.

---

## Operating modes and the live-capital ramp

Three modes, always visible in the banner:

| Mode | What it does | Real money? |
|---|---|---|
| **SANDBOX** | Upstox sandbox order APIs only. | No |
| **PAPER** | Internal simulator. **No broker order call is ever made.** | No |
| **LIVE** | Real orders. Locked until every gate passes. | **Yes** |

Deployment stages, never advanced automatically:

| Stage | Meaning |
|---|---|
| 0 | Backtest |
| 1 | Sandbox |
| 2 | Paper trading |
| 3 | Shadow (live prices, no orders) |
| 4 | Minimum live capital |
| 5 | Small allocation |
| 6 | Normal allocation |

To enter LIVE you must set **all** of:

```env
OPERATING_MODE=LIVE
ALLOW_LIVE_TRADING=true
DEPLOYMENT_STAGE=4            # or higher
UPSTOX_ACCESS_TOKEN=...       # today's token
UPSTOX_STATIC_IP_PRIMARY=...  # registered with Upstox
DASHBOARD_PASSWORD=...        # required so the dashboard is not open
```

…and then pass the preflight checklist on the Settings page. If anything is missing the
system stays in PAPER and tells you exactly what is blocking.

---

## How to run a backtest

**From the dashboard:** *Backtest* → choose dates, tick symbols, set capital and risk →
**RUN BACKTEST**.

**From the command line:**

```bash
.venv/bin/python scripts/run_backtest.py \
    --start 2025-01-01 --end 2025-12-31 \
    --symbols RELIANCE,INFY,HDFCBANK,TATAMOTORS,SBIN \
    --capital 500000 --risk 0.25
```

Reports include: net and annualised return, win/loss rate, average win/loss, expectancy
(in ₹ and in R), profit factor, Sharpe, Sortino, Calmar, recovery factor, maximum
drawdown, volatility, trade count, average holding time, exposure, fees, slippage,
largest win/loss, max consecutive wins/losses, monthly returns, and performance by
regime, sector, side, time of day and exit reason — plus Monte Carlo distributions.

**Costs are mandatory.** The engine refuses to run with `apply_costs=False`; a backtest
without realistic Indian charges is meaningless.

---

## How to paper trade

1. Set `OPERATING_MODE=PAPER` (the default).
2. Make sure `config/execution.yaml → paper.initial_capital` matches what you want to simulate.
3. Leave the app running during market hours. The scheduler runs a scan cycle automatically;
   you can also press **Run trading cycle now** on the Overview page.
4. Watch **Positions** and the **Trade journal**.

Paper trading simulates latency, slippage, spread, partial fills, rejections, stop
triggers (including gap-through fills at the open) and real transaction charges.

---

## Emergency stop

There is a big red **⛔ STOP TRADING** button at the top right of every page.

- It engages the **internal kill switch**, which is instant, local, and works even if the
  broker is unreachable.
- In LIVE mode you can additionally engage the **Upstox account-level kill switch**, which
  cancels open orders at the exchange. Note the documented platform rules: positions must
  be closed first, a **12-hour cooling period** applies before re-enabling, and the access
  token must be regenerated for the change to take effect.
- The switch also engages **automatically** on: daily loss limits, consecutive losses,
  stale data, feed failure, position mismatch, rejection loops, abnormal slippage, a failed
  preflight, or an uncertain order outcome.

Resume with **▶ RESUME TRADING** once you have reviewed what happened.

---

## Deployment to a VPS

A single small VPS (2 vCPU / 4 GB) is enough. The system must keep running when your
personal computer is off.

### 1. Provision

- Ubuntu 22.04+ or Debian 12, 2 vCPU / 4 GB RAM / 40 GB disk.
- **A static IP** — required for Upstox live order placement. Register it with Upstox.
- Docker + Docker Compose, or Python 3.12 + Node 20 for a bare-metal install.

### 2. Configure

```bash
git clone <your-repo> /opt/chief-agent && cd /opt/chief-agent
cp .env.example .env
```

Set at minimum:

```env
OPERATING_MODE=PAPER
DATABASE_URL=postgresql+psycopg://chief:STRONG_PASSWORD@postgres:5432/chief_agent
POSTGRES_PASSWORD=STRONG_PASSWORD
DASHBOARD_PASSWORD=a_long_random_password
SECRET_KEY=a_long_random_string
UPSTOX_API_KEY=...
UPSTOX_API_SECRET=...
UPSTOX_REDIRECT_URI=https://your-domain/broker/upstox/callback
UPSTOX_STATIC_IP_PRIMARY=<your VPS static IP>
```

### 3. Start

```bash
docker compose --profile postgres --profile redis up -d --build
docker compose logs -f app
```

A named volume (`chief_agent_var`) holds the database, the candle cache, backups and
research artefacts, so a container rebuild loses nothing.

### 4. HTTPS and a domain

Put nginx or Caddy in front and terminate TLS there. With Caddy:

```
your-domain.example {
    reverse_proxy 127.0.0.1:8000
}
```

Then set `UPSTOX_REDIRECT_URI=https://your-domain.example/broker/upstox/callback` and
register the same URL on the Upstox app.

### 5. Operations checklist

| Concern | How it is handled |
|---|---|
| Health checks | `GET /health` (also used by the container healthcheck) |
| Automatic restart | `restart: unless-stopped` plus a container healthcheck |
| Database backups | A nightly job writes a compressed backup to `var/backups` and prunes old ones (`monitoring/maintenance.py`) |
| Log rotation | JSON logs to stdout with the Docker `json-file` driver capped at 20 MB × 5 |
| Persistence | Named volumes for `var/` and (optionally) PostgreSQL data |
| Time zone | All internal timestamps are IST-aware; the container TZ does not matter |
| Static IP | Verified by the preflight checklist against the registered IP |

---

## Docker

```bash
docker compose up -d --build                 # SQLite, no external services
docker compose --profile postgres up -d      # PostgreSQL
docker compose --profile dev up -d           # dashboard with hot reload on :5173
docker compose down                          # stop
docker compose down -v                       # stop and delete data (careful!)
```

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite runs entirely offline against an isolated temporary database and simulated
market data. It covers:

- **risk**: sizing, every rejection path, kill switch, fail-closed behaviour;
- **strategy & indicators**: causality (a value never changes when future bars arrive),
  opening-range invisibility before formation, score determinism;
- **backtest & costs**: no same-bar fills, stops capped, costs on every trade, the engine
  refusing to run without costs, determinism, walk-forward window geometry, Monte Carlo
  distributions;
- **broker & execution**: mocked HTTP transport covering 401/400/429/5xx, timeouts,
  uncertain outcomes, rate limiting and the circuit breaker; duplicate-signal
  idempotency; partial fills; rejections; stop-before-target ordering; reconciliation
  mismatches; slippage escalation;
- **research & API**: one-variable challengers, immutability, the status state machine,
  every promotion-gate requirement, hypotheses, memory, and the HTTP surface including
  "no secret is ever returned".

---

## Project layout

```
backend/chief_agent/
    api/            FastAPI app, routers, session auth, CSRF, app state
    broker/         Upstox client: auth, market data, orders, positions,
                    portfolio, instruments, news, options, kill switch,
                    static IP, WebSocket feed, rate limiter
    data/           schema, db, candle store, historical downloader, data-quality
                    engine, provider (live/simulated), migrations
    indicators/     causal indicators, opening range, feature engine
    regime/         market-regime engine
    sectors/        sector engine and ranking
    strategies/     strategy base types + the ORB family
    scanner/        opportunity scanner and explanations
    risk/           risk engine, position sizing, kill switch, preflight
    execution/      lifecycle engine, paper simulator, reconciliation, slippage
    portfolio/      single source of truth for equity/exposure
    costs/          Indian transaction-cost model
    backtest/       event-driven engine, metrics, walk-forward, Monte Carlo,
                    robustness, orchestration
    research/       champion registry, hypotheses, experiments, promotion, memory
    ai/             optional LLM adapter + deterministic explanations
    news/           news classification and context
    monitoring/     health checks, notifications, maintenance
    scheduling/     trading calendar + the daily workflow
frontend/           React + TypeScript dashboard (Vite)
tests/              pytest suite
config/             YAML configuration + the watchlist CSV
scripts/            run_server, dev, migrate, run_backtest, download_data, ...
docker/             deployment helpers
docs/               Upstox notes, architecture, deployment runbook
```

---

## Troubleshooting

**The dashboard says SIMULATION and the data source is SIMULATED.**
No Upstox access token is configured. Everything still works; connect Upstox on the
Settings page to use real data. Simulated results are not evidence of an edge.

**"no cached candles for the current universe".**
Run a historical download first: *Data & universe* → **Download history**.

**The scan finds nothing.**
That is often correct. Look at the **gate census** on the Opportunities page — it lists
exactly which filter blocked each candidate and how many times. Typical answers:
`orb_breakout` (nothing broke the range yet), `relative_volume`, `regime_allowed`.

**Opportunities are empty outside market hours.**
The scanner evaluates the latest session it has data for. Very late in the session the
entry window is closed by design.

**"LIVE mode blocked".**
The Settings page lists every blocking check and how to fix it. This is intentional.

**"instrument master is not loaded".**
*Data & universe* → **Refresh instruments**. This downloads the official BOD file
(`assets.upstox.com`) and needs outbound network access.

**Upstox calls fail with a TLS/network error.**
The host cannot reach `api.upstox.com`. The client opens a circuit breaker after three
consecutive failures and stops retrying for a minute rather than hammering the API.

**A "broker connection is unavailable (circuit open…)" error.**
That is the circuit breaker. Wait for the cooldown, or restart the app if the network has
been fixed.

**Where are the logs?**
Structured JSON on stdout. `docker compose logs -f app` when containerised. Secrets are
scrubbed by a logging filter.

---

## Honest limitations

Stated plainly, because pretending otherwise is how trading systems lose money.

- **The baseline strategy has no proven edge.** On the simulated fixture its gross
  expectancy is slightly positive but **realistic Indian intraday costs turn it negative**
  at 0.25% risk per trade — roughly 0.25R of cost per trade. That is the single most
  important number this system produced, and it is exactly the kind of finding the
  research loop exists to surface. Costs are not a rounding error at this size.
- **Simulated data is a fixture, not a market.** It is a randomised, clearly labelled
  random-walk model. Nothing computed from it demonstrates anything about real markets.
  Connect Upstox and download real history before drawing any conclusion.
- **One year of 1-minute history is a small sample.** The promotion gate asks for 200+
  trades for a reason.
- **Options, futures and F&O strategies are modelled but disabled.** The option-chain
  module exists (OI, PCR, max pain, Greeks where provided) but the first release trades
  NSE cash equities only.
- **No live order has ever been placed by this build.** Live mode is locked, and this
  documentation does not claim it has been validated against a funded account.
- **Broker-side protection depends on the broker.** In PAPER mode the simulator tracks
  stops; in LIVE mode the system places exchange-resident SL-M orders, but a broker
  outage can still leave a position unprotected — which is why reconciliation runs
  continuously.

---

## Licence

MIT. Use at your own risk. This software can lose money; that is what trading systems do.
Nothing here is investment advice.
