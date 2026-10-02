# Upstox API notes

Everything this project uses, taken from the current official documentation
(<https://upstox.com/developer/api-documentation/>). The machine-readable version lives in
`config/broker.yaml`; this file is the human-readable summary.

**Nothing in this project invents an Upstox endpoint, parameter or behaviour.**

## Authentication

OAuth 2.0 authorization-code flow.

1. `GET  /v2/login/authorization/dialog?response_type=code&client_id=…&redirect_uri=…&state=…`
2. Upstox redirects back with `?code=…`
3. `POST /v2/login/authorization/token` (form-encoded, server to server) with
   `code`, `client_id`, `client_secret`, `redirect_uri`, `grant_type=authorization_code`

Notes honoured by the implementation:

- `client_id` is the **API key**; `client_secret` is the **API secret**.
- The authorization code is **single use**, whether or not the exchange succeeds.
- Access tokens are daily; the analytics token is generated once and powers the market-data
  and streaming APIs.

## Market data

| Purpose | Endpoint |
|---|---|
| Historical Candle V3 | `GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}` |
| Intraday Candle V3 | `GET /v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}` |
| LTP Quotes V3 | `GET /v3/market-quote/ltp` |
| OHLC Quotes V3 | `GET /v3/market-quote/ohlc` |
| Full Market Quotes V3 | `GET /v3/market-quote/full` |
| Option Chain | `GET /v2/option/chain?instrument_key=…&expiry_date=…` |
| Option Contracts | `GET /v2/option/contract` |
| News | `GET /v2/news?category=instrument_keys&instrument_keys=…` |
| Market Holidays | `GET /v2/market/holidays` (public) |
| Market Timings | `GET /v2/market/timings/{date}` (public) |
| Exchange Status | `GET /v2/market/status/{exchange}` (public) |

Documented retrieval limits for Historical Candle V3:

| Unit | Intervals | Available from | Max window |
|---|---|---|---|
| `minutes` | 1 … 300 | January 2022 | 1 month for intervals ≤ 15; 1 quarter above |
| `hours` | 1 … 5 | January 2022 | 1 quarter |
| `days` | 1 | January 2000 | 1 decade |
| `weeks` / `months` | 1 | January 2000 | unlimited |

Candle payload shape: `[timestamp, open, high, low, close, volume, open_interest]` where the
timestamp is the candle **start** time in IST.

`India VIX` is available at `NSE_INDEX|India VIX`. The News API returns articles from the
last 7 days and accepts at most **30 instrument keys** per request.

## Orders

| Purpose | Endpoint |
|---|---|
| Place Order V3 | `POST /v3/order/place` (on `api-hft.upstox.com`) |
| Modify Order V3 | `PUT /v3/order/modify` |
| Cancel Order V3 | `DELETE /v3/order/cancel` |
| Order Book | `GET /v2/order/retrieve-all` |
| Order History | `GET /v2/order/history` (`order_id` or `tag`) |
| Order Status | `GET /v2/order/details` |
| Trades for the day | `GET /v2/order/trades/get-trades-for-day` |
| Exit all positions | `POST /v2/order/positions/exit` |
| GTT place / modify / cancel / list | `/v3/order/gtt/{place,modify,cancel,list}` |

Place Order V3 body: `quantity`, `product` (`I`/`D`/`MTF`), `validity` (`DAY`/`IOC`),
`price`, `tag`, `instrument_token`, `order_type` (`MARKET`/`LIMIT`/`SL`/`SL-M`),
`transaction_type`, `disclosed_quantity`, `trigger_price`, `is_amo`, `slice`,
`market_protection`.

Behaviours the implementation accounts for:

- `is_amo` is **ignored** by Upstox — AMO is inferred from the market session.
- `market_protection = 0` causes MARKET/SL-M orders from the API to be **rejected**, so the
  default here is `-1` (automatic protection per guidelines).
- The response returns `data.order_ids` as a **list** (auto-slicing).
- `metadata.latency` reports Upstox-side processing time.
- Order tags are limited to **40 characters**; the tag format here is `CHIEF-<trade_id>-<leg>`.
- `X-Algo-Name` is optional and only used with an exchange-approved algo name.

## Portfolio and account

| Purpose | Endpoint |
|---|---|
| Positions | `GET /v2/portfolio/short-term-positions` |
| Holdings | `GET /v2/portfolio/long-term-holdings` |
| Convert position | `PUT /v2/portfolio/convert-position` |
| Funds and margin | `GET /v2/user/get-funds-and-margin?segment=SEC|COM` |
| Profile | `GET /v2/user/profile` |

## Risk controls, with their platform rules

**Kill switch** — `POST /v2/user/kill-switch`, body an **array** of
`{"segment", "action"}` where action is `ENABLE`/`DISABLE`.

Documented rules, respected and never worked around:

- all open positions in a segment must be closed before disabling it;
- a **12-hour cooling period** applies before a disabled segment can be re-enabled;
- disabling a segment **cancels its open orders**;
- you must **regenerate the access token** for the change to take effect;
- if any segment in a request fails, none are updated.

**Static IP** — `PUT /v2/user/ip` with `{"primary_ip", "secondary_ip"}`.

- can only be changed **once per calendar week**;
- a successful update **invalidates existing access tokens**;
- orders may be rejected unless traffic originates from a registered IP.

## Rate limits (per API, per user)

| Category | Per second | Per minute | Per 30 minutes |
|---|---|---|---|
| Order placement (regular algo) | 10 | 500 | 2000 |
| Order placement (SEBI-registered algo) | 50 | 500 | 2000 |
| Other standard APIs | 50 | 500 | 2000 |
| Payout (standard) | 10 | 500 | 2000 |
| Apply IPO | 1 | 10 | 300 |
| TOTP login | 1 | 10 | 60 |

This project enforces the budgets **client-side at a 70% safety factor** and treats an HTTP
429 as a signal to open a cooling period. The system never intentionally bypasses a broker
or exchange limit.

## WebSocket

| Purpose | Endpoint |
|---|---|
| Market Data Feed V3 authorize | `GET /v3/feed/market-data-feed/authorize` |
| Market Data Feed V3 | `WSS /feed/market-data-feed` |
| Portfolio (order) stream authorize | `GET /v2/feed/portfolio-stream-feed/authorize` |
| Portfolio stream | `WSS /feed/portfolio-stream-feed` |

- The market feed is **protobuf encoded**; the schema is published at
  `https://assets.upstox.com/feed/market-data-feed/v3/MarketDataFeed.proto`.
- Subscription messages are JSON but must be sent as **binary** frames, in the V3 format
  `{"guid", "method": "sub"|"change_mode"|"unsub", "data": {"mode", "instrumentKeys"}}`.
- The authorized URL is **single use**, so the client connects immediately.
- Documented limits (normal plan): 2 connections per user; LTPC 5000 keys (2000 combined),
  Option Greeks 3000 (2000 combined), Full 2000 (1500 combined).

**Important:** because the feed is protobuf, the decoder is only available after the schema
has been compiled locally (`scripts/download_proto.py`, needs `grpcio-tools`). This project
deliberately does **not** hand-roll a decoder from guessed field numbers — inventing a wire
format would silently corrupt market data. Without the decoder the system reports
`DECODER_UNAVAILABLE` and uses the REST market-quote feed instead, which needs no schema.

## Sandbox

- One sandbox app per user; the token is valid for **30 days**.
- Sandbox tokens work only against `https://sandbox.upstox.com` and only for
  **place / modify / cancel order**.
- `/v3/order/place` on the sandbox is documented as sandbox-enabled.

## Instrument master

- `https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz`
- `…/complete.json.gz`, `…/global.json.gz`, `…/suspended-instrument.json.gz`,
  `…/NSE_MIS.json.gz`
- Upstox recommends **JSON over CSV** (more robust, future-proof) and states that
  **`instrument_key` is canonical** because `exchange_token` may be reused by the exchange
  for a different instrument after expiry.
