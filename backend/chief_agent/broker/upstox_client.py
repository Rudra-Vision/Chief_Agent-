"""Low-level Upstox HTTP client.

Responsibilities
----------------
* One place where every Upstox endpoint lives (paths are taken from the current
  official documentation - see ``config/broker.yaml``).
* Rate-limit every call through :mod:`chief_agent.broker.rate_limiter`.
* Classify failures so the execution engine can distinguish
  "request never reached the broker" from "broker may have accepted it".
  That distinction is critical for order idempotency.
* Never log secrets; the auth header is scrubbed by the logging filter and is
  never included in exception messages.

This module performs no trading logic and holds no strategy state.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode

import httpx

from ..logging_setup import get_logger
from ..settings import OperatingMode, Settings, get_settings
from .rate_limiter import RateLimiterRegistry, get_rate_limiters

log = get_logger(__name__, component="upstox_http")


# --------------------------------------------------------------------------- #
# Endpoint catalogue (current official Upstox API)
# --------------------------------------------------------------------------- #
class Endpoints:
    """Every Upstox path the system uses. Documented sources in config/broker.yaml."""

    # --- authentication (v2) ------------------------------------------------
    AUTH_DIALOG = "/v2/login/authorization/dialog"
    AUTH_TOKEN = "/v2/login/authorization/token"
    LOGOUT = "/v2/logout"

    # --- market data --------------------------------------------------------
    LTP_V3 = "/v3/market-quote/ltp"
    OHLC_V3 = "/v3/market-quote/ohlc"
    FULL_QUOTE_V3 = "/v3/market-quote/full"
    QUOTES_V2 = "/v2/market-quote/quotes"
    OPTION_CHAIN = "/v2/option/chain"
    OPTION_CONTRACT = "/v2/option/contract"
    OPTION_GREEK = "/v2/option/greek"
    NEWS = "/v2/news"

    @staticmethod
    def historical_v3(instrument_key: str, unit: str, interval: int, to_date: str, from_date: str) -> str:
        return f"/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"

    @staticmethod
    def intraday_v3(instrument_key: str, unit: str, interval: int) -> str:
        return f"/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"

    # --- orders -------------------------------------------------------------
    ORDER_PLACE_V3 = "/v3/order/place"
    ORDER_MODIFY_V3 = "/v3/order/modify"
    ORDER_CANCEL_V3 = "/v3/order/cancel"
    ORDER_PLACE_V2 = "/v2/order/place"
    ORDER_MODIFY_V2 = "/v2/order/modify"
    ORDER_CANCEL_V2 = "/v2/order/cancel"
    ORDER_MULTI_PLACE = "/v2/order/multi/place"
    ORDER_MULTI_CANCEL = "/v2/order/multi/cancel"
    ORDER_EXIT_ALL = "/v2/order/positions/exit"
    ORDER_BOOK = "/v2/order/retrieve-all"
    ORDER_HISTORY = "/v2/order/history"
    ORDER_STATUS = "/v2/order/details"
    TRADES_FOR_DAY = "/v2/order/trades/get-trades-for-day"
    TRADES_BY_ORDER = "/v2/order/trades"

    # --- GTT ----------------------------------------------------------------
    GTT_PLACE = "/v3/order/gtt/place"
    GTT_MODIFY = "/v3/order/gtt/modify"
    GTT_CANCEL = "/v3/order/gtt/cancel"
    GTT_LIST = "/v3/order/gtt/list"

    # --- portfolio ----------------------------------------------------------
    POSITIONS = "/v2/portfolio/short-term-positions"
    HOLDINGS = "/v2/portfolio/long-term-holdings"
    CONVERT_POSITION = "/v2/portfolio/convert-position"
    MARGIN_REQUIRED = "/v2/portfolio/margin-required"

    # --- user / account -----------------------------------------------------
    PROFILE = "/v2/user/profile"
    FUNDS_AND_MARGIN = "/v2/user/get-funds-and-margin"
    KILL_SWITCH = "/v2/user/kill-switch"
    STATIC_IPS = "/v2/user/ip"

    # --- market information (public, no auth required) ----------------------
    MARKET_HOLIDAYS = "/v2/market/holidays"
    MARKET_TIMINGS = "/v2/market/timings/{date}"
    MARKET_STATUS = "/v2/market/status/{exchange}"

    # --- websocket ----------------------------------------------------------
    FEED_AUTHORIZE_V3 = "/v3/feed/market-data-feed/authorize"
    PORTFOLIO_FEED_AUTHORIZE = "/v2/feed/portfolio-stream-feed/authorize"


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #
class UpstoxError(Exception):
    """Base class for all Upstox client failures."""

    retryable = False
    outcome_uncertain = False

    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class UpstoxAuthError(UpstoxError):
    """401/403 - token missing, expired or lacking scope."""


class UpstoxValidationError(UpstoxError):
    """The request itself is wrong (4xx). Never retry these."""


class UpstoxRateLimitError(UpstoxError):
    """HTTP 429."""

    retryable = True


class UpstoxServerError(UpstoxError):
    """5xx from Upstox."""

    retryable = True


class UpstoxTimeoutError(UpstoxError):
    """Read/connect timeout. For orders this is an UNCERTAIN outcome."""

    retryable = True
    outcome_uncertain = True


class UpstoxNetworkError(UpstoxError):
    """Connection reset / DNS / TLS. For orders this is an UNCERTAIN outcome."""

    retryable = True
    outcome_uncertain = True


UNCERTAIN_STATUS_CODES = frozenset({500, 502, 503, 504, 408, 429})
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 405, 409, 415, 422})


@dataclass
class ApiResponse:
    status_code: int
    payload: Any
    latency_ms: float
    raw_text: str = ""
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def data(self) -> Any:
        if isinstance(self.payload, Mapping):
            return self.payload.get("data")
        return None

    def error_message(self) -> str:
        if isinstance(self.payload, Mapping):
            errors = self.payload.get("errors")
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, Mapping):
                    return str(first.get("message") or first.get("errorCode") or first)
                return str(first)
            if "message" in self.payload:
                return str(self.payload["message"])
        return self.raw_text[:400] or f"HTTP {self.status_code}"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class UpstoxHttpClient:
    """Synchronous, thread-safe Upstox REST client.

    The engine runs on a single asyncio loop for streaming but performs REST
    calls from a small thread pool, which is why this client is synchronous and
    protected by a lock around token access.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        limiters: Optional[RateLimiterRegistry] = None,
        client: Optional[httpx.Client] = None,
        mode: Optional[OperatingMode] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.limiters = limiters or get_rate_limiters()
        self.mode = mode or self.settings.operating_mode
        self._lock = threading.RLock()
        self._access_token: str = ""
        self._token_source: str = "none"
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(self.settings.upstox_http_timeout_seconds, connect=8.0),
            follow_redirects=True,
            headers={"Accept": "application/json", "Api-Version": self.settings.upstox_api_version},
        )
        self._owns_client = client is None
        self._load_token_from_settings()
        self.last_call_stats: Dict[str, Any] = {}
        # Circuit breaker: when the broker network is unreachable we back off
        # instead of hammering it with retries on every dashboard refresh.
        self._consecutive_network_failures = 0
        self._circuit_open_until: float = 0.0
        self._circuit_cooldown_seconds = 60.0
        self._circuit_threshold = 3

    # ------------------------------------------------------------------- token
    def _load_token_from_settings(self) -> None:
        s = self.settings
        if self.mode is OperatingMode.SANDBOX and s.upstox_sandbox_token:
            self._access_token = s.upstox_sandbox_token
            self._token_source = "sandbox_env"
        elif s.upstox_access_token:
            self._access_token = s.upstox_access_token
            self._token_source = "env"
        elif s.upstox_analytics_token:
            self._access_token = s.upstox_analytics_token
            self._token_source = "analytics_env"
        else:
            self._access_token = ""
            self._token_source = "none"

    def set_access_token(self, token: str, source: str = "runtime") -> None:
        with self._lock:
            self._access_token = (token or "").strip()
            self._token_source = source if self._access_token else "none"
        log.info("access token updated", context={"source": source, "present": bool(token)})

    def clear_access_token(self) -> None:
        with self._lock:
            self._access_token = ""
            self._token_source = "none"

    @property
    def has_token(self) -> bool:
        with self._lock:
            return bool(self._access_token)

    @property
    def token_source(self) -> str:
        with self._lock:
            return self._token_source

    def _auth_headers(self) -> Dict[str, str]:
        with self._lock:
            token = self._access_token
        if not token:
            raise UpstoxAuthError("No Upstox access token available. Connect the account first.")
        return {"Authorization": f"Bearer {token}"}

    # -------------------------------------------------------------- base URLs
    def base_url(self, kind: str = "rest") -> str:
        s = self.settings
        if self.mode is OperatingMode.SANDBOX and kind == "rest":
            return s.upstox_sandbox_base_url.rstrip("/")
        if kind == "hft":
            return s.upstox_hft_base_url.rstrip("/")
        return s.upstox_base_url.rstrip("/")

    # ----------------------------------------------------------------- request
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        data: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        limit_class: str = "standard",
        use_hft: bool = False,
        require_auth: bool = True,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        idempotent: bool = True,
    ) -> ApiResponse:
        """Perform a rate-limited request with bounded, classified retries.

        ``idempotent=False`` is used for order placement so that a transport
        failure is surfaced as :class:`UpstoxTimeoutError` /
        :class:`UpstoxNetworkError` (``outcome_uncertain = True``) instead of
        being silently retried - the execution engine must reconcile first.
        """
        method = method.upper()
        base = self.base_url("hft" if use_hft else "rest")
        url = f"{base}{path}"

        # Fail fast while the circuit is open.
        if time.monotonic() < self._circuit_open_until:
            raise UpstoxNetworkError(
                f"broker connection is unavailable (circuit open for another "
                f"{self._circuit_open_until - time.monotonic():.0f}s); refusing to retry {path}"
            )
        hdrs: Dict[str, str] = {
            "Accept": "application/json",
            "Api-Version": self.settings.upstox_api_version,
        }
        if require_auth:
            hdrs.update(self._auth_headers())
        if json_body is not None:
            hdrs["Content-Type"] = "application/json"
        elif data is not None:
            hdrs["Content-Type"] = "application/x-www-form-urlencoded"
        if headers:
            hdrs.update(dict(headers))

        limiter = self.limiters.get(limit_class)
        attempts = max(1, max_retries + 1) if idempotent else 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            if not limiter.acquire():
                raise UpstoxRateLimitError("local rate limiter timeout; refusing to exceed the budget")

            started = time.perf_counter()
            try:
                response = self._client.request(
                    method,
                    url,
                    params=dict(params) if params else None,
                    json=json_body,
                    data=dict(data) if data else None,
                    headers=hdrs,
                    timeout=timeout or self.settings.upstox_http_timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = UpstoxTimeoutError(f"timeout calling {path}: {exc}")
                log.warning("upstox timeout", context={"path": path, "attempt": attempt + 1})
                if not idempotent:
                    raise last_error from exc
                time.sleep(0.5 * (attempt + 1))
                continue
            except httpx.HTTPError as exc:
                self._note_network_failure()
                last_error = UpstoxNetworkError(f"network error calling {path}: {exc}")
                log.warning("upstox network error", context={"path": path, "error": str(exc)})
                if not idempotent or self.circuit_open:
                    raise last_error from exc
                time.sleep(0.5 * (attempt + 1))
                continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            parsed = self._parse(response)
            self.last_call_stats = {
                "path": path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 1),
            }

            if 200 <= response.status_code < 300:
                self._note_network_success()
                return ApiResponse(
                    status_code=response.status_code,
                    payload=parsed,
                    latency_ms=latency_ms,
                    raw_text=response.text[:2000],
                    headers=dict(response.headers),
                )

            api = ApiResponse(
                status_code=response.status_code,
                payload=parsed,
                latency_ms=latency_ms,
                raw_text=response.text[:2000],
                headers=dict(response.headers),
            )
            message = api.error_message()
            error_code = self._extract_error_code(parsed)

            if response.status_code == 429:
                retry_after = self._retry_after_seconds(response.headers)
                limiter.note_429(retry_after)
                last_error = UpstoxRateLimitError(message, status_code=429, payload=parsed)
                if not idempotent or attempt == attempts - 1:
                    raise last_error
                time.sleep(retry_after)
                continue

            if response.status_code in (401, 403):
                raise UpstoxAuthError(
                    f"authentication failed ({response.status_code}): {message}",
                    status_code=response.status_code,
                    payload=parsed,
                )

            if response.status_code in UNCERTAIN_STATUS_CODES and idempotent:
                last_error = UpstoxServerError(message, status_code=response.status_code, payload=parsed)
                if attempt == attempts - 1:
                    raise last_error
                time.sleep(0.5 * (attempt + 1))
                continue

            if response.status_code in UNCERTAIN_STATUS_CODES:
                # Non-idempotent request with an uncertain server response.
                raise UpstoxServerError(
                    f"uncertain outcome for {path} ({response.status_code}): {message}",
                    status_code=response.status_code,
                    payload=parsed,
                )

            raise UpstoxValidationError(
                f"{path} rejected ({response.status_code} / {error_code}): {message}",
                status_code=response.status_code,
                payload=parsed,
            )

        raise last_error or UpstoxError(f"request to {path} failed")

    # ------------------------------------------------------- circuit breaker
    @property
    def circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    @property
    def consecutive_network_failures(self) -> int:
        return self._consecutive_network_failures

    def _note_network_failure(self) -> None:
        self._consecutive_network_failures += 1
        if self._consecutive_network_failures >= self._circuit_threshold:
            self._circuit_open_until = time.monotonic() + self._circuit_cooldown_seconds
            log.warning(
                "broker circuit opened after repeated network failures",
                context={
                    "failures": self._consecutive_network_failures,
                    "cooldown_s": self._circuit_cooldown_seconds,
                },
            )

    def _note_network_success(self) -> None:
        self._consecutive_network_failures = 0
        self._circuit_open_until = 0.0

    def reset_circuit(self) -> None:
        self._consecutive_network_failures = 0
        self._circuit_open_until = 0.0

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _parse(response: httpx.Response) -> Any:
        text = response.text or ""
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text[:1000]}

    @staticmethod
    def _retry_after_seconds(headers: Mapping[str, str]) -> float:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if not raw:
            return 2.0
        try:
            return max(0.5, min(60.0, float(raw)))
        except ValueError:
            return 2.0

    @staticmethod
    def _extract_error_code(payload: Any) -> str:
        if isinstance(payload, Mapping):
            errors = payload.get("errors")
            if isinstance(errors, list) and errors and isinstance(errors[0], Mapping):
                return str(errors[0].get("errorCode") or errors[0].get("error_code") or "")
        return ""

    # -------------------------------------------------------------- public ops
    def get(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> ApiResponse:
        return self.request("DELETE", path, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass

    @staticmethod
    def build_query(params: Mapping[str, Any]) -> str:
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        return urlencode(clean)


__all__ = [
    "Endpoints",
    "UpstoxHttpClient",
    "ApiResponse",
    "UpstoxError",
    "UpstoxAuthError",
    "UpstoxValidationError",
    "UpstoxRateLimitError",
    "UpstoxServerError",
    "UpstoxTimeoutError",
    "UpstoxNetworkError",
]
