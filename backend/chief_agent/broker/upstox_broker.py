"""The Upstox broker facade.

A single object the rest of the system talks to, holding the HTTP client and the
domain modules (auth, market data, orders, positions, portfolio, news, options,
broker-side risk). Nothing above this layer touches HTTP directly.

Mode behaviour
--------------
SANDBOX : REST market data from the production host (read-only), orders to
          ``https://sandbox.upstox.com`` (the sandbox supports place / modify /
          cancel only), authenticated with the sandbox token.
PAPER   : everything simulated locally. No broker order call is ever made.
LIVE    : real endpoints, still subjected to the full preflight gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..logging_setup import get_logger
from ..settings import OperatingMode, Settings, get_settings
from .upstox_auth import UpstoxAuth
from .upstox_client import UpstoxAuthError, UpstoxError, UpstoxHttpClient
from .upstox_instruments import InstrumentMaster, Watchlist
from .upstox_market_data import UpstoxMarketData
from .upstox_news import UpstoxNews
from .upstox_options import UpstoxOptions
from .upstox_orders import OrderRequest, OrderResult, UpstoxOrders
from .upstox_portfolio import UpstoxPortfolio
from .upstox_positions import UpstoxPositions
from .upstox_risk import UpstoxRisk

log = get_logger(__name__, component="broker")


@dataclass
class ConnectionCheck:
    name: str
    ok: bool
    detail: str = ""
    required_for_live: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "required_for_live": self.required_for_live,
        }


class UpstoxBroker:
    """Facade over all Upstox modules."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[UpstoxHttpClient] = None,
        mode: Optional[OperatingMode] = None,
    ) -> None:
        self.settings = settings or get_settings()
        effective = mode or self.settings.effective_mode()
        self.client = client or UpstoxHttpClient(settings=self.settings, mode=effective)
        self.mode = effective

        self.auth = UpstoxAuth(self.client, self.settings)
        self.market_data = UpstoxMarketData(self.client, self.settings)
        self.orders = UpstoxOrders(self.client, self.settings)
        self.positions = UpstoxPositions(self.client, self.settings)
        self.portfolio = UpstoxPortfolio(self.client, self.settings)
        self.news = UpstoxNews(self.client, self.settings)
        self.options = UpstoxOptions(self.client, self.settings)
        self.broker_risk = UpstoxRisk(self.client, self.settings)

        self.instruments = InstrumentMaster(self.settings)
        self.watchlist = Watchlist(self.settings, self.instruments)

    # ------------------------------------------------------------------ status
    @property
    def has_token(self) -> bool:
        return self.client.has_token

    def connect(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Attach a token (optionally supplied) and verify it end-to-end."""
        if token:
            self.auth.use_existing_token(token, source="manual")
        return self.auth.verify()

    def status(self) -> Dict[str, Any]:
        mode = self.settings.effective_mode()
        return {
            "mode": mode.value,
            "mode_is_simulated": mode.is_simulated,
            "requested_mode": self.settings.operating_mode.value,
            "live_permitted": self.settings.live_permitted,
            "live_blocked_reasons": self.settings.live_blocked_reasons,
            "deployment_stage": self.settings.deployment_stage.name,
            "auth": self.auth.status(),
            "rate_limits": self.client.limiters.stats(),
            "instrument_master": {
                "loaded": self.instruments.is_loaded,
                "count": self.instruments.size,
                "cache_age_hours": self.instruments.cache_age_hours("nse"),
                "cache_path": str(self.instruments.cache_path("nse")),
            },
            "static_ip": self.broker_risk.get_registered_ips(),
            "kill_switch_last": self.broker_risk.last_result.to_dict() if self.broker_risk.last_result else None,
        }

    # ------------------------------------------------------------- connection
    def preflight(self, deep: bool = True) -> Dict[str, Any]:
        """The LIVE-mode compliance checklist.

        Returns a structured report. Importing this from the API layer is what
        makes the "no live order without a passing preflight" rule enforceable:
        the execution engine refuses LIVE entries while any required check fails.
        """
        checks: List[ConnectionCheck] = []
        mode = self.settings.effective_mode()

        # 1. Credentials present
        checks.append(
            ConnectionCheck(
                "credentials_configured",
                self.settings.has_upstox_credentials,
                "API key, secret and redirect URI must be set",
            )
        )
        # 2. Token present and valid
        if self.has_token:
            if deep:
                verification = self.auth.verify()
                checks.append(
                    ConnectionCheck(
                        "authentication_valid",
                        bool(verification.get("ok")),
                        verification.get("detail") or verification.get("user_name") or "profile verified",
                    )
                )
            else:
                checks.append(ConnectionCheck("authentication_valid", True, "token present (not verified)"))
        else:
            checks.append(ConnectionCheck("authentication_valid", False, "no access token"))

        # 3. Static IP configured / matches
        static_ip = self.broker_risk.get_registered_ips()
        checks.append(
            ConnectionCheck(
                "static_ip_configured",
                bool(static_ip.get("configured")),
                "A registered static IP is required for API order placement in LIVE mode",
            )
        )
        if deep and static_ip.get("configured"):
            ip_check = self.broker_risk.verify_outbound_ip()
            checks.append(
                ConnectionCheck(
                    "outbound_ip_matches",
                    bool(ip_check.get("ok")),
                    f"actual={ip_check.get('actual_ip')} expected={ip_check.get('expected_ip')}",
                )
            )

        # 4. Market data reachable
        if deep and self.has_token:
            try:
                quote = self.market_data.ltp([self.settings and "NSE_INDEX|Nifty 50"])
                ok = bool(quote)
                checks.append(ConnectionCheck("market_data_reachable", ok, f"{len(quote)} quote(s)"))
            except UpstoxError as exc:
                checks.append(ConnectionCheck("market_data_reachable", False, str(exc)))
        else:
            checks.append(ConnectionCheck("market_data_reachable", False, "not tested (deep=False or no token)"))

        # 5. Order API reachable (read-only probe: order book)
        if deep and self.has_token:
            try:
                book = self.orders.order_book()
                checks.append(ConnectionCheck("order_api_reachable", True, f"{len(book)} order(s) today"))
            except UpstoxAuthError as exc:
                checks.append(ConnectionCheck("order_api_reachable", False, f"auth: {exc}"))
            except UpstoxError as exc:
                checks.append(ConnectionCheck("order_api_reachable", False, str(exc)))
        else:
            checks.append(ConnectionCheck("order_api_reachable", False, "not tested"))

        # 6. Funds accessible
        if deep and self.has_token:
            funds = self.portfolio.funds("SEC")
            checks.append(
                ConnectionCheck(
                    "account_funds_accessible",
                    bool(funds.raw),
                    f"available={funds.available_cash:.2f} equity={funds.total_equity:.2f}",
                )
            )
        else:
            checks.append(ConnectionCheck("account_funds_accessible", False, "not tested"))

        # 7. Broker time (exchange status uses Upstox server timestamps)
        try:
            status = self.market_data.exchange_status("NSE")
            ok = status.get("status") not in (None, "UNKNOWN")
            checks.append(ConnectionCheck("broker_time_synchronized", ok, str(status.get("status"))))
        except Exception as exc:
            checks.append(ConnectionCheck("broker_time_synchronized", False, str(exc)))

        # 8-10 are supplied by other engines and merged in by the caller.
        required = [c for c in checks if c.required_for_live]
        passed = all(c.ok for c in required) if mode is OperatingMode.LIVE else None
        return {
            "mode": mode.value,
            "ran_deep_checks": deep,
            "checks": [c.to_dict() for c in checks],
            "all_required_passed": passed,
            "blocking_checks": [c.name for c in required if not c.ok],
        }

    # ----------------------------------------------------------------- helpers
    def resolve_symbol(self, symbol: str) -> Optional[str]:
        """Symbol -> instrument_key using the official master (never guessed)."""
        record = self.instruments.by_symbol(symbol)
        return record.instrument_key if record else None

    def close(self) -> None:
        self.client.close()


__all__ = ["UpstoxBroker", "ConnectionCheck", "OrderRequest", "OrderResult"]
