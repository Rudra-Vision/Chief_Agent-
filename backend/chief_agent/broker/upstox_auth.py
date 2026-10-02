"""Upstox authentication.

Implements the OAuth 2.0 authorization-code flow exactly as documented at
https://upstox.com/developer/api-documentation/authentication :

    1. GET  /v2/login/authorization/dialog?response_type=code&client_id=...&redirect_uri=...&state=...
    2. Upstox redirects back to ``redirect_uri`` with ``?code=...``
    3. POST /v2/login/authorization/token  (form-encoded, server-to-server)
       with code, client_id, client_secret, redirect_uri, grant_type=authorization_code

The authorization code is SINGLE USE (valid whether or not token generation
succeeds). The access token is regenerated daily.

Sandbox tokens are created in the Upstox developer portal and are valid for
30 days; they can only be used against ``https://sandbox.upstox.com`` and only
for place/modify/cancel order APIs.

Secrets are read from the environment only and are never returned to a client.
"""

from __future__ import annotations

import datetime as dt
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from ..logging_setup import get_logger
from ..settings import OperatingMode, Settings, get_settings
from ..timeutil import IST, now_ist
from .upstox_client import ApiResponse, Endpoints, UpstoxAuthError, UpstoxError, UpstoxHttpClient

log = get_logger(__name__, component="upstox_auth")


def ist_today() -> dt.date:
    return now_ist().date()


@dataclass
class TokenRecord:
    """An access token and when it was obtained (never persisted to disk)."""

    token: str
    obtained_at: dt.datetime
    source: str
    scope: str = ""
    user_id: str = ""
    expires_at: Optional[dt.datetime] = None

    @property
    def age_hours(self) -> float:
        return (now_ist() - self.obtained_at).total_seconds() / 3600.0

    def is_probably_valid_for_today(self) -> bool:
        """Upstox access tokens are effectively daily; treat them as such."""
        return self.obtained_at.astimezone(IST).date() == ist_today()

    @property
    def masked(self) -> str:
        if not self.token:
            return "***unset***"
        return f"{self.token[:4]}...{self.token[-4:]} ({len(self.token)} chars)"


class OAuthStateStore:
    """Short-lived CSRF ``state`` values for the OAuth dialog."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._values: Dict[str, dt.datetime] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def issue(self) -> str:
        value = secrets.token_urlsafe(24)
        with self._lock:
            self._gc()
            self._values[value] = now_ist()
        return value

    def consume(self, value: str) -> bool:
        if not value:
            return False
        with self._lock:
            self._gc()
            return self._values.pop(value, None) is not None

    def _gc(self) -> None:
        cutoff = now_ist() - dt.timedelta(seconds=self._ttl)
        for key in [k for k, v in self._values.items() if v < cutoff]:
            self._values.pop(key, None)


class UpstoxAuth:
    """Owns the token lifecycle for one Upstox account."""

    def __init__(self, client: UpstoxHttpClient, settings: Optional[Settings] = None) -> None:
        self.client = client
        self.settings = settings or get_settings()
        self.states = OAuthStateStore()
        self._token: Optional[TokenRecord] = None
        self._last_error: Optional[str] = None
        if self.client.has_token:
            self._token = TokenRecord(
                token=self.client._access_token,  # noqa: SLF001 - same package, deliberate
                obtained_at=now_ist(),
                source=self.client.token_source,
            )

    # ---------------------------------------------------------------- utilities
    @property
    def credentials_configured(self) -> bool:
        return self.settings.has_upstox_credentials

    @property
    def token(self) -> Optional[str]:
        return self._token.token if self._token else None

    @property
    def token_record(self) -> Optional[TokenRecord]:
        return self._token

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def is_authenticated(self) -> bool:
        return bool(self._token and self._token.token)

    def status(self) -> Dict[str, Any]:
        """A safe, secret-free status view for the dashboard."""
        s = self.settings
        rec = self._token
        return {
            "authenticated": self.is_authenticated,
            "credentials_configured": self.credentials_configured,
            "token_source": rec.source if rec else "none",
            "token_masked": rec.masked if rec else "***unset***",
            "token_obtained_at": rec.obtained_at.isoformat() if rec else None,
            "token_age_hours": round(rec.age_hours, 2) if rec else None,
            "token_probably_valid_today": rec.is_probably_valid_for_today() if rec else False,
            "mode": s.operating_mode.value,
            "redirect_uri": s.upstox_redirect_uri,
            "api_key_configured": bool(s.upstox_api_key),
            "api_secret_configured": bool(s.upstox_api_secret),
            "sandbox_token_configured": bool(s.upstox_sandbox_token),
            "analytics_token_configured": bool(s.upstox_analytics_token),
            "static_ip_primary": s.upstox_static_ip_primary or None,
            "static_ip_secondary": s.upstox_static_ip_secondary or None,
            "last_error": self._last_error,
        }

    # -------------------------------------------------------------- login flow
    def build_login_url(self, state: Optional[str] = None, redirect_uri: Optional[str] = None) -> Dict[str, str]:
        """Step 1: the URL to open in the browser / webview."""
        if not self.settings.upstox_api_key:
            raise UpstoxAuthError("UPSTOX_API_KEY is not configured")
        state = state or self.states.issue()
        redirect = redirect_uri or self.settings.upstox_redirect_uri
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.settings.upstox_api_key,
                "redirect_uri": redirect,
                "state": state,
            }
        )
        return {
            "url": f"{self.client.base_url('rest')}{Endpoints.AUTH_DIALOG}?{query}",
            "state": state,
        }

    def exchange_code_for_token(self, code: str, state: Optional[str] = None) -> TokenRecord:
        """Step 3: exchange the single-use authorization code for an access token."""
        if not code:
            raise UpstoxAuthError("Authorization code is required")
        if state is not None and not self.states.consume(state):
            log.warning("oauth state check failed", context={"state_present": bool(state)})
            raise UpstoxAuthError("OAuth state mismatch - refusing to exchange the code")

        form = {
            "code": code,
            "client_id": self.settings.upstox_api_key,
            "client_secret": self.settings.upstox_api_secret,
            "redirect_uri": self.settings.upstox_redirect_uri,
            "grant_type": "authorization_code",
        }
        # The token endpoint lives on the production host even for sandbox apps.
        original = self.client.mode
        self.client.mode = OperatingMode.LIVE
        try:
            response: ApiResponse = self.client.request(
                "POST",
                Endpoints.AUTH_TOKEN,
                data=form,
                require_auth=False,
                limit_class="standard",
                idempotent=False,
                max_retries=0,
            )
        finally:
            self.client.mode = original
        return self._store_token_response(response, source="oauth")

    def _store_token_response(self, response: ApiResponse, source: str) -> TokenRecord:
        payload = response.payload if isinstance(response.payload, dict) else {}
        token = str(payload.get("access_token") or "").strip()
        if not token:
            self._last_error = response.error_message()
            raise UpstoxAuthError(f"token exchange did not return an access_token: {self._last_error}")
        record = TokenRecord(
            token=token,
            obtained_at=now_ist(),
            source=source,
            scope=str(payload.get("scope") or ""),
            user_id=str(payload.get("user_id") or ""),
            expires_at=None,
        )
        self._token = record
        self._last_error = None
        self.client.set_access_token(token, source=source)
        log.info(
            "upstox access token stored",
            context={"source": source, "masked": record.masked, "scope": record.scope},
        )
        return record

    def use_existing_token(self, token: str, source: str = "manual") -> TokenRecord:
        rec = TokenRecord(token=token.strip(), obtained_at=now_ist(), source=source)
        self._token = rec
        self.client.set_access_token(rec.token, source=source)
        return rec

    def use_sandbox_token(self, token: str) -> TokenRecord:
        return self.use_existing_token(token, source="sandbox_env")

    def logout(self) -> bool:
        """Revoke the session on Upstox and clear the local token."""
        ok = False
        if self.client.has_token:
            try:
                resp = self.client.request("POST", Endpoints.LOGOUT, idempotent=True, max_retries=0)
                ok = resp.ok
            except UpstoxError as exc:
                log.warning("logout call failed", context={"error": str(exc)})
        self._token = None
        self.client.clear_access_token()
        return ok

    # -------------------------------------------------------------- validation
    def verify(self) -> Dict[str, Any]:
        """Call ``/v2/user/profile`` to prove the token actually works."""
        if not self.client.has_token:
            return {"ok": False, "reason": "no_token"}
        try:
            response = self.client.get(Endpoints.PROFILE, limit_class="standard")
        except UpstoxAuthError as exc:
            self._last_error = str(exc)
            return {"ok": False, "reason": "unauthorized", "detail": str(exc)}
        except UpstoxError as exc:
            self._last_error = str(exc)
            return {"ok": False, "reason": "transport", "detail": str(exc)}
        data = response.data if isinstance(response.data, dict) else {}
        self._last_error = None
        return {
            "ok": True,
            "user_id": data.get("user_id"),
            "user_name": data.get("user_name"),
            "email": data.get("email"),
            "broker": data.get("broker"),
            "exchanges": data.get("exchanges"),
            "products": data.get("products"),
            "order_types": data.get("order_types"),
        }


__all__ = ["UpstoxAuth", "TokenRecord", "OAuthStateStore", "ist_today"]
