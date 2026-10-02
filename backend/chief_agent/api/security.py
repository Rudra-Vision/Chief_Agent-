"""Dashboard authentication and request hardening.

Security posture
----------------
* The dashboard can be protected by a password. **LIVE mode requires it** (see
  the preflight checklist) - an open dashboard controlling real money is not
  acceptable.
* Sessions are signed, HTTP-only, SameSite=Lax cookies. The signing key comes
  from ``SECRET_KEY`` (generated per process if unset, which logs everyone out
  on restart - acceptable and safe).
* State-changing requests (POST/PUT/DELETE) from a browser must present a CSRF
  token tied to the session.
* Broker tokens are **never** sent to the browser. Every broker call happens
  server-side; the API returns only sanitised status objects.
* Login is rate-limited to blunt brute-force attempts.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..logging_setup import get_logger
from ..settings import get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="security")

SESSION_MAX_AGE_SECONDS = 12 * 3600
CSRF_HEADER = "x-csrf-token"
LOGIN_ATTEMPT_WINDOW = 300
MAX_LOGIN_ATTEMPTS = 8

_PUBLIC_PATHS = (
    "/health",
    "/api/system/health",
    "/api/system/mode",
    "/api/auth/status",
    "/api/auth/login",
    "/api/auth/logout",
    "/broker/upstox/callback",
    "/docs",
    "/openapi.json",
    "/redoc",
)


@dataclass
class Session:
    user: str
    issued_at: float
    csrf_token: str
    session_id: str = field(default_factory=lambda: secrets.token_hex(8))

    def to_dict(self) -> Dict[str, Any]:
        return {"user": self.user, "issued_at": self.issued_at, "session_id": self.session_id}


class LoginRateLimiter:
    def __init__(self) -> None:
        self._attempts: Dict[str, list] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            attempts = [t for t in self._attempts.get(key, []) if now - t < LOGIN_ATTEMPT_WINDOW]
            self._attempts[key] = attempts
            if len(attempts) >= MAX_LOGIN_ATTEMPTS:
                retry_in = int(LOGIN_ATTEMPT_WINDOW - (now - attempts[0]))
                return False, max(1, retry_in)
            return True, 0

    def record_failure(self, key: str) -> None:
        with self._lock:
            self._attempts.setdefault(key, []).append(time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)


class AuthManager:
    """Password login, signed sessions and CSRF tokens."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.serializer = URLSafeTimedSerializer(self.settings.secret_key, salt="chief-agent-session")
        self.rate_limiter = LoginRateLimiter()

    # ------------------------------------------------------------------ config
    @property
    def auth_enabled(self) -> bool:
        return bool(self.settings.dashboard_password)

    @property
    def auth_required(self) -> bool:
        """In LIVE mode the login wall is mandatory, not optional."""
        if self.settings.effective_mode().value == "LIVE":
            return self.settings.require_dashboard_auth_in_live
        return self.auth_enabled

    @property
    def cookie_name(self) -> str:
        return self.settings.session_cookie_name

    # ------------------------------------------------------------------- login
    def verify_credentials(self, username: str, password: str, client_key: str = "local") -> Tuple[bool, str]:
        allowed, retry_in = self.rate_limiter.check(client_key)
        if not allowed:
            return False, f"too many attempts; try again in {retry_in}s"

        if not self.auth_enabled:
            return False, "no dashboard password is configured"

        expected_user = self.settings.dashboard_user
        user_ok = hmac.compare_digest(username or "", expected_user)
        # Compare digests so the comparison time does not leak the password length.
        password_ok = hmac.compare_digest(
            hashlib.sha256((password or "").encode()).hexdigest(),
            hashlib.sha256(self.settings.dashboard_password.encode()).hexdigest(),
        )
        if user_ok and password_ok:
            self.rate_limiter.reset(client_key)
            return True, ""
        self.rate_limiter.record_failure(client_key)
        log.warning("failed dashboard login", context={"username": username, "client": client_key})
        return False, "invalid username or password"

    def create_session(self, user: str) -> Tuple[str, Session]:
        session = Session(user=user, issued_at=time.time(), csrf_token=secrets.token_urlsafe(32))
        token = self.serializer.dumps({"user": user, "csrf": session.csrf_token, "sid": session.session_id})
        return token, session

    def read_session(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        try:
            payload = self.serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        except (BadSignature, SignatureExpired):
            return None
        if not isinstance(payload, dict) or "user" not in payload:
            return None
        return payload

    def set_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=SESSION_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            secure=False,  # set True behind HTTPS in production
            path="/",
        )

    def clear_cookie(self, response: Response) -> None:
        response.delete_cookie(self.cookie_name, path="/")

    # ------------------------------------------------------------------ checks
    def is_public(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in _PUBLIC_PATHS) or path.startswith("/assets")

    def current_session(self, request: Request) -> Optional[Dict[str, Any]]:
        return self.read_session(request.cookies.get(self.cookie_name))

    def require_session(self, request: Request) -> Dict[str, Any]:
        if not self.auth_required:
            return {"user": "anonymous", "csrf": ""}
        session = self.current_session(request)
        if session is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
        return session

    def require_csrf(self, request: Request, session: Dict[str, Any]) -> None:
        """Double-submit CSRF check for state-changing requests."""
        if not self.auth_required:
            return
        expected = session.get("csrf")
        if not expected:
            return
        provided = request.headers.get(CSRF_HEADER) or request.headers.get(CSRF_HEADER.upper())
        if not provided or not hmac.compare_digest(str(provided), str(expected)):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid CSRF token")

    # ------------------------------------------------------------------ status
    def status(self, request: Request) -> Dict[str, Any]:
        session = self.current_session(request)
        return {
            "auth_enabled": self.auth_enabled,
            "auth_required": self.auth_required,
            "authenticated": session is not None or not self.auth_required,
            "user": session.get("user") if session else None,
            "csrf_token": session.get("csrf") if session else None,
            "note": (
                "The dashboard password is required in LIVE mode. "
                "Broker tokens never leave the server."
            ),
        }


_manager: Optional[AuthManager] = None
_lock = threading.Lock()


def get_auth() -> AuthManager:
    global _manager
    with _lock:
        if _manager is None:
            _manager = AuthManager()
        return _manager


def reset_auth() -> None:
    global _manager
    with _lock:
        _manager = None


def sanitize_for_client(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strip anything credential-shaped before a payload leaves the server."""
    forbidden = {
        "access_token",
        "api_secret",
        "api_key",
        "client_secret",
        "password",
        "secret",
        "secret_key",
        "authorization",
        "token",
        "sandbox_token",
        "analytics_token",
        "dashboard_password",
        "llm_api_key",
    }
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in forbidden:
            out[key] = "***redacted***" if value else ""
        elif isinstance(value, dict):
            out[key] = sanitize_for_client(value)
        elif isinstance(value, list):
            out[key] = [sanitize_for_client(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


__all__ = ["AuthManager", "get_auth", "reset_auth", "Session", "sanitize_for_client", "CSRF_HEADER"]
