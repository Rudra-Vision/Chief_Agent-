"""Application settings.

All configuration comes from environment variables (optionally loaded from a
``.env`` file) and from the YAML files in ``config/``.

Secrets are **never** hard-coded and **never** logged. See
:func:`Settings.redacted` for the single safe rendering path.
"""

from __future__ import annotations

import enum
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file's package directory
_PACKAGE_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _PACKAGE_DIR.parent
REPO_ROOT = _BACKEND_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"
VAR_DIR = REPO_ROOT / "var"
CACHE_DIR = VAR_DIR / "cache"
RESEARCH_DIR = REPO_ROOT / "research_data"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

SECRET_FIELDS = frozenset(
    {
        "upstox_api_secret",
        "upstox_access_token",
        "upstox_sandbox_token",
        "upstox_analytics_token",
        "dashboard_password",
        "secret_key",
    }
)


class OperatingMode(str, enum.Enum):
    """The three operating modes. The UI must always show which one is active."""

    SANDBOX = "SANDBOX"
    PAPER = "PAPER"
    LIVE = "LIVE"

    @property
    def can_place_real_orders(self) -> bool:
        return self is OperatingMode.LIVE

    @property
    def is_simulated(self) -> bool:
        return self in (OperatingMode.SANDBOX, OperatingMode.PAPER)


class DeploymentStage(int, enum.Enum):
    """Live-capital ramp. Never advanced automatically."""

    STAGE_0_BACKTEST = 0
    STAGE_1_SANDBOX = 1
    STAGE_2_PAPER = 2
    STAGE_3_SHADOW = 3
    STAGE_4_MIN_LIVE = 4
    STAGE_5_SMALL_LIVE = 5
    STAGE_6_NORMAL_LIVE = 6


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("CHIEF_AGENT_ENV_FILE", str(REPO_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- identity
    app_name: str = "Chief Agent"
    app_version: str = "0.1.0"
    environment: str = "development"

    # ------------------------------------------------------------ mode / stage
    operating_mode: OperatingMode = OperatingMode.PAPER
    deployment_stage: DeploymentStage = DeploymentStage.STAGE_2_PAPER
    #: Hard master switch. When False, LIVE mode cannot be entered at all.
    allow_live_trading: bool = False
    #: Set by the dashboard STOP TRADING button / kill switch.
    kill_switch_engaged: bool = False

    # ------------------------------------------------------------------ upstox
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_redirect_uri: str = "http://127.0.0.1:8000/broker/upstox/callback"
    upstox_access_token: str = ""
    upstox_sandbox_token: str = ""
    upstox_analytics_token: str = ""
    upstox_base_url: str = "https://api.upstox.com"
    upstox_hft_base_url: str = "https://api-hft.upstox.com"
    upstox_sandbox_base_url: str = "https://sandbox.upstox.com"
    upstox_assets_base_url: str = "https://assets.upstox.com"
    upstox_api_version: str = "2.0"
    upstox_static_ip_primary: str = ""
    upstox_static_ip_secondary: str = ""
    upstox_algo_name: Optional[str] = None
    upstox_http_timeout_seconds: float = 15.0
    #: Fail-closed: if True, any live order attempt without a verified static IP
    #: registration is refused.
    require_static_ip_for_live: bool = True

    # ---------------------------------------------------------------- database
    database_url: str = Field(default_factory=lambda: f"sqlite:///{(VAR_DIR / 'chief_agent.db').as_posix()}")
    database_echo: bool = False

    # ------------------------------------------------------------------- redis
    redis_url: str = ""
    redis_required: bool = False

    # ------------------------------------------------------------------- server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------- dashboard authentication
    #: When set, the dashboard requires a login. In LIVE mode this is mandatory.
    dashboard_user: str = "owner"
    dashboard_password: str = ""
    secret_key: str = Field(default_factory=lambda: os.urandom(24).hex())
    session_cookie_name: str = "chief_agent_session"
    require_dashboard_auth_in_live: bool = True

    # ---------------------------------------------------------------- research
    #: Optional LLM for explanation/research summaries. Trading decisions are
    #: deterministic; an LLM can never place or size a trade.
    llm_provider: str = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""

    # ------------------------------------------------------------- data engine
    historical_years_to_cache: int = 1
    candle_cache_backend: str = "sqlite"
    max_universe_size: int = 150
    market_data_poll_seconds: float = 1.0
    stale_quote_seconds: float = 5.0

    @field_validator("operating_mode", mode="before")
    @classmethod
    def _normalise_mode(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().upper()
        return v

    @field_validator("deployment_stage", mode="before")
    @classmethod
    def _normalise_stage(cls, v: Any) -> Any:
        if isinstance(v, str):
            text = v.strip().upper()
            if text.startswith("STAGE_"):
                try:
                    return int(text.split("_")[1])
                except (IndexError, ValueError):
                    return v
        return v

    @field_validator("upstox_redirect_uri", mode="before")
    @classmethod
    def _strip_redirect(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    # ------------------------------------------------------------------ helpers
    @property
    def has_upstox_credentials(self) -> bool:
        return bool(self.upstox_api_key and self.upstox_api_secret and self.upstox_redirect_uri)

    @property
    def has_live_token(self) -> bool:
        return bool(self.upstox_access_token)

    @property
    def has_sandbox_token(self) -> bool:
        return bool(self.upstox_sandbox_token)

    @property
    def using_simulated_broker(self) -> bool:
        """True when the system must run against the built-in simulated broker."""
        if self.operating_mode is OperatingMode.SANDBOX:
            return not self.has_sandbox_token
        return not self.has_live_token

    @property
    def live_blocked_reasons(self) -> List[str]:
        """Why LIVE mode is currently not permitted (fail-closed list)."""
        reasons: List[str] = []
        if not self.allow_live_trading:
            reasons.append("ALLOW_LIVE_TRADING is not enabled in the environment")
        if not self.has_upstox_credentials:
            reasons.append("Upstox API key/secret/redirect URI are not configured")
        if not self.has_live_token:
            reasons.append("No Upstox access token (UPSTOX_ACCESS_TOKEN)")
        if self.require_static_ip_for_live and not self.upstox_static_ip_primary:
            reasons.append("No registered static IP (UPSTOX_STATIC_IP_PRIMARY)")
        if self.deployment_stage.value < DeploymentStage.STAGE_4_MIN_LIVE.value:
            reasons.append(
                f"Deployment stage {self.deployment_stage.name} is below STAGE_4_MIN_LIVE"
            )
        if self.require_dashboard_auth_in_live and not self.dashboard_password:
            reasons.append("DASHBOARD_PASSWORD is required for LIVE mode")
        return reasons

    @property
    def live_permitted(self) -> bool:
        return not self.live_blocked_reasons

    def effective_mode(self) -> OperatingMode:
        """The mode the engine will actually run in, after fail-closed checks.

        Requesting LIVE without satisfying every gate silently downgrades to
        PAPER - the system fails CLOSED, never open.
        """
        if self.kill_switch_engaged:
            return OperatingMode.PAPER
        if self.operating_mode is OperatingMode.LIVE and not self.live_permitted:
            return OperatingMode.PAPER
        return self.operating_mode

    def redacted(self) -> Dict[str, Any]:
        """Settings as a dict with every secret masked. Safe for logs and the UI."""
        out: Dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if name in SECRET_FIELDS:
                out[name] = "***set***" if value else "***unset***"
            elif isinstance(value, enum.Enum):
                out[name] = value.name
            else:
                out[name] = value
        return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return loaded


class ConfigStore:
    """Typed-ish accessor over the YAML files in ``config/``.

    Kept deliberately simple: the files are the source of truth for trading
    parameters, and everything is overridable at runtime through the dashboard
    (which writes back to the same files under ``var/overrides``).
    """

    NAMES = ("risk", "universe", "strategy", "broker", "research", "execution")

    def __init__(self, config_dir: Path = CONFIG_DIR, override_dir: Optional[Path] = None) -> None:
        self.config_dir = Path(config_dir)
        self.override_dir = Path(override_dir or (VAR_DIR / "overrides"))
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load(self, name: str, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and name in self._cache:
            return self._cache[name]
        base = _load_yaml(self.config_dir / f"{name}.yaml")
        override = _load_yaml(self.override_dir / f"{name}.yaml")
        merged = _deep_merge(base, override)
        self._cache[name] = merged
        return merged

    def all(self, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        return {name: self.load(name, use_cache=use_cache) for name in self.NAMES}

    def set_override(self, name: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a runtime override (used by the dashboard settings page)."""
        if name not in self.NAMES:
            raise KeyError(f"unknown config domain: {name}")
        current = _load_yaml(self.override_dir / f"{name}.yaml")
        merged = _deep_merge(current, patch)
        self.override_dir.mkdir(parents=True, exist_ok=True)
        with (self.override_dir / f"{name}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=False)
        self._cache[name] = _deep_merge(_load_yaml(self.config_dir / f"{name}.yaml"), merged)
        return self._cache[name]

    def reload(self) -> None:
        self._cache.clear()


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()


@lru_cache(maxsize=1)
def get_config_store() -> ConfigStore:
    return ConfigStore()


def reset_caches() -> None:
    """Test helper: drop cached settings/config."""
    get_settings.cache_clear()
    get_config_store.cache_clear()


__all__ = [
    "OperatingMode",
    "DeploymentStage",
    "Settings",
    "ConfigStore",
    "get_settings",
    "get_config_store",
    "reset_caches",
    "REPO_ROOT",
    "CONFIG_DIR",
    "VAR_DIR",
    "CACHE_DIR",
    "RESEARCH_DIR",
    "FRONTEND_DIST",
]
