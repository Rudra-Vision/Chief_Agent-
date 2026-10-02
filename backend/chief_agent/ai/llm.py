"""Optional LLM adapter.

The system is **fully functional with no LLM at all** (``LLM_PROVIDER=none``,
the default). When one is configured it is used only for:

    summarising market conditions, analysing journals, reading news, proposing
    hypotheses, explaining signals, producing daily reports.

Hard limits enforced by construction - the LLM has no access to:

    order placement, risk limits, the live strategy configuration, leverage,
    promotion decisions, or market data.

:class:`LLMClient` therefore only exposes ``complete()``. There is no tool
calling, no function execution and no writable path back into the trading stack.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..logging_setup import get_logger
from ..settings import get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="llm")

#: Everything the LLM is permitted to help with (see config/research.yaml -> llm).
ALLOWED_USES = frozenset(
    {
        "market_condition_summary",
        "trade_journal_analysis",
        "news_classification",
        "hypothesis_proposal",
        "signal_explanation",
        "daily_report",
    }
)

#: Explicitly forbidden capabilities. Documented here so the boundary is auditable.
FORBIDDEN_USES = frozenset(
    {
        "bypass_risk_engine",
        "modify_live_strategy",
        "increase_leverage",
        "declare_candidate_successful",
        "fabricate_market_data",
        "place_orders",
    }
)


class LLMNotConfigured(RuntimeError):
    """Raised when an LLM call is attempted without a provider configured."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    created_at: Any = None
    usage: Optional[Dict[str, Any]] = None


class LLMClient:
    """Minimal, read-only LLM client."""

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = (settings.llm_provider or "none").lower()
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.max_tokens = int(
            ((__import__("chief_agent.settings", fromlist=["get_config_store"]).get_config_store().load("research"))
             .get("llm") or {}).get("max_tokens", 1200)
        )
        self.temperature = float(
            ((__import__("chief_agent.settings", fromlist=["get_config_store"]).get_config_store().load("research"))
             .get("llm") or {}).get("temperature", 0.2)
        )
        self._calls = 0
        self._errors = 0

    @property
    def available(self) -> bool:
        return self.provider not in ("", "none") and bool(self.model)

    def guard(self, use_case: str) -> None:
        """Raise if the caller asks for something the LLM is not allowed to do."""
        if use_case in FORBIDDEN_USES:
            raise PermissionError(
                f"the LLM layer is not permitted to perform '{use_case}'. "
                f"Trading decisions, risk limits and promotions are deterministic."
            )
        if use_case not in ALLOWED_USES:
            log.warning("unrecognised LLM use case", context={"use_case": use_case})

    # ---------------------------------------------------------------- complete
    def complete(self, prompt: str, *, use_case: str = "signal_explanation", max_tokens: Optional[int] = None) -> str:
        self.guard(use_case)
        if not self.available:
            raise LLMNotConfigured(
                "no LLM provider is configured (LLM_PROVIDER=none). "
                "The system falls back to deterministic explanations."
            )
        self._calls += 1
        try:
            if self.provider in ("openai_compatible", "openai", "groq", "together"):
                return self._openai_compatible(prompt, max_tokens or self.max_tokens)
            raise LLMNotConfigured(f"unsupported LLM provider '{self.provider}'")
        except LLMNotConfigured:
            raise
        except Exception as exc:
            self._errors += 1
            log.warning("LLM call failed", context={"provider": self.provider, "error": str(exc)})
            raise

    def _openai_compatible(self, prompt: str, max_tokens: int) -> str:
        import httpx

        base = self.base_url or "https://api.openai.com/v1"
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a quantitative research assistant for an Indian equity trading system. "
                            "You explain and summarise only. You never place orders, never change strategy "
                            "configuration, never change risk limits, and never claim a strategy is "
                            "successful - those decisions belong to a deterministic validation engine. "
                            "If the data does not support a statement, say so."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "calls": self._calls,
            "errors": self._errors,
            "allowed_uses": sorted(ALLOWED_USES),
            "forbidden_uses": sorted(FORBIDDEN_USES),
            "note": (
                "The LLM is optional. Every user-facing explanation and every research decision "
                "has a deterministic fallback."
            ),
        }


_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm() -> None:
    global _client
    _client = None


__all__ = ["LLMClient", "LLMResponse", "LLMNotConfigured", "get_llm", "reset_llm", "ALLOWED_USES", "FORBIDDEN_USES"]
