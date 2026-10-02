"""Structured logging.

JSON lines to stdout (machine friendly, works with Docker log drivers), plain
readable lines when ``LOG_FORMAT=text``. Secrets are scrubbed by a filter so an
accidental ``logger.info(token)`` cannot leak into a log file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Dict, Iterable, Optional

_REDACT_PATTERNS = [
    re.compile(r"(?i)(access[_-]?token|api[_-]?secret|api[_-]?key|password|secret|authorization)\s*[=:]\s*['\"]?([A-Za-z0-9._\-|]{6,})"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{10,}"),
]
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_secret",
        "api_key",
        "password",
        "secret",
        "secret_key",
        "authorization",
        "sandbox_token",
        "analytics_token",
        "dashboard_password",
    }
)

_CONFIGURED = False


class SecretScrubber(logging.Filter):
    """Never let credentials reach a log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = scrub(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: scrub(v) for k, v in record.args.items()}
                elif isinstance(record.args, tuple):
                    record.args = tuple(scrub(a) for a in record.args)
        except Exception:  # pragma: no cover - logging must never raise
            pass
        return True


def scrub(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pattern in _REDACT_PATTERNS:
            if pattern.groups >= 2:
                out = pattern.sub(lambda m: f"{m.group(1)}=***REDACTED***", out)
            else:
                out = pattern.sub("Bearer ***REDACTED***", out)
        return out
    return value


def redact_mapping(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _SENSITIVE_KEYS:
            out[key] = "***REDACTED***" if value else ""
        elif isinstance(value, dict):
            out[key] = redact_mapping(value)
        elif isinstance(value, str):
            out[key] = scrub(value)
        else:
            out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload["context"] = redact_mapping(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            base += " " + json.dumps(redact_mapping(extra), default=str, ensure_ascii=False)
        return base


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger that carries a persistent context dict (e.g. ``component=risk``)."""

    def process(self, msg: Any, kwargs: Dict[str, Any]):
        merged: Dict[str, Any] = dict(self.extra or {})
        ctx = kwargs.pop("context", None)
        if isinstance(ctx, dict):
            merged.update(ctx)
        kwargs["extra"] = {"context": merged}
        return msg, kwargs

    def bind(self, **kwargs: Any) -> "ContextLoggerAdapter":
        merged = dict(self.extra or {})
        merged.update(kwargs)
        return ContextLoggerAdapter(self.logger, merged)


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    global _CONFIGURED
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.environ.get("LOG_FORMAT", "json")).lower()

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else TextFormatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    handler.addFilter(SecretScrubber())
    root.addHandler(handler)

    for noisy in ("urllib3", "httpx", "httpcore", "websockets", "asyncio", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str, **context: Any) -> ContextLoggerAdapter:
    if not _CONFIGURED:
        configure_logging()
    return ContextLoggerAdapter(logging.getLogger(name), dict(context))


def iter_context(logger: ContextLoggerAdapter) -> Iterable[str]:
    return list((logger.extra or {}).keys())


__all__ = [
    "configure_logging",
    "get_logger",
    "scrub",
    "redact_mapping",
    "SecretScrubber",
    "ContextLoggerAdapter",
]
