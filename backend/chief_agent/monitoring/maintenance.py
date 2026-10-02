"""Backups, log rotation and database maintenance.

The brief requires database backups and log rotation for the VPS deployment, and
"database backups work" is a LIVE-mode prerequisite. Backups are written to
``var/backups`` and are safe to run while the system is live (SQLite uses the
online backup API; PostgreSQL shells out to ``pg_dump`` when available).
"""

from __future__ import annotations

import datetime as dt
import gzip
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_setup import get_logger
from ..settings import VAR_DIR, get_settings
from ..timeutil import now_ist

log = get_logger(__name__, component="maintenance")

BACKUP_DIR = VAR_DIR / "backups"


def backup_database(keep_days: int = 14) -> Dict[str, Any]:
    """Create a timestamped backup and prune old ones."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    dsn = settings.database_url or ""
    stamp = now_ist().strftime("%Y%m%d-%H%M%S")

    try:
        if dsn.startswith("sqlite") or not dsn:
            from ..data.db import get_engine

            engine = get_engine()
            raw_path = engine.url.database
            if not raw_path or raw_path == ":memory:":
                return {"ok": False, "reason": "in-memory database cannot be backed up"}
            source = Path(raw_path)
            if not source.exists():
                return {"ok": False, "reason": f"database file not found at {source}"}

            target = BACKUP_DIR / f"chief_agent-{stamp}.db"
            # SQLite online backup is safe while the app is running.
            source_conn = engine.raw_connection()
            try:
                import sqlite3

                destination = sqlite3.connect(str(target))
                try:
                    source_conn.driver_connection.backup(destination)
                finally:
                    destination.close()
            finally:
                source_conn.close()
        else:
            target = BACKUP_DIR / f"chief_agent-{stamp}.sql"
            result = subprocess.run(
                ["pg_dump", "--no-owner", "--format=plain", dsn],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                return {"ok": False, "reason": result.stderr[:400]}
            target.write_text(result.stdout, encoding="utf-8")

        with open(target, "rb") as handle:
            payload = handle.read()
        compressed = target.with_suffix(target.suffix + ".gz")
        with gzip.open(compressed, "wb") as handle:
            handle.write(payload)
        target.unlink(missing_ok=True)

        pruned = _prune_backups(keep_days)
        size_mb = compressed.stat().st_size / (1024 * 1024)
        log.info("database backup created", context={"path": str(compressed), "size_mb": round(size_mb, 2)})
        return {
            "ok": True,
            "path": str(compressed),
            "size_mb": round(size_mb, 3),
            "pruned": pruned,
            "created_at": now_ist().isoformat(),
        }
    except Exception as exc:
        log.error("database backup failed", context={"error": str(exc)})
        return {"ok": False, "reason": str(exc)}


def _prune_backups(keep_days: int) -> List[str]:
    cutoff = now_ist().timestamp() - keep_days * 86400
    removed: List[str] = []
    for path in sorted(BACKUP_DIR.glob("chief_agent-*")):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(path.name)
        except OSError:
            continue
    return removed


def list_backups() -> List[Dict[str, Any]]:
    if not BACKUP_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(BACKUP_DIR.glob("chief_agent-*"), reverse=True):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append(
            {
                "name": path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "created_at": dt.datetime.fromtimestamp(stat.st_mtime).astimezone(now_ist().tzinfo).isoformat(),
            }
        )
    return out


def prune_old_events(
    quote_days: int = 30,
    candle_days: int = 0,
    event_days: int = 180,
) -> Dict[str, int]:
    """Trim high-volume tables.

    ``quote_days=30`` keeps a month of tick snapshots for slippage research.
    ``candle_days=0`` means NEVER prune candles - source market data is retained.
    """
    from sqlalchemy import delete

    from ..data.db import session_scope
    from ..data.schema import Quote, SystemEvent

    deleted: Dict[str, int] = {}
    cutoff_quotes = now_ist() - dt.timedelta(days=quote_days)
    cutoff_events = now_ist() - dt.timedelta(days=event_days)

    with session_scope() as session:
        result = session.execute(delete(Quote).where(Quote.ts < cutoff_quotes))
        deleted["quotes"] = int(result.rowcount or 0)
        result = session.execute(delete(SystemEvent).where(SystemEvent.ts < cutoff_events))
        deleted["system_events"] = int(result.rowcount or 0)

    if candle_days > 0:
        log.warning("candle pruning is disabled by policy: source market data is never deleted")
        deleted["candles"] = 0

    log.info("event pruning complete", context=deleted)
    return deleted


def disk_usage() -> Dict[str, Any]:
    """Report the size of the data directory so the dashboard can warn early."""
    total = 0
    breakdown: Dict[str, int] = {}
    for root, _dirs, files in os.walk(VAR_DIR):
        for name in files:
            path = Path(root) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            total += size
            top = Path(root).relative_to(VAR_DIR).parts[0] if Path(root) != VAR_DIR else name
            breakdown[top] = breakdown.get(top, 0) + size
    return {
        "var_dir": str(VAR_DIR),
        "total_mb": round(total / (1024 * 1024), 2),
        "by_area_mb": {k: round(v / (1024 * 1024), 3) for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])[:15]},
        "backups": len(list_backups()),
    }


__all__ = ["backup_database", "list_backups", "prune_old_events", "disk_usage", "BACKUP_DIR"]
