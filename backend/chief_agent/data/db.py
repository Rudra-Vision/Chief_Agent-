"""Database engine and session management.

Default local backend is SQLite so the project runs with zero setup; setting
``DATABASE_URL`` to a PostgreSQL DSN switches to Postgres with no code change.

Timestamps are stored as timezone-aware ISO strings with an IST offset. SQLite
has no native timezone support, so we never rely on the database to normalise
time - see :mod:`chief_agent.timeutil`.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..logging_setup import get_logger
from ..settings import VAR_DIR, get_settings

log = get_logger(__name__, component="database")

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None
_lock = threading.Lock()


def _normalise_url(url: str) -> str:
    if not url:
        (VAR_DIR).mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(VAR_DIR / 'chief_agent.db').as_posix()}"
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine(url: Optional[str] = None, echo: Optional[bool] = None) -> Engine:
    """Process-wide engine (created lazily, thread-safe)."""
    global _engine, _SessionFactory
    with _lock:
        if _engine is not None and url is None:
            return _engine
        settings = get_settings()
        dsn = _normalise_url(url or settings.database_url)
        echo = settings.database_echo if echo is None else echo

        if dsn.startswith("sqlite"):
            db_path = dsn.split("///", 1)[-1]
            if db_path and db_path not in (":memory:", ""):
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                dsn,
                echo=echo,
                future=True,
                connect_args={"check_same_thread": False, "timeout": 30},
            )

            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        else:
            engine = create_engine(dsn, echo=echo, future=True, pool_pre_ping=True, pool_size=10, max_overflow=20)

        _engine = engine
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        log.info("database engine created", context={"dialect": engine.dialect.name})
        return _engine


def get_session_factory() -> sessionmaker:
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on exception."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def session_dependency() -> Iterator[Session]:
    """FastAPI dependency."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all tables (used in tests and first-run bootstrap).

    Production deployments should use Alembic migrations (``scripts/migrate.py``).
    """
    from .schema import Base

    eng = engine or get_engine()
    Base.metadata.create_all(eng)
    log.info("database schema ensured", context={"tables": len(Base.metadata.tables)})


def healthcheck() -> dict:
    from sqlalchemy import text

    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        return {"ok": True, "engine": get_engine().dialect.name}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": str(exc)}


def reset_engine() -> None:
    """Test helper: drop the cached engine/session factory."""
    global _engine, _SessionFactory
    with _lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:  # pragma: no cover
                pass
        _engine = None
        _SessionFactory = None


__all__ = [
    "get_engine",
    "get_session_factory",
    "session_scope",
    "session_dependency",
    "init_db",
    "healthcheck",
    "reset_engine",
]
