#!/usr/bin/env python
"""Apply database migrations.

Usage:
    python scripts/migrate.py            # apply all migrations (upgrade head)
    python scripts/migrate.py --stamp    # mark the DB as current without running
    python scripts/migrate.py --init     # create the schema directly (first run)

For brand-new installations ``--init`` is the quickest path: it creates every
table from the SQLAlchemy models and then stamps Alembic at head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chief Agent database migrations")
    parser.add_argument("--init", action="store_true", help="create all tables directly and stamp head")
    parser.add_argument("--stamp", action="store_true", help="stamp the database at head without migrating")
    parser.add_argument("--revision", default="head", help="target revision (default: head)")
    args = parser.parse_args()

    from alembic import command
    from alembic.config import Config

    from chief_agent.data.db import init_db
    from chief_agent.logging_setup import configure_logging
    from chief_agent.settings import get_settings

    configure_logging()
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "backend" / "chief_agent" / "data" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)

    if args.init:
        print("Creating the schema directly from the models ...")
        init_db()
        command.stamp(cfg, "head")
        print("Schema created and stamped at head.")
        return 0

    if args.stamp:
        command.stamp(cfg, args.revision)
        print(f"Stamped at {args.revision}.")
        return 0

    command.upgrade(cfg, args.revision)
    print(f"Migrated to {args.revision}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
