"""Initial schema for Chief Agent.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-01-01

This migration creates every table the system needs. It is written as an
explicit ``Base.metadata.create_all`` against the live bind so the schema can
never drift from the models it is derived from; subsequent migrations should be
generated with ``alembic revision --autogenerate``.
"""

from __future__ import annotations

from alembic import op

from chief_agent.data.schema import Base

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
