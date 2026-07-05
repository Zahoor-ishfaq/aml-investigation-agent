"""add unique constraint on alerts rule_code + transaction_id

Revision ID: a1b2c3d4e5f6
Revises: 48d69600f361
Create Date: 2026-07-05 15:30:00.000000

Purpose: prevent duplicate alerts when the rule engine is re-run.
DB-layer enforcement is atomic; a pre-check in Python could race with
a concurrent engine run and let both writes through.

Constraint scope: (rule_code, transaction_id). Same transaction can
generate alerts from multiple rules (e.g. structuring + fan-in), which
we want. Same rule matching the same transaction twice is a bug.

Nullable rule_code: agent-originated alerts (source = investigation_agent)
have rule_code = NULL. Postgres treats NULLs as distinct in UNIQUE
constraints — two agent alerts on the same tx would NOT be caught here.
Acceptable: agent alerts have their own dedup logic in step 7.
"""
from typing import Sequence, Union

from alembic import op

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '48d69600f361'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        'uq_alerts_rule_code_transaction_id',
        'alerts',
        ['rule_code', 'transaction_id'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_alerts_rule_code_transaction_id',
        'alerts',
        type_='unique',
    )