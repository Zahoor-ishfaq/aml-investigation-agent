"""add pseudonym_map table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-05 22:15:00.000000

Purpose: reversible mapping of real values (customer names, external refs)
to opaque tokens (CUST_xxx, AEXT_xxx) that can safely be included in LLM
prompts. Same-DB deployment chosen for simplicity; production defense-in-depth
would isolate this in a separate DB with its own access controls.

Design:
- Composite PK (entity_type, real_value) makes get-or-create idempotent
  under concurrent lookups.
- Unique constraint on (entity_type, token) enforces per-type token
  uniqueness so depseudonymization is unambiguous. Cross-type collisions
  are impossible by construction (tokens carry a type prefix).
- No FK to source tables: source rows (customers/accounts) can be deleted
  without cascading the mapping. The mapping is a lookup, not a relationship.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'pseudonym_map',
        sa.Column('entity_type', sa.Text(), nullable=False),
        sa.Column('real_value', sa.Text(), nullable=False),
        sa.Column('token', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('entity_type', 'real_value'),
        sa.UniqueConstraint('entity_type', 'token',
                            name='uq_pseudonym_map_entity_type_token'),
    )
    # Reverse-lookup index: depseudonymize() queries by (entity_type, token).
    # The unique constraint above creates one implicitly, but naming it
    # explicitly documents intent and lets us drop it independently later.


def downgrade() -> None:
    op.drop_table('pseudonym_map')