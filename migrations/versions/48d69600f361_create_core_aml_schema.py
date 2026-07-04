"""create core aml schema

Revision ID: 48d69600f361
Revises: 
Create Date: 2026-07-04 17:58:50.198104

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '48d69600f361'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### Alembic autogenerate, patched by hand ###
    # Fixes applied beyond raw autogenerate output:
    #   1. server_default restored on enum/currency columns (autogenerate
    #      captures NOT NULL but drops DEFAULT) — without this, inserts
    #      omitting these columns fail instead of using the intended default.
    #   2. Indexes added at the end — autogenerate only diffs table/column
    #      structure, not indexes. These match sql/001_schema.sql and exist
    #      so the rule engine's per-account/time-window queries (step 2)
    #      don't hit full table scans as transaction volume grows.
    op.create_table('audit_log',
    sa.Column('audit_id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('entity_type', sa.Text(), nullable=False),
    sa.Column('entity_id', sa.BigInteger(), nullable=False),
    sa.Column('action', sa.Text(), nullable=False),
    sa.Column('actor', sa.Text(), nullable=False),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('occurred_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('audit_id')
    )
    op.create_table('customers',
    sa.Column('customer_id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('external_ref', sa.Text(), nullable=False),
    sa.Column('customer_type', sa.Enum('individual', 'business', name='customer_type'), nullable=False),
    sa.Column('full_name', sa.Text(), nullable=False),
    sa.Column('date_of_birth', sa.Date(), nullable=True),
    sa.Column('country_code', sa.String(length=2), nullable=False),
    # FIX: default 'low' — FATF risk-based approach assigns baseline risk
    # at onboarding rather than leaving it unset.
    sa.Column('risk_rating', sa.Enum('low', 'medium', 'high', name='customer_risk_rating'), server_default='low', nullable=False),
    sa.Column('onboarding_date', sa.Date(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('customer_id'),
    sa.UniqueConstraint('external_ref')
    )
    op.create_table('accounts',
    sa.Column('account_id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('external_ref', sa.Text(), nullable=False),
    sa.Column('customer_id', sa.BigInteger(), nullable=False),
    sa.Column('account_type', sa.Enum('checking', 'savings', 'business', 'correspondent', name='account_type'), nullable=False),
    # FIX: default 'active' — matches DDL.
    sa.Column('status', sa.Enum('active', 'dormant', 'frozen', 'closed', name='account_status'), server_default='active', nullable=False),
    sa.Column('opened_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
    # FIX: default 'USD' — matches DDL.
    sa.Column('currency', sa.String(length=3), server_default='USD', nullable=False),
    sa.CheckConstraint("(status = 'closed' AND closed_at IS NOT NULL) OR (status != 'closed' AND closed_at IS NULL)", name='chk_closed_consistency'),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ),
    sa.PrimaryKeyConstraint('account_id'),
    sa.UniqueConstraint('external_ref')
    )
    op.create_table('transactions',
    sa.Column('transaction_id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('external_ref', sa.Text(), nullable=False),
    sa.Column('sender_account_id', sa.BigInteger(), nullable=False),
    sa.Column('receiver_account_id', sa.BigInteger(), nullable=False),
    sa.Column('amount', sa.Numeric(precision=18, scale=2), nullable=False),
    # FIX: default 'USD' — matches DDL.
    sa.Column('currency', sa.String(length=3), server_default='USD', nullable=False),
    sa.Column('channel', sa.Enum('wire', 'ach', 'card', 'cash', 'internal_transfer', name='transaction_channel'), nullable=False),
    sa.Column('executed_at', sa.DateTime(timezone=True), nullable=False),
    # FIX: default False — ground-truth label defaults to legitimate absent
    # AMLSim's explicit labeling during ingestion.
    sa.Column('is_laundering_ground_truth', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.CheckConstraint('amount > 0', name='chk_positive_amount'),
    sa.CheckConstraint('sender_account_id != receiver_account_id', name='chk_not_self_transfer'),
    sa.ForeignKeyConstraint(['receiver_account_id'], ['accounts.account_id'], ),
    sa.ForeignKeyConstraint(['sender_account_id'], ['accounts.account_id'], ),
    sa.PrimaryKeyConstraint('transaction_id'),
    sa.UniqueConstraint('external_ref')
    )
    op.create_table('alerts',
    sa.Column('alert_id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('transaction_id', sa.BigInteger(), nullable=False),
    sa.Column('account_id', sa.BigInteger(), nullable=False),
    sa.Column('source', sa.Enum('rule_engine', 'investigation_agent', name='alert_source'), nullable=False),
    sa.Column('rule_code', sa.Text(), nullable=True),
    # FIX: default 'open' — entry state of the FATF-aligned alert lifecycle.
    sa.Column('status', sa.Enum('open', 'under_review', 'escalated', 'closed_false_positive', 'closed_sar_filed', 'closed_no_action', name='alert_status'), server_default='open', nullable=False),
    sa.Column('severity', sa.SmallInteger(), nullable=False),
    sa.Column('narrative', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('severity BETWEEN 1 AND 5', name='chk_severity_range'),
    sa.ForeignKeyConstraint(['account_id'], ['accounts.account_id'], ),
    sa.ForeignKeyConstraint(['transaction_id'], ['transactions.transaction_id'], ),
    sa.PrimaryKeyConstraint('alert_id')
    )

    # --- Indexes (not captured by autogenerate; match sql/001_schema.sql) ---
    op.create_index('idx_accounts_customer_id', 'accounts', ['customer_id'])
    op.create_index('idx_txn_sender_time', 'transactions', ['sender_account_id', 'executed_at'])
    op.create_index('idx_txn_receiver_time', 'transactions', ['receiver_account_id', 'executed_at'])
    op.create_index('idx_txn_executed_at', 'transactions', ['executed_at'])
    op.create_index('idx_alerts_status', 'alerts', ['status'])
    op.create_index('idx_alerts_account_id', 'alerts', ['account_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('idx_alerts_account_id', table_name='alerts')
    op.drop_index('idx_alerts_status', table_name='alerts')
    op.drop_index('idx_txn_executed_at', table_name='transactions')
    op.drop_index('idx_txn_receiver_time', table_name='transactions')
    op.drop_index('idx_txn_sender_time', table_name='transactions')
    op.drop_index('idx_accounts_customer_id', table_name='accounts')
    op.drop_table('alerts')
    op.drop_table('transactions')
    op.drop_table('accounts')
    op.drop_table('customers')
    op.drop_table('audit_log')
    # ### end Alembic commands ###