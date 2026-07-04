"""
SQLAlchemy ORM models for the AML investigation schema.

These models are the source of truth Alembic diffs against to generate
migrations (see migrations/versions/). They mirror sql/001_schema.sql
exactly — that file remains as a human-readable reference copy of the
schema, but this file is what actually drives migrations and the app's
DB access layer.

Design note: enums are defined as Python Enum classes and mapped via
SQLAlchemy's Enum type, which creates a native Postgres ENUM type —
matches the raw DDL's CREATE TYPE statements and gets the same DB-level
validation (invalid values rejected at insert, not just in application code).
"""

import enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base — every model inherits from this so Alembic
    can discover all tables via Base.metadata in a single place."""
    pass


# ----------------------------------------------------------------------------
# Enums — mirror the CREATE TYPE statements in sql/001_schema.sql exactly.
# Defined once here and reused in the relevant Column definitions below.
# ----------------------------------------------------------------------------

class CustomerType(str, enum.Enum):
    INDIVIDUAL = "individual"
    BUSINESS = "business"


class CustomerRiskRating(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AccountType(str, enum.Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    BUSINESS = "business"
    CORRESPONDENT = "correspondent"


class AccountStatus(str, enum.Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    FROZEN = "frozen"
    CLOSED = "closed"


class TransactionChannel(str, enum.Enum):
    WIRE = "wire"
    ACH = "ach"
    CARD = "card"
    CASH = "cash"
    INTERNAL_TRANSFER = "internal_transfer"


class AlertStatus(str, enum.Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    ESCALATED = "escalated"
    CLOSED_FALSE_POSITIVE = "closed_false_positive"
    CLOSED_SAR_FILED = "closed_sar_filed"
    CLOSED_NO_ACTION = "closed_no_action"


class AlertSource(str, enum.Enum):
    RULE_ENGINE = "rule_engine"
    INVESTIGATION_AGENT = "investigation_agent"


# ----------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------

class Customer(Base):
    """
    KYC record. risk_rating drives rule-engine thresholds — FATF
    Recommendation 10 requires risk-based due diligence, so not every
    customer is scrutinized with the same intensity.
    """
    __tablename__ = "customers"

    customer_id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_ref = Column(Text, unique=True, nullable=False)
    customer_type = Column(
        Enum(CustomerType, name="customer_type", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    full_name = Column(Text, nullable=False)
    date_of_birth = Column(Date, nullable=True)  # null allowed: businesses have no DOB
    country_code = Column(String(2), nullable=False)
    risk_rating = Column(
        Enum(CustomerRiskRating, name="customer_risk_rating", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=CustomerRiskRating.LOW,
    )
    onboarding_date = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    accounts = relationship("Account", back_populates="customer")


# ----------------------------------------------------------------------------
# Accounts
# ----------------------------------------------------------------------------

class Account(Base):
    """
    One customer can hold multiple accounts. status/closed_at consistency
    enforced via CheckConstraint — mirrors the raw DDL's chk_closed_consistency,
    preventing a closed account from lacking a closed_at timestamp (or vice versa).
    """
    __tablename__ = "accounts"

    account_id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_ref = Column(Text, unique=True, nullable=False)
    customer_id = Column(BigInteger, ForeignKey("customers.customer_id"), nullable=False)
    account_type = Column(
        Enum(AccountType, name="account_type", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    status = Column(
        Enum(AccountStatus, name="account_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=AccountStatus.ACTIVE,
    )
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    currency = Column(String(3), nullable=False, default="USD")

    __table_args__ = (
        CheckConstraint(
            "(status = 'closed' AND closed_at IS NOT NULL) OR (status != 'closed' AND closed_at IS NULL)",
            name="chk_closed_consistency",
        ),
    )

    customer = relationship("Customer", back_populates="accounts")


# ----------------------------------------------------------------------------
# Transactions
# ----------------------------------------------------------------------------

class Transaction(Base):
    """
    Core transaction graph, generated by AMLSim. sender/receiver reference
    accounts (not customers) because AML typologies operate at account
    granularity — e.g. structuring across multiple accounts of one customer.

    is_laundering_ground_truth is the AMLSim-injected label. It must never
    be read by the rule engine or agent (would leak the answer) — it exists
    solely for the eval harness (step 9) to score detection accuracy.
    """
    __tablename__ = "transactions"

    transaction_id = Column(BigInteger, primary_key=True, autoincrement=True)
    external_ref = Column(Text, unique=True, nullable=False)
    sender_account_id = Column(BigInteger, ForeignKey("accounts.account_id"), nullable=False)
    receiver_account_id = Column(BigInteger, ForeignKey("accounts.account_id"), nullable=False)
    amount = Column(Numeric(18, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    channel = Column(
        Enum(TransactionChannel, name="transaction_channel", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    executed_at = Column(DateTime(timezone=True), nullable=False)
    is_laundering_ground_truth = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint("amount > 0", name="chk_positive_amount"),
        CheckConstraint("sender_account_id != receiver_account_id", name="chk_not_self_transfer"),
    )


# ----------------------------------------------------------------------------
# Alerts
# ----------------------------------------------------------------------------

class Alert(Base):
    """
    Written by the rule engine (step 2) and Investigation Agent (step 7).
    status lifecycle mirrors FATF's suspicious-activity workflow:
    detection -> review -> decision -> (optional) SAR filing.
    """
    __tablename__ = "alerts"

    alert_id = Column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id = Column(BigInteger, ForeignKey("transactions.transaction_id"), nullable=False)
    account_id = Column(BigInteger, ForeignKey("accounts.account_id"), nullable=False)
    source = Column(
        Enum(AlertSource, name="alert_source", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    rule_code = Column(Text, nullable=True)  # populated only when source = rule_engine
    status = Column(
        Enum(AlertStatus, name="alert_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=AlertStatus.OPEN,
    )
    severity = Column(SmallInteger, nullable=False)
    narrative = Column(Text, nullable=True)  # filled by agent's case file (step 7)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("severity BETWEEN 1 AND 5", name="chk_severity_range"),
    )


# ----------------------------------------------------------------------------
# Audit log
# ----------------------------------------------------------------------------

class AuditLog(Base):
    """
    Append-only, immutable trail — architecture requires every pipeline
    step to log here. Immutability itself is enforced at the DB layer via
    trigger (see sql/002_audit_immutability.sql / its Alembic equivalent),
    not in this model — SQLAlchemy can't express "reject UPDATE/DELETE"
    declaratively, so that logic lives in a migration's raw SQL instead.
    """
    __tablename__ = "audit_log"

    audit_id = Column(BigInteger, primary_key=True, autoincrement=True)
    entity_type = Column(Text, nullable=False)
    entity_id = Column(BigInteger, nullable=False)
    action = Column(Text, nullable=False)
    actor = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())