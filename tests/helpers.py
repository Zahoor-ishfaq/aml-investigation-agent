"""
Test data seeding helpers.

Keeps tests focused on the assertion, not on boilerplate row construction.
Every helper commits within the test's SAVEPOINT — the outer transaction
rollback still cleans up.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from aml_agent.db.models import Account, Customer, Transaction


def make_customer(db: Session, ext: str) -> Customer:
    c = Customer(
        external_ref=ext,
        customer_type="individual",
        full_name=f"Test {ext}",
        country_code="SA",
        risk_rating="low",
        onboarding_date=datetime(2024, 1, 1).date(),
    )
    db.add(c)
    db.flush()  # populate customer_id without committing
    return c


def make_account(db: Session, ext: str, customer_id: int) -> Account:
    a = Account(
        external_ref=ext,
        customer_id=customer_id,
        account_type="checking",
        status="active",
        opened_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        currency="USD",
    )
    db.add(a)
    db.flush()
    return a


def make_txn(
    db: Session,
    ext: str,
    sender_id: int,
    receiver_id: int,
    amount: float,
    executed_at: datetime,
    is_laundering: bool = False,
) -> Transaction:
    t = Transaction(
        external_ref=ext,
        sender_account_id=sender_id,
        receiver_account_id=receiver_id,
        amount=amount,
        currency="USD",
        channel="wire",
        executed_at=executed_at,
        is_laundering_ground_truth=is_laundering,
    )
    db.add(t)
    db.flush()
    return t


def seed_pair(db: Session, tag: str) -> tuple[int, int]:
    """Convenience: create two accounts (each with own customer) and return
    their account_ids. Most rule tests need at least a sender and receiver."""
    c1 = make_customer(db, f"{tag}-c1")
    c2 = make_customer(db, f"{tag}-c2")
    a1 = make_account(db, f"{tag}-a1", c1.customer_id)
    a2 = make_account(db, f"{tag}-a2", c2.customer_id)
    return a1.account_id, a2.account_id