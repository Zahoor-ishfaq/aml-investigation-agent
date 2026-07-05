"""
Pseudonymization service.

Wraps the low-level tokenizer with domain-specific operations the tool
layer (Phase 4) calls before handing data to the LLM. Every function
takes an ORM object and returns a plain dict — the dict is what crosses
the LLM boundary, so any field kept in it is field the LLM will see.

Rule of thumb: if it's directly identifying (name, external ref) → token.
If it's quasi-identifying (DOB) → bucket. If it's a reasoning signal
(amount, country, timestamp) → keep raw.
"""

from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from aml_agent.db.models import Account, Customer, Transaction
from aml_agent.pseudonymization.tokenizer import pseudonymize


# Age buckets balance signal preservation vs. re-identification risk.
# Bands of 10 years give the agent "elderly / working-age / young"
# discrimination without exposing a linkable birth year.
def _age_bucket(dob: date | None) -> str | None:
    """Convert exact DOB to a 10-year age band. Returns None if DOB
    absent (businesses have no DOB). Uses a fixed reference year for
    determinism — pseudonymized output shouldn't drift with wall clock."""
    if dob is None:
        return None
    # Reference year matches the simulation base_date. In production this
    # would be `date.today().year`, but that would make test output
    # nondeterministic across the year boundary.
    ref_year = 2024
    age = ref_year - dob.year
    if age < 0:
        return "unknown"
    lo = (age // 10) * 10
    return f"{lo}-{lo + 9}"


def pseudonymize_customer(db: Session, customer: Customer) -> dict[str, Any]:
    """
    Return an LLM-safe view of a customer.

    full_name is tokenized (direct PII). date_of_birth is bucketed
    (quasi-identifier — exact date is re-linkable). external_ref is
    tokenized (linkable identifier). Country / risk_rating / customer_type
    are kept raw — these are precisely the signals the agent must reason
    over (FATF Recommendation 10's risk-based due diligence).
    """
    return {
        "customer_token": pseudonymize(db, "customer_external_ref", customer.external_ref),
        "name_token": pseudonymize(db, "customer_name", customer.full_name),
        "customer_type": customer.customer_type.value if hasattr(customer.customer_type, "value") else customer.customer_type,
        "age_bucket": _age_bucket(customer.date_of_birth),
        "country_code": customer.country_code,
        "risk_rating": customer.risk_rating.value if hasattr(customer.risk_rating, "value") else customer.risk_rating,
        "onboarding_date": customer.onboarding_date.isoformat() if customer.onboarding_date else None,
    }


def pseudonymize_account(db: Session, account: Account) -> dict[str, Any]:
    """
    Return an LLM-safe view of an account.

    external_ref is tokenized. Account type / status / currency are kept
    raw — dormant-then-active or foreign-currency accounts are legitimate
    AML signals the agent needs to reason about.
    """
    return {
        "account_token": pseudonymize(db, "account_external_ref", account.external_ref),
        "account_type": account.account_type.value if hasattr(account.account_type, "value") else account.account_type,
        "status": account.status.value if hasattr(account.status, "value") else account.status,
        "currency": account.currency,
        "opened_at": account.opened_at.isoformat() if account.opened_at else None,
        "closed_at": account.closed_at.isoformat() if account.closed_at else None,
    }


def pseudonymize_transaction(db: Session, transaction: Transaction) -> dict[str, Any]:
    """
    Return an LLM-safe view of a transaction.

    sender/receiver need to become account tokens so the agent can reason
    about who-transacted-with-whom without seeing real account refs. We
    look up each account row and tokenize its external_ref (rather than
    tokenizing the internal integer account_id) — this keeps the token
    namespace stable if internal IDs ever get regenerated. Costs two
    extra SELECTs per transaction; acceptable for the single-object path.

    is_laundering_ground_truth is deliberately NOT included: it exists
    only for the eval harness. Including it here would leak the answer
    into the LLM context.
    """
    sender = db.execute(
        select(Account).where(Account.account_id == transaction.sender_account_id)
    ).scalar_one()
    receiver = db.execute(
        select(Account).where(Account.account_id == transaction.receiver_account_id)
    ).scalar_one()

    return {
        "transaction_token": pseudonymize(db, "transaction_external_ref", transaction.external_ref),
        "sender_account_token": pseudonymize(db, "account_external_ref", sender.external_ref),
        "receiver_account_token": pseudonymize(db, "account_external_ref", receiver.external_ref),
        "amount": float(transaction.amount),
        "currency": transaction.currency,
        "channel": transaction.channel.value if hasattr(transaction.channel, "value") else transaction.channel,
        "executed_at": transaction.executed_at.isoformat() if transaction.executed_at else None,
    }