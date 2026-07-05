"""
Pseudonymization tokenizer.

Two operations:
    pseudonymize(db, entity_type, real_value) -> token
    depseudonymize(db, entity_type, token) -> real_value | None

Same real_value always yields the same token — critical for entity
continuity across LLM turns and separate agent sessions. New values get
a fresh random token; retries protect against the vanishingly rare
collision at the current scale.

The (entity_type, real_value) primary key makes get-or-create atomic
under concurrent lookups: two threads inserting the same pair see one
succeed and one hit the PK conflict, at which point the loser re-reads
the existing token. No locking required.
"""

import secrets
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aml_agent.pseudonymization.models import PseudonymMap


# Prefix per entity type. Tokens are self-describing when they appear in
# LLM context — the model sees "CUST_a1b2c3d4" and can reason about it
# as a customer identifier without knowing the real value. Cross-type
# collisions are impossible by construction because prefixes differ.
PREFIXES: dict[str, str] = {
    "customer_name": "CUST",
    "customer_external_ref": "CEXT",
    "account_external_ref": "AEXT",
    "transaction_external_ref": "TEXT",
}

# 4 hex bytes = 8 chars = 2^32 possible tokens. At 10K entities, collision
# probability per generation is ~2.3e-6 — one retry loop handles it.
# For >1M entities per type, increase to 6 bytes.
_TOKEN_BYTES = 4
_MAX_COLLISION_RETRIES = 5


def _generate_token(entity_type: str) -> str:
    """Generate a fresh <PREFIX>_<hex> token. Uses secrets (CSPRNG) not
    random — token unpredictability limits an attacker's ability to
    enumerate or forge tokens if they gain access to LLM logs."""
    prefix = PREFIXES.get(entity_type)
    if prefix is None:
        raise ValueError(f"unknown entity_type: {entity_type}")
    return f"{prefix}_{secrets.token_hex(_TOKEN_BYTES)}"


def pseudonymize(db: Session, entity_type: str, real_value: str) -> str:
    """
    Return the token for (entity_type, real_value), creating one if absent.

    Idempotent: repeated calls with the same inputs return the same token.
    Uses INSERT ... ON CONFLICT DO NOTHING then SELECT to make this
    concurrency-safe without an explicit transaction: two concurrent
    creators can't race because PK conflict is resolved atomically at
    the DB layer.
    """
    if real_value is None:
        raise ValueError("real_value cannot be None")

    for _ in range(_MAX_COLLISION_RETRIES):
        token = _generate_token(entity_type)
        stmt = pg_insert(PseudonymMap).values(
            entity_type=entity_type,
            real_value=str(real_value),
            token=token,
        ).on_conflict_do_nothing(index_elements=["entity_type", "real_value"])

        try:
            db.execute(stmt)
            db.commit()
        except IntegrityError:
            # Token collided within the type namespace (uq_ constraint).
            # Rare (see collision math above); retry with a new token.
            db.rollback()
            continue

        # Read back the token that landed — either the one we just inserted
        # or a pre-existing one from a concurrent creator or prior call.
        result = db.execute(
            select(PseudonymMap.token).where(
                PseudonymMap.entity_type == entity_type,
                PseudonymMap.real_value == str(real_value),
            )
        ).scalar_one()
        return result

    raise RuntimeError(
        f"pseudonymize: {_MAX_COLLISION_RETRIES} consecutive token collisions — "
        f"consider increasing _TOKEN_BYTES"
    )


def depseudonymize(db: Session, entity_type: str, token: str) -> str | None:
    """
    Return the real_value for (entity_type, token), or None if not found.

    None is a valid outcome (not an error): the caller may pass tokens
    the LLM hallucinated that don't correspond to any real entity. The
    human-review path (substep 3.4) uses this to resolve agent output
    back to real customers.
    """
    result = db.execute(
        select(PseudonymMap.real_value).where(
            PseudonymMap.entity_type == entity_type,
            PseudonymMap.token == token,
        )
    ).scalar_one_or_none()
    return result


def pseudonymize_batch(
    db: Session, entity_type: str, real_values: Iterable[str]
) -> dict[str, str]:
    """
    Batch variant: return {real_value: token} for a list of inputs.

    Load-time optimization for pseudonymizing whole result sets (e.g. a
    query returning 500 customers to hand to the LLM). One SELECT for
    already-mapped values + one INSERT per new value beats N round-trips
    through pseudonymize().
    """
    real_values = list({str(v) for v in real_values if v is not None})
    if not real_values:
        return {}

    # Fetch existing mappings in one query.
    existing = {
        row.real_value: row.token
        for row in db.execute(
            select(PseudonymMap.real_value, PseudonymMap.token).where(
                PseudonymMap.entity_type == entity_type,
                PseudonymMap.real_value.in_(real_values),
            )
        )
    }

    # Create tokens for the remainder via the single-item function so
    # collision retry logic is shared, not duplicated.
    for v in real_values:
        if v not in existing:
            existing[v] = pseudonymize(db, entity_type, v)

    return existing