"""
Audit-log write helper.

Centralises the pattern currently duplicated across the ingestion
script, rule engine runner, tool dispatch, guardrail engine, and
orchestrator. New code should call write_audit() instead of hand-
building AuditLog rows. Existing call sites remain unchanged — the
schema and semantics are identical, so this is non-breaking.

Reference: audit_log immutability is enforced at DB layer (see
migration c3d4e5f6a7b8). This helper doesn't add any application-level
enforcement — the DB is the trust boundary.
"""

from typing import Any, Optional

from sqlalchemy.orm import Session

from aml_agent.db.models import AuditLog


def write_audit(
    db: Session,
    *,
    entity_type: str,
    entity_id: int,
    action: str,
    actor: str,
    details: Optional[dict[str, Any]] = None,
) -> AuditLog:
    """
    Append one row to audit_log and commit.

    Keyword-only args (enforced by the `*`) prevent silent swaps between
    same-typed fields — entity_type and action are both strings, and a
    positional call reversing them would land the wrong value in each
    column without any type error.

    The commit is inline because every existing call site already
    commits immediately after building the row. Callers wanting batch
    commits should skip this helper and use db.add(AuditLog(...))
    directly; the helper is the "single-row atomic audit" path.

    Returns the persisted AuditLog so callers can pick up audit_id
    if they need to correlate a follow-up entry (rare, but cheap to
    provide).
    """
    row = AuditLog(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        actor=actor,
        details=details,
    )
    db.add(row)
    db.commit()
    # Refresh so audit_id and server-side defaults (occurred_at) populate
    # on the returned object — callers otherwise see None for these until
    # the row is next queried.
    db.refresh(row)
    return row