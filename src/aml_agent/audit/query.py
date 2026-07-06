"""
Audit-log read interface.

Consumed by the eval harness (Phase 9) and monitoring dashboard
(Phase 10). Composable filter function — any subset of predicates
can be combined; unspecified filters are omitted from the WHERE
clause so callers don't have to reason about "None means all".

Implemented as a plain function rather than a class because there's
no configuration to carry between calls: filters are stateless.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from aml_agent.db.models import AuditLog


def query_audit(
    db: Session,
    *,
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    limit: int = 100,
) -> list[AuditLog]:
    """
    Return audit_log rows matching the supplied filters, most recent first.

    Keyword-only args (enforced by `*`) prevent silent argument swaps
    between the four string filters — entity_type, action, and actor
    are all TEXT columns, and a positional call reversing them would
    return the wrong subset silently with no type error.

    Ordering: occurred_at DESC as primary sort (natural review order —
    "what happened most recently"), audit_id DESC as tiebreaker. The
    tiebreaker matters because occurred_at has microsecond resolution;
    two rows can collide, and eval reproducibility depends on
    deterministic ordering.

    Default limit=100 prevents accidental full-table scans from
    dashboard queries. Callers wanting a full sweep pass explicit
    larger values with awareness of the memory cost.
    """
    stmt = select(AuditLog)

    # Conditional filter application — each predicate is added only if
    # its argument was supplied. A monolithic .filter(and_(...)) with
    # `x == arg or True` no-ops would still send the trivially-true
    # clauses to Postgres, wasting planner cycles.
    if entity_type is not None:
        stmt = stmt.where(AuditLog.entity_type == entity_type)
    if entity_id is not None:
        stmt = stmt.where(AuditLog.entity_id == entity_id)
    if action is not None:
        stmt = stmt.where(AuditLog.action == action)
    if actor is not None:
        stmt = stmt.where(AuditLog.actor == actor)
    if after is not None:
        stmt = stmt.where(AuditLog.occurred_at >= after)
    if before is not None:
        stmt = stmt.where(AuditLog.occurred_at < before)

    stmt = stmt.order_by(
        AuditLog.occurred_at.desc(),
        AuditLog.audit_id.desc(),
    ).limit(limit)

    return list(db.execute(stmt).scalars())


def get_alert_history(db: Session, alert_id: int, limit: int = 500) -> list[AuditLog]:
    """
    Convenience wrapper: full audit trail for one alert, in ascending
    chronological order (first-event-first, matching how a reviewer
    reads a case).

    Higher default limit than query_audit because a full investigation
    trace can produce dozens of tool_call + guardrail rows plus the
    orchestrator's own writes — 100 might truncate the middle of a
    single alert's history.

    Ordering is intentionally opposite to query_audit's default:
    reviewers want the story from the beginning; query_audit's callers
    want the latest events.
    """
    stmt = (
        select(AuditLog)
        .where(AuditLog.entity_type == "alert", AuditLog.entity_id == alert_id)
        .order_by(AuditLog.occurred_at.asc(), AuditLog.audit_id.asc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars())