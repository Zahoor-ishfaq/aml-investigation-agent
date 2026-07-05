"""
Rule execution engine.

Orchestrates loading, running, and persisting output of all registered
rules. Rules themselves are pure logic (yield candidates); this module
owns dedup, batching, and audit_log integration — separation keeps rules
trivially unit-testable and prevents each new rule from re-implementing
the same DB plumbing.
"""

import logging
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from aml_agent.db.models import Alert, AuditLog
from aml_agent.rule_engine.base import AlertCandidate, REGISTRY

# Importing the rules modules triggers @register, populating REGISTRY.
# Explicit imports (not dynamic discovery via pkgutil) keep the rule set
# auditable — a reviewer can grep for @register calls to see every rule
# that will run, no reflection surprises.
from aml_agent.rule_engine.rules import structuring  # noqa: F401
from aml_agent.rule_engine.rules import fan  # noqa: F401
from aml_agent.rule_engine.rules import rapid_movement  # noqa: F401

logger = logging.getLogger("aml_agent.rule_engine")

# Bulk insert batch size. Aligns with ingestion script's BATCH_SIZE.
BATCH_SIZE = 5000


def _write_audit(db: Session, action: str, details: dict) -> None:
    """Append an audit_log row for engine lifecycle events.

    entity_type='rule_engine_run' — engine runs aren't first-class entities
    in the schema (no runs table), timestamp-int gives a stable identifier
    consistent with the ingestion script's approach.
    """
    db.add(AuditLog(
        entity_type="rule_engine_run",
        entity_id=int(datetime.now(timezone.utc).timestamp() * 1000),
        action=action,
        actor="rule_engine",
        details=details,
    ))
    db.commit()


def _persist_candidates(db: Session, candidates: Iterable[AlertCandidate]) -> int:
    """Bulk-insert alert candidates with ON CONFLICT DO NOTHING.

    Uses Postgres-specific insert().on_conflict_do_nothing keyed on the
    unique constraint uq_alerts_rule_code_transaction_id. DB-level dedup
    is atomic; a Python-side pre-check would race with concurrent engine
    runs. Returns the number of rows actually inserted (excludes skipped
    duplicates).
    """
    batch = []
    inserted = 0

    def flush() -> int:
        if not batch:
            return 0
        stmt = pg_insert(Alert).values(batch).on_conflict_do_nothing(
            constraint="uq_alerts_rule_code_transaction_id",
        )
        # rowcount reflects rows actually inserted after conflict resolution.
        result = db.execute(stmt)
        db.commit()
        return result.rowcount or 0

    for cand in candidates:
        batch.append({
            "transaction_id": cand.transaction_id,
            "account_id": cand.account_id,
            "source": "rule_engine",
            "rule_code": cand.rule_code,
            "severity": cand.severity,
            "narrative": cand.narrative,
        })
        if len(batch) >= BATCH_SIZE:
            inserted += flush()
            batch = []
    inserted += flush()
    return inserted


def run_all_rules(db: Session, dry_run: bool = False) -> dict[str, int]:
    """Instantiate and execute every registered rule.

    One transaction per rule (commit after each) rather than one big
    transaction across all rules: partial progress is preferable to
    all-or-nothing at this scale — if rule 3 hits a bug, rules 1 and 2's
    alerts stay persisted, and reruns are safe thanks to the unique
    constraint.

    Returns {rule_code: candidates_written} for the caller / audit log.
    """
    _write_audit(db, "rule_engine_started", {"dry_run": dry_run, "rule_count": len(REGISTRY)})

    per_rule_counts: dict[str, int] = {}

    for rule_cls in REGISTRY:
        rule = rule_cls()  # instantiate with defaults — parameter tuning deferred to step 9
        logger.info("Running rule %s", rule.code)

        candidates = list(rule.evaluate(db))
        logger.info("Rule %s produced %d candidates", rule.code, len(candidates))

        if dry_run:
            per_rule_counts[rule.code] = len(candidates)
            continue

        n = _persist_candidates(db, candidates)
        per_rule_counts[rule.code] = n
        _write_audit(db, "rule_completed", {
            "rule_code": rule.code,
            "candidates": len(candidates),
            "inserted": n,
        })
        logger.info("Rule %s inserted %d alerts (deduped %d)", rule.code, n, len(candidates) - n)

    _write_audit(db, "rule_engine_completed", {
        "dry_run": dry_run,
        "per_rule_inserted": per_rule_counts,
    })
    return per_rule_counts