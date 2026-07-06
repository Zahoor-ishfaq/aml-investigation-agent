"""
Guardrail execution point.

Runs every registered guardrail against an AgentDecision and aggregates
the results. Aggregation rule: BLOCK > REQUIRES_REVIEW > PASS. If any
single guardrail blocks, the whole decision is blocked. If any requires
review (and none block), the decision is downgraded to review. Only if
all guardrails pass is the agent's original action allowed to proceed.

Every guardrail evaluation writes to audit_log with actor='guardrail'
so a regulator can reconstruct exactly which guardrails ran, in what
order, and why the final aggregate landed on its verdict.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from aml_agent.db.models import AuditLog
from aml_agent.guardrails.base import (
    AgentDecision,
    GuardrailDecision,
    GuardrailResult,
    REGISTRY,
)

# Explicit module imports trigger @register at import time. Same
# rationale as the rule engine runner and tool dispatch — a reviewer
# can grep for these to see the exact guardrail set applied. No
# dynamic discovery, no reflection surprises.
from aml_agent.guardrails import policies  # noqa: F401
from aml_agent.guardrails import validation  # noqa: F401


# Ordering matters for aggregation: BLOCK dominates REQUIRES_REVIEW
# dominates PASS. Kept as an explicit map instead of enum ordering so
# the precedence is self-documenting.
_PRECEDENCE = {
    GuardrailDecision.PASS: 0,
    GuardrailDecision.REQUIRES_REVIEW: 1,
    GuardrailDecision.BLOCK: 2,
}


def _worse(a: GuardrailDecision, b: GuardrailDecision) -> GuardrailDecision:
    """Return whichever decision is more restrictive."""
    return a if _PRECEDENCE[a] >= _PRECEDENCE[b] else b


def evaluate_all(
    db: Session,
    decision: AgentDecision,
    actor: str = "guardrail",
) -> tuple[GuardrailDecision, list[GuardrailResult]]:
    """
    Run every registered guardrail against `decision`, write audit, and
    return (aggregate_verdict, per_guardrail_results).

    Runs all guardrails even after a BLOCK is seen. Rationale: the audit
    trail should show that every relevant policy was evaluated, not just
    the first one that objected. Short-circuiting on first BLOCK would
    hide subsequent objections and make policy analysis harder later.
    """
    results: list[GuardrailResult] = []
    aggregate = GuardrailDecision.PASS

    for guardrail_cls in REGISTRY:
        guardrail = guardrail_cls()
        result = guardrail.evaluate(decision)
        results.append(result)
        aggregate = _worse(aggregate, result.decision)

    _audit(db, decision, aggregate, results, actor)
    return aggregate, results


def _audit(
    db: Session,
    decision: AgentDecision,
    aggregate: GuardrailDecision,
    results: list[GuardrailResult],
    actor: str,
) -> None:
    """
    Append one audit_log row summarising the full guardrail evaluation.

    Structured details capture:
      - the alert being decided
      - what the agent proposed
      - the aggregate verdict
      - every guardrail's individual outcome + reason

    One row per evaluation (not one per guardrail) keeps audit_log
    compact while preserving the full traceable rationale — regulators
    reviewing an alert's history see the full evaluation as a single
    coherent event, matching how they think about it.
    """
    entity_id = decision.alert_id if decision.alert_id else int(
        datetime.now(timezone.utc).timestamp() * 1000
    )
    entity_type = "alert" if decision.alert_id else "guardrail_evaluation"

    db.add(AuditLog(
        entity_type=entity_type,
        entity_id=entity_id,
        action="guardrail_evaluated",
        actor=actor,
        details={
            "proposed_action": decision.action.value,
            "current_status": decision.current_status,
            "severity": decision.severity,
            "aggregate_verdict": aggregate.value,
            "guardrails": [
                {
                    "code": r.guardrail_code,
                    "decision": r.decision.value,
                    "reason": r.reason,
                }
                for r in results
            ],
        },
    ))
    db.commit()