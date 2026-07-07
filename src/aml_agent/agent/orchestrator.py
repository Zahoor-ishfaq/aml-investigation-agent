"""
End-to-end agent orchestrator.

Wires together every prior phase into one call per alert:
  Phase 3 (pseudonymization) — mask account/transaction refs
  Phase 5 (RAG) — retrieved implicitly via a future rag tool; not
                  wired here yet, prompt already tells the agent about it
  Phase 4 (tools) — invoked inside InvestigationSession
  Phase 6 (guardrails) — evaluate the agent's parsed decision
  Phase 1 audit_log — investigation_completed row per alert

Safe-default policy: any downstream failure (parse fail, guardrail BLOCK,
session iteration cap without decision) degrades to `escalate` with a
diagnostic narrative. The whole point of the guardrail layer is that
model failures land in human review, not in silent close. Never
auto-close on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from aml_agent.agent.case_file import build_agent_decision
from aml_agent.agent.prompts import SYSTEM_PROMPT, build_initial_user_message
from aml_agent.agent.session import InvestigationSession, ToolCallRecord
from aml_agent.db.models import Account, Alert, AuditLog, Transaction
from aml_agent.guardrails.base import AgentAction, AgentDecision, GuardrailDecision
from aml_agent.guardrails.engine import evaluate_all
from aml_agent.pseudonymization.tokenizer import pseudonymize


logger = logging.getLogger("aml_agent.agent.orchestrator")


@dataclass
class InvestigationOutcome:
    """
    Full trace of one agent-driven investigation. Kept structurally
    rich (not just a status enum) so the eval harness (Phase 9) has
    everything needed for offline scoring without re-running the loop.
    """
    alert_id: int
    applied_status: str                         # what we actually set in DB
    agent_decision: Optional[AgentDecision]     # parsed model output; None if parse failed
    guardrail_verdict: Optional[GuardrailDecision]  # None if we never reached guardrails
    guardrail_reasons: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stopped_reason: str = ""                    # from InvestigationSession
    case_file_parse_ok: bool = False
    diagnostic: str = ""                        # human-readable summary of why we landed here
    # Per-Groq-call token usage from the ReAct loop. Carried all the way
    # out to the eval harness so cost-per-alert can be broken down by
    # iteration rather than only seen as an opaque total — this is what
    # lets us tell "resent history is dominating cost" apart from
    # "one turn is unusually expensive" (e.g. an oversized tool result).
    token_usage: list[dict[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Status mapping: aggregate guardrail verdict + agent action -> DB status
#
# Kept as an explicit function rather than an inline if/else so the policy
# is easy to review in one place and testable in isolation.
# ---------------------------------------------------------------------------

_ACTION_TO_STATUS: dict[AgentAction, str] = {
    AgentAction.ESCALATE: "escalated",
    AgentAction.CLOSE_FALSE_POSITIVE: "closed_false_positive",
    AgentAction.CLOSE_NO_ACTION: "closed_no_action",
    AgentAction.CLOSE_SAR_FILED: "closed_sar_filed",
}


def _resolve_final_status(
    verdict: GuardrailDecision,
    decision: AgentDecision,
) -> str:
    """
    Map (guardrail verdict, agent action) to the alert_status value we
    persist. See the substep 7.5 policy summary:
      PASS -> apply agent's action verbatim
      REQUIRES_REVIEW -> set to under_review regardless of action
      BLOCK -> force escalate (safe default)
    """
    if verdict is GuardrailDecision.PASS:
        return _ACTION_TO_STATUS[decision.action]
    if verdict is GuardrailDecision.REQUIRES_REVIEW:
        return "under_review"
    return "escalated"  # BLOCK -> safe default


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def investigate_alert(db: Session, alert_id: int) -> InvestigationOutcome:
    """
    Run one full agent investigation over one alert. All side effects
    (alert status update, audit rows) commit before return.

    Failure modes are absorbed as escalations, not raised: the compliance
    system must remain live even when the model misbehaves. Callers that
    need programmatic access to failure detail can inspect the returned
    InvestigationOutcome (case_file_parse_ok, guardrail_verdict, diagnostic).
    """
    alert = db.execute(select(Alert).where(Alert.alert_id == alert_id)).scalar_one_or_none()
    if alert is None:
        # Not an escalatable outcome — there's nothing to escalate.
        # Return a sentinel so the caller (batch loop) can skip cleanly.
        return InvestigationOutcome(
            alert_id=alert_id,
            applied_status="",
            agent_decision=None,
            guardrail_verdict=None,
            diagnostic=f"alert {alert_id} not found",
        )

    # Load related account + transaction so we can pass pseudonymized
    # refs into the initial user message. Account is used by the agent's
    # tools throughout; we resolve it once here.
    account = db.execute(
        select(Account).where(Account.account_id == alert.account_id)
    ).scalar_one()
    transaction = db.execute(
        select(Transaction).where(Transaction.transaction_id == alert.transaction_id)
    ).scalar_one()

    account_token = pseudonymize(db, "account_external_ref", account.external_ref)
    transaction_token = pseudonymize(db, "transaction_external_ref", transaction.external_ref)

    initial_user_message = build_initial_user_message(
        alert_id=alert.alert_id,
        account_token=account_token,
        rule_code=alert.rule_code or "unknown",
        severity=alert.severity,
        rule_narrative=alert.narrative or "",
        transaction_token=transaction_token,
    )

    # Snapshot the pre-run status — feeds LifecycleStateGuardrail.
    current_status = (
        alert.status.value if hasattr(alert.status, "value") else str(alert.status)
    )

    # Run the ReAct loop.
    session = InvestigationSession(
        db=db,
        alert_id=alert.alert_id,
        system_prompt=SYSTEM_PROMPT,
        initial_user_message=initial_user_message,
    )
    result = session.run()

    # Parse the agent's final JSON. None means "unusable output".
    decision = build_agent_decision(result, current_status=current_status)

    if decision is None:
        # Safe default: forced escalate. Never silently close an alert
        # because the model produced bad output — the whole reason we
        # have guardrails is that model failures should route to humans.
        return _apply_forced_escalate(
            db, alert,
            reason=(
                f"case file parse failed (stopped_reason={result.stopped_reason}); "
                f"defaulting to escalate for human review"
            ),
            result=result,
            case_file_parse_ok=False,
        )

    # Guardrail evaluation. evaluate_all writes its own audit row.
    verdict, gr_results = evaluate_all(db, decision)

    final_status = _resolve_final_status(verdict, decision)
    _apply_status(db, alert, final_status, decision.narrative)

    _write_investigation_audit(
        db, alert.alert_id,
        applied_status=final_status,
        agent_action=decision.action.value,
        guardrail_verdict=verdict.value,
        tool_call_count=len(result.tool_calls),
        stopped_reason=result.stopped_reason,
    )

    return InvestigationOutcome(
        alert_id=alert.alert_id,
        applied_status=final_status,
        agent_decision=decision,
        guardrail_verdict=verdict,
        guardrail_reasons=[r.reason for r in gr_results if r.reason],
        tool_calls=result.tool_calls,
        stopped_reason=result.stopped_reason,
        case_file_parse_ok=True,
        diagnostic=f"agent proposed {decision.action.value}, guardrail {verdict.value}, applied {final_status}",
        token_usage=result.token_usage,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_forced_escalate(
    db: Session,
    alert: Alert,
    reason: str,
    result,
    case_file_parse_ok: bool,
) -> InvestigationOutcome:
    """Fallback path for irrecoverable agent failures. Sets alert to
    'escalated' with a diagnostic narrative that captures why the
    fallback fired — critical for the human reviewer to know they're
    seeing an escalation due to model failure, not agent judgment."""
    narrative = f"[FORCED ESCALATION] {reason}"
    _apply_status(db, alert, "escalated", narrative)
    _write_investigation_audit(
        db, alert.alert_id,
        applied_status="escalated",
        agent_action=None,
        guardrail_verdict=None,
        tool_call_count=len(result.tool_calls),
        stopped_reason=result.stopped_reason,
        note=reason,
    )
    return InvestigationOutcome(
        alert_id=alert.alert_id,
        applied_status="escalated",
        agent_decision=None,
        guardrail_verdict=None,
        tool_calls=result.tool_calls,
        stopped_reason=result.stopped_reason,
        case_file_parse_ok=case_file_parse_ok,
        diagnostic=reason,
        # Token usage is recorded even on the forced-escalate path —
        # the Groq spend happened regardless of whether parsing
        # succeeded, and the eval harness needs it to explain runs
        # that burn budget without producing a usable decision.
        token_usage=result.token_usage,
    )


def _apply_status(db: Session, alert: Alert, new_status: str, narrative: str) -> None:
    """Update the alert row's status + narrative + updated_at. Committed
    inline so an exception in the audit write below doesn't leave the
    alert change unsaved. Trade-off: audit lag by one commit; acceptable
    because the guardrail evaluation already wrote its own audit row
    before we got here."""
    alert.status = new_status
    alert.narrative = narrative
    alert.updated_at = datetime.now(timezone.utc)
    db.commit()


def _write_investigation_audit(
    db: Session,
    alert_id: int,
    applied_status: str,
    agent_action: Optional[str],
    guardrail_verdict: Optional[str],
    tool_call_count: int,
    stopped_reason: str,
    note: Optional[str] = None,
) -> None:
    """Single audit_log row per completed investigation. Captures the
    decision surface — agent proposal, guardrail verdict, applied status —
    rather than the full transcript. Full transcripts live in the
    InvestigationSession's message_history for eval, not in audit."""
    db.add(AuditLog(
        entity_type="alert",
        entity_id=alert_id,
        action="investigation_completed",
        actor="investigation_agent",
        details={
            "applied_status": applied_status,
            "agent_action": agent_action,
            "guardrail_verdict": guardrail_verdict,
            "tool_call_count": tool_call_count,
            "stopped_reason": stopped_reason,
            "note": note,
        },
    ))
    db.commit()