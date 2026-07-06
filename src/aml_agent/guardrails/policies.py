"""
Concrete guardrail implementations.

Each class enforces one specific policy from substep 6.1. Kept as
separate classes (not one big if/else) so each guardrail's audit_log
entry references a single, meaningful code and reviewers can enable /
disable individual policies without editing shared logic.
"""

from aml_agent.guardrails.base import (
    AgentAction,
    AgentDecision,
    Guardrail,
    GuardrailDecision,
    GuardrailResult,
    register,
)


# ---------------------------------------------------------------------------
# Structural guardrails — malformed / hallucinated agent output
# ---------------------------------------------------------------------------

@register
class ActionEnumGuardrail(Guardrail):
    """
    Reject actions that don't match the AgentAction enum.

    Rationale: LLMs occasionally emit near-miss strings ('escalated',
    'file_sar', 'closefp'). Catching them here prevents corrupt state
    from reaching the alerts table's alert_status enum, which would
    fail loudly at write time but with a less traceable audit trail.
    """
    code = "ACTION_INVALID_ENUM"
    description = "Rejects agent actions outside the AgentAction enum"

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        # AgentDecision.action is already typed as AgentAction, so if the
        # decision was constructed at all the enum check has passed.
        # Defensive check: guard against callers passing raw strings.
        if not isinstance(decision.action, AgentAction):
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                f"action is not a valid AgentAction value: {decision.action!r}",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)


@register
class NarrativeRequiredGuardrail(Guardrail):
    """
    Every agent action must carry a written justification.

    FATF R20 requires suspicious-activity determinations be documented
    and reproducible. An action with no narrative is undocumented by
    definition; block regardless of what the action is.
    """
    code = "NARRATIVE_MISSING"
    description = "Blocks actions with missing or empty narrative"

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if not decision.narrative or not decision.narrative.strip():
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                "agent action must include a non-empty narrative justification",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)


@register
class EvidenceTrailGuardrail(Guardrail):
    """
    Every action must reference at least one tool call.

    An action with zero tool_calls has no observable basis for its
    conclusion — the agent claims a determination without having looked
    at anything. Regulators auditing the decision would find no evidence
    trail. Block.
    """
    code = "NO_EVIDENCE_TRAIL"
    description = "Blocks actions with no supporting tool calls"

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if not decision.tool_calls:
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                "agent action has no tool call evidence trail",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)


@register
class LifecycleStateGuardrail(Guardrail):
    """
    Agent can only act on alerts in 'open' or 'under_review'.

    Prevents double-processing: an already-closed alert being re-decided
    would create conflicting audit history and, worse, could reopen a
    filed SAR without proper regulatory workflow.
    """
    code = "INVALID_LIFECYCLE_STATE"
    description = "Blocks actions on alerts not in open / under_review"

    _ACTIONABLE_STATES = {"open", "under_review"}

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if decision.current_status not in self._ACTIONABLE_STATES:
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                f"alert is in state '{decision.current_status}'; agent may only "
                f"act on: {sorted(self._ACTIONABLE_STATES)}",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)


# ---------------------------------------------------------------------------
# Policy guardrails — from the substep 6.1 matrix
# ---------------------------------------------------------------------------

@register
class HighSeverityDismissalGuardrail(Guardrail):
    """
    Block autonomous dismissal (close_false_positive / close_no_action)
    of severity 4-5 alerts.

    FATF R20 requires documented justification for dismissing suspicious
    activity; high-severity dismissals by an autonomous agent without
    human sign-off are a regulatory red flag. Force human review.
    """
    code = "HIGH_SEV_DISMISSAL_BLOCK"
    description = "Blocks autonomous dismissal of severity 4-5 alerts"

    _DISMISSAL_ACTIONS = {AgentAction.CLOSE_FALSE_POSITIVE, AgentAction.CLOSE_NO_ACTION}

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if decision.action in self._DISMISSAL_ACTIONS and decision.severity >= 4:
            return GuardrailResult(
                self.code, GuardrailDecision.BLOCK,
                f"cannot autonomously {decision.action.value} on severity "
                f"{decision.severity} alert; human review required",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)


@register
class MidSeverityDismissalReviewGuardrail(Guardrail):
    """
    Downgrade severity-3 dismissals to requires_review.

    Mid-severity dismissals are not clearly wrong, but at 50/50 signal
    they benefit from a human's judgment before persisting. Cheaper
    than a false-negative SAR at scale.
    """
    code = "MID_SEV_DISMISSAL_REVIEW"
    description = "Requires human review for severity-3 dismissals"

    _DISMISSAL_ACTIONS = {AgentAction.CLOSE_FALSE_POSITIVE, AgentAction.CLOSE_NO_ACTION}

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if decision.action in self._DISMISSAL_ACTIONS and decision.severity == 3:
            return GuardrailResult(
                self.code, GuardrailDecision.REQUIRES_REVIEW,
                f"severity-3 {decision.action.value} requires human review",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)


@register
class SARFilingReviewGuardrail(Guardrail):
    """
    SAR filing always requires human review, regardless of severity.

    Rationale: SAR filings carry legal consequences under FATF R20 and
    Wolfsberg guidance — financial institutions face penalties for both
    over- and under-filing. Never autonomous, at any severity.
    """
    code = "SAR_FILING_REVIEW"
    description = "Requires human review for any SAR-filing recommendation"

    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        if decision.action == AgentAction.CLOSE_SAR_FILED:
            return GuardrailResult(
                self.code, GuardrailDecision.REQUIRES_REVIEW,
                "SAR filings require human review regardless of severity",
            )
        return GuardrailResult(self.code, GuardrailDecision.PASS)