from aml_agent.db.session import get_db
from aml_agent.guardrails.base import AgentDecision, AgentAction
from aml_agent.guardrails.engine import evaluate_all

# 4 test decisions covering pass, review, and block outcomes
cases = [
    ("clean escalate (should PASS)", AgentDecision(
        alert_id=1, current_status="open", action=AgentAction.ESCALATE,
        severity=4, narrative="Structuring pattern with 6 sub-threshold txns.",
        tool_calls=["get_transaction_history"])),
    ("high-sev dismissal (should BLOCK)", AgentDecision(
        alert_id=2, current_status="open", action=AgentAction.CLOSE_FALSE_POSITIVE,
        severity=5, narrative="Legitimate payroll.", tool_calls=["get_customer_profile"])),
    ("SAR filing (should REQUIRES_REVIEW)", AgentDecision(
        alert_id=3, current_status="under_review", action=AgentAction.CLOSE_SAR_FILED,
        severity=4, narrative="Clear layering evidence.", tool_calls=["get_linked_accounts"])),
    ("PII leak (should BLOCK)", AgentDecision(
        alert_id=4, current_status="open", action=AgentAction.ESCALATE,
        severity=3, narrative="Customer John Smith transferred funds.",
        tool_calls=["get_customer_profile"])),
]

with get_db() as db:
    for label, decision in cases:
        verdict, results = evaluate_all(db, decision)
        print(f"\n{label}")
        print(f"  aggregate: {verdict.value}")
        for r in results:
            if r.decision.value != "pass":
                print(f"    {r.guardrail_code}: {r.decision.value} — {r.reason}")