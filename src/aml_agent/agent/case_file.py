"""
Case file parser — InvestigationResult -> AgentDecision.

Extracts the JSON verdict from the model's final message and hands
it to the guardrail layer as an AgentDecision. Malformed output is
reported as None (not raised) so the dispatcher (7.5) can fall back
to a safe default (forced escalate) rather than crash the pipeline —
a compliance system that halts on model misbehaviour is worse than one
that defaults conservatively.

The tool-call names come from the InvestigationResult (session's log),
not from the JSON. The model doesn't need to redeclare what it called —
we already tracked it, and trusting the model's self-report would let
it lie to the EvidenceTrailGuardrail.
"""

import json
import logging
import re
from typing import Optional

from aml_agent.agent.session import InvestigationResult
from aml_agent.guardrails.base import AgentAction, AgentDecision


logger = logging.getLogger("aml_agent.agent.case_file")


# Fenced code block: ```json ... ``` or ``` ... ```. Non-greedy so
# multiple blocks don't collapse into one match; DOTALL so newlines
# inside the JSON are captured.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Try three extraction strategies in order of specificity:
      1. Full-string parse (model followed the prompt exactly)
      2. Fenced code block regex (model wrapped in ```json ... ```)
      3. First-{-to-last-} slice (model surrounded JSON with prose)

    Returns the parsed dict or None if all three fail. Kept as its own
    function so the strategy chain is testable in isolation.
    """
    text = text.strip()

    # Strategy 1
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strategy 2 — fenced code block
    match = _FENCED_JSON_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 3 — greedy brace slice. Not bulletproof (breaks if the
    # narrative itself contains an unescaped `}`), but good enough as
    # last resort; failures fall through to None and get caught upstream.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def build_agent_decision(
    result: InvestigationResult,
    current_status: str,
) -> Optional[AgentDecision]:
    """
    Parse the agent's final message into an AgentDecision.

    Returns None if:
      - final_message is missing (session hit iteration cap without deciding)
      - JSON cannot be extracted from the message
      - required keys are missing or the action is not a valid AgentAction
      - severity is not an integer in [1, 5]

    None is a valid outcome, not an error — it signals "agent produced
    unusable output" and the caller applies the safe default (escalate
    with an audit note).
    """
    if not result.final_message:
        logger.warning("no final_message on result alert_id=%d", result.alert_id)
        return None

    payload = _extract_json_object(result.final_message)
    if payload is None:
        logger.warning(
            "could not extract JSON from final_message alert_id=%d",
            result.alert_id,
        )
        return None

    # Validate required keys individually so error logs identify exactly
    # what was missing — helps debug prompt-drift issues.
    try:
        raw_action = payload["action"]
        severity = payload["severity"]
        narrative = payload["narrative"]
    except KeyError as e:
        logger.warning("missing key %s in agent JSON alert_id=%d", e, result.alert_id)
        return None

    try:
        action = AgentAction(raw_action)
    except ValueError:
        logger.warning(
            "unknown action %r in agent JSON alert_id=%d",
            raw_action, result.alert_id,
        )
        return None

    if not isinstance(severity, int) or not (1 <= severity <= 5):
        logger.warning(
            "invalid severity %r in agent JSON alert_id=%d",
            severity, result.alert_id,
        )
        return None

    if not isinstance(narrative, str):
        logger.warning(
            "non-string narrative %r in agent JSON alert_id=%d",
            type(narrative).__name__, result.alert_id,
        )
        return None

    # Deduplicate tool names while preserving order — EvidenceTrailGuardrail
    # only cares that tools were called, not how many times each. Distinct
    # list gives a cleaner audit record without losing signal.
    seen: set[str] = set()
    unique_tools: list[str] = []
    for tc in result.tool_calls:
        if tc.tool_name not in seen:
            seen.add(tc.tool_name)
            unique_tools.append(tc.tool_name)

    return AgentDecision(
        alert_id=result.alert_id,
        current_status=current_status,
        action=action,
        severity=severity,
        narrative=narrative,
        tool_calls=unique_tools,
    )