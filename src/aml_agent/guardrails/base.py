"""
Guardrail layer — base contract.

Deterministic, non-LLM safety checks over agent decisions. Every guardrail
inspects the agent's proposed action (close / escalate / SAR) plus its
supporting case file and returns pass / block. Any block downgrades the
decision to `requires_human_review`.

Design rationale: LLMs are probabilistic; regulators need a deterministic
answer to "what stops the agent from misbehaving?". That answer must not
itself depend on an LLM. Same architectural principle FATF R20 assumes
for suspicious-activity determinations: decisions must be documented and
reproducible, which means implementable in code, not by inference.

Mirrors the rule-engine pattern (base.py in Phase 2): ABC + registry +
run-all execution point. Reusing the pattern keeps the codebase's control
surfaces consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentAction(str, Enum):
    """
    Agent's proposed disposition for an alert. Values mirror the
    alert_status enum from Phase 1's schema, minus the intermediate
    states the agent cannot set directly (`open`, `under_review` are
    lifecycle-managed, not agent-selected).
    """
    ESCALATE = "escalate"
    CLOSE_FALSE_POSITIVE = "close_false_positive"
    CLOSE_NO_ACTION = "close_no_action"
    CLOSE_SAR_FILED = "close_sar_filed"


class GuardrailDecision(str, Enum):
    """
    Outcome of running a guardrail.

    - PASS: guardrail did not object; the agent's action may proceed.
    - REQUIRES_REVIEW: action needs human sign-off before taking effect.
    - BLOCK: action is prohibited; cannot proceed even with review.
    """
    PASS = "pass"
    REQUIRES_REVIEW = "requires_review"
    BLOCK = "block"


@dataclass(frozen=True)
class GuardrailResult:
    """One guardrail's verdict on one agent decision.

    reason is required whenever decision != PASS. Regulators auditing a
    blocked action need to know WHICH guardrail objected and WHY —
    otherwise the guardrail layer is opaque and defeats its own purpose.
    """
    guardrail_code: str
    decision: GuardrailDecision
    reason: str | None = None


@dataclass(frozen=True)
class AgentDecision:
    """
    The complete agent output a guardrail evaluates.

    alert_id / current_status: what the agent is acting on.
    action: what the agent wants to do (from AgentAction).
    severity: severity of the alert (1..5).
    narrative: the agent's written justification.
    tool_calls: names of tools invoked during investigation — evidence
                trail. An action with zero tool_calls has no observable
                basis.
    """
    alert_id: int
    current_status: str
    action: AgentAction
    severity: int
    narrative: str | None
    tool_calls: list[str]


class Guardrail(ABC):
    """
    Abstract base for every guardrail.

    Subclasses declare `code` + `description` and implement `evaluate()`.
    Guardrails are pure Python — no I/O, no DB, no LLM. Any guardrail
    that needs external state should fetch it before construction and
    pass it in via __init__.
    """

    code: str            # unique identifier, e.g. 'HIGH_SEV_DISMISSAL_BLOCK'
    description: str     # one-line human-readable purpose

    @abstractmethod
    def evaluate(self, decision: AgentDecision) -> GuardrailResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry — parallel to the rule engine's REGISTRY and the tool layer's
# TOOL_REGISTRY. Same trade-off: explicit register + explicit imports keep
# the guardrail set auditable and grep-friendly.
# ---------------------------------------------------------------------------

REGISTRY: list[type[Guardrail]] = []


def register(guardrail_cls: type[Guardrail]) -> type[Guardrail]:
    """
    Class decorator that registers a Guardrail with the layer.

    Enforces uniqueness on `code`: audit_log entries reference guardrail
    codes to explain why an action was blocked; duplicates would corrupt
    that trace.
    """
    if any(g.code == guardrail_cls.code for g in REGISTRY):
        raise ValueError(f"duplicate guardrail code: {guardrail_cls.code}")
    REGISTRY.append(guardrail_cls)
    return guardrail_cls