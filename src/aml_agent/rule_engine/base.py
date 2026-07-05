"""
Rule engine — base class and contract.

Every rule inherits from `Rule` and implements `evaluate()`, returning
`AlertCandidate` instances. Rules are pure logic: they read from the DB
and yield candidates. The execution engine (rule_engine/runner.py, step 2.5)
handles persisting alerts and writing audit_log entries — this separation
keeps rules trivially testable and prevents each rule from re-implementing
audit/dedup logic.

Rules encode FATF-recognized typologies (structuring, layering, integration
patterns per FATF's 2012 Recommendations and typology reports).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class AlertCandidate:
    """
    One candidate alert produced by a rule.

    frozen=True makes instances hashable and prevents accidental mutation
    between rule.evaluate() and the execution engine writing to DB.
    narrative is required (not optional) — regulators need a human-readable
    reason for every alert per FATF Recommendation 20.
    """
    transaction_id: int
    account_id: int
    severity: int          # 1..5, matches alerts.severity CHECK constraint
    narrative: str
    rule_code: str         # populated by the engine from Rule.code, kept here so
                           # the execution engine can dedup without inspecting the source rule


class Rule(ABC):
    """
    Abstract base class every rule implements.

    Subclasses declare `code`, `description`, `default_severity` as class
    attributes and implement `evaluate()`. The engine discovers rules via
    the `REGISTRY` list below (populated by the @register decorator).
    """

    code: str                    # unique identifier, e.g. 'STRUCTURING_001'
    description: str             # one-line explanation shown in alert narratives
    default_severity: int = 3    # 1..5; individual matches can override

    @abstractmethod
    def evaluate(self, db: Session) -> Iterator[AlertCandidate]:
        """
        Scan the DB and yield candidate alerts.

        Implementations are expected to be batch-mode (single SQL query per
        rule where possible) rather than per-row Python loops — with 80K+
        transactions the difference is seconds vs minutes.

        Rules must NOT read is_laundering_ground_truth. That column exists
        only for the eval harness (step 9); using it here would leak the
        answer and make rule performance meaningless.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry — populated by @register decorator applied to each concrete rule.
# The runner (2.5) iterates REGISTRY to know which rules to execute. Using
# a decorator + module import beats hardcoding a list because new rules
# self-register when their module is imported.
# ---------------------------------------------------------------------------

REGISTRY: list[type[Rule]] = []


def register(rule_cls: type[Rule]) -> type[Rule]:
    """
    Class decorator that registers a Rule subclass with the engine.

    Usage:
        @register
        class StructuringRule(Rule):
            code = "STRUCTURING_001"
            ...

    Enforces uniqueness on `code` — two rules sharing a code would corrupt
    the (rule_code, transaction_id) dedup constraint on alerts.
    """
    existing = {r.code for r in REGISTRY}
    if rule_cls.code in existing:
        raise ValueError(f"duplicate rule code: {rule_cls.code}")
    REGISTRY.append(rule_cls)
    return rule_cls