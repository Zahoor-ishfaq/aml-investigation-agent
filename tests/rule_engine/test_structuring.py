"""
Tests for StructuringRule.

Positive: sender sends 5 sub-threshold transactions within 7 days summing
above the threshold — rule must fire on the triggering transaction.

Negative: sender sends transactions above the threshold — rule must not
fire regardless of count.
"""

from datetime import datetime, timezone

import pytest

from aml_agent.rule_engine.rules.structuring import StructuringRule
from tests.helpers import make_customer, make_account, make_txn, seed_pair


def test_structuring_fires_when_pattern_matches(db):
    sender, receiver = seed_pair(db, "struct-pos")

    # 5 transactions of 900 (below 1000 threshold) within 3 days,
    # summing to 4500 (above min_total default of 1000).
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for i in range(5):
        make_txn(db, f"struct-pos-t{i}", sender, receiver, 900.0,
                 base.replace(hour=i * 4))

    rule = StructuringRule()
    candidates = list(rule.evaluate(db))

    # At least the 5th tx should trigger (window_count >= 5 for the last tx
    # in the sequence). Earlier ones may or may not depending on ordering.
    assert len(candidates) >= 1, "structuring pattern was not detected"
    assert all(c.rule_code == "STRUCTURING_001" for c in candidates)
    assert all(c.severity == 4 for c in candidates)


def test_structuring_does_not_fire_above_threshold(db):
    sender, receiver = seed_pair(db, "struct-neg")

    # 5 transactions of 5000 — all ABOVE the sub-threshold ceiling of 1000.
    # These should never be counted as structuring regardless of frequency.
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for i in range(5):
        make_txn(db, f"struct-neg-t{i}", sender, receiver, 5000.0,
                 base.replace(hour=i * 4))

    rule = StructuringRule()
    # Filter to only alerts for this sender — other seed data from the DB
    # (if the outer transaction contains any) shouldn't affect the assertion.
    candidates = [c for c in rule.evaluate(db) if c.account_id == sender]

    assert candidates == [], "structuring should not fire on above-threshold txns"