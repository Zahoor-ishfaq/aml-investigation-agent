"""
Tests for RapidMovementRule.

Positive: account receives large inflow, then sends out 80%+ within 48h.
Negative: account receives inflow but sends out only a small fraction.
"""

from datetime import datetime, timedelta, timezone

from aml_agent.rule_engine.rules.rapid_movement import RapidMovementRule
from tests.helpers import make_customer, make_account, make_txn, seed_pair


def test_rapid_movement_fires_on_passthrough(db):
    # Three accounts: source funds transit, transit forwards to destination.
    src_a, transit_a = seed_pair(db, "rm-pos-1")
    # Second pair — reuse make_customer/make_account directly for the dest.
    dest_c = make_customer(db, "rm-pos-destc")
    dest_a = make_account(db, "rm-pos-desta", dest_c.customer_id)

    t0 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    # Inflow of 10000 to transit account.
    make_txn(db, "rm-pos-in", src_a, transit_a, 10000.0, t0)
    # Outflow of 9000 (90% of inflow) 24h later — should trigger.
    make_txn(db, "rm-pos-out", transit_a, dest_a.account_id, 9000.0,
             t0 + timedelta(hours=24))

    rule = RapidMovementRule()
    candidates = [c for c in rule.evaluate(db) if c.account_id == transit_a]
    assert len(candidates) == 1, f"expected 1 pass-through alert, got {len(candidates)}"
    assert candidates[0].rule_code == "RAPID_MOVEMENT_001"


def test_rapid_movement_does_not_fire_on_low_ratio(db):
    src_a, transit_a = seed_pair(db, "rm-neg-1")
    dest_c = make_customer(db, "rm-neg-destc")
    dest_a = make_account(db, "rm-neg-desta", dest_c.customer_id)

    t0 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    # Inflow 10000, outflow only 2000 (20%) — well below 80% ratio.
    make_txn(db, "rm-neg-in", src_a, transit_a, 10000.0, t0)
    make_txn(db, "rm-neg-out", transit_a, dest_a.account_id, 2000.0,
             t0 + timedelta(hours=24))

    rule = RapidMovementRule()
    candidates = [c for c in rule.evaluate(db) if c.account_id == transit_a]
    assert candidates == [], "rapid movement should not fire below ratio threshold"