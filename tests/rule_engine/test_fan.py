"""
Tests for FanInRule and FanOutRule.

Fan-in positive: one receiver, many distinct senders in short window.
Fan-out positive: one sender, many distinct receivers in short window.
Negative: below min_distinct_counterparties threshold.
"""

from datetime import datetime, timezone

from aml_agent.rule_engine.rules.fan import FanInRule, FanOutRule
from tests.helpers import make_customer, make_account, make_txn


def _make_hub_and_spokes(db, tag: str, n_spokes: int):
    """Create a hub account plus n_spokes counterparty accounts. Returns
    (hub_account_id, list_of_spoke_account_ids)."""
    hub_c = make_customer(db, f"{tag}-hub-c")
    hub_a = make_account(db, f"{tag}-hub-a", hub_c.customer_id)
    spokes = []
    for i in range(n_spokes):
        sc = make_customer(db, f"{tag}-sp-c{i}")
        sa = make_account(db, f"{tag}-sp-a{i}", sc.customer_id)
        spokes.append(sa.account_id)
    return hub_a.account_id, spokes


def test_fan_in_fires_when_many_distinct_senders(db):
    hub, spokes = _make_hub_and_spokes(db, "fanin-pos", n_spokes=6)

    # 6 distinct senders each send 1000 to the hub within 1 day.
    # window_days default 3 and min_distinct default 5 → should fire.
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for i, s in enumerate(spokes):
        make_txn(db, f"fanin-pos-t{i}", s, hub, 1000.0, base.replace(hour=i))

    rule = FanInRule()
    candidates = [c for c in rule.evaluate(db) if c.account_id == hub]
    assert len(candidates) >= 1, "fan-in was not detected"
    assert all(c.rule_code == "FAN_IN_001" for c in candidates)


def test_fan_in_does_not_fire_with_few_senders(db):
    hub, spokes = _make_hub_and_spokes(db, "fanin-neg", n_spokes=3)

    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for i, s in enumerate(spokes):
        make_txn(db, f"fanin-neg-t{i}", s, hub, 1000.0, base.replace(hour=i))

    rule = FanInRule()
    candidates = [c for c in rule.evaluate(db) if c.account_id == hub]
    assert candidates == [], "fan-in should not fire below min_distinct=5"


def test_fan_out_fires_when_many_distinct_receivers(db):
    hub, spokes = _make_hub_and_spokes(db, "fanout-pos", n_spokes=6)

    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for i, s in enumerate(spokes):
        make_txn(db, f"fanout-pos-t{i}", hub, s, 1000.0, base.replace(hour=i))

    rule = FanOutRule()
    candidates = [c for c in rule.evaluate(db) if c.account_id == hub]
    assert len(candidates) >= 1, "fan-out was not detected"
    assert all(c.rule_code == "FAN_OUT_001" for c in candidates)


def test_fan_out_does_not_fire_with_few_receivers(db):
    hub, spokes = _make_hub_and_spokes(db, "fanout-neg", n_spokes=3)

    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for i, s in enumerate(spokes):
        make_txn(db, f"fanout-neg-t{i}", hub, s, 1000.0, base.replace(hour=i))

    rule = FanOutRule()
    candidates = [c for c in rule.evaluate(db) if c.account_id == hub]
    assert candidates == [], "fan-out should not fire below min_distinct=5"