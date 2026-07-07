"""
Labeled test set curation for the eval harness.

Two functions:
  ground_truth_account_days(db) -> frozenset[(account_id, date)]
  sample_alerts_stratified(db, target_size=100) -> list[int]

Read-only — no writes. The ground truth already lives in
transactions.is_laundering_ground_truth (from AMLSim ingestion in
Phase 1); we only aggregate and sample.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session


# Fixed seed so repeated eval runs produce identical samples. Same value
# every time the module is imported — deterministic evaluation is a
# standard scikit-learn / academic-eval practice; without it, comparing
# eval runs before/after a code change confounds real signal with sampling
# noise.
RANDOM_SEED = 42


def ground_truth_account_days(db: Session) -> frozenset[tuple[int, date]]:
    """
    Return the set of (sender_account_id, date) pairs where any laundering
    transaction occurred, per AMLSim's ground-truth labels.

    Aggregation at (account, day) granularity rather than per-transaction
    matches how rules operate: a rule that fires on the 5th transaction
    in a structuring pattern should score as one detection, not four
    misses plus one hit. Same practice adopted in the AMLSim reference
    paper (Suzumura & Kanezashi, 2021).

    Returned as frozenset for O(1) membership checks — the rule engine
    eval will do tens of thousands of `(a, d) in truth_set` lookups.
    """
    rows = db.execute(text("""
        SELECT DISTINCT
            sender_account_id AS account_id,
            executed_at::date AS event_date
        FROM transactions
        WHERE is_laundering_ground_truth = TRUE
    """)).all()
    return frozenset((r.account_id, r.event_date) for r in rows)


def alert_account_days(db: Session) -> dict[str, frozenset[tuple[int, date]]]:
    """
    Return {rule_code: frozenset[(account_id, date)]} for every alert
    ever raised, aggregated by (account, day).

    Same granularity as ground_truth_account_days so precision/recall
    computation is a straight set intersection. Grouped by rule_code so
    the eval script can compute per-rule metrics without re-querying.
    """
    rows = db.execute(text("""
        SELECT
            a.rule_code,
            a.account_id,
            t.executed_at::date AS event_date
        FROM alerts a
        JOIN transactions t ON t.transaction_id = a.transaction_id
        WHERE a.source = 'rule_engine'
    """)).all()

    grouped: dict[str, set[tuple[int, date]]] = {}
    for r in rows:
        grouped.setdefault(r.rule_code, set()).add((r.account_id, r.event_date))
    return {code: frozenset(s) for code, s in grouped.items()}


def sample_alerts_stratified(
    db: Session,
    target_size: int = 100,
) -> list[int]:
    """
    Return a stratified sample of alert_ids for agent eval.

    Stratification dimensions: (rule_code, severity, is_laundering_ground_truth).
    Ensures the agent is evaluated on a representative mix — without
    stratification, STRUCTURING_001 (8382 alerts) would dominate any
    uniform random sample and rare typologies (FAN_IN_001, 87 alerts)
    might not appear at all.

    Uses SQL-side sampling with setseed() so the sample is reproducible
    across runs. Alternative (pull all alerts to Python then bucket)
    wastes a round-trip on 12K rows we don't need.

    target_size is a soft ceiling — actual sample may be slightly smaller
    if some strata have fewer members than the per-stratum allocation.
    """
    # Set session-level random seed for Postgres's random() so DISTINCT ON
    # + ORDER BY random() is reproducible. setseed accepts [-1, 1];
    # normalize RANDOM_SEED into that range deterministically.
    seed_normalized = (RANDOM_SEED % 1000) / 1000.0
    db.execute(text("SELECT setseed(:s)"), {"s": seed_normalized})

    # First: count strata to decide per-stratum quota. Even allocation
    # across strata beats proportional — the goal is representative
    # coverage of edge cases, not mirroring the population distribution
    # (which is what the rule-engine eval already measures).
    strata_rows = db.execute(text("""
        SELECT
            a.rule_code,
            a.severity,
            t.is_laundering_ground_truth AS truth,
            COUNT(*) AS n
        FROM alerts a
        JOIN transactions t ON t.transaction_id = a.transaction_id
        WHERE a.source = 'rule_engine'
        GROUP BY a.rule_code, a.severity, t.is_laundering_ground_truth
    """)).all()

    if not strata_rows:
        return []

    per_stratum = max(1, target_size // len(strata_rows))

    # For each stratum, take up to `per_stratum` alerts. LIMIT + ORDER
    # BY random() is fine at 12K rows; at 100K+ we'd switch to
    # TABLESAMPLE.
    sampled: list[int] = []
    for r in strata_rows:
        rows = db.execute(text("""
            SELECT a.alert_id
            FROM alerts a
            JOIN transactions t ON t.transaction_id = a.transaction_id
            WHERE a.source = 'rule_engine'
              AND a.rule_code = :rc
              AND a.severity = :sev
              AND t.is_laundering_ground_truth = :truth
            ORDER BY random()
            LIMIT :n
        """), {
            "rc": r.rule_code,
            "sev": r.severity,
            "truth": r.truth,
            "n": per_stratum,
        }).all()
        sampled.extend(row.alert_id for row in rows)

    return sampled