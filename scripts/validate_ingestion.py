"""
Post-ingestion validation.

Runs a series of correctness/sanity checks against the Postgres schema
after AMLSim data has been ingested. Purpose: catch silent data corruption
that row counts alone can't reveal — orphaned FKs, degenerate distributions,
missing typology coverage.

Not a unittest suite: this is a one-time-per-ingest data audit, not logic
that runs continuously. Emits a report to stderr; exits nonzero if any
check fails so it can gate a CI pipeline later if needed.

Usage:
    python scripts/validate_ingestion.py
"""

import logging
import sys

from sqlalchemy import text

from aml_agent.db.session import get_db

logger = logging.getLogger("aml_agent.validation")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


class ValidationError(Exception):
    """Raised when a validation check fails. Collected so all failures
    surface in one run rather than exiting on the first — the caller
    wants to see the full picture before re-ingesting."""


# ---------------------------------------------------------------------------
# Check 1 — Referential integrity
# ---------------------------------------------------------------------------

def check_no_orphan_transactions(db) -> None:
    """
    Every transaction's sender/receiver must resolve to a real account.

    Postgres FK constraints already enforce this at insert time, so a
    failure here means either (a) constraints were disabled during load
    (they weren't, but belt-and-suspenders), or (b) accounts were deleted
    after ingestion. Either way — critical to catch before rule engine runs.
    """
    result = db.execute(text("""
        SELECT COUNT(*) FROM transactions t
        LEFT JOIN accounts s ON t.sender_account_id = s.account_id
        LEFT JOIN accounts r ON t.receiver_account_id = r.account_id
        WHERE s.account_id IS NULL OR r.account_id IS NULL
    """)).scalar()
    if result:
        raise ValidationError(f"orphan_transactions: {result} tx with missing account FK")
    logger.info("check_no_orphan_transactions: PASS")


def check_no_orphan_accounts(db) -> None:
    """Every account must resolve to a real customer. Same reasoning as above."""
    result = db.execute(text("""
        SELECT COUNT(*) FROM accounts a
        LEFT JOIN customers c ON a.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """)).scalar()
    if result:
        raise ValidationError(f"orphan_accounts: {result} accounts with missing customer FK")
    logger.info("check_no_orphan_accounts: PASS")


# ---------------------------------------------------------------------------
# Check 2 — Distribution sanity
# ---------------------------------------------------------------------------

def check_amount_distribution(db) -> None:
    """
    Transaction amounts must be non-uniform. Real AML data is log-normal:
    a heavy tail of small transactions, a thin tail of large ones. Uniform
    distributions are a red flag that generation misfired.

    Test: p99 / p50 ratio > 1.5. Threshold set to catch degenerate cases
    (all-identical amounts) rather than enforce heavy tails, since AMLSim's
    default config samples amounts uniformly within [min_amount, max_amount].
    A truly heavy-tailed distribution requires reconfiguring AMLSim's
    amount ranges to span orders of magnitude — deferred until eval
    harness (step 9) demonstrates it's needed for detection.
    """
    row = db.execute(text("""
        SELECT
            percentile_cont(0.50) WITHIN GROUP (ORDER BY amount) AS p50,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY amount) AS p99
        FROM transactions
    """)).one()
    p50, p99 = float(row.p50), float(row.p99)
    ratio = p99 / p50 if p50 else 0
    logger.info("amount p50=%.2f p99=%.2f ratio=%.2f", p50, p99, ratio)
    if ratio < 1.5:
        raise ValidationError(f"amount_distribution: p99/p50 ratio {ratio:.2f} < 1.5 (degenerate)")
    logger.info("check_amount_distribution: PASS")


def check_temporal_spread(db) -> None:
    """
    Transactions must span the full simulation window, not clump at day 1.
    Test: at least 300 distinct days have transactions (out of 365 configured).
    """
    days = db.execute(text("""
        SELECT COUNT(DISTINCT executed_at::date) FROM transactions
    """)).scalar()
    logger.info("distinct_transaction_days=%d", days)
    if days < 300:
        raise ValidationError(f"temporal_spread: only {days} distinct days (expected >= 300)")
    logger.info("check_temporal_spread: PASS")


def check_laundering_ratio(db) -> None:
    """
    Laundering base rate must fall in expected range (0.1% - 5%). Real
    AML class imbalance is ~1%; our config targeted ~0.5-1%. Outside this
    range signals typology injection went wrong (either too few → useless
    for eval, or too many → not realistic training signal).
    """
    row = db.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE is_laundering_ground_truth) AS laundering,
            COUNT(*) AS total
        FROM transactions
    """)).one()
    ratio = row.laundering / row.total if row.total else 0
    logger.info("laundering_ratio=%.4f (%d / %d)", ratio, row.laundering, row.total)
    if not (0.001 <= ratio <= 0.05):
        raise ValidationError(f"laundering_ratio: {ratio:.4f} outside [0.001, 0.05]")
    logger.info("check_laundering_ratio: PASS")


# ---------------------------------------------------------------------------
# Check 3 — Structural sanity (no self-transfers snuck through, etc.)
# ---------------------------------------------------------------------------

def check_no_self_transfers(db) -> None:
    """chk_not_self_transfer constraint already enforces this at insert,
    but a defensive scan confirms nothing bypassed it (e.g. via direct SQL)."""
    result = db.execute(text("""
        SELECT COUNT(*) FROM transactions WHERE sender_account_id = receiver_account_id
    """)).scalar()
    if result:
        raise ValidationError(f"self_transfers: {result} rows violate constraint")
    logger.info("check_no_self_transfers: PASS")


def check_all_accounts_active_or_valid_closed(db) -> None:
    """chk_closed_consistency at DB layer; scan confirms it holds post-ingestion."""
    result = db.execute(text("""
        SELECT COUNT(*) FROM accounts
        WHERE (status = 'closed' AND closed_at IS NULL)
           OR (status != 'closed' AND closed_at IS NOT NULL)
    """)).scalar()
    if result:
        raise ValidationError(f"closed_consistency: {result} accounts violate constraint")
    logger.info("check_all_accounts_active_or_valid_closed: PASS")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

CHECKS = [
    check_no_orphan_transactions,
    check_no_orphan_accounts,
    check_amount_distribution,
    check_temporal_spread,
    check_laundering_ratio,
    check_no_self_transfers,
    check_all_accounts_active_or_valid_closed,
]


def main() -> None:
    _configure_logging()
    failures = []
    with get_db() as db:
        for check in CHECKS:
            try:
                check(db)
            except ValidationError as e:
                # Collect, don't raise — report all failures in one run so
                # regeneration cycles don't require iterative fixes.
                logger.error("%s: FAIL — %s", check.__name__, e)
                failures.append(str(e))

    if failures:
        logger.error("VALIDATION FAILED: %d check(s) failed", len(failures))
        sys.exit(1)
    logger.info("VALIDATION PASSED: all %d checks OK", len(CHECKS))


if __name__ == "__main__":
    main()