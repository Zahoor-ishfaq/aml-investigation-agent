"""
CLI: evaluate rule engine detection quality against AMLSim ground truth.

Runs offline over already-persisted alerts and transactions — no rule
re-execution, no DB writes. Fast enough to run after every rule tweak.

Metrics computed per rule and in aggregate:
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  F1        = 2 * P * R / (P + R)

TP = alert (account, day) pairs that ARE in ground truth
FP = alert (account, day) pairs NOT in ground truth
FN = ground-truth (account, day) pairs NOT alerted

Set-based computation follows standard precision/recall definitions
(scikit-learn docs). Aggregate uses set union across rules (not sum):
if multiple rules flag the same (account, day), that's one detection.
"""

import argparse
import logging
import sys

from aml_agent.db.session import get_db
from aml_agent.eval.labels import alert_account_days, ground_truth_account_days


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _metrics(
    alerts: frozenset[tuple],
    truth: frozenset[tuple],
) -> dict[str, float | int]:
    """
    Compute precision, recall, F1 given alert and truth sets.

    Guarded division: precision/recall are undefined when the
    denominator is zero (no alerts fired, or no ground-truth events
    exist in the window). Returning 0.0 in those cases lets aggregate
    reports print without special-case branching downstream.
    """
    tp = len(alerts & truth)
    fp = len(alerts - truth)
    fn = len(truth - alerts)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _print_row(name: str, m: dict) -> None:
    """One table row. Fixed widths so columns align in a terminal."""
    print(
        f"{name:<22} "
        f"{m['tp']:>6} {m['fp']:>7} {m['fn']:>6} "
        f"{m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rule engine against AMLSim ground truth."
    )
    parser.parse_args()

    _configure_logging()
    logger = logging.getLogger("aml_agent.eval.rules")

    with get_db() as db:
        logger.info("Loading ground truth and alert coverage")
        truth = ground_truth_account_days(db)
        per_rule = alert_account_days(db)

    # Aggregate = union across all rule sets. A (account, day) flagged
    # by multiple rules counts once — matches how a human reviewer
    # would experience the alert (one review per account per day).
    aggregate: frozenset[tuple] = frozenset().union(*per_rule.values())

    print(f"\nGround truth (account, day) pairs: {len(truth):,}")
    print(f"Alert (account, day) pairs (union): {len(aggregate):,}\n")

    header = (
        f"{'Rule':<22} {'TP':>6} {'FP':>7} {'FN':>6} "
        f"{'Precision':>10} {'Recall':>8} {'F1':>6}"
    )
    print(header)
    print("-" * len(header))

    for rule_code, alerts in sorted(per_rule.items()):
        _print_row(rule_code, _metrics(alerts, truth))

    print("-" * len(header))
    _print_row("AGGREGATE (union)", _metrics(aggregate, truth))


if __name__ == "__main__":
    main()