"""
Structuring / smurfing detection rule.

FATF typology reports (e.g. FATF Report on ML/TF Typologies) identify
structuring as one of the most common placement-stage techniques:
deliberately breaking transactions into amounts below reporting or
scrutiny thresholds to evade detection. Classic manifestation is multiple
sub-threshold deposits totalling well over the threshold within a short
window.
"""

from typing import Iterator

from sqlalchemy import text
from sqlalchemy.orm import Session

from aml_agent.rule_engine.base import AlertCandidate, Rule, register


@register
class StructuringRule(Rule):
    """
    Detect same-sender rolling-window structuring.

    A transaction triggers this rule if, within a look-back window ending
    at that transaction's timestamp, the same sender account has issued
    at least `min_count` transactions each at or below `threshold_amount`,
    summing to at least `min_total`.

    Emits one alert per triggering transaction (every tx that satisfies
    the count/sum condition in its window), not one per incident — keeps
    alerts tied to individual transactions for FATF R20 traceability.
    Dedup across engine reruns is handled by the DB unique constraint on
    (rule_code, transaction_id).
    """

    code = "STRUCTURING_001"
    description = "Multiple sub-threshold transactions from the same sender within a short window"
    default_severity = 4

    def __init__(
        self,
        threshold_amount: float = 1000.0,
        window_days: int = 7,
        min_count: int = 5,
        min_total: float = 1000.0,
    ):
        # Parameters kept as instance state, not class attrs, so the same
        # rule class can be instantiated with different thresholds for
        # different jurisdictions (e.g. USD 10K CTR vs SAR 3K) without
        # subclassing.
        self.threshold_amount = threshold_amount
        self.window_days = window_days
        self.min_count = min_count
        self.min_total = min_total

    def evaluate(self, db: Session) -> Iterator[AlertCandidate]:
        # Single query using Postgres window functions with a time-range
        # frame. Alternative approaches (per-tx correlated subquery or a
        # Python loop) issue O(N) queries or force full-table scans in
        # Python — this is O(N log N) and fully server-side.
        #
        # We restrict to sub-threshold transactions in the CTE so the
        # window function only aggregates over relevant rows. Filtering
        # after the window function would still count ineligible rows.
        sql = text("""
            WITH sub_threshold AS (
                SELECT
                    transaction_id,
                    sender_account_id,
                    executed_at,
                    amount
                FROM transactions
                WHERE amount <= :threshold
            ),
            windowed AS (
                SELECT
                    transaction_id,
                    sender_account_id,
                    executed_at,
                    amount,
                    COUNT(*) OVER (
                        PARTITION BY sender_account_id
                        ORDER BY executed_at
                        RANGE BETWEEN :window_interval PRECEDING AND CURRENT ROW
                    ) AS window_count,
                    SUM(amount) OVER (
                        PARTITION BY sender_account_id
                        ORDER BY executed_at
                        RANGE BETWEEN :window_interval PRECEDING AND CURRENT ROW
                    ) AS window_sum
                FROM sub_threshold
            )
            SELECT
                transaction_id,
                sender_account_id,
                window_count,
                window_sum
            FROM windowed
            WHERE window_count >= :min_count
              AND window_sum >= :min_total
        """)

        result = db.execute(
            sql,
            {
                "threshold": self.threshold_amount,
                # Postgres accepts an INTERVAL literal parameter for RANGE frames.
                "window_interval": f"{self.window_days} days",
                "min_count": self.min_count,
                "min_total": self.min_total,
            },
        )

        for row in result:
            yield AlertCandidate(
                transaction_id=row.transaction_id,
                account_id=row.sender_account_id,
                severity=self.default_severity,
                narrative=(
                    f"Structuring pattern: sender issued {row.window_count} "
                    f"transactions totalling {row.window_sum:.2f} at or below "
                    f"threshold {self.threshold_amount:.2f} within {self.window_days} days"
                ),
                rule_code=self.code,
            )