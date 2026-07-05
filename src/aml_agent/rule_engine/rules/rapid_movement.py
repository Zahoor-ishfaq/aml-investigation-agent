"""
Rapid movement / layering detection rule.

FATF's Interpretive Note to Recommendation 3 defines layering as the
process of moving funds through a chain of accounts and transactions
to distance them from their illicit origin. The observable signature
of a transit (or "pass-through") account is a tight temporal coupling
between inflows and outflows: money arrives, sits briefly, leaves —
the account holds little residual balance because incoming and outgoing
amounts nearly cancel within a short window.
"""

from typing import Iterator

from sqlalchemy import text
from sqlalchemy.orm import Session

from aml_agent.rule_engine.base import AlertCandidate, Rule, register


@register
class RapidMovementRule(Rule):
    """
    Detect outbound transactions where the sender received a large
    inflow within the immediately preceding window and is sending out
    a high fraction of it.

    Alert is emitted on the outbound transaction (not the inbound):
    the outbound is what completes the layering pattern and is the
    actionable moment. One alert per completed pass-through, not two,
    which also aligns with FATF R20's per-transaction reporting model.
    """

    code = "RAPID_MOVEMENT_001"
    description = "Account rapidly forwards most of a recent inflow (pass-through / layering pattern)"
    default_severity = 4

    def __init__(
        self,
        passthrough_window_hours: int = 48,
        passthrough_ratio: float = 0.8,
        min_amount: float = 500.0,
    ):
        # Parameterized so the same rule can be tuned per-jurisdiction or
        # per-risk-tier without subclassing. Defaults reflect FATF typology
        # descriptions of layering (<72h transit is characteristic).
        self.passthrough_window_hours = passthrough_window_hours
        self.passthrough_ratio = passthrough_ratio
        self.min_amount = min_amount

    def evaluate(self, db: Session) -> Iterator[AlertCandidate]:
        # LATERAL join to compute the inflow total per outbound transaction.
        # For each candidate outbound tx, we look back at inflows to the
        # same account within the window. Hits idx_txn_receiver_time
        # (defined in migration 48d69600f361), so this stays index-bound
        # even at 80K+ rows.
        #
        # Filter min_amount at both ends: outbound side removes noise
        # (small consumer payments); inflow side ensures we don't trigger
        # on tiny incoming amounts that a legitimate outbound would
        # coincidentally exceed the ratio against.
        sql = text("""
            SELECT
                t.transaction_id,
                t.sender_account_id,
                t.amount AS outbound_amount,
                w.inbound_total,
                t.amount / NULLIF(w.inbound_total, 0) AS ratio
            FROM transactions t
            CROSS JOIN LATERAL (
                SELECT SUM(t2.amount) AS inbound_total
                FROM transactions t2
                WHERE t2.receiver_account_id = t.sender_account_id
                  AND t2.executed_at BETWEEN
                        t.executed_at - CAST(:window_interval AS INTERVAL)
                    AND t.executed_at
                  AND t2.amount >= :min_amount
            ) w
            WHERE t.amount >= :min_amount
              AND w.inbound_total IS NOT NULL
              AND w.inbound_total > 0
              AND (t.amount / w.inbound_total) >= :ratio
        """)

        result = db.execute(
            sql,
            {
                "window_interval": f"{self.passthrough_window_hours} hours",
                "min_amount": self.min_amount,
                "ratio": self.passthrough_ratio,
            },
        )

        for row in result:
            yield AlertCandidate(
                transaction_id=row.transaction_id,
                account_id=row.sender_account_id,
                severity=self.default_severity,
                narrative=(
                    f"Rapid movement: sender forwarded {row.outbound_amount:.2f} "
                    f"({row.ratio * 100:.1f}% of recent inflows totalling {row.inbound_total:.2f}) "
                    f"within {self.passthrough_window_hours}h — consistent with layering / pass-through"
                ),
                rule_code=self.code,
            )