"""
Fan-in / fan-out detection rules.

FATF typology reports identify these as core layering-stage patterns:

- Fan-in: one account receives from many distinct senders in a short
  window — signals aggregation of illicit proceeds ("gathering").
- Fan-out: one account sends to many distinct recipients in a short
  window — signals distribution / dispersal to obscure the trail
  ("scattering", "smurfing distribution").

Both share the same shape: hub account + many distinct counterparties +
short window + material total. Only the direction and rule code differ.
"""

from typing import Iterator

from sqlalchemy import text
from sqlalchemy.orm import Session

from aml_agent.rule_engine.base import AlertCandidate, Rule, register


# ---------------------------------------------------------------------------
# SQL template shared by both rules.
#
# We use LATERAL join rather than a window function because Postgres does
# not support COUNT(DISTINCT ...) OVER (RANGE ...) — window functions
# don't accept DISTINCT. LATERAL executes the aggregate per outer row and
# hits idx_txn_sender_time / idx_txn_receiver_time (created in migration
# 48d69600f361), so at 80K rows this remains an indexed range scan per
# transaction rather than a full table scan.
#
# hub_col: account column defining the hub (receiver for fan-in, sender
# for fan-out). counterparty_col: the other side, counted DISTINCT.
# ---------------------------------------------------------------------------
_SQL_TEMPLATE = """
    SELECT
        t.transaction_id,
        t.{hub_col} AS hub_account_id,
        w.distinct_counterparties,
        w.window_sum
    FROM transactions t
    CROSS JOIN LATERAL (
        SELECT
            COUNT(DISTINCT t2.{counterparty_col}) AS distinct_counterparties,
            SUM(t2.amount) AS window_sum
        FROM transactions t2
        WHERE t2.{hub_col} = t.{hub_col}
          AND t2.executed_at BETWEEN t.executed_at - CAST(:window_interval AS INTERVAL)
                                 AND t.executed_at
    ) w
    WHERE w.distinct_counterparties >= :min_distinct
      AND w.window_sum >= :min_total
"""


class _FanRuleBase(Rule):
    """
    Shared implementation. Subclasses set hub_col / counterparty_col /
    direction_label; everything else — the SQL, the parameter binding,
    the narrative shape — lives here so fan-in and fan-out stay in sync
    when we tune thresholds.
    """

    hub_col: str            # "receiver_account_id" or "sender_account_id"
    counterparty_col: str   # the opposite
    direction_label: str    # human-readable direction for narratives

    def __init__(
        self,
        min_distinct_counterparties: int = 5,
        window_days: int = 3,
        min_total: float = 5000.0,
    ):
        self.min_distinct = min_distinct_counterparties
        self.window_days = window_days
        self.min_total = min_total

    def evaluate(self, db: Session) -> Iterator[AlertCandidate]:
        sql = text(_SQL_TEMPLATE.format(
            hub_col=self.hub_col,
            counterparty_col=self.counterparty_col,
        ))
        result = db.execute(
            sql,
            {
                "window_interval": f"{self.window_days} days",
                "min_distinct": self.min_distinct,
                "min_total": self.min_total,
            },
        )
        for row in result:
            yield AlertCandidate(
                transaction_id=row.transaction_id,
                account_id=row.hub_account_id,
                severity=self.default_severity,
                narrative=(
                    f"{self.direction_label} pattern: hub account interacted with "
                    f"{row.distinct_counterparties} distinct counterparties totalling "
                    f"{row.window_sum:.2f} within {self.window_days} days"
                ),
                rule_code=self.code,
            )


@register
class FanInRule(_FanRuleBase):
    """Hub account receiving from many distinct senders in short window."""
    code = "FAN_IN_001"
    description = "Account receives from many distinct senders in a short window (aggregation)"
    default_severity = 3
    hub_col = "receiver_account_id"
    counterparty_col = "sender_account_id"
    direction_label = "Fan-in"


@register
class FanOutRule(_FanRuleBase):
    """Hub account sending to many distinct receivers in short window."""
    code = "FAN_OUT_001"
    description = "Account sends to many distinct receivers in a short window (distribution)"
    default_severity = 3
    hub_col = "sender_account_id"
    counterparty_col = "receiver_account_id"
    direction_label = "Fan-out"