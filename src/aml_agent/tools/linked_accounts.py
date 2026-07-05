"""
Tool: query linked / counterparty accounts for a given account.

Returns aggregated per-counterparty statistics — one row per distinct
counterparty rather than raw transactions. Rationale: fifty raw
transactions showing "same three counterparties" wastes agent context;
one summarized row per counterparty is the actual signal the agent
reasons over.
"""

from datetime import timedelta

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from aml_agent.db.models import Account, Transaction
from aml_agent.pseudonymization.tokenizer import depseudonymize, pseudonymize
from aml_agent.tools.base import Tool, ToolResult, register_tool
from sqlalchemy import or_, select


class LinkedAccountsArgs(BaseModel):
    """
    Args for get_linked_accounts. All bounds validated at the schema
    boundary so the underlying SQL always sees sane values.
    """
    account_token: str = Field(
        ..., description="Pseudonymized account token, format AEXT_<hex>."
    )
    days_back: int = Field(
        90, ge=1, le=365,
        description="Look-back window in days. Default 90 — wider than "
                    "transaction-history default because relationship view "
                    "needs a longer horizon to distinguish one-off from recurring."
    )
    min_interactions: int = Field(
        1, ge=1,
        description="Minimum number of transactions with the counterparty to include. "
                    "Raise to filter out one-off transactions when hunting for "
                    "recurring relationships."
    )
    limit: int = Field(
        20, ge=1, le=100,
        description="Max number of counterparties to return, ordered by total "
                    "volume descending. Default 20."
    )


@register_tool
class LinkedAccountsTool(Tool):
    """Return per-counterparty aggregates for an account: distinct
    counterparties, their direction of interaction, total volume, count,
    and first/last interaction dates."""

    name = "get_linked_accounts"
    description = (
        "Retrieve the distinct counterparty accounts that have transacted with a "
        "given account, aggregated per counterparty (total volume, interaction count, "
        "direction, date range). Use to map an account's relationship network before "
        "assessing whether interactions with any specific counterparty are anomalous. "
        "Results are ordered by total volume descending."
    )
    args_schema = LinkedAccountsArgs

    def _run(self, db: Session, args: LinkedAccountsArgs) -> ToolResult:
        real_ext_ref = depseudonymize(db, "account_external_ref", args.account_token)
        if real_ext_ref is None:
            return ToolResult(
                status="error",
                error=f"account token '{args.account_token}' not found",
            )

        account = db.execute(
            select(Account).where(Account.external_ref == real_ext_ref)
        ).scalar_one_or_none()
        if account is None:
            return ToolResult(
                status="error",
                error=f"account for token '{args.account_token}' not found in current dataset",
            )

        # Anchor the window on this account's most recent transaction rather
        # than wall-clock, same reasoning as transaction_history — the
        # simulation dataset is fixed to 2024 and wall-clock queries would
        # return empty. Flag for production swap to datetime.utcnow().
        latest = db.execute(
            select(Transaction.executed_at)
            .where(or_(
                Transaction.sender_account_id == account.account_id,
                Transaction.receiver_account_id == account.account_id,
            ))
            .order_by(Transaction.executed_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest is None:
            return ToolResult(
                status="ok", data=[],
                metadata={"account_token": args.account_token, "count": 0},
            )
        window_start = latest - timedelta(days=args.days_back)

        # Single query: UNION ALL both directions into a per-edge stream,
        # then GROUP BY counterparty. FILTER clauses on COUNT distinguish
        # "sent only" / "received only" / "both" without a second pass —
        # standard Postgres aggregate-filter idiom, index-friendly because
        # each UNION arm hits idx_txn_sender_time / idx_txn_receiver_time.
        sql = text("""
            WITH edges AS (
                SELECT
                    receiver_account_id AS counterparty_id,
                    'sent' AS direction,
                    amount,
                    executed_at
                FROM transactions
                WHERE sender_account_id = :aid
                  AND executed_at >= :window_start

                UNION ALL

                SELECT
                    sender_account_id AS counterparty_id,
                    'received' AS direction,
                    amount,
                    executed_at
                FROM transactions
                WHERE receiver_account_id = :aid
                  AND executed_at >= :window_start
            )
            SELECT
                counterparty_id,
                CASE
                    WHEN COUNT(*) FILTER (WHERE direction = 'sent') > 0
                     AND COUNT(*) FILTER (WHERE direction = 'received') > 0 THEN 'both'
                    WHEN COUNT(*) FILTER (WHERE direction = 'sent') > 0 THEN 'sent_only'
                    ELSE 'received_only'
                END AS direction_summary,
                SUM(amount) AS total_volume,
                COUNT(*) AS interaction_count,
                MIN(executed_at) AS first_interaction,
                MAX(executed_at) AS last_interaction
            FROM edges
            GROUP BY counterparty_id
            HAVING COUNT(*) >= :min_interactions
            ORDER BY total_volume DESC
            LIMIT :limit
        """)

        rows = db.execute(sql, {
            "aid": account.account_id,
            "window_start": window_start,
            "min_interactions": args.min_interactions,
            "limit": args.limit,
        }).all()

        if not rows:
            return ToolResult(
                status="ok", data=[],
                metadata={
                    "account_token": args.account_token,
                    "count": 0,
                    "window_days": args.days_back,
                },
            )

        # Bulk-load counterparty accounts in one query so we can tokenize
        # their external_refs without N+1 lookups.
        counterparty_ids = [r.counterparty_id for r in rows]
        counterparties = {
            a.account_id: a
            for a in db.execute(
                select(Account).where(Account.account_id.in_(counterparty_ids))
            ).scalars()
        }

        data = []
        for r in rows:
            cp = counterparties.get(r.counterparty_id)
            if cp is None:
                # Defensive skip; shouldn't happen given FK constraints.
                continue
            data.append({
                "counterparty_account_token": pseudonymize(
                    db, "account_external_ref", cp.external_ref
                ),
                "direction_summary": r.direction_summary,
                "total_volume": float(r.total_volume),
                "interaction_count": int(r.interaction_count),
                "first_interaction": r.first_interaction.isoformat(),
                "last_interaction": r.last_interaction.isoformat(),
            })

        return ToolResult(
            status="ok",
            data=data,
            metadata={
                "account_token": args.account_token,
                "count": len(data),
                "window_days": args.days_back,
                "window_start": window_start.isoformat(),
                "window_end": latest.isoformat(),
            },
        )