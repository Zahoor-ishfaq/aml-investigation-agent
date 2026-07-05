"""
Tool: query transaction history for an account.

Given a pseudonymized account token, returns the most recent transactions
where the account was sender OR receiver. Output is pseudonymized before
return.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from aml_agent.db.models import Account, Transaction
from aml_agent.pseudonymization.depseudonymize import _PREFIX_TO_TYPE  # for token type check
from aml_agent.pseudonymization.service import pseudonymize_transaction
from aml_agent.pseudonymization.tokenizer import depseudonymize
from aml_agent.tools.base import Tool, ToolResult, register_tool


class TransactionHistoryArgs(BaseModel):
    """
    Args for get_transaction_history. Field descriptions appear in the
    JSON schema sent to the LLM — write them as instructions to the model,
    not as internal comments.
    """
    account_token: str = Field(
        ..., description="Pseudonymized account token, format AEXT_<hex>."
    )
    limit: int = Field(
        50, ge=1, le=200,
        description="Max number of transactions to return (default 50, max 200)."
    )
    days_back: Optional[int] = Field(
        30, ge=1, le=365,
        description="Look-back window in days from the most recent transaction. "
                    "Default 30. Increase only when investigating longer-horizon patterns."
    )


@register_tool
class TransactionHistoryTool(Tool):
    """Return recent transactions for a pseudonymized account, both
    directions (sender and receiver), most recent first."""

    name = "get_transaction_history"
    description = (
        "Retrieve recent transactions involving a specific account (as sender or "
        "receiver). Use to understand an account's activity pattern before deciding "
        "whether alert-triggering behavior is anomalous or consistent with baseline. "
        "Returns transactions sorted most-recent-first."
    )
    args_schema = TransactionHistoryArgs

    def _run(self, db: Session, args: TransactionHistoryArgs) -> ToolResult:
        # Resolve token -> real external_ref -> internal account_id.
        # We resolve via external_ref (not directly to account_id) because
        # the pseudonym map is keyed on external_ref — internal integer IDs
        # never leave the DB and never enter the pseudonym namespace.
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
            # Defensive: shouldn't happen if pseudonym map and accounts stay
            # in sync, but if pseudonym predates a data reset, gracefully report.
            return ToolResult(
                status="error",
                error=f"account for token '{args.account_token}' not found in current dataset",
            )

        # Time window: relative to most-recent tx for THIS account, not
        # wall-clock. Reason: our simulation dataset is fixed to 2024;
        # anchoring on wall-clock would return an empty window on every call.
        # In production this would be `datetime.utcnow()` — flag for later.
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
                status="ok",
                data=[],
                metadata={"account_token": args.account_token, "count": 0},
            )

        window_start = latest - timedelta(days=args.days_back)

        # Single query, both directions, ordered, limited. Uses
        # idx_txn_sender_time and idx_txn_receiver_time (migration 48d69600f361)
        # so this is index-bound even for high-degree hub accounts.
        txns = db.execute(
            select(Transaction)
            .where(
                or_(
                    Transaction.sender_account_id == account.account_id,
                    Transaction.receiver_account_id == account.account_id,
                ),
                Transaction.executed_at >= window_start,
            )
            .order_by(Transaction.executed_at.desc())
            .limit(args.limit)
        ).scalars().all()

        data = [pseudonymize_transaction(db, t) for t in txns]
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