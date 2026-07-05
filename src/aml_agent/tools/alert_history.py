"""
Tool: query alert history for an account.

Returns prior alerts written against the account with a lifecycle
summary. No field pseudonymization needed — rule_code, status, severity,
narrative all lack direct PII. Narratives from the rule engine contain
only counts and amounts; narratives from the agent (Phase 7) already
contain tokens, which stay as tokens.
"""

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aml_agent.db.models import Account, Alert
from aml_agent.pseudonymization.tokenizer import depseudonymize
from aml_agent.tools.base import Tool, ToolResult, register_tool


class AlertHistoryArgs(BaseModel):
    account_token: str = Field(
        ..., description="Pseudonymized account token, format AEXT_<hex>."
    )
    days_back: int = Field(
        180, ge=1, le=730,
        description="Look-back window in days. Default 180 — longer than "
                    "transaction lookups because alert history informs "
                    "pattern-of-scrutiny signal (3 prior escalations vs. 3 "
                    "prior false positives are very different contexts)."
    )
    limit: int = Field(
        20, ge=1, le=100,
        description="Max number of alerts to return. Default 20."
    )


@register_tool
class AlertHistoryTool(Tool):
    """Return prior alerts for an account, most recent first, plus a
    lifecycle-status summary in metadata."""

    name = "get_alert_history"
    description = (
        "Retrieve prior alerts raised against a given account, with a lifecycle "
        "summary (counts by status). Use to understand whether the account has been "
        "previously scrutinized and how those investigations were resolved — repeat "
        "false-positives shift priors differently than repeat SAR filings."
    )
    args_schema = AlertHistoryArgs

    def _run(self, db: Session, args: AlertHistoryArgs) -> ToolResult:
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

        # Window anchored on wall-clock: alerts are created by ongoing
        # rule engine runs, not tied to the simulation's fixed date window.
        # Using now() here is correct (unlike transaction_history / linked_
        # accounts, which anchor on data timestamps to avoid empty windows).
        window_start = datetime.now(timezone.utc) - timedelta(days=args.days_back)

        alerts = db.execute(
            select(Alert)
            .where(
                Alert.account_id == account.account_id,
                Alert.created_at >= window_start,
            )
            .order_by(Alert.created_at.desc())
            .limit(args.limit)
        ).scalars().all()

        # Lifecycle summary: one query over the FULL window (not limited)
        # so the counts are exhaustive even if `alerts` was truncated by
        # the display limit. This is what actually informs the agent's
        # prior — "5 prior false positives" is meaningful even if we only
        # return the top 20 for detail.
        summary_rows = db.execute(
            select(Alert.status, func.count().label("n"))
            .where(
                Alert.account_id == account.account_id,
                Alert.created_at >= window_start,
            )
            .group_by(Alert.status)
        ).all()
        summary = {str(row.status.value if hasattr(row.status, "value") else row.status): int(row.n)
                   for row in summary_rows}
        total = sum(summary.values())

        data = [
            {
                "alert_id": a.alert_id,
                "rule_code": a.rule_code,
                "source": a.source.value if hasattr(a.source, "value") else a.source,
                "severity": a.severity,
                "status": a.status.value if hasattr(a.status, "value") else a.status,
                "narrative": a.narrative,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in alerts
        ]

        return ToolResult(
            status="ok",
            data=data,
            metadata={
                "account_token": args.account_token,
                "returned_count": len(data),
                "window_days": args.days_back,
                "lifecycle_summary": summary,
                "total_in_window": total,
            },
        )