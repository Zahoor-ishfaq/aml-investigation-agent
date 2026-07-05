"""
Tool: query customer KYC profile.

Given an account token OR a customer token, returns the pseudonymized
KYC profile (name token, age bucket, country, risk rating, etc). Accepts
either identifier because the agent may reach this tool from different
starting points: an alert (has account_token) or a linked-accounts result
(has customer_token).
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from aml_agent.db.models import Account, Customer
from aml_agent.pseudonymization.service import pseudonymize_customer
from aml_agent.pseudonymization.tokenizer import depseudonymize
from aml_agent.tools.base import Tool, ToolResult, register_tool


class CustomerProfileArgs(BaseModel):
    """
    Args for get_customer_profile. Exactly one of account_token /
    customer_token must be provided — enforced via validator so bad
    calls surface as tool errors (agent-actionable) rather than DB errors.
    """
    account_token: Optional[str] = Field(
        None,
        description="Pseudonymized account token (AEXT_<hex>). Use when starting "
                    "from an alert or transaction and needing to know who owns the account.",
    )
    customer_token: Optional[str] = Field(
        None,
        description="Pseudonymized customer external-ref token (CEXT_<hex>). Use "
                    "when the customer identity is already known from a prior tool call.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "CustomerProfileArgs":
        if bool(self.account_token) == bool(self.customer_token):
            raise ValueError(
                "provide exactly one of account_token or customer_token, not both / neither"
            )
        return self


@register_tool
class CustomerProfileTool(Tool):
    """Return the KYC profile for the customer owning a given account
    or referenced by a given customer token."""

    name = "get_customer_profile"
    description = (
        "Retrieve the KYC profile (customer type, age bucket, country, risk rating, "
        "onboarding date) for a customer. Provide EITHER an account_token OR a "
        "customer_token — not both. Use to establish baseline risk context before "
        "assessing whether observed transactional behavior is anomalous."
    )
    args_schema = CustomerProfileArgs

    def _run(self, db: Session, args: CustomerProfileArgs) -> ToolResult:
        # Two resolution paths converge on a Customer row.
        if args.account_token is not None:
            real_ext_ref = depseudonymize(db, "account_external_ref", args.account_token)
            if real_ext_ref is None:
                return ToolResult(
                    status="error",
                    error=f"account token '{args.account_token}' not found",
                )
            # Account -> customer via FK. Single join, not two round trips.
            customer = db.execute(
                select(Customer)
                .join(Account, Account.customer_id == Customer.customer_id)
                .where(Account.external_ref == real_ext_ref)
            ).scalar_one_or_none()
        else:
            real_ext_ref = depseudonymize(db, "customer_external_ref", args.customer_token)
            if real_ext_ref is None:
                return ToolResult(
                    status="error",
                    error=f"customer token '{args.customer_token}' not found",
                )
            customer = db.execute(
                select(Customer).where(Customer.external_ref == real_ext_ref)
            ).scalar_one_or_none()

        if customer is None:
            return ToolResult(
                status="error",
                error="referenced entity resolved but no matching customer row (stale pseudonym map)",
            )

        return ToolResult(
            status="ok",
            data=[pseudonymize_customer(db, customer)],
            metadata={"count": 1},
        )