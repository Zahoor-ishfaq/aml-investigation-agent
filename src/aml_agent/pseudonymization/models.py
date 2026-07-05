"""
Pseudonymization models.

Kept in its own module rather than db/models.py because the
pseudonymization layer is architecturally isolated from the rest of the
data model — treating it as a separate subsystem enforces that boundary
in the codebase, not just in documentation.
"""

from sqlalchemy import Column, DateTime, PrimaryKeyConstraint, Text, UniqueConstraint, func

from aml_agent.db.models import Base


class PseudonymMap(Base):
    """
    Reversible mapping (entity_type, real_value) <-> token.

    entity_type examples: 'customer_name', 'customer_external_ref',
    'account_external_ref', 'transaction_external_ref'. The type namespace
    is enforced in the tokenizer (not the DB) so adding a new type doesn't
    require a migration.
    """
    __tablename__ = "pseudonym_map"

    entity_type = Column(Text, nullable=False)
    real_value = Column(Text, nullable=False)
    token = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("entity_type", "real_value"),
        UniqueConstraint("entity_type", "token", name="uq_pseudonym_map_entity_type_token"),
    )