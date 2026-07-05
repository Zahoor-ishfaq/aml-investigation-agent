from sqlalchemy import select
from aml_agent.db.session import get_db
from aml_agent.db.models import Customer
from aml_agent.pseudonymization.service import pseudonymize_customer
from aml_agent.pseudonymization.depseudonymize import (
    depseudonymize_case_file,
    resolve_tokens_in_text,
)

with get_db() as db:
    c = db.execute(select(Customer).limit(1)).scalar_one()
    p = pseudonymize_customer(db, c)
    print("pseudonymized :", p)
    print("depseudonymized:", depseudonymize_case_file(db, p))

    narrative = f"Alert: customer {p['name_token']} (ref {p['customer_token']}) flagged."
    print("narrative in :", narrative)
    print("narrative out:", resolve_tokens_in_text(db, narrative))