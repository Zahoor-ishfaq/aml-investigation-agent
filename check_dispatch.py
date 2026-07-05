import json
from sqlalchemy import select
from aml_agent.db.session import get_db
from aml_agent.db.models import Account
from aml_agent.pseudonymization.tokenizer import pseudonymize
from aml_agent.tools.dispatch import dispatch, all_tool_schemas

with get_db() as db:
    # Grab a real account, tokenize its external_ref to simulate what the agent sees.
    a = db.execute(select(Account).limit(1)).scalar_one()
    tok = pseudonymize(db, "account_external_ref", a.external_ref)
    print(f"Testing with account token: {tok}\n")

    print("Registered tool schemas:")
    for s in all_tool_schemas():
        print(f"  - {s['function']['name']}: {s['function']['description'][:60]}...")
    print()

    # Call each tool
    for name, args in [
        ("get_transaction_history", {"account_token": tok, "limit": 3}),
        ("get_customer_profile", {"account_token": tok}),
        ("get_linked_accounts", {"account_token": tok, "limit": 3}),
        ("get_alert_history", {"account_token": tok, "limit": 5}),
    ]:
        r = dispatch(db, name, args)
        print(f"{name}: status={r.status}, count={len(r.data)}")