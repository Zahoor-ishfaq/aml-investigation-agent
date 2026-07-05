from aml_agent.db.session import get_db
from sqlalchemy import text

with get_db() as db:
    print(db.execute(text("SELECT 1")).scalar())