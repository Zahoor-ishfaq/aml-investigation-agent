"""
Shared pytest fixtures.

Uses the SQLAlchemy-documented "join a nested transaction to an external
transaction" pattern for test isolation:

  1. Session-scoped connection opens a top-level transaction.
  2. Each test starts a SAVEPOINT (via begin_nested).
  3. On session.commit(), we restart the SAVEPOINT so rule code can
     legally call commit() and the test still rolls back cleanly.
  4. Test teardown rolls back the outer transaction — nothing persists.

Reference:
  https://docs.sqlalchemy.org/en/20/orm/session_transaction.html
    #joining-a-session-into-an-external-transaction-such-as-for-test-suites

This is the standard production pattern across mature Python codebases;
it's fast (no schema teardown), catches real Postgres constraint/trigger
behavior, and requires no separate test database.
"""

import pytest
from sqlalchemy.orm import Session

from aml_agent.db.session import engine


@pytest.fixture(scope="session")
def _connection():
    """Session-scoped DB connection — one per test session, not per test.
    Opening a new connection per test would dominate runtime at 100+ tests."""
    conn = engine.connect()
    yield conn
    conn.close()


@pytest.fixture
def db(_connection):
    """
    Function-scoped session bound to a rolled-back outer transaction.

    Any writes (including commits) inside the test are captured in a
    SAVEPOINT and never persisted — safe to run against dev Postgres
    without contaminating real data.
    """
    outer_txn = _connection.begin()
    session = Session(bind=_connection)
    # Start a SAVEPOINT that rule code's commit() will land in.
    nested = _connection.begin_nested()

    # If the code under test commits, restart the SAVEPOINT so subsequent
    # operations remain isolated within the outer transaction.
    from sqlalchemy import event

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, trans):
        nonlocal nested
        if trans.nested and not trans._parent.nested:
            nested = _connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        outer_txn.rollback()