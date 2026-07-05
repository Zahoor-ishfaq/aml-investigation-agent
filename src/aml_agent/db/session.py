"""
Database connection layer.

Single source of the SQLAlchemy engine and session factory for the entire
project. Every module that touches Postgres (ingestion, rule engine, tool
layer, audit writes) imports from here — one engine means one connection
pool, one place to tune pooling, and one place to fix a bug.

Reference: SQLAlchemy 2.0 session usage patterns,
https://docs.sqlalchemy.org/en/20/orm/session_basics.html
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aml_agent.config import settings


# ---------------------------------------------------------------------------
# Engine — one process-wide instance.
#
# pool_pre_ping=True issues a lightweight SELECT 1 before handing out any
# pooled connection. Without it, connections dropped by Postgres (idle
# timeout, Docker container restart, network blip) are handed to callers
# and only fail at first query with an opaque OperationalError. Cost is one
# extra round-trip on checkout, which is negligible for our workload but
# eliminates a whole class of flaky failures. Per SQLAlchemy docs' explicit
# recommendation for containerized/long-lived processes.
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,  # opt into SQLAlchemy 2.0-style API explicitly (defensive against future default flips)
)


# ---------------------------------------------------------------------------
# Session factory.
#
# expire_on_commit=False is deliberate: default behavior expires all ORM
# objects after commit, forcing a re-fetch on next access. For our batched
# ingestion and rule-engine writes we want to keep working with the objects
# post-commit (e.g. read the auto-assigned PK) without extra queries.
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session scoped to a single unit of work.

    Usage:
        with get_db() as db:
            db.add(some_row)
            db.commit()

    Guarantees the session is closed even if the caller raises — SQLAlchemy
    docs warn that leaked sessions hold connections out of the pool, so
    close() must run unconditionally. Commit is left to the caller: this
    layer manages *lifecycle*, not *transaction boundaries*, since ingestion
    (batch commit) and per-alert writes (single commit) have different needs.

    The generator form also matches FastAPI's dependency-injection contract
    (step 4's tool layer), so the same function serves scripts and endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()