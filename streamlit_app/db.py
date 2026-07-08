"""
Read-only database connection for the Streamlit dashboard.

Reuses the project's existing Settings.database_url so there's one
source of truth for connection params. Engine created once per process
via st.cache_resource (Streamlit's singleton cache) to avoid
reconnecting on every widget interaction — Streamlit reruns the full
script on each user action, so uncached connections would churn.
"""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from aml_agent.config import settings


@st.cache_resource
def _engine() -> Engine:
    """Singleton engine — cached for the lifetime of the Streamlit process."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame.

    All dashboard queries go through this function so the DB access
    pattern is uniform and auditable. Using raw SQL (not ORM) because
    dashboard queries are aggregation-heavy (GROUP BY, COUNT, joins)
    which are cleaner as SQL than as chained ORM calls.
    """
    with _engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)