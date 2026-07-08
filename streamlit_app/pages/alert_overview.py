"""
Alert Overview page.

Top-level view of the alert pipeline: how many alerts exist, their
current status distribution, and severity breakdown. Designed as the
first thing an AML compliance officer sees — answers "what's the
current state of my alert queue?" at a glance.
"""

import streamlit as st
from streamlit_app.db import query_df


def render() -> None:
    st.header("Alert Overview")

    # --- Metric cards: headline counts ---
    totals = query_df("""
        SELECT
            COUNT(*)                                           AS total,
            COUNT(*) FILTER (WHERE status = 'open')            AS open,
            COUNT(*) FILTER (WHERE status = 'escalated')       AS escalated,
            COUNT(*) FILTER (WHERE status = 'under_review')    AS under_review,
            COUNT(*) FILTER (WHERE status IN (
                'closed_false_positive','closed_no_action','closed_sar_filed'
            ))                                                 AS closed
        FROM alerts
    """)

    if totals.empty:
        st.warning("No alerts found in database.")
        return

    row = totals.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts", int(row["total"]))
    col2.metric("Open", int(row["open"]))
    col3.metric("Escalated", int(row["escalated"]))
    col4.metric("Closed", int(row["closed"]))

    # --- Status distribution bar chart ---
    st.subheader("Alerts by Status")
    status_df = query_df("""
        SELECT status, COUNT(*) AS count
        FROM alerts
        GROUP BY status
        ORDER BY count DESC
    """)
    st.bar_chart(status_df.set_index("status"))

    # --- Severity distribution ---
    st.subheader("Alerts by Severity")
    severity_df = query_df("""
        SELECT severity, COUNT(*) AS count
        FROM alerts
        GROUP BY severity
        ORDER BY severity
    """)
    st.bar_chart(severity_df.set_index("severity"))

    # --- Recent alerts table ---
    st.subheader("Recent Alerts")
    recent_df = query_df("""
        SELECT alert_id, status, severity, rule_code, created_at, updated_at
        FROM alerts
        ORDER BY created_at DESC
        LIMIT 20
    """)
    st.dataframe(recent_df, use_container_width=True)