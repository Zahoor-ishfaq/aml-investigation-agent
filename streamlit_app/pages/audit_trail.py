"""
Audit Trail viewer page.

Searchable, filterable view over the immutable audit_log table.
Designed for compliance review: a regulator or internal auditor can
filter by alert_id, actor, or action type and see the full chain of
events that led to a disposition — from rule engine trigger through
agent investigation to guardrail evaluation.
"""

import json

import pandas as pd
import streamlit as st
from streamlit_app.db import query_df


def render() -> None:
    st.header("Audit Trail")

    # --- Filters ---
    col1, col2, col3 = st.columns(3)

    with col1:
        alert_filter = st.text_input("Filter by Alert ID", placeholder="e.g. 8321")

    with col2:
        actions = query_df("SELECT DISTINCT action FROM audit_log ORDER BY action")
        action_list = ["All"] + actions["action"].tolist()
        action_filter = st.selectbox("Action", action_list)

    with col3:
        actors = query_df("SELECT DISTINCT actor FROM audit_log ORDER BY actor")
        actor_list = ["All"] + actors["actor"].tolist()
        actor_filter = st.selectbox("Actor", actor_list)

    # --- Build query with filters ---
    conditions = []
    params = {}

    if alert_filter.strip():
        conditions.append("entity_id = :alert_id")
        params["alert_id"] = int(alert_filter.strip())

    if action_filter != "All":
        conditions.append("action = :action")
        params["action"] = action_filter

    if actor_filter != "All":
        conditions.append("actor = :actor")
        params["actor"] = actor_filter

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    results = query_df(
        f"""
        SELECT audit_id, entity_type, entity_id AS alert_id,
               action, actor, details, occurred_at
        FROM audit_log
        {where}
        ORDER BY occurred_at DESC
        LIMIT 100
        """,
        params=params if params else None,
    )

    if results.empty:
        st.info("No audit records match the current filters.")
        return

    st.caption(f"Showing {len(results)} records (max 100)")

    # --- Format details column for readability ---
    def format_details(val):
        if val is None:
            return ""
        if isinstance(val, dict):
            return json.dumps(val, indent=2)
        try:
            return json.dumps(json.loads(val), indent=2)
        except (json.JSONDecodeError, TypeError):
            return str(val)

    results["details"] = results["details"].apply(format_details)
    st.dataframe(results, use_container_width=True)

    # --- Expandable detail view ---
    st.subheader("Inspect Record")
    selected = st.selectbox(
        "Select audit_id to inspect",
        results["audit_id"].tolist(),
    )
    if selected:
        row = results[results["audit_id"] == selected].iloc[0]
        st.json(json.loads(row["details"]) if row["details"] else {})