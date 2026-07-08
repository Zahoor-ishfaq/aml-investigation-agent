"""
Guardrail Activity page.

Reads audit_log rows where action='guardrail_evaluated' to show which
guardrails are firing, how often, and whether they're blocking or
passing. Answers: "is my guardrail layer too aggressive (blocking
everything) or too permissive (passing everything)?" — the key
calibration question before trusting the agent in production.
"""

import json

import pandas as pd
import streamlit as st
from streamlit_app.db import query_df


def render() -> None:
    st.header("Guardrail Activity")

    # --- Load guardrail audit rows ---
    raw = query_df("""
        SELECT entity_id AS alert_id, details, occurred_at
        FROM audit_log
        WHERE action = 'guardrail_evaluated'
        ORDER BY occurred_at DESC
    """)

    if raw.empty:
        st.warning("No guardrail evaluations found in audit_log.")
        return

    # Parse the JSONB details column into structured data.
    # Each row's details contains: proposed_action, aggregate_verdict,
    # and a guardrails[] array with per-guardrail code/decision/reason.
    parsed_rows = []
    per_guardrail_rows = []
    for _, row in raw.iterrows():
        details = row["details"] if isinstance(row["details"], dict) else json.loads(row["details"])
        parsed_rows.append({
            "alert_id": row["alert_id"],
            "proposed_action": details.get("proposed_action"),
            "aggregate_verdict": details.get("aggregate_verdict"),
            "created_at": row["occurred_at"],
        })
        for g in details.get("guardrails", []):
            per_guardrail_rows.append({
                "alert_id": row["alert_id"],
                "guardrail_code": g.get("code"),
                "decision": g.get("decision"),
                "reason": g.get("reason", ""),
            })

    evals_df = pd.DataFrame(parsed_rows)
    guardrails_df = pd.DataFrame(per_guardrail_rows)

    # --- Aggregate verdict metric cards ---
    st.subheader("Aggregate Verdicts")
    total = len(evals_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluations", total)
    c2.metric("Pass", int((evals_df["aggregate_verdict"] == "pass").sum()))
    c3.metric("Block", int((evals_df["aggregate_verdict"] == "block").sum()))
    c4.metric("Review", int((evals_df["aggregate_verdict"] == "requires_review").sum()))

    # --- Per-guardrail block/pass counts ---
    if not guardrails_df.empty:
        st.subheader("Blocks by Guardrail Code")
        blocks = guardrails_df[guardrails_df["decision"] == "block"]
        if not blocks.empty:
            block_counts = blocks.groupby("guardrail_code").size().reset_index(name="blocks")
            block_counts = block_counts.sort_values("blocks", ascending=False)
            st.bar_chart(block_counts.set_index("guardrail_code"))
        else:
            st.success("No guardrail blocks recorded.")

        # --- Full per-guardrail stats table ---
        st.subheader("Per-Guardrail Summary")
        summary = guardrails_df.groupby(["guardrail_code", "decision"]).size().reset_index(name="count")
        pivot = summary.pivot_table(
            index="guardrail_code",
            columns="decision",
            values="count",
            fill_value=0,
        )
        st.dataframe(pivot, use_container_width=True)

    # --- Recent evaluations table ---
    st.subheader("Recent Evaluations")
    st.dataframe(evals_df.head(20), use_container_width=True)