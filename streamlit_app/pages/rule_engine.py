"""
Rule Engine breakdown page.

Answers "which rules are firing, how often, and at what severity?" —
the first question an AML ops team asks when tuning detection thresholds.
A rule that fires thousands of times at low severity is a noise source;
a rule that fires rarely at high severity is working as intended.
"""

import streamlit as st
from streamlit_app.db import query_df


def render() -> None:
    st.header("Rule Engine Breakdown")

    # --- Alerts per rule code ---
    st.subheader("Alerts by Rule Code")
    rule_counts = query_df("""
        SELECT rule_code, COUNT(*) AS count
        FROM alerts
        GROUP BY rule_code
        ORDER BY count DESC
    """)
    if rule_counts.empty:
        st.warning("No alerts found.")
        return

    st.bar_chart(rule_counts.set_index("rule_code"))

    # --- Severity distribution per rule (cross-tab) ---
    # Crosstab lets an analyst spot rules that consistently fire at
    # high severity (potential tuning targets) vs rules that fire
    # broadly across all severities (potential noise sources).
    st.subheader("Severity Distribution per Rule")
    rule_severity = query_df("""
        SELECT rule_code, severity, COUNT(*) AS count
        FROM alerts
        GROUP BY rule_code, severity
        ORDER BY rule_code, severity
    """)
    if not rule_severity.empty:
        pivot = rule_severity.pivot_table(
            index="rule_code",
            columns="severity",
            values="count",
            fill_value=0,
        )
        pivot.columns = [f"Sev {int(c)}" for c in pivot.columns]
        st.dataframe(pivot, use_container_width=True)

    # --- Rule-level stats table ---
    # Escalation rate per rule = fraction of alerts that left 'open'
    # status. High escalation rate on a rule means either the rule is
    # well-targeted or the agent is over-escalating on that rule's
    # alerts — distinguishing requires cross-referencing with eval
    # results (Phase 9), but surfacing the rate here is the first step.
    st.subheader("Rule Statistics")
    rule_stats = query_df("""
        SELECT
            rule_code,
            COUNT(*)                                        AS total_alerts,
            ROUND(AVG(severity), 1)                         AS avg_severity,
            COUNT(*) FILTER (WHERE status = 'escalated')    AS escalated,
            COUNT(*) FILTER (WHERE status IN (
                'closed_false_positive','closed_no_action','closed_sar_filed'
            ))                                              AS closed,
            ROUND(
                COUNT(*) FILTER (WHERE status = 'escalated') * 100.0
                / NULLIF(COUNT(*), 0), 1
            )                                               AS escalation_pct
        FROM alerts
        GROUP BY rule_code
        ORDER BY total_alerts DESC
    """)
    st.dataframe(rule_stats, use_container_width=True)