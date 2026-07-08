"""
Agent Eval metrics page.

Reads the persisted eval_summary.json from Phase 9's score_eval.py —
no DB queries, no Groq/Claude calls, no token spend. Displays the
detection correctness metrics that tell an AML ops team whether the
agent is safe to deploy: recall (are we catching laundering?) and
precision (are we drowning the human queue in false positives?).
"""

import json
from pathlib import Path

import streamlit as st


_SUMMARY_PATH = Path("eval_results/eval_summary.json")


def render() -> None:
    st.header("Agent Eval Metrics")

    if not _SUMMARY_PATH.exists():
        st.warning(
            f"`{_SUMMARY_PATH}` not found. "
            "Run `python scripts/score_eval.py` to generate it."
        )
        return

    summary = json.loads(_SUMMARY_PATH.read_text())
    det = summary["detection"]
    qual = summary["quality"]
    cost = summary["cost"]
    lat = summary["latency"]

    # --- Detection correctness headline ---
    st.subheader("Detection Correctness")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{det['precision']:.4f}")
    c2.metric("Recall", f"{det['recall']:.4f}")
    c3.metric("F1", f"{det['f1']:.4f}")
    c4.metric("Scored Alerts", det["scored_count"])

    # --- Confusion matrix ---
    st.subheader("Confusion Matrix")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            f"""
            |  | **Predicted Positive** | **Predicted Negative** |
            |---|---|---|
            | **Actually Laundering** | TP = {det['tp']} | FN = {det['fn']} |
            | **Not Laundering** | FP = {det['fp']} | TN = {det['tn']} |
            """
        )
    with col_right:
        # Visual emphasis on the two numbers that matter most in AML:
        # FN (missed laundering) and FP (wasted analyst time).
        if det["fn"] == 0:
            st.success("Zero false negatives — no laundering missed.")
        else:
            st.error(f"{det['fn']} false negative(s) — laundering alerts dismissed.")
        if det["fp"] > 0:
            st.warning(f"{det['fp']} false positive(s) — over-escalation.")

    # --- Case file quality ---
    st.subheader("Case File Quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Parse OK Rate", f"{qual['parse_ok_rate']:.1%}")
    q2.metric("Complete Rate", f"{qual['complete_rate']:.1%}")
    q3.metric("Error Rate", f"{qual['error_rate']:.1%}")

    g1, g2, g3 = st.columns(3)
    g1.metric("Guardrail Pass", f"{qual['guardrail_pass_rate']:.1%}")
    g2.metric("Guardrail Block", f"{qual['guardrail_block_rate']:.1%}")
    g3.metric("Guardrail Review", f"{qual['guardrail_review_rate']:.1%}")

    # --- Cost & latency ---
    st.subheader("Cost & Latency")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Tokens", f"{cost['total_tokens']:,}")
    t2.metric("Mean / Alert", f"{cost['mean_tokens_per_alert']:,.0f}")
    t3.metric("Mean Latency", f"{lat['mean_sec']:.1f}s")
    t4.metric("Max Latency", f"{lat['max_sec']:.1f}s")