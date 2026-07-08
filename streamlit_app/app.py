"""
Streamlit monitoring dashboard — entry point.

Phase 10: read-only dashboard over the AML investigation pipeline.
Run: streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# Project root on sys.path so both streamlit_app.* and aml_agent.*
# imports resolve when Streamlit runs this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AML Investigation Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.sidebar.title("AML Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Alert Overview", "Rule Engine", "Agent Eval", "Guardrails", "Audit Trail"],
    label_visibility="collapsed",
)

if page == "Alert Overview":
    from streamlit_app.pages.alert_overview import render
    render()
elif page == "Rule Engine":
    from streamlit_app.pages.rule_engine import render
    render()
elif page == "Agent Eval":
    from streamlit_app.pages.agent_eval import render
    render()
elif page == "Guardrails":
    from streamlit_app.pages.guardrails import render
    render()
elif page == "Audit Trail":
    from streamlit_app.pages.audit_trail import render
    render()