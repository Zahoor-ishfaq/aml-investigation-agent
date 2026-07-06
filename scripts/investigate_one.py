"""
CLI: invoke the agent on one alert end-to-end.

Picks an open alert (either the highest-severity open one by default,
or a specific alert_id if passed), runs investigate_alert, prints the
outcome. Intended for manual verification of the Phase 7 integration
before running the full eval harness (Phase 9).

Usage:
    python scripts/investigate_one.py               # highest-severity open alert
    python scripts/investigate_one.py --alert-id 5  # specific alert
"""

import argparse
import logging
import sys

from sqlalchemy import select

from aml_agent.agent.orchestrator import investigate_alert
from aml_agent.db.models import Alert
from aml_agent.db.session import get_db


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _pick_alert_id(db) -> int | None:
    """Pick the highest-severity open alert with the most recent
    created_at as tiebreaker. Deterministic-ish and gives us a
    meaningful signal to investigate on first run."""
    row = db.execute(
        select(Alert.alert_id)
        .where(Alert.status == "open")
        .order_by(Alert.severity.desc(), Alert.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on one alert.")
    parser.add_argument("--alert-id", type=int, default=None,
                        help="Specific alert ID. If omitted, picks the highest-severity open alert.")
    args = parser.parse_args()

    _configure_logging()

    with get_db() as db:
        alert_id = args.alert_id or _pick_alert_id(db)
        if alert_id is None:
            print("No open alerts to investigate.")
            sys.exit(1)

        print(f"Investigating alert #{alert_id}...\n")
        outcome = investigate_alert(db, alert_id)

    # Print outcome. Kept structured but not JSON since a human is reading
    # this in the terminal, not a machine.
    print("=" * 70)
    print(f"Alert ID:            {outcome.alert_id}")
    print(f"Applied status:      {outcome.applied_status}")
    print(f"Case file parse OK:  {outcome.case_file_parse_ok}")
    print(f"Stopped reason:      {outcome.stopped_reason}")
    if outcome.agent_decision:
        d = outcome.agent_decision
        print(f"Agent action:        {d.action.value}")
        print(f"Agent severity:      {d.severity}")
        print(f"Agent narrative:     {d.narrative}")
    if outcome.guardrail_verdict:
        print(f"Guardrail verdict:   {outcome.guardrail_verdict.value}")
    if outcome.guardrail_reasons:
        print(f"Guardrail reasons:")
        for r in outcome.guardrail_reasons:
            print(f"  - {r}")
    print(f"\nTool calls ({len(outcome.tool_calls)}):")
    for tc in outcome.tool_calls:
        print(f"  - {tc.tool_name}({tc.arguments}) -> {tc.result_status}, {tc.result_row_count} rows")
    print(f"\nDiagnostic: {outcome.diagnostic}")


if __name__ == "__main__":
    main()