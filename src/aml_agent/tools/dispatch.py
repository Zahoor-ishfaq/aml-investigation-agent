"""
Tool dispatch layer.

Single entry point (`dispatch`) that the agent (Phase 7) will call for
every LLM-issued tool_call. Responsibilities: look up the tool by name,
execute it, write an audit_log entry, return the ToolResult.

Tools themselves are pure logic. This layer owns the operational
concerns (audit, error boundary, unknown-tool handling) so every tool
implementation stays trivially unit-testable.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from aml_agent.db.models import AuditLog
from aml_agent.tools.base import Tool, ToolResult, TOOL_REGISTRY

# Importing each tool module triggers @register_tool at import time,
# populating TOOL_REGISTRY. Explicit imports (not pkgutil discovery)
# keep the tool set auditable — a reviewer can grep for these to see
# exactly which tools the agent has access to. Same pattern as the
# rule engine's runner.
from aml_agent.tools import transaction_history  # noqa: F401
from aml_agent.tools import customer_profile  # noqa: F401
from aml_agent.tools import linked_accounts  # noqa: F401
from aml_agent.tools import alert_history  # noqa: F401


def all_tool_schemas() -> list[dict[str, Any]]:
    """
    Return the OpenAI/Groq-format function schemas for every registered
    tool. Passed to Groq's `tools=` parameter when invoking the model —
    this is what the LLM reads to know what tools exist and how to call
    them.
    """
    return [cls.json_schema() for cls in TOOL_REGISTRY.values()]


def dispatch(
    db: Session,
    tool_name: str,
    arguments: dict[str, Any],
    alert_id: Optional[int] = None,
    actor: str = "investigation_agent",
) -> ToolResult:
    """
    Execute a tool by name, audit the call, return the result.

    alert_id anchors the audit entry to the investigation-in-progress
    when supplied — makes it trivial later to reconstruct "every tool
    call the agent made while investigating alert 12345". Absent an
    alert_id, we fall back to a timestamp-based entity_id so the audit
    row still has a stable identifier (consistent with the ingestion
    script and rule engine).

    Errors from unknown tools are returned as ToolResult(status='error')
    rather than raised — the agent must be able to observe and self-correct
    on bad calls, not have the reasoning loop crash.
    """
    tool_cls = TOOL_REGISTRY.get(tool_name)

    # Compute entity_id once. Timestamp uses millisecond precision so
    # concurrent tool calls within the same alert investigation don't
    # collide in audit_log.
    entity_id = alert_id if alert_id is not None else int(
        datetime.now(timezone.utc).timestamp() * 1000
    )
    entity_type = "alert" if alert_id is not None else "tool_call"

    if tool_cls is None:
        # Log the bad call — the audit trail must capture what the agent
        # tried, not only what succeeded. Regulators care about the full
        # trace of attempted actions, not the sanitized happy path.
        result = ToolResult(status="error", error=f"unknown tool: {tool_name}")
        _audit(db, entity_type, entity_id, actor, tool_name, arguments, result)
        return result

    tool = tool_cls()

    try:
        result = tool.execute(db, **arguments)
    except Exception as e:
        # Defensive catch: a tool raising instead of returning ToolResult
        # is a bug in that tool, but we still want an audit row and a
        # structured error the agent can react to. The exception message
        # is included so debugging doesn't require reading logs separately.
        result = ToolResult(status="error", error=f"tool raised exception: {e!r}")

    _audit(db, entity_type, entity_id, actor, tool_name, arguments, result)
    return result


def _audit(
    db: Session,
    entity_type: str,
    entity_id: int,
    actor: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> None:
    """
    Append the tool-call audit entry.

    details captures tool name, args (already pseudonymized since the
    agent only ever sees tokens), result status, and result row count.
    Not the full result payload — audit rows should be small; the
    reconstructable claim ("call X returned N rows with status Y") is
    what regulators need. Full payloads live in the LLM transcript.
    """
    db.add(AuditLog(
        entity_type=entity_type,
        entity_id=entity_id,
        action="tool_call",
        actor=actor,
        details={
            "tool_name": tool_name,
            "arguments": arguments,
            "result_status": result.status,
            "result_count": len(result.data),
            "error": result.error,
        },
    ))
    db.commit()