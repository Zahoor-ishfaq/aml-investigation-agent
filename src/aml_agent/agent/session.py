"""
Agent investigation loop (ReAct-style).

Given an alert, iterates LLM calls with tool schemas until the model
either produces a plain-text response (done) or hits the iteration cap.
Each tool_call in a response is dispatched via Phase 4's dispatch(),
its result appended to the message history for the next turn.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in
Language Models", arXiv:2210.03629.

Design notes:
- Message history and tool-call log are kept as separate structures.
  The message history is what the LLM needs (its conversational state);
  the tool-call log is what downstream consumers need (guardrail's
  EvidenceTrailGuardrail, the case file, the audit trail). Different
  shapes for different consumers.
- The LLM provider is stateless per request. The full conversation is
  passed on every turn — nothing is stored server-side. This is also
  why token cost grows with iteration count: each turn re-sends the
  entire accumulated history plus tool schemas.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

# --- Provider swap: change this single import to switch providers ---
# groq_client and cerebras_client expose the same chat() signature
# and return the same OpenAI-compatible response shape.
from aml_agent.agent.claude_client import chat
from aml_agent.tools.dispatch import all_tool_schemas, dispatch


logger = logging.getLogger("aml_agent.agent.session")

MAX_ITERATIONS = 15


@dataclass
class ToolCallRecord:
    """One tool call the agent made during an investigation."""
    tool_name: str
    arguments: dict[str, Any]
    result_status: str            # "ok" | "error"
    result_row_count: int
    result_error: str | None = None


@dataclass
class InvestigationResult:
    """Final output of a session."""
    alert_id: int
    final_message: str | None
    tool_calls: list[ToolCallRecord]
    message_history: list[dict[str, Any]]
    stopped_reason: str
    token_usage: list[dict[str, int]] = field(default_factory=list)


def _build_valid_tool_names(tool_schemas: list[dict[str, Any]]) -> set[str]:
    """Extract the set of tool names the agent is allowed to call.

    Used to detect when the model invents a bogus tool name (e.g.
    'json', 'JSON') instead of returning plain-text content — a known
    failure mode in several models where the final JSON decision gets
    wrapped as a fake tool call."""
    names: set[str] = set()
    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name")
        if name:
            names.add(name)
    return names


class InvestigationSession:
    """Encapsulates one agent investigation of one alert."""

    def __init__(
        self,
        db: Session,
        alert_id: int,
        system_prompt: str,
        initial_user_message: str,
    ):
        self.db = db
        self.alert_id = alert_id
        self.tool_calls: list[ToolCallRecord] = []
        self.token_usage: list[dict[str, int]] = []
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

    def run(self) -> InvestigationResult:
        """Execute the ReAct loop. Returns InvestigationResult regardless
        of how the loop terminated."""
        tool_schemas = all_tool_schemas()
        valid_tool_names = _build_valid_tool_names(tool_schemas)
        stopped_reason = "iteration_cap"
        final_message: str | None = None

        for iteration in range(MAX_ITERATIONS):
            logger.info("iteration=%d alert_id=%d", iteration + 1, self.alert_id)

            response = chat(messages=self.messages, tools=tool_schemas)
            choice = response.choices[0]
            msg = choice.message

            # Capture token usage before any parsing — the spend
            # happened regardless of what we do with the response.
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.token_usage.append({
                    "iteration": iteration + 1,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                })

            # Append assistant turn to history.
            assistant_turn: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_turn["content"] = msg.content
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(assistant_turn)

            # Plain text with no tool calls = agent is done.
            if not msg.tool_calls:
                final_message = msg.content or ""
                stopped_reason = "complete"
                break

            # --- Bogus tool-call recovery ---
            # Some models wrap their final JSON decision as a tool call
            # named 'json'/'JSON' not in request.tools. Recover the
            # arguments as the final decision instead of crashing.
            bogus_calls = [
                tc for tc in msg.tool_calls
                if tc.function.name not in valid_tool_names
            ]
            if bogus_calls:
                bogus = bogus_calls[0]
                logger.warning(
                    "alert_id=%d: unrecognized tool call '%s' — "
                    "treating arguments as final decision",
                    self.alert_id, bogus.function.name,
                )
                final_message = bogus.function.arguments
                stopped_reason = "complete"
                break

            # Execute each valid tool call.
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    arguments = {}
                    result_content = json.dumps({
                        "status": "error",
                        "error": f"invalid JSON in arguments: {e}",
                    })
                    self.tool_calls.append(ToolCallRecord(
                        tool_name=tc.function.name,
                        arguments={},
                        result_status="error",
                        result_row_count=0,
                        result_error=f"invalid JSON: {e}",
                    ))
                else:
                    result = dispatch(
                        self.db,
                        tool_name=tc.function.name,
                        arguments=arguments,
                        alert_id=self.alert_id,
                    )
                    result_content = json.dumps(result.to_dict())
                    self.tool_calls.append(ToolCallRecord(
                        tool_name=tc.function.name,
                        arguments=arguments,
                        result_status=result.status,
                        result_row_count=len(result.data),
                        result_error=result.error,
                    ))

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                })

        else:
            logger.warning(
                "iteration cap reached alert_id=%d tool_calls=%d",
                self.alert_id, len(self.tool_calls),
            )

        return InvestigationResult(
            alert_id=self.alert_id,
            final_message=final_message,
            tool_calls=self.tool_calls,
            message_history=self.messages,
            stopped_reason=stopped_reason,
            token_usage=self.token_usage,
        )