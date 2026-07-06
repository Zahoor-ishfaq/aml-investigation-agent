"""
Agent investigation loop (ReAct-style).

Given an alert, iterates Groq calls with tool schemas until the model
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
- Groq is stateless per request. The full conversation is passed on
  every turn — nothing is stored server-side.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from aml_agent.agent.groq_client import chat
from aml_agent.tools.dispatch import all_tool_schemas, dispatch


logger = logging.getLogger("aml_agent.agent.session")

# Hard cap on tool-calling turns per alert. Prevents unbounded loops
# from cost, latency, and a subtler guardrail-adjacency: an agent that
# never terminates never reaches the guardrail layer. 15 is enough for
# multi-hop investigation across 4 tools; forcing termination at 15
# and escalating is safer than letting an under-decided agent close
# an alert autonomously.
MAX_ITERATIONS = 15


@dataclass
class ToolCallRecord:
    """One tool call the agent made during an investigation.

    Structured separately from the LLM's chat history because downstream
    consumers (guardrails, case file, audit_log) need the call/result
    pair as data, not as opaque conversation strings."""
    tool_name: str
    arguments: dict[str, Any]
    result_status: str            # "ok" | "error"
    result_row_count: int
    result_error: str | None = None


@dataclass
class InvestigationResult:
    """Final output of a session.

    - final_message: the agent's last non-tool-call response (natural
      language). Case file generation (7.4) will parse this into a
      structured decision.
    - tool_calls: chronological log of every tool the agent invoked.
    - message_history: full LLM transcript, retained for debugging /
      eval / audit reconstruction.
    - stopped_reason: 'complete' if the model finished naturally,
      'iteration_cap' if we forced termination at MAX_ITERATIONS.
    """
    alert_id: int
    final_message: str | None
    tool_calls: list[ToolCallRecord]
    message_history: list[dict[str, Any]]
    stopped_reason: str


class InvestigationSession:
    """
    Encapsulates one agent investigation of one alert.

    Kept as a class (not a bare function) because session state — history,
    tool-call log — must remain inspectable after failures for debugging
    and eval. A raise-and-lose-state design would obscure exactly where
    the loop went wrong.
    """

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
        # Message history seeded with system + first user turn. Every
        # subsequent turn appends to this list.
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

    def run(self) -> InvestigationResult:
        """
        Execute the ReAct loop. Returns InvestigationResult regardless
        of how the loop terminated — the caller checks stopped_reason
        to distinguish natural completion from forced cap termination.
        """
        tool_schemas = all_tool_schemas()
        stopped_reason = "iteration_cap"
        final_message: str | None = None

        for iteration in range(MAX_ITERATIONS):
            logger.info("iteration=%d alert_id=%d", iteration + 1, self.alert_id)

            response = chat(messages=self.messages, tools=tool_schemas)
            choice = response.choices[0]
            msg = choice.message

            # Append assistant turn to history in Groq/OpenAI shape.
            # tool_calls may be None (plain text) or a list.
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

            # Termination: plain text with no tool calls means the agent
            # is done reasoning and delivering its verdict / narrative.
            if not msg.tool_calls:
                final_message = msg.content or ""
                stopped_reason = "complete"
                break

            # Execute each tool call, append per-tool result message.
            # Groq requires exactly one tool result message per tool_call,
            # each keyed by the tool_call_id — mismatch causes 400.
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    # Malformed JSON args from the LLM — feed the error
                    # back so the model can self-correct next turn
                    # rather than crashing the whole session.
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
            # for..else fires when the loop exhausts without break —
            # here that means iteration cap hit without natural completion.
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
        )