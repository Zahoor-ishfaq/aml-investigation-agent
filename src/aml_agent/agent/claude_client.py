"""
Anthropic Claude API client wrapper.

Drop-in replacement for groq_client.chat() using the Anthropic SDK.
Returns an OpenAI-compatible response shape via adapter dataclasses
so session.py needs zero changes.

Anthropic tool-use docs: https://docs.claude.com/en/docs/build-with-claude/tool-use/overview
Anthropic messages API:  https://docs.claude.com/en/api/messages
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from anthropic import Anthropic, RateLimitError, APIConnectionError, InternalServerError

from aml_agent.config import settings


logger = logging.getLogger("aml_agent.agent.claude")


# ---------------------------------------------------------------------------
# Adapter dataclasses: Anthropic response → OpenAI-compatible shape
#
# session.py accesses response.choices[0].message.content,
# .message.tool_calls[i].id/.function.name/.function.arguments,
# and response.usage.prompt_tokens/completion_tokens/total_tokens.
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    name: str
    arguments: str  # JSON string per OpenAI convention


@dataclass
class _ToolCall:
    id: str
    type: str  # always "function"
    function: _Function


@dataclass
class _Message:
    content: Optional[str]
    tool_calls: Optional[list[_ToolCall]]


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage


def _convert_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool schemas to Anthropic format.

    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}
    """
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for tool in openai_tools
    ]


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Anthropic format.

    Key differences from OpenAI/Groq:
      - System prompt is a separate parameter, not a message role.
      - Tool results use role="user" with content type="tool_result",
        not a dedicated "tool" role.
      - Assistant tool calls are content blocks with type="tool_use",
        not a separate tool_calls field.
    """
    system = None
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system = msg.get("content", "")

        elif role == "user":
            anthropic_messages.append({
                "role": "user",
                "content": msg.get("content", ""),
            })

        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args,
                })
            if content_blocks:
                anthropic_messages.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            # Anthropic expects tool results as user messages with
            # tool_result content blocks, keyed by tool_use_id.
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })

    return system, anthropic_messages


def _adapt_response(anthropic_response) -> _Response:
    """Translate Anthropic MessageResponse to OpenAI-compatible shape."""
    text_parts = []
    tool_calls = []

    for block in anthropic_response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(_ToolCall(
                id=block.id,
                type="function",
                function=_Function(
                    name=block.name,
                    arguments=json.dumps(block.input),
                ),
            ))

    content = "\n".join(text_parts) if text_parts else None
    usage = _Usage(
        prompt_tokens=anthropic_response.usage.input_tokens,
        completion_tokens=anthropic_response.usage.output_tokens,
        total_tokens=anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
    )

    return _Response(
        choices=[_Choice(message=_Message(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        ))],
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Public interface — same signature as groq_client.chat()
# ---------------------------------------------------------------------------

def _client() -> Anthropic:
    return Anthropic(api_key=settings.claude_api_key)


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    response_format: Optional[dict[str, Any]] = None,
) -> _Response:
    """
    Send a messages request to Claude and return an OpenAI-compatible
    response object.

    Retry: one retry on transient errors. RateLimitError propagates
    for eval_agent.py to handle via sleep-and-retry.
    """
    client = _client()
    system, anthropic_messages = _convert_messages(messages)

    kwargs: dict[str, Any] = {
        "model": settings.claude_model_name,
        "messages": anthropic_messages,
        "temperature": settings.claude_temperature,
        "max_tokens": settings.claude_max_tokens,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _convert_tools(tools)
        # Anthropic tool_choice format differs from OpenAI:
        # {"type": "auto"} not just "auto"
        kwargs["tool_choice"] = {"type": "auto"}

    try:
        response = client.messages.create(**kwargs)
        return _adapt_response(response)
    except (APIConnectionError, InternalServerError) as e:
        logger.warning("Claude transient error, retrying once: %r", e)
        time.sleep(2)
        response = client.messages.create(**kwargs)
        return _adapt_response(response)