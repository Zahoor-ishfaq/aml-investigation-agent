"""
Google Gemini API client wrapper (native SDK).

Uses google-genai (not the OpenAI-compatible endpoint) because Google's
new Auth keys (AQ. prefix, mandatory since June 2026 per Google's
api-key docs) are rejected by the OpenAI-compatible gateway.

Exposes the same chat() signature as groq_client — returns an object
with .choices[0].message.tool_calls, .usage.prompt_tokens, etc. so
session.py needs zero changes. The adapter dataclasses below translate
Gemini's native response shape into OpenAI's, absorbing the provider
difference at the boundary rather than leaking it into the agent loop.

Gemini free tier (per Google AI rate-limits docs, 2026):
  15 RPM, 1,500 RPD, 1M TPM — no credit card, no expiration.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from google import genai
from google.genai import types as gemini_types

from aml_agent.config import settings


logger = logging.getLogger("aml_agent.agent.gemini")


# ---------------------------------------------------------------------------
# Adapter dataclasses: Gemini response → OpenAI-compatible shape
#
# session.py accesses response.choices[0].message.content,
# .message.tool_calls[i].id/.function.name/.function.arguments,
# and response.usage.prompt_tokens/completion_tokens/total_tokens.
# These thin wrappers provide exactly that interface without importing
# or depending on the openai package.
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    name: str
    arguments: str  # JSON string, matching OpenAI's convention


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
    """Minimal OpenAI-compatible response wrapper over a Gemini result."""
    choices: list[_Choice]
    usage: _Usage


def _convert_tools(openai_tools: list[dict[str, Any]]) -> list[gemini_types.Tool]:
    """Convert OpenAI-format tool schemas to Gemini function declarations.

    OpenAI shape:
      {"type": "function", "function": {"name": ..., "description": ...,
       "parameters": {...json-schema...}}}

    Gemini shape:
      types.Tool(function_declarations=[types.FunctionDeclaration(
        name=..., description=..., parameters=...)])

    Kept as a pure data transform — no side effects, no API calls —
    so it's safe to call on every chat() invocation without caching
    concerns.
    """
    declarations = []
    for tool in openai_tools:
        func = tool.get("function", {})
        declarations.append(gemini_types.FunctionDeclaration(
            name=func.get("name", ""),
            description=func.get("description", ""),
            parameters=func.get("parameters"),
        ))
    return [gemini_types.Tool(function_declarations=declarations)]


def _build_contents(messages: list[dict[str, Any]]) -> tuple[Optional[str], list[gemini_types.Content]]:
    """Convert OpenAI-format message history to Gemini contents.

    Returns (system_instruction, contents) because Gemini handles
    system prompts separately from the conversation history, unlike
    OpenAI/Groq which include them as a message role.

    Role mapping:
      OpenAI "system"    → extracted as system_instruction (string)
      OpenAI "user"      → Gemini "user"
      OpenAI "assistant"  → Gemini "model"
      OpenAI "tool"      → Gemini "user" with FunctionResponse part
    """
    system_instruction = None
    contents: list[gemini_types.Content] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system_instruction = msg.get("content", "")

        elif role == "user":
            contents.append(gemini_types.Content(
                role="user",
                parts=[gemini_types.Part.from_text(text=msg.get("content", ""))],
            ))

        elif role == "assistant":
            parts: list[gemini_types.Part] = []
            if msg.get("content"):
                parts.append(gemini_types.Part.from_text(text=msg["content"]))
            # Reconstruct function_call parts from tool_calls if present,
            # so Gemini sees the assistant's prior tool invocations in
            # the conversation history (required for multi-turn tool use).
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                parts.append(gemini_types.Part.from_function_call(
                    name=func.get("name", ""),
                    args=args,
                ))
            if parts:
                contents.append(gemini_types.Content(role="model", parts=parts))

        elif role == "tool":
            # Tool results map to FunctionResponse parts. Gemini expects
            # these as "user" role content, keyed by function name.
            # We parse the result JSON to extract the function name from
            # the preceding assistant turn's tool_calls via tool_call_id
            # matching — but Gemini only needs the content, so we pass
            # it as a generic function response.
            tool_call_id = msg.get("tool_call_id", "")
            result_content = msg.get("content", "{}")
            try:
                result_data = json.loads(result_content)
            except json.JSONDecodeError:
                result_data = {"raw": result_content}

            # Find the function name from the preceding assistant message's
            # tool_calls by matching tool_call_id.
            func_name = _find_func_name(messages, tool_call_id)
            parts = [gemini_types.Part.from_function_response(
                name=func_name,
                response=result_data,
            )]
            contents.append(gemini_types.Content(role="user", parts=parts))

    return system_instruction, contents


def _find_func_name(messages: list[dict[str, Any]], tool_call_id: str) -> str:
    """Resolve a tool_call_id back to its function name by scanning
    prior assistant messages. Falls back to 'unknown' if unresolvable
    (defensive — should never happen with well-formed history)."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            if tc.get("id") == tool_call_id:
                return tc.get("function", {}).get("name", "unknown")
    return "unknown"


def _adapt_response(gemini_response) -> _Response:
    """Translate a Gemini GenerateContentResponse into the OpenAI-
    compatible shape that session.py expects."""
    candidate = gemini_response.candidates[0]
    parts = candidate.content.parts

    text_parts = []
    tool_calls = []

    for part in parts:
        if part.text is not None:
            text_parts.append(part.text)
        elif part.function_call is not None:
            fc = part.function_call
            tool_calls.append(_ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=_Function(
                    name=fc.name,
                    arguments=json.dumps(dict(fc.args) if fc.args else {}),
                ),
            ))

    content = "\n".join(text_parts) if text_parts else None

    # Usage metadata — Gemini uses different field names.
    usage_meta = gemini_response.usage_metadata
    usage = _Usage(
        prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
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

def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    response_format: Optional[dict[str, Any]] = None,
) -> _Response:
    """
    Send a chat completion request to Gemini and return an
    OpenAI-compatible response object.

    Retry policy: one retry on transient errors. RateLimitError
    propagates for eval_agent.py to handle via sleep-and-retry.
    """
    client = _client()

    system_instruction, contents = _build_contents(messages)

    config: dict[str, Any] = {
        "temperature": settings.gemini_temperature,
        "max_output_tokens": settings.gemini_max_tokens,
    }
    if response_format and response_format.get("type") == "json_object":
        config["response_mime_type"] = "application/json"

    kwargs: dict[str, Any] = {
        "model": settings.gemini_model_name,
        "contents": contents,
        "config": gemini_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
            tools=_convert_tools(tools) if tools else None,
        ),
    }

    try:
        # Pace requests to stay under Gemini's free-tier RPM limit.
        time.sleep(4)
        response = client.models.generate_content(**kwargs)
        return _adapt_response(response)
    except Exception as e:
        error_str = str(e).lower()
        # 429 / quota errors must propagate to eval_agent for
        # sleep-and-retry — do NOT retry here.
        if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
            raise
        # Transient errors (503, network) — one retry.
        if "503" in error_str or "unavailable" in error_str or "internal" in error_str:
            logger.warning("Gemini transient error, retrying once: %r", e)
            time.sleep(3)
            response = client.models.generate_content(**kwargs)
            return _adapt_response(response)
        raise