"""
Groq API client wrapper.

Thin wrapper around the official groq SDK. Centralises model name,
sampling params, and one-time retry on transient network errors so the
agent loop (Phase 7.2) can focus on reasoning logic rather than API
plumbing.

Reference: Groq tool-use docs, https://console.groq.com/docs/tool-use
"""

import logging
import time
from typing import Any, Optional

from groq import Groq
from groq import APIConnectionError, InternalServerError

from aml_agent.config import settings


logger = logging.getLogger("aml_agent.agent.groq")


def _client() -> Groq:
    """
    Instantiate the SDK client. The Groq SDK reads GROQ_API_KEY from env
    by default, but we pass it explicitly so misconfigured .env files
    surface as a config error at startup (via pydantic-settings) rather
    than as a silent 401 at first request.
    """
    return Groq(api_key=settings.groq_api_key)


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    response_format: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Send a chat completion request and return the raw SDK response.

    Returns the SDK object directly (not a DTO) because the ReAct loop
    (7.2) needs access to tool_calls, finish_reason, and message content
    — wrapping it would just re-expose the same fields with more code.

    Retry policy: one retry on transient errors (connection reset, 5xx).
    The SDK already retries rate limits internally per Groq's own
    guidance, so we only defend against network hiccups here. No
    exponential backoff — at our request rate a fixed 2-second delay
    handles the failure mode without adding complexity.
    """
    client = _client()

    kwargs: dict[str, Any] = {
        "model": settings.groq_model_name,
        "messages": messages,
        "temperature": settings.groq_temperature,
        "max_tokens": settings.groq_max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if response_format:
        # JSON mode used for the final case file (7.4) to enforce a
        # schema the guardrail layer accepts directly. Not set on
        # tool-calling turns since Groq's function-calling and JSON
        # mode are mutually exclusive.
        kwargs["response_format"] = response_format

    try:
        return client.chat.completions.create(**kwargs)
    except (APIConnectionError, InternalServerError) as e:
        logger.warning("Groq transient error, retrying once: %r", e)
        time.sleep(2)
        return client.chat.completions.create(**kwargs)