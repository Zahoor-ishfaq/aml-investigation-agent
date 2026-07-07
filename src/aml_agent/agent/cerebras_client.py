"""
Cerebras API client wrapper.

Drop-in replacement for groq_client.chat() using Cerebras's
OpenAI-compatible endpoint. Same function signature, same return
shape (choices[0].message, tool_calls, usage) — session.py only
needs a one-line import swap.

Cerebras free tier (https://inference-docs.cerebras.ai/):
  5 RPM, 30K TPM, 1M TPD — 10x Groq's daily token budget.

Uses the openai SDK pointed at Cerebras's base URL rather than a
Cerebras-specific SDK, because the response object shape must match
what session.py already parses (choices, message.tool_calls,
response.usage). Adding a new SDK would require adapting every
field access downstream.
"""

import logging
import time
from typing import Any, Optional

from openai import OpenAI, APIConnectionError, InternalServerError, RateLimitError

from aml_agent.config import settings


logger = logging.getLogger("aml_agent.agent.cerebras")

# Cerebras's OpenAI-compatible inference endpoint.
_BASE_URL = "https://api.cerebras.ai/v1"


def _client() -> OpenAI:
    """Instantiate the OpenAI SDK client pointed at Cerebras.
    Explicit api_key so misconfigured .env surfaces at startup."""
    return OpenAI(
        api_key=settings.cerebras_api_key,
        base_url=_BASE_URL,
    )


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    response_format: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Send a chat completion request to Cerebras and return the raw
    OpenAI-SDK response object.

    Retry policy mirrors groq_client: one retry on transient errors
    (connection reset, 5xx). RateLimitError is not retried here —
    it propagates to eval_agent.py which handles sleep-and-retry
    across the full alert loop.
    """
    client = _client()

    kwargs: dict[str, Any] = {
        "model": settings.cerebras_model_name,
        "messages": messages,
        "temperature": settings.cerebras_temperature,
        "max_tokens": settings.cerebras_max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format

    try:
        return client.chat.completions.create(**kwargs)
    except (APIConnectionError, InternalServerError) as e:
        logger.warning("Cerebras transient error, retrying once: %r", e)
        time.sleep(2)
        return client.chat.completions.create(**kwargs)