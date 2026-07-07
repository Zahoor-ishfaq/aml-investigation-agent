"""
Groq API client wrapper with multi-key rotation.

Thin wrapper around the official groq SDK. Centralises model name,
sampling params, and retry logic so the agent loop (Phase 7.2) can
focus on reasoning logic rather than API plumbing.

Key rotation rationale: Groq's free tier enforces a 100K tokens/day
limit per organization (per Groq's rate-limits docs). Since rate limits
are per-org (not per-key), multiple keys from the *same* org don't help.
But keys from *different* orgs each get their own 100K pool. This module
cycles through a list of keys on TPD exhaustion, transparent to callers
— the ReAct loop sees a single chat() function regardless of how many
keys back it.

Reference: Groq rate-limits docs, https://console.groq.com/docs/rate-limits
"""

import logging
import os
import time
from typing import Any, Optional

from groq import Groq
from groq import APIConnectionError, InternalServerError, RateLimitError

from aml_agent.config import settings


logger = logging.getLogger("aml_agent.agent.groq")


class _KeyPool:
    """Manages a pool of Groq API keys from different orgs.

    Rotation policy: on RateLimitError, advance to the next key.
    Once every key has been tried in a single chat() call, re-raise
    so the caller (eval harness) can decide whether to sleep or abort.
    """

    def __init__(self) -> None:
        # GROQ_API_KEYS (comma-separated) takes precedence over the
        # single-key settings.groq_api_key. This avoids touching
        # config.py / pydantic-settings schema for what is a
        # run-time-only eval concern, not a production config change.
        raw = os.environ.get("GROQ_API_KEYS", "")
        if raw.strip():
            self._keys = [k.strip() for k in raw.split(",") if k.strip()]
        else:
            self._keys = [settings.groq_api_key]
        self._index = 0
        logger.info("Groq key pool initialised with %d key(s)", len(self._keys))

    @property
    def current_key(self) -> str:
        return self._keys[self._index]

    @property
    def size(self) -> int:
        return len(self._keys)

    def rotate(self) -> str:
        """Advance to the next key. Returns the new key."""
        self._index = (self._index + 1) % len(self._keys)
        return self._keys[self._index]


# Module-level singleton — initialised once on first import.
_pool = _KeyPool()


def _client(api_key: str) -> Groq:
    """Instantiate the SDK client with an explicit key so
    misconfigured .env surfaces as a startup error (via
    pydantic-settings) rather than a silent 401 at first request."""
    return Groq(api_key=api_key)


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: str = "auto",
    response_format: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Send a chat completion request, rotating keys on rate-limit errors.

    Retry policy:
      - Transient errors (connection reset, 5xx): one retry with a
        2-second delay on the same key.
      - RateLimitError (TPD/TPM exhaustion): rotate to the next key
        and retry immediately. If all keys in the pool have been
        tried, re-raise so the caller can sleep-and-retry.

    Returns the raw SDK response object (not a DTO) because the ReAct
    loop needs access to tool_calls, finish_reason, usage, and message
    content directly.
    """
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
        kwargs["response_format"] = response_format

    attempts = 0
    last_error: Optional[Exception] = None

    while attempts < _pool.size:
        client = _client(_pool.current_key)
        try:
            return client.chat.completions.create(**kwargs)
        except (APIConnectionError, InternalServerError) as e:
            # Transient network/server error — one retry on same key.
            logger.warning("Groq transient error, retrying once: %r", e)
            time.sleep(2)
            try:
                return client.chat.completions.create(**kwargs)
            except (APIConnectionError, InternalServerError):
                # Second failure on same key — treat as exhausted,
                # rotate rather than raising immediately.
                last_error = e
                attempts += 1
                next_key = _pool.rotate()
                logger.warning(
                    "Transient error persisted, rotated to key %d/%d",
                    attempts + 1, _pool.size,
                )
        except RateLimitError as e:
            last_error = e
            attempts += 1
            if attempts < _pool.size:
                next_key = _pool.rotate()
                logger.warning(
                    "Rate limit on key %d/%d, rotating to next key",
                    attempts, _pool.size,
                )
            else:
                # All keys exhausted — re-raise so the eval harness
                # can parse retry-after and sleep.
                logger.error("All %d keys exhausted", _pool.size)
                raise

    # Should only reach here if all keys hit transient errors.
    raise last_error  # type: ignore[misc]