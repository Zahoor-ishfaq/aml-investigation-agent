"""
Tool layer — base contract.

Every capability the agent can invoke is expressed as a Tool subclass.
Tools are pure functions: given validated arguments, they read from the
DB, pseudonymize the output, and return a ToolResult. Wrapping (audit
logging, guardrail interception, error boundaries) lives in the dispatch
layer (substep 4.6), not here — keeps each tool trivially unit-testable.

The contract mirrors the OpenAI / Groq function-calling convention: each
tool exposes a JSON schema (auto-generated from a Pydantic args model)
that the LLM reads to know what to pass. Pydantic is single source of
truth: what the LLM sees and what the code accepts cannot drift.

Reference: Groq API tool-use docs, https://console.groq.com/docs/tool-use
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Type

from pydantic import BaseModel
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Standard output envelope.
#
# All tool results share this shape so the agent (Phase 7) has a single
# parser regardless of which tool it called. status distinguishes normal
# empty results ("no transactions in window") from errors ("account token
# not found") — the LLM reasons differently about each.
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Container for tool output. `data` holds already-pseudonymized dicts."""
    status: str                          # "ok" | "error"
    data: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # count, window used, etc.

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON transport to the LLM. dataclass -> dict without
        including None fields for compactness (LLM context is expensive)."""
        d = asdict(self)
        if d["error"] is None:
            d.pop("error")
        if not d["metadata"]:
            d.pop("metadata")
        return d


# ---------------------------------------------------------------------------
# Tool base class.
#
# Subclasses declare `name`, `description`, `args_schema` (Pydantic model)
# and implement `_run()`. The public `execute()` handles arg validation
# once, at the boundary — no tool re-implements schema enforcement.
# ---------------------------------------------------------------------------

class Tool(ABC):
    """
    Abstract base for every agent-callable tool.

    Class attributes:
      name          — snake_case identifier the LLM uses in tool calls
      description   — natural-language description shown to the LLM (must
                      convey WHEN to call this tool, not just what it does)
      args_schema   — Pydantic model defining the tool's argument contract
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[Type[BaseModel]]

    def execute(self, db: Session, **raw_args: Any) -> ToolResult:
        """
        Public entry point. Validates raw_args against args_schema, then
        delegates to the subclass's _run() with the parsed model.

        Validation errors are returned as ToolResult(status='error') rather
        than raised — the agent needs to see and react to bad calls (e.g.
        "you passed an invalid account token, try again"), not have the
        whole reasoning loop crash on a malformed argument.
        """
        try:
            args = self.args_schema(**raw_args)
        except Exception as e:
            return ToolResult(status="error", error=f"invalid arguments: {e}")
        return self._run(db, args)

    @abstractmethod
    def _run(self, db: Session, args: BaseModel) -> ToolResult:
        """Subclass implementation. Called with validated args."""
        raise NotImplementedError

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """
        Return the OpenAI/Groq-style function schema for this tool.

        Pydantic's model_json_schema() generates the args JSON schema; we
        wrap it with the outer function-call envelope both Groq and OpenAI
        expect. Used by the dispatch layer (4.6) when constructing the
        tool list to send to the LLM.
        """
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.args_schema.model_json_schema(),
            },
        }


# ---------------------------------------------------------------------------
# Registry — parallel to the rule engine's REGISTRY.
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Type[Tool]] = {}


def register_tool(tool_cls: Type[Tool]) -> Type[Tool]:
    """
    Class decorator that registers a Tool with the dispatch layer.

    Enforces uniqueness on `name`: two tools sharing a name would break
    dispatch since the LLM identifies tools by name in its tool_calls.
    """
    if tool_cls.name in TOOL_REGISTRY:
        raise ValueError(f"duplicate tool name: {tool_cls.name}")
    TOOL_REGISTRY[tool_cls.name] = tool_cls
    return tool_cls