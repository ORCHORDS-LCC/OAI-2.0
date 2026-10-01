"""Tool-calling schemas.

These are the **wire shapes** the agent uses to talk to a host. A host
(e.g. an IDE or agent runtime) declares its tools as ``ToolDefinition``
and receives ``ToolCall`` requests with validated ``ToolArgument`` maps,
returning ``ToolResult`` to the agent.

Validation rules from ``docs/agent-architecture/TOOL_CALLING.md`` are
implemented as Pydantic constraints here. Schema validation is the first
gate in the policy pipeline; the next gates (capability, scope, budget,
high-impact approval) live in :mod:`oai2.tools.dispatch`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core import ToolId


class ToolArgument(BaseModel):
    """A single tool argument: name, JSON-schema-like type, value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    type: str = Field(min_length=1, max_length=64)
    value: Any = None

    @field_validator("type")
    @classmethod
    def _type_is_known(cls, v: str) -> str:
        allowed = {
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "object",
            "null",
            "uri",
            "path",
            "binary",
        }
        if v not in allowed:
            raise ValueError(f"unknown argument type: {v!r}")
        return v


class ToolDefinition(BaseModel):
    """A host-declared tool the agent can call."""

    model_config = ConfigDict(extra="forbid")

    id: ToolId
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=4096)
    arguments: tuple[ToolArgument, ...] = Field(default_factory=tuple)
    # Capability is a free-form host-side label (e.g. "fs.read", "net.fetch").
    capability: str = Field(min_length=1, max_length=128)
    # If True, calls must pass a resource-scope check before execution.
    scoped: bool = False
    # High-impact tools require explicit approval in addition to policy checks.
    high_impact: bool = False

    def signature(self) -> str:
        """Stable string used for caches and prompt hashing."""
        parts = [f"{a.name}:{a.type}" for a in self.arguments]
        return f"{self.name}({', '.join(parts)})"


class ToolPolicy(BaseModel):
    """Per-call policy constraints; the policy pipeline enforces these."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_capabilities: frozenset[str] = Field(default_factory=frozenset)
    deny_capabilities: frozenset[str] = Field(default_factory=frozenset)
    budget_calls: int = Field(default=64, ge=0, le=100_000)
    budget_seconds: float = Field(default=60.0, ge=0.0, le=86_400.0)
    high_impact_approved: bool = False


class ToolCall(BaseModel):
    """A request from the agent to the host to run a tool."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    tool_id: ToolId
    arguments: dict[str, Any] = Field(default_factory=dict)
    policy: ToolPolicy = Field(default_factory=ToolPolicy)


class ToolResult(BaseModel):
    """A response from the host after running a tool."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    ok: bool
    output: Any = None
    error: str | None = None
    elapsed_ms: float = Field(default=0.0, ge=0.0)


__all__ = [
    "ToolArgument",
    "ToolDefinition",
    "ToolPolicy",
    "ToolCall",
    "ToolResult",
]
