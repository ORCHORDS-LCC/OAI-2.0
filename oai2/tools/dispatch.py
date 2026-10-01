"""Tool dispatch pipeline.

Implements the gate sequence from ``TOOL_CALLING.md``. The pipeline is
pure — it inspects the call against the host's policy and returns a
:class:`DispatchDecision`. The actual side-effecting call happens
downstream when ``decision.action is DispatchStage.EXECUTE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status
from ..protocols import ToolCall, ToolDefinition, ToolPolicy, ToolResult


class DispatchStage(str, Enum):
    EXECUTE = "execute"
    REPAIR = "repair_arguments"
    CHOOSE_ALT = "choose_alternative"
    RETRY = "retry"
    REPLAN = "replan"
    DENY = "deny"


@dataclass(slots=True, frozen=True)
class DispatchDecision:
    stage: DispatchStage
    reason: str
    result: ToolResult | None = None


class DispatchPolicy(BaseModel):
    """Static host-side policy the dispatcher enforces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_capabilities: frozenset[str] = Field(default_factory=frozenset)
    deny_capabilities: frozenset[str] = Field(default_factory=frozenset)
    high_impact_approved: bool = False
    resource_scopes: frozenset[str] = Field(default_factory=frozenset)
    budget_calls: int = 64
    status: Status = Status.PROPOSED


class ToolDispatcher:
    """Pure pipeline; no side effects, easy to unit-test."""

    STATUS = Status.PROPOSED

    def __init__(self, registry: tuple[ToolDefinition, ...], policy: DispatchPolicy) -> None:
        self.registry = {td.id: td for td in registry}
        self.policy = policy

    def check(self, call: ToolCall, *, calls_used: int) -> DispatchDecision:
        # Gate 1: tool exists.
        td = self.registry.get(call.tool_id)
        if td is None:
            return DispatchDecision(DispatchStage.REPLAN, f"unknown tool {call.tool_id!r}")

        # Gate 2: arguments validate.
        declared = {a.name for a in td.arguments}
        provided = set(call.arguments.keys())
        if not provided.issubset(declared):
            return DispatchDecision(
                DispatchStage.REPAIR,
                f"unknown arguments: {sorted(provided - declared)}",
            )

        # Gate 3: capability allowed.
        if td.capability in self.policy.deny_capabilities:
            return DispatchDecision(DispatchStage.DENY, f"denied capability {td.capability!r}")
        if self.policy.allow_capabilities and td.capability not in self.policy.allow_capabilities:
            return DispatchDecision(
                DispatchStage.CHOOSE_ALT,
                f"capability {td.capability!r} not in allow list",
            )

        # Gate 4: resource in scope.
        if td.scoped:
            scope = call.arguments.get("scope") or call.arguments.get("path")
            if scope not in self.policy.resource_scopes:
                return DispatchDecision(DispatchStage.DENY, "resource out of scope")

        # Gate 5: budget respected.
        if calls_used + 1 > self.policy.budget_calls:
            return DispatchDecision(DispatchStage.RETRY, "budget exceeded")

        # Gate 6: high-impact approval.
        if td.high_impact and not (
            self.policy.high_impact_approved or call.policy.high_impact_approved
        ):
            return DispatchDecision(DispatchStage.DENY, "high-impact tool not approved")

        # If we get here, the policy also needs to permit this specific
        # call's per-call policy (overrides the host policy).
        per_call = call.policy
        if per_call.deny_capabilities and td.capability in per_call.deny_capabilities:
            return DispatchDecision(
                DispatchStage.DENY,
                f"per-call deny {td.capability!r}",
            )
        if per_call.allow_capabilities and td.capability not in per_call.allow_capabilities:
            return DispatchDecision(
                DispatchStage.CHOOSE_ALT,
                f"per-call not in allow {td.capability!r}",
            )

        return DispatchDecision(DispatchStage.EXECUTE, "all gates passed")


def default_dispatcher() -> ToolDispatcher:
    """Permissive dispatcher used in tests and offline runs."""
    policy = DispatchPolicy()
    return ToolDispatcher(registry=(), policy=policy)


__all__ = [
    "DispatchDecision",
    "DispatchPolicy",
    "DispatchStage",
    "ToolDispatcher",
    "default_dispatcher",
]
