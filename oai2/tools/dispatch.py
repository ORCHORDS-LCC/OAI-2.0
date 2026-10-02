"""Tool dispatch pipeline.

Implements the gate sequence from ``TOOL_CALLING.md``. The pipeline is
pure — it inspects the call against the host's policy and returns a
:class:`DispatchDecision`. The actual side-effecting call happens
downstream when ``decision.action is DispatchStage.EXECUTE``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status
from ..protocols import ToolCall, ToolDefinition, ToolResult


class DispatchStage(StrEnum):
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
    # Capabilities permitted to act WITHOUT a resource scope. Empty by
    # default: a tool that takes no path (a shell, a cwd-relative glob) can
    # reach anything, so treating "not scoped" as an implicit yes would make
    # ``resource_scopes`` decorative. Listing a capability here is the
    # explicit, auditable statement that the host accepts that reach.
    allow_unscoped_capabilities: frozenset[str] = Field(default_factory=frozenset)
    budget_calls: int = 64
    status: Status = Status.PROPOSED


class ToolDispatcher:
    """Pure pipeline; no side effects, easy to unit-test."""

    STATUS = Status.PROPOSED

    def __init__(
        self,
        registry: tuple[ToolDefinition, ...],
        policy: DispatchPolicy,
        cwd: Path | None = None,
    ) -> None:
        self.registry = {td.id: td for td in registry}
        self.policy = policy
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()

    # -- gate 4 helpers ------------------------------------------------

    def _resolve(self, raw: Any) -> Path | None:
        """Normalise a requested resource to an absolute, traversal-free path.

        ``..`` is collapsed *before* the scope test, so a path such as
        ``<scope>/../etc/passwd`` normalises to ``/etc/passwd`` and is
        correctly judged out of scope. Normalisation is lexical on purpose:
        it must not depend on the target existing, and a symlink the model
        cannot see must not be able to widen its own scope.
        """
        if not isinstance(raw, str) or not raw:
            return None
        expanded = os.path.expanduser(raw)
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        return Path(os.path.normpath(str(candidate)))

    def _in_scope(self, target: Path) -> bool:
        """Whether ``target`` is at or beneath any declared scope root."""
        for scope in self.policy.resource_scopes:
            if not isinstance(scope, str) or not scope:
                continue
            root = self._resolve(scope)
            if root is None:
                continue
            if target == root or root in target.parents:
                return True
        return False

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
            target = self._resolve(scope)
            if target is None or not self._in_scope(target):
                return DispatchDecision(DispatchStage.DENY, "resource out of scope")

        # Gate 4b: an unscoped tool is not checked against resource_scopes
        # at all, so when the host HAS declared scopes, its capability must
        # be explicitly granted unscoped reach. Without this, `Bash` is a
        # hole straight past gate 4 and the declared scopes are decoration.
        # A host that declares no scopes has expressed no scoping intent, so
        # the gate does not apply.
        elif self.policy.resource_scopes:
            if td.capability not in self.policy.allow_unscoped_capabilities:
                return DispatchDecision(
                    DispatchStage.DENY,
                    f"unscoped capability {td.capability!r} not explicitly permitted "
                    "(add it to allow_unscoped_capabilities to grant this reach)",
                )

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
