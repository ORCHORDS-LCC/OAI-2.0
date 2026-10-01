"""Shared types, IDs, and small helpers for the OAI-2.0 package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

# Stable identifier aliases — opaque strings to the rest of the codebase.
KnowledgeId = NewType("KnowledgeId", str)
EvidenceId = NewType("EvidenceId", str)
TaskId = NewType("TaskId", str)
ClaimId = NewType("ClaimId", str)
AgentId = NewType("AgentId", str)
ToolId = NewType("ToolId", str)


class Status(StrEnum):
    """Public lifecycle status for OAI-2.0 components.

    Mirrors the status language used in ``docs/agent-architecture/``:

    - ``IMPLEMENTED`` — running, observed working code.
    - ``EXPERIMENTAL`` — wired but not validated end-to-end.
    - ``PROPOSED`` — design only; no observable behavior yet.
    - ``BLOCKED`` — cannot proceed without an external dependency.
    """

    IMPLEMENTED = "IMPLEMENTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    PROPOSED = "PROPOSED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Static description of a component, used for self-description and logs."""

    name: str
    status: Status
    summary: str


def component_info(name: str, status: Status, summary: str) -> ComponentInfo:
    """Helper for declaring component metadata in one line."""
    return ComponentInfo(name=name, status=status, summary=summary)


__all__ = [
    "KnowledgeId",
    "EvidenceId",
    "TaskId",
    "ClaimId",
    "AgentId",
    "ToolId",
    "Status",
    "ComponentInfo",
    "component_info",
]
