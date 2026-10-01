"""Multi-agent orchestration scaffold.

The orchestrator is responsible for spawning specialized agents (planner,
implementer, critic, verifier), collecting their outputs, and merging
them into a single response.

Status: PROPOSED.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..core import AgentId, Status
from ..reasoning import SwarmRole


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: AgentId
    role: SwarmRole
    description: str = Field(min_length=1, max_length=1024)
    status: Status = Status.PROPOSED


class OrchestratorContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: tuple[AgentSpec, ...] = Field(default_factory=tuple)
    merge_strategy: str = Field(default="critic_pick", min_length=1, max_length=64)
    status: Status = Status.PROPOSED


class Orchestrator:
    STATUS = Status.PROPOSED

    def __init__(self, ctx: OrchestratorContext) -> None:
        self.ctx = ctx

    def by_role(self, role: SwarmRole) -> tuple[AgentSpec, ...]:
        return tuple(a for a in self.ctx.agents if a.role is role)


__all__ = ["AgentSpec", "Orchestrator", "OrchestratorContext"]
