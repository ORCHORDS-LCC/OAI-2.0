"""SWARM multi-agent controller scaffold.

SWARM runs several specialized agents in parallel, each with its own
context slice, then merges their outputs through a critic.

Status: PROPOSED.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..core import AgentId, Status


class SwarmRole(StrEnum):
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    CRITIC = "critic"
    VERIFIER = "verifier"


class SwarmContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: tuple[AgentId, ...] = Field(default_factory=tuple)
    roles: dict[AgentId, SwarmRole] = Field(default_factory=dict)
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    status: Status = Status.PROPOSED


class SwarmController:
    STATUS = Status.PROPOSED

    def __init__(self, ctx: SwarmContext) -> None:
        self.ctx = ctx

    def critic_pick(self, votes: dict[AgentId, float]) -> AgentId | None:
        if not votes:
            return None
        winner, score = max(votes.items(), key=lambda kv: kv[1])
        return winner if score >= self.ctx.consensus_threshold else None


__all__ = ["SwarmContext", "SwarmController", "SwarmRole"]
