"""FURIOUS fast-path controller.

FURIOUS is the single-agent, low-budget reasoning mode. It plans a
single next action per step, runs it, observes, and decides whether to
continue, escalate to DEEP, or hand off to SWARM.

Status: PROPOSED. The state machine is wired; live decisions need an
IMPLEMENTED :class:`~oai2.runtime.InferenceRuntime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status


class FuriousState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    CONTINUE = "continue"
    ESCALATE = "escalate"
    HANDOFF = "handoff"
    DONE = "done"


@dataclass(slots=True, frozen=True)
class FuriousDecision:
    next_state: FuriousState
    reason: str
    escalate_to: str | None = None  # e.g. "DEEP" or "SWARM".


class FuriousContext(BaseModel):
    """Observable state used by the FURIOUS controller."""

    model_config = ConfigDict(extra="forbid")

    budget_remaining: float = 1.0
    last_action: str | None = None
    last_observation: str | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    status: Status = Status.PROPOSED


class FuriousController:
    """Pure-function state machine for FURIOUS. No side effects."""

    STATUS = Status.PROPOSED

    @staticmethod
    def decide(ctx: FuriousContext, *, current: FuriousState) -> FuriousDecision:
        if ctx.budget_remaining <= 0.0:
            return FuriousDecision(FuriousState.DONE, "budget exhausted")
        if ctx.consecutive_failures >= 3:
            return FuriousDecision(
                FuriousState.ESCALATE,
                f"{ctx.consecutive_failures} consecutive failures",
                escalate_to="DEEP",
            )
        match current:
            case FuriousState.IDLE:
                return FuriousDecision(FuriousState.PLANNING, "start")
            case FuriousState.PLANNING:
                return FuriousDecision(FuriousState.ACTING, "plan ready")
            case FuriousState.ACTING:
                return FuriousDecision(FuriousState.OBSERVING, "action sent")
            case FuriousState.OBSERVING:
                return FuriousDecision(FuriousState.CONTINUE, "observed")
            case FuriousState.CONTINUE:
                return FuriousDecision(FuriousState.PLANNING, "loop")
            case _:
                return FuriousDecision(FuriousState.DONE, "terminal")


__all__ = [
    "FuriousController",
    "FuriousContext",
    "FuriousDecision",
    "FuriousState",
]
