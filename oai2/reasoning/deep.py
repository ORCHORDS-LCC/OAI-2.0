"""DEEP controller scaffold.

DEEP is the branch + verify reasoning mode. It produces N candidate
next actions, scores them against the evidence graph, picks the highest-
confidence one, and re-verifies after execution.

Status: PROPOSED.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status


@dataclass(slots=True, frozen=True)
class Candidate:
    action: str
    confidence: float


class DeepContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch_factor: int = Field(default=4, ge=1, le=32)
    evidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    status: Status = Status.PROPOSED


class DeepController:
    STATUS = Status.PROPOSED

    @staticmethod
    def select(candidates: list[Candidate], ctx: DeepContext) -> Candidate | None:
        viable = [c for c in candidates if c.confidence >= ctx.evidence_threshold]
        if not viable:
            return None
        return max(viable, key=lambda c: c.confidence)


__all__ = ["Candidate", "DeepContext", "DeepController"]
