"""Claim and verification state machine.

A :class:`ClaimState` tracks a single claim and which evidence gate
("source fact", "external fact", "runtime behavior", "visual behavior",
"code correctness") is required to resolve it. ``ClaimStatus`` is the
state-machine result.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import ClaimId


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNVERIFIED = "unverified"
    CONFLICTING = "conflicting"
    BLOCKED = "blocked"


class SupportNeed(str, Enum):
    SOURCE_FACT = "source_fact"
    EXTERNAL_FACT = "external_fact"
    RUNTIME = "runtime_behavior"
    VISUAL = "visual_behavior"
    CODE = "code_correctness"
    NONE = "none"


class ClaimState(BaseModel):
    """Mutable state for a single claim inside a verification context."""

    model_config = ConfigDict(extra="forbid")

    id: ClaimId
    text: str = Field(min_length=1, max_length=4096)
    support_need: SupportNeed = SupportNeed.NONE
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None


class VerificationContext(BaseModel):
    """Container for a set of claims the agent is currently asserting."""

    model_config = ConfigDict(extra="forbid")

    claims: dict[str, ClaimState] = Field(default_factory=dict)

    def add(self, claim: ClaimState) -> None:
        self.claims[claim.id] = claim

    def update(self, claim_id: str, **changes: Any) -> None:
        if claim_id not in self.claims:
            raise KeyError(claim_id)
        node = self.claims[claim_id]
        self.claims[claim_id] = node.model_copy(update=changes)

    def open(self) -> list[ClaimState]:
        return [c for c in self.claims.values() if c.status is ClaimStatus.UNVERIFIED]


__all__ = ["ClaimState", "ClaimStatus", "SupportNeed", "VerificationContext"]
