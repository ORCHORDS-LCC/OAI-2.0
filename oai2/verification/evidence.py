"""Evidence graph primitives.

Implements the public shapes for evidence and the evidence graph used to
support or refute a claim. References ``VERIFICATION_AND_EVIDENCE.md``.
"""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core import EvidenceId


class EvidenceClass(StrEnum):
    USER_INTENT = "user_intent"
    REPO_SOURCE = "repo_source"
    CHANGE_HISTORY = "change_history"
    DETERMINISTIC = "deterministic"
    RUNTIME_OBS = "runtime_observation"
    VISUAL_OBS = "visual_observation"
    EXTERNAL = "external_reference"
    HYPOTHESIS = "model_hypothesis"


class EvidenceStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONFLICTING = "conflicting"


class Evidence(BaseModel):
    """A single piece of evidence supporting or refuting a claim."""

    model_config = ConfigDict(extra="forbid")

    id: EvidenceId
    cls: EvidenceClass
    summary: str = Field(min_length=1, max_length=1024)
    source_uri: str | None = None
    artifact_ref: str | None = None
    embedding_ref: str | None = None
    observed_at: float = 0.0  # Unix epoch seconds; 0 = unset.


class EvidenceNode(BaseModel):
    """A claim-level container that holds supporting Evidence nodes.

    ``status`` is derived from the current supporting/refuting contents and is
    recomputed every time the node is constructed, merged, or updated. It is
    intentionally not a free-form input: an EvidenceNode with empty
    supporting and refuting is UNVERIFIED; supporting-only or refuting-only
    is VERIFIED (the verdict is settled, even when negative); both is
    CONFLICTING.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    claim_id: str = Field(min_length=1, max_length=128)
    supporting: tuple[Evidence, ...] = Field(default_factory=tuple)
    refuting: tuple[Evidence, ...] = Field(default_factory=tuple)
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED

    @staticmethod
    def _derive_status(
        supporting: tuple[Evidence, ...], refuting: tuple[Evidence, ...]
    ) -> EvidenceStatus:
        has_support = bool(supporting)
        has_refute = bool(refuting)
        if has_support and has_refute:
            return EvidenceStatus.CONFLICTING
        if has_support or has_refute:
            return EvidenceStatus.VERIFIED
        return EvidenceStatus.UNVERIFIED

    @model_validator(mode="after")
    def _enforce_status_invariant(self) -> EvidenceNode:
        # `status` is documented as derived from supporting/refuting contents
        # (UNVERIFIED / VERIFIED / CONFLICTING). Every construction path —
        # direct constructor, model_validate, model_copy — must end with the
        # derived value, not whatever the caller happened to pass. Without
        # this validator a caller could build EvidenceNode(claim_id="c",
        # status=EvidenceStatus.VERIFIED) and produce a node whose status
        # lies about empty supporting/refuting contents. The single source of
        # truth is _derive_status, shared with EvidenceGraph.add/merge.
        derived = self._derive_status(self.supporting, self.refuting)
        if self.status is not derived:
            object.__setattr__(self, "status", derived)
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy through validation so derived status stays consistent."""
        if update is None:
            return super().model_copy(update=None, deep=deep)
        payload = self.model_dump(mode="python")
        if deep:
            payload = deepcopy(payload)
        payload.update(update)
        return type(self).model_validate(payload)

    @property
    def net_count(self) -> int:
        return len(self.supporting) - len(self.refuting)

    def merge(self, other: EvidenceNode) -> EvidenceNode:
        """Combine two nodes that reference the same claim."""
        if other.claim_id != self.claim_id:
            raise ValueError("cannot merge nodes with different claim_id")
        seen = {e.id for e in self.supporting}
        supporting = list(self.supporting) + [
            e for e in other.supporting if e.id not in seen
        ]
        seen = {e.id for e in self.refuting}
        refuting = list(self.refuting) + [e for e in other.refuting if e.id not in seen]
        return EvidenceNode(
            claim_id=self.claim_id,
            supporting=tuple(supporting),
            refuting=tuple(refuting),
            status=self._derive_status(tuple(supporting), tuple(refuting)),
        )


class EvidenceGraph(BaseModel):
    """An ordered set of claim nodes."""

    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, EvidenceNode] = Field(default_factory=dict)

    def upsert(self, node: EvidenceNode) -> None:
        if node.claim_id in self.nodes:
            node = self.nodes[node.claim_id].merge(node)
        self.nodes[node.claim_id] = node

    def add(self, claim_id: str, evidence: Evidence, supports: bool) -> None:
        node = self.nodes.get(claim_id) or EvidenceNode(claim_id=claim_id)
        if supports:
            supporting = node.supporting + (evidence,)
            refuting = node.refuting
        else:
            supporting = node.supporting
            refuting = node.refuting + (evidence,)
        node = node.model_copy(
            update={
                "supporting": supporting,
                "refuting": refuting,
                "status": EvidenceNode._derive_status(supporting, refuting),
            }
        )
        self.upsert(node)

    def status_for(self, claim_id: str) -> EvidenceStatus:
        node = self.nodes.get(claim_id)
        if node is None:
            return EvidenceStatus.UNVERIFIED
        return node.status

    def as_dict(self) -> dict[str, Any]:
        return {cid: n.model_dump() for cid, n in self.nodes.items()}


__all__ = [
    "Evidence",
    "EvidenceClass",
    "EvidenceGraph",
    "EvidenceNode",
    "EvidenceStatus",
]
