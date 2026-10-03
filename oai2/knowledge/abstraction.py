"""Knowledge store abstraction.

Defines the public contract for OAI-2.0's external knowledge store. The
agent stores typed :class:`KnowledgeObject` records keyed by
``knowledge_id`` and retrieves them through :class:`RetrievalRequest`.
A real deployment backs this with a Cloudflare R2 + D1 binding; the
reference implementation here is a deterministic in-memory store used
by tests and local development.

Reference: ``docs/agent-architecture/SYSTEM_ARCHITECTURE.md`` —
"knowledge layer" subsection.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ..core import KnowledgeId, Status

MAX_CLAIM_KEY_LENGTH: Final[int] = 128
MAX_CONTENT_VERSION_LENGTH: Final[int] = 128
MAX_SOURCE_VERSION_LENGTH: Final[int] = 128
# A supersession link is a *derived* identifier: producers such as the q-pipe
# importer build it as ``<prefix>:<external_id>``. It is bounded here so a
# producer can refuse to emit an unrepresentable link at its own gate, instead
# of constructing one that its own model then rejects.
MAX_SUPERSEDED_BY_LENGTH: Final[int] = 128


class KnowledgeObject(BaseModel):
    """A single typed piece of external knowledge."""

    model_config = ConfigDict(extra="forbid")

    knowledge_id: KnowledgeId
    topic: str = Field(min_length=1, max_length=256)
    content: str
    content_hash: str = Field(min_length=8, max_length=128)
    source_uri: str | None = None
    retrieved_at: float = 0.0  # Unix epoch seconds.
    authority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: Status = Status.PROPOSED
    artifact_ref: str | None = None
    embedding_ref: str | None = None

    # --- Temporal / supersession (OAI-2.0 #200 REQ-TEMP-001/011/012, #207) ---
    # `claim_key` is the stable identity of the CLAIM (polarity-independent
    # subject signature) and is deliberately distinct from `content_hash`,
    # which identifies one exact revision of it. A claim whose
    # recommendation was flipped keeps its claim_key and gains a new
    # content_hash, which is what makes supersession and conflict
    # reconciliation possible at all. Conflating the two would make every
    # reworded revision look like an unrelated claim.
    claim_key: str | None = Field(default=None, max_length=MAX_CLAIM_KEY_LENGTH)
    content_version: str | None = Field(
        default=None, max_length=MAX_CONTENT_VERSION_LENGTH
    )
    source_version: str | None = Field(
        default=None, max_length=MAX_SOURCE_VERSION_LENGTH
    )
    effective_at: float | None = None
    superseded_by: str | None = Field(default=None, max_length=MAX_SUPERSEDED_BY_LENGTH)
    superseded_at: float | None = None

    # --- Trust and isolation (OAI-2.0 #185) ---
    # Retrieved evidence is never equivalent to an instruction. `trust_class`
    # records that in the authoritative store, and `scope_class` records who
    # may receive it, so a lesson promoted from one client's private context
    # cannot be served to another by virtue of being globally indexed.
    trust_class: str = Field(default="retrieved_evidence", max_length=64)
    scope_class: str = Field(default="global", max_length=32)


@dataclass(slots=True, frozen=True)
class RetrievalRequest:
    topic: str
    limit: int = 8
    min_authority: float = 0.0
    include_status: tuple[Status, ...] = (
        Status.IMPLEMENTED,
        Status.EXPERIMENTAL,
    )
    # REQ-TEMP-025: a historical question must still be able to retrieve
    # superseded evidence, so exclusion is the default and inclusion is an
    # explicit opt-in rather than an unconditional filter.
    include_superseded: bool = False


@dataclass(slots=True, frozen=True)
class RetrievalCandidate:
    """One eligible retrieval candidate with explicit ranking/provenance evidence."""

    knowledge_id: KnowledgeId
    content_hash: str
    source_uri: str | None
    score: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    topic: str
    objects: list[KnowledgeObject] = field(default_factory=list)
    candidates: list[RetrievalCandidate] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.objects)


class KnowledgeStore(ABC):
    """Abstract base — subclasses must persist and retrieve."""

    STATUS: Status = Status.PROPOSED

    @abstractmethod
    def put(self, obj: KnowledgeObject) -> None: ...

    @abstractmethod
    def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None: ...

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...

    @abstractmethod
    def all(self) -> Iterable[KnowledgeObject]: ...


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_epoch() -> float:
    return time.time()


class InMemoryKnowledgeStore(KnowledgeStore):
    """Deterministic, network-free implementation used in tests."""

    STATUS = Status.EXPERIMENTAL

    def __init__(self) -> None:
        self._items: dict[KnowledgeId, KnowledgeObject] = {}

    def put(self, obj: KnowledgeObject) -> None:
        self._items[obj.knowledge_id] = obj

    def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:
        return self._items.get(knowledge_id)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        out: list[KnowledgeObject] = []
        for obj in self._items.values():
            if request.topic.lower() not in obj.topic.lower():
                continue
            if obj.authority < request.min_authority:
                continue
            if obj.status not in request.include_status:
                continue
            # REQ-TEMP-014: a superseded record stays retrievable for audit,
            # never by default. This is the gate `is_active` promised and no
            # caller performed. It cannot be delegated to `include_status`,
            # because a superseded row is deliberately demoted to
            # EXPERIMENTAL and EXPERIMENTAL is in the default set -- so the
            # status filter never excluded it. Filtering on the field itself
            # is what makes the exclusion independent of the status mapping.
            if obj.superseded_by is not None and not request.include_superseded:
                continue
            out.append(obj)
        out.sort(key=lambda o: (o.authority, o.retrieved_at), reverse=True)
        selected = out[: request.limit]
        return RetrievalResult(
            topic=request.topic,
            objects=selected,
            candidates=[
                RetrievalCandidate(
                    knowledge_id=obj.knowledge_id,
                    content_hash=obj.content_hash,
                    source_uri=obj.source_uri,
                )
                for obj in selected
            ],
        )

    def all(self) -> Iterable[KnowledgeObject]:
        return tuple(self._items.values())


__all__ = [
    "KnowledgeObject",
    "RetrievalRequest",
    "RetrievalCandidate",
    "RetrievalResult",
    "KnowledgeStore",
    "InMemoryKnowledgeStore",
    "sha256_hex",
    "now_epoch",
]
