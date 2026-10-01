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
from dataclasses import dataclass, field
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..core import KnowledgeId, Status


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


@dataclass(slots=True, frozen=True)
class RetrievalRequest:
    topic: str
    limit: int = 8
    min_authority: float = 0.0
    include_status: tuple[Status, ...] = (
        Status.IMPLEMENTED,
        Status.EXPERIMENTAL,
    )


@dataclass(slots=True)
class RetrievalResult:
    topic: str
    objects: list[KnowledgeObject] = field(default_factory=list)

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
            out.append(obj)
        out.sort(key=lambda o: (o.authority, o.retrieved_at), reverse=True)
        return RetrievalResult(topic=request.topic, objects=out[: request.limit])

    def all(self) -> Iterable[KnowledgeObject]:
        return tuple(self._items.values())


__all__ = [
    "KnowledgeObject",
    "RetrievalRequest",
    "RetrievalResult",
    "KnowledgeStore",
    "InMemoryKnowledgeStore",
    "sha256_hex",
    "now_epoch",
]
