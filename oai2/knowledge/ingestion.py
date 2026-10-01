"""Knowledge ingestion pipeline (scaffold only).

A real pipeline pulls from allow-listed sources and emits
:class:`KnowledgeObject` records. The reference implementation here is
deterministic and synchronous so tests can exercise the shape.

Status: PROPOSED. No external fetches happen in v0.1.0.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol

from ..core import KnowledgeId, Status
from .abstraction import KnowledgeObject, KnowledgeStore, now_epoch, sha256_hex


class IngestionStatus(str, Enum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True, frozen=True)
class IngestionJob:
    source_uri: str
    topic: str
    content: str
    authority: float = 0.5
    status: Status = Status.EXPERIMENTAL

    def build(self) -> KnowledgeObject:
        return KnowledgeObject(
            knowledge_id=KnowledgeId(f"ko_{uuid.uuid4().hex[:12]}"),
            topic=self.topic,
            content=self.content,
            content_hash=sha256_hex(self.content),
            source_uri=self.source_uri,
            retrieved_at=now_epoch(),
            authority=self.authority,
            status=self.status,
        )


class _Source(Protocol):
    def fetch(self, source_uri: str) -> Iterable[IngestionJob]: ...


@dataclass(slots=True)
class IngestionPipeline:
    store: KnowledgeStore
    sources: list[_Source] = field(default_factory=list)

    def ingest(self, jobs: Iterable[IngestionJob]) -> list[IngestionStatus]:
        results: list[IngestionStatus] = []
        for job in jobs:
            try:
                obj = job.build()
                self.store.put(obj)
                results.append(IngestionStatus.OK)
            except Exception:
                results.append(IngestionStatus.FAILED)
        return results


__all__ = ["IngestionJob", "IngestionPipeline", "IngestionStatus"]
