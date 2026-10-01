"""External knowledge abstraction backed by Cloudflare (scaffold only).

Status: PROPOSED. No live network calls in v0.1.0; the abstraction is
defined, the contract is fixed, and the local in-memory implementation is
exercised by unit tests. Real Cloudflare binding lands in a later release.
"""

from __future__ import annotations

from .abstraction import (
    InMemoryKnowledgeStore,
    KnowledgeObject,
    KnowledgeStore,
    RetrievalRequest,
    RetrievalResult,
    now_epoch,
    sha256_hex,
)
from .ingestion import IngestionJob, IngestionPipeline, IngestionStatus

__all__ = [
    "InMemoryKnowledgeStore",
    "KnowledgeObject",
    "KnowledgeStore",
    "RetrievalRequest",
    "RetrievalResult",
    "IngestionJob",
    "IngestionPipeline",
    "IngestionStatus",
    "sha256_hex",
    "now_epoch",
]
