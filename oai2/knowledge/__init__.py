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
from .cloudflare import (
    CFPlan,
    CFPrimitive,
    CFRow,
    CloudflareKnowledgeStore,
    MockCloudflareBindings,
    cache_key_for,
    hash_topic,
    object_to_row,
    r2_blob_key_for,
)
from .ingestion import IngestionJob, IngestionPipeline, IngestionStatus
from .qpipe_import import (
    ImportReport,
    QPipeRow,
    QPipeSource,
    QPipeStatus,
    derive_authority,
    import_qpipe_rows,
    row_to_knowledge_object,
)

__all__ = [
    "InMemoryKnowledgeStore",
    "KnowledgeObject",
    "KnowledgeStore",
    "RetrievalRequest",
    "RetrievalResult",
    "IngestionJob",
    "IngestionPipeline",
    "IngestionStatus",
    "CFPlan",
    "CFPrimitive",
    "CFRow",
    "CloudflareKnowledgeStore",
    "MockCloudflareBindings",
    "cache_key_for",
    "hash_topic",
    "object_to_row",
    "r2_blob_key_for",
    "sha256_hex",
    "now_epoch",
    "ImportReport",
    "QPipeRow",
    "QPipeSource",
    "QPipeStatus",
    "derive_authority",
    "import_qpipe_rows",
    "row_to_knowledge_object",
]
