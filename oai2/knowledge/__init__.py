"""External knowledge contracts for OAI-2.0.

The current package contains a tested application-level Cloudflare storage
contract, deterministic mock bindings, and a strict q-pipe import gate.
Live Cloudflare network wiring remains PROPOSED until a Worker transport is
implemented and verified against production-safe bindings.
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
    CloudflareBindingAdapter,
    CloudflareKnowledgeStore,
    MockCloudflareBindings,
    cache_key_for,
    hash_topic,
    object_to_row,
    r2_blob_key_for,
)
from .ingestion import IngestionJob, IngestionPipeline, IngestionStatus
from .qpipe_import import (
    ImportPolicy,
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
    "CloudflareBindingAdapter",
    "CloudflareKnowledgeStore",
    "MockCloudflareBindings",
    "cache_key_for",
    "hash_topic",
    "object_to_row",
    "r2_blob_key_for",
    "sha256_hex",
    "now_epoch",
    "ImportPolicy",
    "ImportReport",
    "QPipeRow",
    "QPipeSource",
    "QPipeStatus",
    "derive_authority",
    "import_qpipe_rows",
    "row_to_knowledge_object",
]
