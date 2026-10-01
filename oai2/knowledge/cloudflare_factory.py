"""Assembly of Cloudflare knowledge components from bound Worker resources.

This module contains no deployment identifiers or secrets. A Worker entrypoint
supplies its already-bound D1/R2/Vectorize/KV capabilities and receives a fully
wired knowledge runtime + transport handler.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cloudflare_bindings_runtime import (
    CloudflareKvCache,
    CloudflareR2Store,
    CloudflareVectorizeStore,
    KvNamespaceBinding,
    R2BucketBinding,
    VectorizeIndexBinding,
)
from .cloudflare_runtime import AsyncCloudflareKnowledgeRuntime
from .gc_lease_d1_runtime import D1DatabaseBinding, D1GcLeaseStore
from .knowledge_d1_runtime import D1KnowledgeReader, D1KnowledgeWriter
from .worker_transport import EmbeddingProvider, KnowledgeWorkerTransport


@dataclass(slots=True, frozen=True)
class CloudflareKnowledgeComponents:
    reader: D1KnowledgeReader
    writer: D1KnowledgeWriter
    lease_store: D1GcLeaseStore
    r2: CloudflareR2Store
    vectorize: CloudflareVectorizeStore
    kv: CloudflareKvCache
    runtime: AsyncCloudflareKnowledgeRuntime
    transport: KnowledgeWorkerTransport


async def build_cloudflare_knowledge_components(
    *,
    d1: D1DatabaseBinding,
    r2: R2BucketBinding,
    vectorize: VectorizeIndexBinding,
    kv: KvNamespaceBinding,
    embedding_version: str,
    embedding_digest: str,
    embedding_provider: EmbeddingProvider | None = None,
    require_auth: bool = True,
    ensure_schema: bool = True,
) -> CloudflareKnowledgeComponents:
    """Wire bound Worker resources into one public-safe knowledge component set."""
    if not isinstance(ensure_schema, bool):
        raise ValueError("ensure_schema must be a boolean")

    reader = D1KnowledgeReader(d1)
    writer = D1KnowledgeWriter(d1)
    lease_store = D1GcLeaseStore(d1)

    if ensure_schema:
        # Knowledge tables/revision authority first; GC lease SQL references
        # authoritative knowledge_index liveness.
        await writer.ensure_schema()
        await lease_store.ensure_schema()

    r2_store = CloudflareR2Store(r2)
    vectorize_store = CloudflareVectorizeStore(vectorize)
    kv_cache = CloudflareKvCache(kv)
    runtime = AsyncCloudflareKnowledgeRuntime(
        reader=reader,
        writer=writer,
        r2=r2_store,
        vectorize=vectorize_store,
        kv=kv_cache,
        embedding_version=embedding_version,
        embedding_digest=embedding_digest,
    )
    transport = KnowledgeWorkerTransport(
        runtime,
        embedding_provider=embedding_provider,
        require_auth=require_auth,
    )
    return CloudflareKnowledgeComponents(
        reader=reader,
        writer=writer,
        lease_store=lease_store,
        r2=r2_store,
        vectorize=vectorize_store,
        kv=kv_cache,
        runtime=runtime,
        transport=transport,
    )


__all__ = [
    "CloudflareKnowledgeComponents",
    "build_cloudflare_knowledge_components",
]
