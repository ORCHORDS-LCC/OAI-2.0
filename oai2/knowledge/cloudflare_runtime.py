"""Async Cloudflare knowledge runtime assembled from binding-facing primitives.

This module is the source-level live transport core behind WI-KNOW-002. It
combines authoritative D1 metadata/revision state, R2 content-addressed bodies,
Vectorize semantic search, and best-effort KV caching without embedding any
deployment identifiers or credentials.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace

from pydantic import ValidationError

from ..core import KnowledgeId
from .abstraction import (
    KnowledgeObject,
    RetrievalRequest,
    RetrievalResult,
    sha256_hex,
)
from .cloudflare import CFRow, cache_key_for, object_to_row
from .cloudflare_bindings_runtime import (
    CloudflareKvCache,
    CloudflareR2Store,
    CloudflareVectorizeStore,
)
from .knowledge_d1_runtime import D1KnowledgeReader, D1KnowledgeWriter
from .transport import (
    KnowledgeCacheRef,
    QueryCacheEnvelope,
    R2BodyDescriptor,
    VectorizeMetadata,
)


class KnowledgeRuntimeError(RuntimeError):
    """Base class for explicit live knowledge runtime failures."""


class KnowledgeConflictError(KnowledgeRuntimeError):
    """Authoritative revision/lease state changed during an operation."""


class KnowledgeIntegrityError(KnowledgeRuntimeError):
    """Authoritative metadata and external storage disagree."""


class AsyncCloudflareKnowledgeRuntime:
    """Async D1/R2/Vectorize/KV runtime with D1-authoritative lifecycle state."""

    def __init__(
        self,
        *,
        reader: D1KnowledgeReader,
        writer: D1KnowledgeWriter,
        r2: CloudflareR2Store,
        vectorize: CloudflareVectorizeStore,
        kv: CloudflareKvCache,
        embedding_version: str,
        embedding_digest: str,
    ) -> None:
        if not embedding_version or embedding_version != embedding_version.strip():
            raise ValueError("embedding_version must be a non-empty normalized string")
        if not embedding_digest or embedding_digest != embedding_digest.strip():
            raise ValueError("embedding_digest must be a non-empty normalized string")
        self._reader = reader
        self._writer = writer
        self._r2 = r2
        self._vectorize = vectorize
        self._kv = kv
        self._embedding_version = embedding_version
        self._embedding_digest = embedding_digest

    async def put(
        self,
        obj: KnowledgeObject,
        *,
        vector: Sequence[float] | None = None,
        now: float | None = None,
    ) -> int:
        """Persist one object and return the new authoritative corpus revision."""
        descriptor = R2BodyDescriptor.from_knowledge(obj)
        timestamp = time.time() if now is None else _non_negative_number(now, "now")
        expected_revision = await self._writer.corpus_revision()

        if obj.content:
            await self._r2.put_text(descriptor.object_key, obj.content)

        row = object_to_row(obj)
        if vector is None:
            row = replace(row, vectorize_id=None)
        else:
            metadata = VectorizeMetadata.from_knowledge(
                obj,
                embedding_version=self._embedding_version,
            ).model_dump(mode="json")
            await self._vectorize.upsert(
                str(obj.knowledge_id),
                vector,
                metadata=metadata,
            )

        revision = await self._writer.write_metadata(
            row,
            expected_revision=expected_revision,
            now=timestamp,
        )
        if revision is None:
            raise KnowledgeConflictError(
                "D1 metadata write was denied by revision or deletion-lease state"
            )
        return revision

    async def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:
        row = await self._reader.get_row(knowledge_id)
        if row is None:
            return None
        return await self._hydrate(row)

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult:
        """Retrieve through revisioned KV cache, D1/R2, and optional Vectorize."""
        start_revision = await self._writer.corpus_revision()
        cache_key = cache_key_for(
            request,
            self._embedding_digest,
            start_revision,
        )

        cached = await self._cache_get(cache_key)
        if cached is not None:
            cached_result = await self._try_cached_result(
                request,
                cache_key,
                start_revision,
                cached,
            )
            if cached_result is not None:
                end_revision = await self._writer.corpus_revision()
                if end_revision != start_revision:
                    raise KnowledgeConflictError(
                        "corpus revision changed during cached retrieval; retry against fresh state"
                    )
                return cached_result

        if query_vector is None:
            rows = await self._reader.query_rows(request)
        else:
            rows = await self._semantic_rows(request, query_vector)

        objects = [await self._hydrate(row) for row in rows]
        end_revision = await self._writer.corpus_revision()
        if end_revision != start_revision:
            raise KnowledgeConflictError(
                "corpus revision changed during retrieval; retry against fresh state"
            )

        envelope = QueryCacheEnvelope(
            corpus_revision=start_revision,
            embedding_digest=self._embedding_digest,
            request_fingerprint=cache_key,
            refs=[
                KnowledgeCacheRef(
                    knowledge_id=obj.knowledge_id,
                    content_hash=obj.content_hash,
                )
                for obj in objects
            ],
        )
        try:
            await self._kv.put_text(
                cache_key,
                envelope.model_dump_json(),
                ttl_seconds=300,
            )
        except Exception:
            # KV is explicitly non-authoritative and best-effort.
            pass

        return RetrievalResult(topic=request.topic, objects=objects)

    async def _semantic_rows(
        self,
        request: RetrievalRequest,
        query_vector: Sequence[float],
    ) -> list[CFRow]:
        top_k = min(max(int(request.limit), 1), 100)
        matches = await self._vectorize.query(query_vector, top_k=top_k)
        rows: list[CFRow] = []
        for vector_id, _score in matches:
            row = await self._reader.get_row(KnowledgeId(vector_id))
            if row is None:
                raise KnowledgeIntegrityError(
                    f"Vectorize result {vector_id!r} has no authoritative D1 row"
                )
            if row.vectorize_id != vector_id:
                raise KnowledgeIntegrityError(
                    f"Vectorize result {vector_id!r} disagrees with D1 vectorize_id"
                )
            if row.authority < request.min_authority:
                continue
            if row.status not in request.include_status:
                continue
            rows.append(row)
            if len(rows) >= request.limit:
                break
        return rows

    async def _hydrate(self, row: CFRow) -> KnowledgeObject:
        if row.r2_blob_key is None:
            content = ""
        else:
            content = await self._r2.get_text(row.r2_blob_key)
            if content is None:
                raise KnowledgeIntegrityError(
                    f"R2 body is missing for knowledge_id={row.knowledge_id}"
                )

        if sha256_hex(content) != row.content_hash:
            raise KnowledgeIntegrityError(
                f"R2 body hash mismatch for knowledge_id={row.knowledge_id}"
            )

        return KnowledgeObject(
            knowledge_id=row.knowledge_id,
            topic=row.topic,
            content=content,
            content_hash=row.content_hash,
            source_uri=row.source_uri,
            retrieved_at=row.retrieved_at,
            authority=row.authority,
            status=row.status,
            artifact_ref=row.r2_blob_key,
            embedding_ref=row.vectorize_id,
        )

    async def _cache_get(self, cache_key: str) -> str | None:
        try:
            return await self._kv.get_text(cache_key)
        except Exception:
            return None

    async def _try_cached_result(
        self,
        request: RetrievalRequest,
        cache_key: str,
        revision: int,
        raw: str,
    ) -> RetrievalResult | None:
        try:
            envelope = QueryCacheEnvelope.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError):
            return None
        if (
            envelope.corpus_revision != revision
            or envelope.embedding_digest != self._embedding_digest
            or envelope.request_fingerprint != cache_key
        ):
            return None

        objects: list[KnowledgeObject] = []
        for ref in envelope.refs:
            row = await self._reader.get_row(ref.knowledge_id)
            if row is None or row.content_hash != ref.content_hash:
                return None
            try:
                obj = await self._hydrate(row)
            except KnowledgeIntegrityError:
                return None
            objects.append(obj)
        return RetrievalResult(topic=request.topic, objects=objects)


def _non_negative_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


__all__ = [
    "KnowledgeRuntimeError",
    "KnowledgeConflictError",
    "KnowledgeIntegrityError",
    "AsyncCloudflareKnowledgeRuntime",
]
