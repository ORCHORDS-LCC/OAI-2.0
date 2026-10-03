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
from ..observability import TraceRecorder
from .abstraction import (
    KnowledgeObject,
    RetrievalCandidate,
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
from .observability import KV_OPERATION_GET, KV_OPERATION_PUT, emit_kv_degraded
from .transport import (
    KnowledgeCacheRef,
    QueryCacheEnvelope,
    R2BodyDescriptor,
    VectorizeMetadata,
    vector_id_for,
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
        trace: TraceRecorder | None = None,
    ) -> int:
        """Persist one object and return the new authoritative corpus revision.

        ORDERING AND WHY IT IS SAFE (#19). R2 -> Vectorize -> D1, with D1 last
        and D1 authoritative.

        The R2 body is content-addressed (``oai2-blobs/<content_hash>``), so a
        new generation lands at a new key and the previously committed body is
        never overwritten.

        The vector is identified by generation (see ``vector_id_for``), so a new
        generation is a NEW vector id. That is the whole point: the vector the
        committed D1 row currently references is never mutated by a put. The D1
        write then performs the switch, atomically, in one row.

        Consequences, which are the invariant rather than a best effort:

        * Every step succeeding — the row now points at the new vector, and the
          old vector is an unreferenced orphan rather than a corrupted one.
        * R2 or Vectorize failing — D1 is never written, the old row and the old
          vector stay authoritative and mutually consistent, and the new body
          or vector is an orphan.
        * The D1 write being DENIED (stale expected revision, or a deletion
          lease) — the old row still points at the old vector, which still holds
          the old embedding, and the new vector is an orphan. The next
          successful update re-derives the identical vector id for identical
          inputs, so the orphan is reclaimed by reuse rather than accumulating.

        A failed update therefore never has to be undone: there is nothing to
        roll back, because nothing authoritative was overwritten. No compensating
        delete is issued, and none should be — a delete here would race a
        concurrent writer that legitimately adopted the same generation.
        """
        descriptor = R2BodyDescriptor.from_knowledge(obj)
        timestamp = time.time() if now is None else _non_negative_number(now, "now")
        expected_revision = await self._writer.corpus_revision()

        if obj.content:
            await self._r2.put_text(descriptor.object_key, obj.content)

        row = object_to_row(obj)
        if vector is None:
            row = replace(row, vectorize_id=None)
        else:
            vector_id = vector_id_for(
                knowledge_id=obj.knowledge_id,
                content_hash=obj.content_hash,
                embedding_version=self._embedding_version,
            )
            metadata = VectorizeMetadata.from_knowledge(
                obj,
                embedding_version=self._embedding_version,
            ).model_dump(mode="json")
            await self._vectorize.upsert(vector_id, vector, metadata=metadata)
            # Only now is the row allowed to name the new vector. Until this
            # D1 write commits, the new vector is referenced by nobody.
            row = replace(row, vectorize_id=vector_id)

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

    async def get(
        self, knowledge_id: KnowledgeId, *, trace: TraceRecorder | None = None
    ) -> KnowledgeObject | None:
        row = await self._reader.get_row(knowledge_id)
        if row is None:
            return None
        return await self._hydrate(row)

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        query_vector: Sequence[float] | None = None,
        trace: TraceRecorder | None = None,
    ) -> RetrievalResult:
        """Retrieve through revisioned KV cache, D1/R2, and optional Vectorize."""
        start_revision = await self._writer.corpus_revision()
        cache_key = cache_key_for(
            request,
            self._embedding_digest,
            start_revision,
        )

        if query_vector is None:
            cached = await self._cache_get(cache_key, trace=trace)
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

        semantic_scores: dict[KnowledgeId, float] = {}
        if query_vector is None:
            rows = await self._reader.query_rows(request)
        else:
            semantic_rows = await self._semantic_rows(request, query_vector)
            rows = [row for row, _score in semantic_rows]
            semantic_scores = {
                row.knowledge_id: score for row, score in semantic_rows
            }

        objects = [await self._hydrate(row) for row in rows]
        end_revision = await self._writer.corpus_revision()
        if end_revision != start_revision:
            raise KnowledgeConflictError(
                "corpus revision changed during retrieval; retry against fresh state"
            )

        if query_vector is None:
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
                # KV is explicitly non-authoritative and best-effort. The write
                # is not retried and the request does not fail; a raised write
                # is reported as degradation so a persistently failing cache is
                # visible rather than silent.
                emit_kv_degraded(trace, operation=KV_OPERATION_PUT, timestamp=time.time())

        return RetrievalResult(
            topic=request.topic,
            objects=objects,
            candidates=[
                RetrievalCandidate(
                    knowledge_id=obj.knowledge_id,
                    content_hash=obj.content_hash,
                    source_uri=obj.source_uri,
                    score=semantic_scores.get(obj.knowledge_id),
                )
                for obj in objects
            ],
        )

    async def _semantic_rows(
        self,
        request: RetrievalRequest,
        query_vector: Sequence[float],
    ) -> list[tuple[CFRow, float]]:
        """Resolve Vectorize matches to authoritative D1 rows, fail-closed.

        Four rules, in order. Each one is a refusal to attach a score to an
        object without knowing the score belongs to that object.

        1. RESOLVE THROUGH METADATA, NEVER THROUGH THE ID. The D1 row is looked
           up by the match's own ``knowledge_id``. The old code did
           ``get_row(KnowledgeId(vector_id))``, which is only correct while
           vector_id happens to equal knowledge_id — an assumption this fix
           removes rather than encodes.

        2. NO D1 ROW, NO CANDIDATE. A vector without an authoritative row is
           eventually-consistent leftovers or a post-delete orphan. It is
           excluded, and R2 is never read for it.

        3. THE SCORE BELONGS TO ONE SPECIFIC VECTOR. It is carried only when the
           resolved row is the row that currently references ``match.vector_id``.
           A match naming any other vector is a previous generation or an
           orphan, and is dropped WITHOUT its score. This is the rule that
           closed the split-brain: a failed put leaves the committed row
           pointing at the old vector, and the new vector that D1 does not
           reference can no longer contribute a score to the old object.

           Note this is an exclusion, not an integrity error. A mismatch here is
           the expected steady state while orphans exist, so raising would make
           an ordinary retry loop look like corruption. The invariant is
           "never returned", not "never observed".

        4. ONE VECTOR, ONE CONTENT. If the id matches but the vector's own
           ``content_hash`` disagrees with the row it claims to represent, that
           is genuine corruption — one vector asserting two contents — and it
           raises rather than being silently dropped.

        A match whose ``embedding_version`` differs from the version this query
        was embedded with is also excluded: its score was computed in a
        different vector space and is not comparable to this query's. That is
        the documented embedding-migration transition (#73), and it is why the
        read contract carries the version at all.
        """
        # Over-fetch before authoritative D1 filtering so an ineligible top
        # vector cannot hide an eligible lower-ranked candidate.
        top_k = min(max(int(request.limit) * 4, 8), 100)
        matches = await self._vectorize.query(query_vector, top_k=top_k)
        rows: list[tuple[CFRow, float]] = []
        seen: set[KnowledgeId] = set()
        for match in matches:
            row = await self._reader.get_row(match.knowledge_id)
            if row is None:
                # Vectorize is eventually consistent and non-authoritative.
                # A stale/deleted vector must not become a final candidate.
                continue
            if str(row.knowledge_id) != str(match.knowledge_id):
                # Unreachable via get_row, but the id is what D1 was asked for
                # and the id is what the score is attributed to.
                raise KnowledgeIntegrityError(
                    "D1 row identity disagrees with the Vectorize match knowledge_id"
                )
            if row.vectorize_id is None or row.vectorize_id != match.vector_id:
                # Rule 3: not the vector D1 currently references. Excluded, and
                # excluded WITH its score.
                continue
            if match.content_hash != row.content_hash:
                # Rule 4: same vector id, two different contents.
                raise KnowledgeIntegrityError(
                    f"Vectorize match {match.vector_id!r} carries content_hash "
                    f"{match.content_hash!r} but D1 holds {row.content_hash!r}"
                )
            if match.embedding_version != self._embedding_version:
                # Scored in a different embedding space; not comparable.
                continue
            if row.authority < request.min_authority:
                continue
            if row.status not in request.include_status:
                continue
            if row.knowledge_id in seen:
                # One authoritative row, one candidate, even if an index ever
                # returned two vector ids that D1 both point at.
                continue
            seen.add(row.knowledge_id)
            rows.append((row, match.score))
            if len(rows) >= request.limit:
                break
        return rows

    async def _hydrate(self, row: CFRow) -> KnowledgeObject:
        if row.r2_blob_key is None:
            content = ""
        else:
            text = await self._r2.get_text(row.r2_blob_key)
            if text is None:
                raise KnowledgeIntegrityError(
                    f"R2 body is missing for knowledge_id={row.knowledge_id}"
                )
            content = text

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

    async def _cache_get(
        self, cache_key: str, *, trace: TraceRecorder | None = None
    ) -> str | None:
        try:
            return await self._kv.get_text(cache_key)
        except Exception:
            # A KV read that RAISED is not a cache miss. Collapsing the two
            # made a KV outage look like a perfectly healthy empty cache, which
            # is how a silent cache failure survives for a long time: nothing
            # errors, every request is just slower. The operation continues
            # against authoritative D1 either way, so this stays a degraded
            # signal and never a request failure.
            #
            # The recorder is a PARAMETER, never stored on self. Self is
            # rebuilt per request today, so a stored recorder would appear to
            # work; that is exactly the shape that leaks a stale request's
            # trace the moment anything caches the runtime. A parameter cannot
            # outlive the call it was given to.
            emit_kv_degraded(trace, operation=KV_OPERATION_GET, timestamp=time.time())
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
        return RetrievalResult(
            topic=request.topic,
            objects=objects,
            candidates=[
                RetrievalCandidate(
                    knowledge_id=obj.knowledge_id,
                    content_hash=obj.content_hash,
                    source_uri=obj.source_uri,
                )
                for obj in objects
            ],
        )


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
