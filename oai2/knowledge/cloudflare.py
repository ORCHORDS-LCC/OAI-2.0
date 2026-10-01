"""Cloudflare-backed knowledge adapter (logical schema, no live network).

This module defines the *logical* binding shape for OAI-2.0's external
knowledge store on Cloudflare. It does NOT call Cloudflare in v0.2.0.
The goal is to lock the contract so the rest of the agent can be written
against a real binding surface, while a mock implementation is used by
tests and local development.

Logical layout (one Worker, five Cloudflare primitives):

    +-----------------------+
    |  Cloudflare Worker    |  <-- public HTTP entrypoint
    +-----------------------+
              |
              | (binding calls)
              v
    +---------+--------+----------------+----------------+
    | D1 (SQL)        | R2 (objects)   | Vectorize      | KV (cache)
    +-----------------+----------------+----------------+
    knowledge_index  artifact_blob    embeddings       query_cache
                                       (chunked)

Responsibilities per primitive (logical, not implementation):

* ``D1.knowledge_index`` — row per :class:`KnowledgeObject`. Stores only
  metadata + pointers (no large bodies). Public-safe: no real
  account_id / bucket name / token appears here.
* ``R2.artifact_blob`` — opaque, content-addressed chunks. Bodies are
  referenced by SHA-256, never inlined into D1.
* ``Vectorize`` — embedding index keyed by ``knowledge_id``. Populated
  by the ingestion pipeline after chunking.
* ``KV.query_cache`` — best-effort query results, keyed by the hash of
  the (topic, min_authority, status_set) triple + embedding digest.

Status: PROPOSED. Wire to live Cloudflare only after secrets are
moved to a deploy-only path; the contract is stable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from ..core import KnowledgeId, Status
from .abstraction import (
    KnowledgeObject,
    KnowledgeStore,
    RetrievalRequest,
    RetrievalResult,
    sha256_hex,
)


class CFPrimitive(StrEnum):
    """Names of the Cloudflare primitives that hold OAI knowledge."""

    D1_INDEX = "D1.knowledge_index"
    R2_BLOB = "R2.artifact_blob"
    VECTORIZE = "Vectorize.embeddings"
    KV_CACHE = "KV.query_cache"


@dataclass(slots=True, frozen=True)
class CFRow:
    """One logical row in ``D1.knowledge_index``.

    Bodies never live in D1 — they are content-addressed in R2.
    """

    knowledge_id: KnowledgeId
    topic: str
    content_hash: str
    authority: float
    status: Status
    source_uri: str | None
    retrieved_at: float
    r2_blob_key: str | None
    vectorize_id: str | None


@dataclass(slots=True, frozen=True)
class CFPlan:
    """Description of the binding layout.

    Public-safe: only logical names. Real binding IDs are environment
    variables resolved by the deploy step, never checked into git.
    """

    account_id: str = "<account_id:env>"
    d1_database_id: str = "<d1_database_id:env>"
    r2_bucket: str = "<r2_bucket:env>"
    vectorize_index: str = "oai2-knowledge-v1"
    kv_namespace: str = "<kv_namespace_id:env>"
    worker_name: str = "oai2-knowledge-worker"


def row_to_d1(row: CFRow) -> dict[str, Any]:
    """Serialize a :class:`CFRow` to a D1 INSERT/UPSERT parameter map."""
    return {
        "knowledge_id": row.knowledge_id,
        "topic": row.topic,
        "content_hash": row.content_hash,
        "authority": row.authority,
        "status": row.status.value,
        "source_uri": row.source_uri,
        "retrieved_at": row.retrieved_at,
        "r2_blob_key": row.r2_blob_key,
        "vectorize_id": row.vectorize_id,
    }


def object_to_row(obj: KnowledgeObject) -> CFRow:
    """Map a :class:`KnowledgeObject` onto a :class:`CFRow`.

    The body lives in R2 under ``oai2-blobs/<content_hash>``. The
    embedding record uses ``knowledge_id`` as its Vectorize id so we can
    delete from Vectorize when a row is removed.
    """
    return CFRow(
        knowledge_id=obj.knowledge_id,
        topic=obj.topic,
        content_hash=obj.content_hash,
        authority=obj.authority,
        status=obj.status,
        source_uri=obj.source_uri,
        retrieved_at=obj.retrieved_at,
        r2_blob_key=f"oai2-blobs/{obj.content_hash}" if obj.content else None,
        vectorize_id=str(obj.knowledge_id),
    )


def r2_blob_key_for(content_hash: str) -> str:
    """Compute the canonical R2 object key for a content hash."""
    return f"oai2-blobs/{content_hash}"


# ---------------------------------------------------------------------------
# Mock bindings — exercised by tests, identical shape to live Worker calls.
# ---------------------------------------------------------------------------


class _D1(Protocol):
    def prepare(self, sql: str) -> _D1Stmt: ...


class _D1Stmt(Protocol):
    def bind(self, *args: Any) -> _D1Stmt: ...
    def all(self) -> list[dict[str, Any]]: ...
    def first(self) -> dict[str, Any] | None: ...
    def run(self) -> None: ...


class _R2(Protocol):
    def put(self, key: str, value: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def delete(self, key: str) -> None: ...


class _Vectorize(Protocol):
    def insert(self, ids: list[str], vectors: list[list[float]]) -> None: ...
    def query(
        self, vector: list[float], top_k: int
    ) -> list[tuple[str, float]]: ...
    def delete_by_ids(self, ids: list[str]) -> None: ...


class _KV(Protocol):
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str, expiration_ttl: int | None = None) -> None: ...


@dataclass(slots=True)
class MockCloudflareBindings:
    """In-memory stand-in for the live Worker bindings.

    Every method maps 1:1 to the binding call a real Worker would make.
    No network involved; fully deterministic.
    """

    rows: dict[KnowledgeId, dict[str, Any]] = None  # type: ignore[assignment]
    blobs: dict[str, bytes] = None  # type: ignore[assignment]
    vectors: dict[str, list[float]] = None  # type: ignore[assignment]
    cache: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = {}
        if self.blobs is None:
            self.blobs = {}
        if self.vectors is None:
            self.vectors = {}
        if self.cache is None:
            self.cache = {}

    # -- D1.knowledge_index --
    def d1_upsert(self, row: CFRow) -> None:
        self.rows[row.knowledge_id] = row_to_d1(row)

    def d1_get(self, knowledge_id: KnowledgeId) -> CFRow | None:
        raw = self.rows.get(knowledge_id)
        if raw is None:
            return None
        return CFRow(
            knowledge_id=raw["knowledge_id"],
            topic=raw["topic"],
            content_hash=raw["content_hash"],
            authority=float(raw["authority"]),
            status=Status(raw["status"]),
            source_uri=raw.get("source_uri"),
            retrieved_at=float(raw.get("retrieved_at", 0.0)),
            r2_blob_key=raw.get("r2_blob_key"),
            vectorize_id=raw.get("vectorize_id"),
        )

    def d1_query(
        self,
        topic_substring: str,
        min_authority: float,
        statuses: Iterable[Status],
        limit: int,
    ) -> list[CFRow]:
        allowed = {s.value for s in statuses}
        out: list[CFRow] = []
        for raw in self.rows.values():
            if topic_substring.lower() not in str(raw["topic"]).lower():
                continue
            if float(raw["authority"]) < min_authority:
                continue
            if str(raw["status"]) not in allowed:
                continue
            row = self.d1_get(raw["knowledge_id"])
            if row is not None:
                out.append(row)
        out.sort(key=lambda r: (r.authority, r.retrieved_at), reverse=True)
        return out[:limit]

    # -- R2.artifact_blob --
    def r2_put(self, key: str, body: bytes) -> None:
        self.blobs[key] = body

    def r2_get(self, key: str) -> bytes | None:
        return self.blobs.get(key)

    # -- Vectorize.embeddings --
    def vectorize_upsert(self, vid: str, vector: list[float]) -> None:
        self.vectors[vid] = vector

    def vectorize_query(
        self, vector: list[float], top_k: int
    ) -> list[tuple[str, float]]:
        if not self.vectors:
            return []
        # Cosine similarity against every stored vector. Deterministic and
        # cheap for the mock; the live Vectorize does this server-side.
        scored: list[tuple[str, float]] = []
        for vid, stored in self.vectors.items():
            denom = (
                sum(a * a for a in vector) ** 0.5
                * sum(a * a for a in stored) ** 0.5
            )
            score = (
                sum(a * b for a, b in zip(vector, stored, strict=True)) / denom
                if denom > 0.0
                else 0.0
            )
            scored.append((vid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # -- KV.query_cache --
    def kv_get(self, key: str) -> str | None:
        return self.cache.get(key)

    def kv_put(self, key: str, value: str, ttl: int | None = None) -> None:
        # ttl is ignored in the mock; live Worker enforces expiry.
        self.cache[key] = value


# ---------------------------------------------------------------------------
# Adapter — translates KnowledgeStore operations into binding calls.
# ---------------------------------------------------------------------------


def cache_key_for(request: RetrievalRequest, embedding_digest: str) -> str:
    """Stable cache key for a retrieval request + embedding digest.

    The embedding digest is content-addressed; mixing it in means a
    re-embedding of the same content does NOT invalidate the cache.
    """
    payload = {
        "topic": request.topic,
        "limit": request.limit,
        "min_authority": request.min_authority,
        "include_status": sorted(s.value for s in request.include_status),
        "embedding_digest": embedding_digest,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "q:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class CloudflareKnowledgeStore(KnowledgeStore):
    """Adapter implementing :class:`KnowledgeStore` on top of Cloudflare bindings."""

    STATUS = Status.PROPOSED

    def __init__(
        self,
        bindings: MockCloudflareBindings,
        *,
        embedding_digest: str = "stub-v0",
    ) -> None:
        self._b = bindings
        self._embedding_digest = embedding_digest

    def put(self, obj: KnowledgeObject) -> None:
        row = object_to_row(obj)
        self._b.d1_upsert(row)
        if obj.content:
            self._b.r2_put(r2_blob_key_for(obj.content_hash), obj.content.encode("utf-8"))

    def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:
        row = self._b.d1_get(knowledge_id)
        if row is None:
            return None
        blob = self._b.r2_get(row.r2_blob_key) if row.r2_blob_key else None
        content = blob.decode("utf-8") if blob is not None else ""
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

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        cache_key = cache_key_for(request, self._embedding_digest)
        cached = self._b.kv_get(cache_key)
        if cached is not None:
            payload = json.loads(cached)
            cached_objs = [
                KnowledgeObject.model_validate(o) for o in payload["objects"]
            ]
            return RetrievalResult(topic=request.topic, objects=cached_objs)

        rows = self._b.d1_query(
            topic_substring=request.topic,
            min_authority=request.min_authority,
            statuses=request.include_status,
            limit=request.limit,
        )
        objs: list[KnowledgeObject] = []
        for row in rows:
            obj = self.get(row.knowledge_id)
            if obj is not None:
                objs.append(obj)

        # Best-effort cache write; TTL 5 minutes matches the Worker default.
        self._b.kv_put(
            cache_key,
            json.dumps(
                {"objects": [o.model_dump() for o in objs]},
                separators=(",", ":"),
            ),
            ttl=300,
        )
        return RetrievalResult(topic=request.topic, objects=objs)

    def all(self) -> Iterable[KnowledgeObject]:
        out: list[KnowledgeObject] = []
        for kid in tuple(self._b.rows.keys()):
            obj = self.get(kid)
            if obj is not None:
                out.append(obj)
        return tuple(out)


def hash_topic(topic: str) -> str:
    """Stable, public-safe digest for a topic string.

    Used to seed embedding-digest without leaking topic text.
    """
    return sha256_hex(topic)[:16]


__all__ = [
    "CFPrimitive",
    "CFRow",
    "CFPlan",
    "MockCloudflareBindings",
    "CloudflareKnowledgeStore",
    "cache_key_for",
    "object_to_row",
    "row_to_d1",
    "r2_blob_key_for",
    "hash_topic",
]
