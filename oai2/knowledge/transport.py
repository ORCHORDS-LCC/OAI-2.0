"""Versioned public-safe transport schemas for the live knowledge Worker.

These models define the application/wire contract that a future live Cloudflare
Worker adapter must implement. They do not contain credentials, account IDs,
private endpoints, or native binding objects.

Authentication credentials belong to the transport layer (for example an
Authorization header). :class:`TransportAuthContext` represents only the
normalized identity/capability context after authentication.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core import KnowledgeId, Status
from .abstraction import KnowledgeObject, sha256_hex

TRANSPORT_VERSION = "1"

# ---------------------------------------------------------------------------
# Vector identity (#19)
#
# The vector id is what makes the D1 row an ATOMIC authority switch rather than
# a description that can drift from the bytes it points at.
#
# The previous identity was the bare knowledge_id, so a new embedding generation
# overwrote the very vector the last committed D1 row still referenced. If the
# following D1 write was then denied by revision or deletion-lease state, D1
# still described generation N-1 while Vectorize already held generation N, and
# a semantic read scored the NEW embedding but returned the OLD body.
#
# A generation-specific id makes that state unreachable instead of merely
# unlikely: writing generation N creates a NEW vector, so the vector the
# committed row points at is never mutated. D1 then either switches to it
# atomically, or does not — and if it does not, the old vector is untouched and
# still authoritative and the new one is a non-authoritative orphan.
#
# CANONICAL SERIALIZATION (stable; changing it changes every id)
#     "\n".join([
#         "oai2-vector-id-v1",     # scheme version, bumped if this changes
#         <knowledge_id>,
#         <content_hash>,
#         <embedding_version>,
#     ])
# joined by LF and hashed with SHA-256. Field values are rejected if they are
# empty, padded, or contain a line break, so the encoding is unambiguous: no two
# distinct triples can produce the same byte string.
#
# COLLISIONS: 56 hex characters of SHA-256 = 224 bits. The birthday bound puts a
# collision beyond 2^112 distinct vectors, so it is not a practical concern.
#
# LENGTH: the "oai2v1-" prefix plus 56 hex characters is 63 bytes, under
# Cloudflare's 64-byte Vectorize id limit. The prefix is a namespace and scheme
# marker, so a future canonicalization change produces a disjoint id space
# rather than a silent collision with ids written under this scheme.
# ---------------------------------------------------------------------------
VECTOR_ID_MAX_BYTES = 64
_VECTOR_ID_PREFIX = "oai2v1-"
_VECTOR_ID_SCHEME = "oai2-vector-id-v1"
_VECTOR_ID_HEX_CHARS = 56


def vector_id_for(
    *,
    knowledge_id: KnowledgeId,
    content_hash: str,
    embedding_version: str,
) -> str:
    """Deterministic vector id for one embedding generation of one object.

    Deterministic and content-addressed: the same triple always yields the same
    id, so a retried put re-upserts the same vector rather than accumulating
    duplicates, and no random or time-based component is involved.
    """
    parts: list[str] = []
    for name, value in (
        ("knowledge_id", knowledge_id),
        ("content_hash", content_hash),
        ("embedding_version", embedding_version),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a non-empty normalized string")
        if "\n" in value or "\r" in value:
            # Would make the canonical encoding ambiguous.
            raise ValueError(f"{name} must not contain a line break")
        parts.append(value)

    canonical = "\n".join([_VECTOR_ID_SCHEME, *parts])
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    vector_id = f"{_VECTOR_ID_PREFIX}{digest[:_VECTOR_ID_HEX_CHARS]}"
    if len(vector_id.encode("utf-8")) > VECTOR_ID_MAX_BYTES:
        # Unreachable with a fixed-width hex digest; asserted so a future
        # change to the prefix or digest width cannot silently break the limit.
        raise ValueError("derived vector id exceeds the Vectorize id length limit")
    return vector_id



class TransportOperation(StrEnum):
    PUT = "put"
    GET = "get"
    RETRIEVE = "retrieve"


class TransportErrorCode(StrEnum):
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNAVAILABLE_DEPENDENCY = "unavailable_dependency"
    INTEGRITY = "integrity"
    INTERNAL = "internal"


class TransportAuthContext(BaseModel):
    """Normalized authenticated identity; never a bearer token/credential."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=256)
    capabilities: tuple[str, ...] = ()
    expires_at: float | None = None


class KnowledgeTransportRequest(BaseModel):
    """Versioned request envelope understood by the future Worker transport."""

    model_config = ConfigDict(extra="forbid")

    version: str = TRANSPORT_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    operation: TransportOperation
    auth: TransportAuthContext | None = None

    knowledge: KnowledgeObject | None = None
    knowledge_id: KnowledgeId | None = None
    topic: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=8, ge=1, le=200)
    min_authority: float = Field(default=0.0, ge=0.0, le=1.0)
    include_status: tuple[Status, ...] = (
        Status.IMPLEMENTED,
        Status.EXPERIMENTAL,
    )

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.version != TRANSPORT_VERSION:
            raise ValueError(
                f"unsupported knowledge transport version: {self.version!r}"
            )

        if self.operation is TransportOperation.PUT:
            if self.knowledge is None:
                raise ValueError("put requires knowledge")
            if self.knowledge_id is not None or self.topic is not None:
                raise ValueError("put must not set knowledge_id/topic")
        elif self.operation is TransportOperation.GET:
            if self.knowledge_id is None or not str(self.knowledge_id).strip():
                raise ValueError("get requires knowledge_id")
            if self.knowledge is not None or self.topic is not None:
                raise ValueError("get must not set knowledge/topic")
        elif self.operation is TransportOperation.RETRIEVE:
            if self.topic is None or not self.topic.strip():
                raise ValueError("retrieve requires topic")
            if self.knowledge is not None or self.knowledge_id is not None:
                raise ValueError("retrieve must not set knowledge/knowledge_id")
        return self


class TransportError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: TransportErrorCode
    message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False


class KnowledgeTransportResponse(BaseModel):
    """Versioned response envelope with explicit success/error invariants."""

    model_config = ConfigDict(extra="forbid")

    version: str = TRANSPORT_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    ok: bool
    corpus_revision: int | None = Field(default=None, ge=0)
    knowledge: KnowledgeObject | None = None
    objects: list[KnowledgeObject] = Field(default_factory=list)
    error: TransportError | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.version != TRANSPORT_VERSION:
            raise ValueError(
                f"unsupported knowledge transport version: {self.version!r}"
            )
        if self.ok and self.error is not None:
            raise ValueError("successful response must not contain error")
        if not self.ok:
            if self.error is None:
                raise ValueError("failed response requires error")
            if self.knowledge is not None or self.objects:
                raise ValueError("failed response must not contain knowledge objects")
        return self


class D1KnowledgeIndexRecord(BaseModel):
    """Authoritative metadata shape intended for D1 persistence.

    D1 is authoritative, so every field retrieval filters on must live here
    and not only in the vector store: claim identity, content version,
    supersession, trust class and scope class. A Vectorize record missing its
    D1 row is removed from results (REQ-RET-013), so a candidate that cannot
    be checked against these fields is not eligible.
    """

    model_config = ConfigDict(extra="forbid")

    knowledge_id: KnowledgeId
    topic: str = Field(min_length=1, max_length=256)
    content_hash: str = Field(min_length=8, max_length=128)
    authority: float = Field(ge=0.0, le=1.0)
    status: Status
    source_uri: str | None = None
    retrieved_at: float
    r2_blob_key: str | None = None
    vectorize_id: str | None = None
    corpus_revision: int = Field(ge=0)

    # Temporal / supersession / trust — see KnowledgeObject for why claim_key
    # and content_hash are different things.
    claim_key: str | None = Field(default=None, max_length=128)
    content_version: str | None = Field(default=None, max_length=128)
    source_version: str | None = Field(default=None, max_length=128)
    effective_at: float | None = None
    superseded_by: str | None = Field(default=None, max_length=128)
    superseded_at: float | None = None
    trust_class: str = Field(default="retrieved_evidence", max_length=64)
    scope_class: str = Field(default="global", max_length=32)

    @property
    def is_active(self) -> bool:
        """A superseded record stays retrievable for audit, never by default."""
        return self.superseded_by is None

    @classmethod
    def from_knowledge(
        cls,
        obj: KnowledgeObject,
        *,
        corpus_revision: int,
        r2_blob_key: str | None,
        vectorize_id: str | None,
    ) -> D1KnowledgeIndexRecord:
        return cls(
            knowledge_id=obj.knowledge_id,
            topic=obj.topic,
            content_hash=obj.content_hash,
            authority=obj.authority,
            status=obj.status,
            source_uri=obj.source_uri,
            retrieved_at=obj.retrieved_at,
            r2_blob_key=r2_blob_key,
            vectorize_id=vectorize_id,
            corpus_revision=corpus_revision,
            claim_key=obj.claim_key,
            content_version=obj.content_version,
            source_version=obj.source_version,
            effective_at=obj.effective_at,
            superseded_by=obj.superseded_by,
            superseded_at=obj.superseded_at,
            trust_class=obj.trust_class,
            scope_class=obj.scope_class,
        )


class R2BodyDescriptor(BaseModel):
    """Content-addressed R2 body descriptor; body bytes are not embedded here."""

    model_config = ConfigDict(extra="forbid")

    knowledge_id: KnowledgeId
    content_hash: str = Field(min_length=8, max_length=128)
    object_key: str = Field(min_length=1, max_length=512)
    size_bytes: int = Field(ge=0)

    @classmethod
    def from_knowledge(cls, obj: KnowledgeObject) -> R2BodyDescriptor:
        actual = sha256_hex(obj.content)
        if actual != obj.content_hash:
            raise ValueError("knowledge content_hash does not match content")
        body = obj.content.encode("utf-8")
        return cls(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            object_key=f"oai2-blobs/{obj.content_hash}",
            size_bytes=len(body),
        )


class VectorizeMetadata(BaseModel):
    """Public-safe metadata stored alongside an embedding/vector record."""

    model_config = ConfigDict(extra="forbid")

    knowledge_id: KnowledgeId
    content_hash: str = Field(min_length=8, max_length=128)
    status: Status
    authority: float = Field(ge=0.0, le=1.0)
    source_uri: str | None = None
    embedding_version: str = Field(min_length=1, max_length=128)

    @classmethod
    def from_knowledge(
        cls,
        obj: KnowledgeObject,
        *,
        embedding_version: str,
    ) -> VectorizeMetadata:
        return cls(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            status=obj.status,
            authority=obj.authority,
            source_uri=obj.source_uri,
            embedding_version=embedding_version,
        )


class VectorMatch(BaseModel):
    """One Vectorize match, parsed fail-closed into a typed read contract.

    A score on its own is not a candidate. It is a number computed from a
    specific vector, for a specific knowledge_id, content_hash and
    embedding_version, and every one of those is needed to decide whether the
    score may be attached to an object at all. Carrying them together makes it
    possible to refuse the pairing rather than to assert it later.

    Public-safe by construction: no body bytes, no R2 key, no credential, no
    deployment identifier, and no field a caller could use to reach anything
    other than the object it already named. ``extra="forbid"`` so a widened
    binding payload cannot smuggle extra state into this contract.

    Parsing is fail-closed. A malformed id, a non-finite score, missing
    metadata, or an invalid knowledge_id / content_hash / embedding_version is
    an error, not a degraded match: a match that cannot be fully attributed must
    not become a candidate that merely looks fine.
    """

    model_config = ConfigDict(extra="forbid")

    vector_id: str = Field(min_length=1, max_length=VECTOR_ID_MAX_BYTES)
    score: float
    knowledge_id: KnowledgeId
    content_hash: str = Field(min_length=8, max_length=128)
    embedding_version: str = Field(min_length=1, max_length=128)

    @classmethod
    def from_vectorize_parts(
        cls,
        vector_id: object,
        score: object,
        metadata: object,
    ) -> VectorMatch:
        """Build a match from raw binding fields, or raise.

        Raises ``ValueError`` for anything that cannot be attributed. Callers
        translate that into a dependency failure; it must never be downgraded to
        a skipped row, because a skipped row is indistinguishable from "no
        results" and would turn a broken dependency into a quiet empty answer.
        """
        if (
            not isinstance(vector_id, str)
            or not vector_id
            or vector_id != vector_id.strip()
        ):
            raise ValueError("vector match has a missing or unnormalized id")
        if len(vector_id.encode("utf-8")) > VECTOR_ID_MAX_BYTES:
            raise ValueError("vector match id exceeds the Vectorize id length limit")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("vector match has a non-finite or non-numeric score")
        if not isinstance(metadata, Mapping):
            raise ValueError("vector match is missing its metadata")

        def required(name: str) -> str:
            value = metadata.get(name)
            # KnowledgeId is a NewType over str, so pydantic imposes no
            # constraint on it: an empty or padded id would validate. These
            # three fields are the whole attribution, so each is checked here
            # rather than trusted to the field constraints.
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"vector match metadata has a missing or unnormalized {name}"
                )
            return value

        return cls(
            vector_id=vector_id,
            score=float(score),
            knowledge_id=KnowledgeId(required("knowledge_id")),
            content_hash=required("content_hash"),
            embedding_version=required("embedding_version"),
        )


class KnowledgeCacheRef(BaseModel):
    """Compact cache entry reference; authoritative bodies remain in R2/D1."""

    model_config = ConfigDict(extra="forbid")

    knowledge_id: KnowledgeId
    content_hash: str = Field(min_length=8, max_length=128)


class QueryCacheEnvelope(BaseModel):
    """Best-effort KV cache envelope, versioned by corpus + embeddings."""

    model_config = ConfigDict(extra="forbid")

    version: str = TRANSPORT_VERSION
    corpus_revision: int = Field(ge=0)
    embedding_digest: str = Field(min_length=1, max_length=128)
    request_fingerprint: str = Field(min_length=1, max_length=128)
    refs: list[KnowledgeCacheRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if self.version != TRANSPORT_VERSION:
            raise ValueError(
                f"unsupported knowledge transport version: {self.version!r}"
            )
        return self


__all__ = [
    "TRANSPORT_VERSION",
    "TransportOperation",
    "TransportErrorCode",
    "TransportAuthContext",
    "KnowledgeTransportRequest",
    "TransportError",
    "KnowledgeTransportResponse",
    "D1KnowledgeIndexRecord",
    "R2BodyDescriptor",
    "VectorizeMetadata",
    "VectorMatch",
    "VECTOR_ID_MAX_BYTES",
    "vector_id_for",
    "KnowledgeCacheRef",
    "QueryCacheEnvelope",
]
