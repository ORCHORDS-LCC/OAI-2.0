"""Versioned public-safe transport schemas for the live knowledge Worker.

These models define the application/wire contract that a future live Cloudflare
Worker adapter must implement. They do not contain credentials, account IDs,
private endpoints, or native binding objects.

Authentication credentials belong to the transport layer (for example an
Authorization header). :class:`TransportAuthContext` represents only the
normalized identity/capability context after authentication.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core import KnowledgeId, Status
from .abstraction import KnowledgeObject, sha256_hex

TRANSPORT_VERSION = "1"


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
    """Authoritative metadata shape intended for D1 persistence."""

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
    "KnowledgeCacheRef",
    "QueryCacheEnvelope",
]
