"""Tests for the versioned Cloudflare Worker transport contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai2.core import KnowledgeId, Status
from oai2.knowledge import KnowledgeObject, sha256_hex
from oai2.knowledge.transport import (
    TRANSPORT_VERSION,
    D1KnowledgeIndexRecord,
    KnowledgeCacheRef,
    KnowledgeTransportRequest,
    KnowledgeTransportResponse,
    QueryCacheEnvelope,
    R2BodyDescriptor,
    TransportAuthContext,
    TransportError,
    TransportErrorCode,
    TransportOperation,
    VectorizeMetadata,
)


def _obj() -> KnowledgeObject:
    body = "verified knowledge"
    return KnowledgeObject(
        knowledge_id=KnowledgeId("ko_transport_1"),
        topic="transport",
        content=body,
        content_hash=sha256_hex(body),
        source_uri="https://example.test/source",
        retrieved_at=123.0,
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )


def test_put_request_round_trip_preserves_knowledge_fields() -> None:
    req = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        auth=TransportAuthContext(subject="local-user", capabilities=("knowledge.write",)),
        knowledge=_obj(),
    )
    restored = KnowledgeTransportRequest.model_validate_json(req.model_dump_json())
    assert restored.version == TRANSPORT_VERSION
    assert restored.knowledge is not None
    assert restored.knowledge.knowledge_id == "ko_transport_1"
    assert restored.knowledge.content_hash == _obj().content_hash
    assert restored.knowledge.source_uri == "https://example.test/source"
    assert restored.knowledge.authority == 0.9
    assert restored.knowledge.status is Status.EXPERIMENTAL


def test_request_rejects_unsupported_version() -> None:
    with pytest.raises(ValidationError, match="unsupported knowledge transport version"):
        KnowledgeTransportRequest(
            version="999",
            request_id="r1",
            operation=TransportOperation.GET,
            knowledge_id=KnowledgeId("ko_1"),
        )


@pytest.mark.parametrize(
    ("operation", "kwargs", "message"),
    [
        (TransportOperation.PUT, {}, "put requires knowledge"),
        (TransportOperation.GET, {}, "get requires knowledge_id"),
        (TransportOperation.RETRIEVE, {}, "retrieve requires topic"),
    ],
)
def test_operation_specific_required_fields(
    operation: TransportOperation,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        KnowledgeTransportRequest(
            request_id="r1",
            operation=operation,
            **kwargs,
        )


def test_success_response_cannot_contain_error() -> None:
    with pytest.raises(ValidationError, match="successful response"):
        KnowledgeTransportResponse(
            request_id="r1",
            ok=True,
            error=TransportError(
                code=TransportErrorCode.INTERNAL,
                message="should not coexist",
            ),
        )


def test_failed_response_requires_error_and_no_objects() -> None:
    with pytest.raises(ValidationError, match="failed response requires error"):
        KnowledgeTransportResponse(request_id="r1", ok=False)

    with pytest.raises(ValidationError, match="must not contain knowledge"):
        KnowledgeTransportResponse(
            request_id="r1",
            ok=False,
            objects=[_obj()],
            error=TransportError(
                code=TransportErrorCode.UNAVAILABLE_DEPENDENCY,
                message="D1 unavailable",
                retryable=True,
            ),
        )


def test_error_codes_cover_contract_failure_classes() -> None:
    expected = {
        "authorization",
        "validation",
        "not_found",
        "conflict",
        "unavailable_dependency",
        "integrity",
        "internal",
    }
    assert {code.value for code in TransportErrorCode} == expected


def test_d1_record_preserves_authoritative_metadata() -> None:
    obj = _obj()
    row = D1KnowledgeIndexRecord.from_knowledge(
        obj,
        corpus_revision=7,
        r2_blob_key=f"oai2-blobs/{obj.content_hash}",
        vectorize_id=str(obj.knowledge_id),
    )
    assert row.knowledge_id == obj.knowledge_id
    assert row.content_hash == obj.content_hash
    assert row.source_uri == obj.source_uri
    assert row.authority == obj.authority
    assert row.status == obj.status
    assert row.corpus_revision == 7


def test_r2_descriptor_is_content_addressed_and_checks_integrity() -> None:
    obj = _obj()
    desc = R2BodyDescriptor.from_knowledge(obj)
    assert desc.object_key == f"oai2-blobs/{obj.content_hash}"
    assert desc.size_bytes == len(obj.content.encode("utf-8"))

    obj.content_hash = sha256_hex("different")
    with pytest.raises(ValueError, match="content_hash"):
        R2BodyDescriptor.from_knowledge(obj)


def test_vectorize_metadata_preserves_provenance_and_embedding_version() -> None:
    meta = VectorizeMetadata.from_knowledge(_obj(), embedding_version="embed-v1")
    assert meta.knowledge_id == "ko_transport_1"
    assert meta.source_uri == "https://example.test/source"
    assert meta.embedding_version == "embed-v1"


def test_query_cache_envelope_is_revision_and_embedding_versioned() -> None:
    obj = _obj()
    env = QueryCacheEnvelope(
        corpus_revision=4,
        embedding_digest="embed-v1-digest",
        request_fingerprint="query-fingerprint",
        refs=[
            KnowledgeCacheRef(
                knowledge_id=obj.knowledge_id,
                content_hash=obj.content_hash,
            )
        ],
    )
    restored = QueryCacheEnvelope.model_validate_json(env.model_dump_json())
    assert restored.corpus_revision == 4
    assert restored.embedding_digest == "embed-v1-digest"
    assert restored.refs[0].content_hash == obj.content_hash


def test_auth_context_contains_identity_not_credentials() -> None:
    auth = TransportAuthContext(
        subject="local-user",
        capabilities=("knowledge.read", "knowledge.write"),
        expires_at=999.0,
    )
    dumped = auth.model_dump()
    assert set(dumped) == {"subject", "capabilities", "expires_at"}


def test_transport_symbols_are_exported_from_knowledge_package() -> None:
    from oai2.knowledge import (
        TRANSPORT_VERSION as ExportedTransportVersion,
    )
    from oai2.knowledge import (
        D1KnowledgeIndexRecord as ExportedD1Record,
    )
    from oai2.knowledge import (
        KnowledgeCacheRef as ExportedCacheRef,
    )
    from oai2.knowledge import (
        KnowledgeTransportRequest as ExportedRequest,
    )
    from oai2.knowledge import (
        KnowledgeTransportResponse as ExportedResponse,
    )
    from oai2.knowledge import (
        QueryCacheEnvelope as ExportedQueryEnvelope,
    )
    from oai2.knowledge import (
        R2BodyDescriptor as ExportedR2Descriptor,
    )
    from oai2.knowledge import (
        TransportAuthContext as ExportedAuthContext,
    )
    from oai2.knowledge import (
        TransportError as ExportedTransportError,
    )
    from oai2.knowledge import (
        TransportErrorCode as ExportedErrorCode,
    )
    from oai2.knowledge import (
        TransportOperation as ExportedOperation,
    )
    from oai2.knowledge import (
        VectorizeMetadata as ExportedVectorizeMetadata,
    )
    from oai2.knowledge.transport import (
        TRANSPORT_VERSION,
        D1KnowledgeIndexRecord,
        KnowledgeCacheRef,
        KnowledgeTransportRequest,
        KnowledgeTransportResponse,
        QueryCacheEnvelope,
        R2BodyDescriptor,
        TransportAuthContext,
        TransportError,
        TransportErrorCode,
        TransportOperation,
        VectorizeMetadata,
    )

    assert ExportedTransportVersion is TRANSPORT_VERSION
    assert ExportedD1Record is D1KnowledgeIndexRecord
    assert ExportedCacheRef is KnowledgeCacheRef
    assert ExportedRequest is KnowledgeTransportRequest
    assert ExportedResponse is KnowledgeTransportResponse
    assert ExportedQueryEnvelope is QueryCacheEnvelope
    assert ExportedR2Descriptor is R2BodyDescriptor
    assert ExportedAuthContext is TransportAuthContext
    assert ExportedTransportError is TransportError
    assert ExportedErrorCode is TransportErrorCode
    assert ExportedOperation is TransportOperation
    assert ExportedVectorizeMetadata is VectorizeMetadata
