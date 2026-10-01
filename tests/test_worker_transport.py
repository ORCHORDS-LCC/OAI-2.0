from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import KnowledgeObject, RetrievalResult, sha256_hex
from oai2.knowledge.cloudflare_runtime import (
    KnowledgeConflictError,
    KnowledgeIntegrityError,
)
from oai2.knowledge.transport import (
    KnowledgeTransportRequest,
    TransportAuthContext,
    TransportErrorCode,
    TransportOperation,
)
from oai2.knowledge.worker_transport import KnowledgeWorkerTransport


def _obj() -> KnowledgeObject:
    body = "worker transport body"
    return KnowledgeObject(
        knowledge_id=KnowledgeId("ko_worker_1"),
        topic="worker",
        content=body,
        content_hash=sha256_hex(body),
        source_uri="https://example.test/worker",
        retrieved_at=10.0,
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )


@dataclass
class FakeRuntime:
    put_revision: int = 8
    get_result: KnowledgeObject | None = field(default_factory=_obj)
    retrieve_result: RetrievalResult = field(
        default_factory=lambda: RetrievalResult(topic="worker", objects=[_obj()])
    )
    error: Exception | None = None
    puts: list[tuple[KnowledgeObject, object, float]] = field(default_factory=list)
    retrieves: list[tuple[object, object]] = field(default_factory=list)

    async def put(
        self,
        obj: KnowledgeObject,
        *,
        vector: object = None,
        now: float | None = None,
    ) -> int:
        if self.error is not None:
            raise self.error
        assert now is not None
        self.puts.append((obj, vector, now))
        return self.put_revision

    async def get(self, _knowledge_id: KnowledgeId) -> KnowledgeObject | None:
        if self.error is not None:
            raise self.error
        return self.get_result

    async def retrieve(self, request: object, *, query_vector: object = None) -> RetrievalResult:
        if self.error is not None:
            raise self.error
        self.retrieves.append((request, query_vector))
        return self.retrieve_result


class FakeEmbedder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [1.0, 0.0]


def _auth(*caps: str, expires_at: float | None = 1000.0) -> TransportAuthContext:
    return TransportAuthContext(
        subject="local-user",
        capabilities=tuple(caps),
        expires_at=expires_at,
    )


@pytest.mark.asyncio
async def test_put_requires_write_capability_and_dispatches_embedding() -> None:
    runtime = FakeRuntime()
    embedder = FakeEmbedder()
    transport = KnowledgeWorkerTransport(
        runtime,  # type: ignore[arg-type]
        embedding_provider=embedder,
    )
    obj = _obj()
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.PUT,
        auth=_auth("knowledge.write"),
        knowledge=obj,
    )

    response = await transport.handle(request, now=100.0)

    assert response.ok is True
    assert response.corpus_revision == 8
    assert response.knowledge == obj
    assert embedder.inputs == [obj.content]
    assert runtime.puts[0][1] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_missing_auth_and_capability_fail_with_authorization_error() -> None:
    runtime = FakeRuntime()
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]

    missing = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    missing_response = await transport.handle(missing, now=100.0)
    assert missing_response.ok is False
    assert missing_response.error is not None
    assert missing_response.error.code is TransportErrorCode.AUTHORIZATION

    wrong = KnowledgeTransportRequest(
        request_id="r2",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.write"),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )
    wrong_response = await transport.handle(wrong, now=100.0)
    assert wrong_response.error is not None
    assert wrong_response.error.code is TransportErrorCode.AUTHORIZATION


@pytest.mark.asyncio
async def test_expired_auth_context_is_rejected() -> None:
    runtime = FakeRuntime()
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read", expires_at=100.0),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )

    response = await transport.handle(request, now=100.0)

    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.AUTHORIZATION
    assert "expired" in response.error.message


@pytest.mark.asyncio
async def test_get_returns_not_found_explicitly() -> None:
    runtime = FakeRuntime(get_result=None)
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read"),
        knowledge_id=KnowledgeId("missing"),
    )

    response = await transport.handle(request, now=100.0)

    assert response.ok is False
    assert response.error is not None
    assert response.error.code is TransportErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_retrieve_builds_existing_retrieval_contract_and_query_embedding() -> None:
    runtime = FakeRuntime()
    embedder = FakeEmbedder()
    transport = KnowledgeWorkerTransport(
        runtime,  # type: ignore[arg-type]
        embedding_provider=embedder,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.RETRIEVE,
        auth=_auth("knowledge.read"),
        topic="worker",
        limit=4,
        min_authority=0.6,
        include_status=(Status.EXPERIMENTAL,),
    )

    response = await transport.handle(request, now=100.0)

    assert response.ok is True
    assert len(response.objects) == 1
    retrieval, vector = runtime.retrieves[0]
    assert retrieval.topic == "worker"
    assert retrieval.limit == 4
    assert retrieval.min_authority == 0.6
    assert retrieval.include_status == (Status.EXPERIMENTAL,)
    assert vector == [1.0, 0.0]
    assert embedder.inputs == ["worker"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (KnowledgeConflictError("revision changed"), TransportErrorCode.CONFLICT, True),
        (KnowledgeIntegrityError("hash mismatch"), TransportErrorCode.INTEGRITY, False),
        (RuntimeError("D1 unavailable"), TransportErrorCode.UNAVAILABLE_DEPENDENCY, True),
    ],
)
async def test_runtime_failures_map_to_explicit_transport_errors(
    error: Exception,
    code: TransportErrorCode,
    retryable: bool,
) -> None:
    runtime = FakeRuntime(error=error)
    transport = KnowledgeWorkerTransport(runtime)  # type: ignore[arg-type]
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        auth=_auth("knowledge.read"),
        knowledge_id=KnowledgeId("ko_worker_1"),
    )

    response = await transport.handle(request, now=100.0)

    assert response.ok is False
    assert response.error is not None
    assert response.error.code is code
    assert response.error.retryable is retryable


@pytest.mark.asyncio
async def test_transport_can_be_configured_for_trusted_no_auth_local_path() -> None:
    runtime = FakeRuntime()
    transport = KnowledgeWorkerTransport(
        runtime,  # type: ignore[arg-type]
        require_auth=False,
    )
    request = KnowledgeTransportRequest(
        request_id="r1",
        operation=TransportOperation.GET,
        knowledge_id=KnowledgeId("ko_worker_1"),
    )

    response = await transport.handle(request, now=100.0)

    assert response.ok is True
    assert response.knowledge is not None



def test_worker_transport_exports_from_knowledge_package() -> None:
    from oai2.knowledge import EmbeddingProvider as ExportedEmbeddingProvider
    from oai2.knowledge import KnowledgeWorkerTransport as ExportedTransport
    from oai2.knowledge.worker_transport import (
        EmbeddingProvider,
        KnowledgeWorkerTransport,
    )

    assert ExportedEmbeddingProvider is EmbeddingProvider
    assert ExportedTransport is KnowledgeWorkerTransport
