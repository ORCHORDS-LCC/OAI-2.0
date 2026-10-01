from __future__ import annotations

from dataclasses import replace

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    CFRow,
    KnowledgeObject,
    RetrievalRequest,
    sha256_hex,
)
from oai2.knowledge.cloudflare_runtime import (
    AsyncCloudflareKnowledgeRuntime,
    KnowledgeConflictError,
    KnowledgeIntegrityError,
)


class FakeReader:
    def __init__(self) -> None:
        self.rows: dict[str, CFRow] = {}
        self.query_result: list[CFRow] = []

    async def get_row(self, knowledge_id: KnowledgeId) -> CFRow | None:
        return self.rows.get(str(knowledge_id))

    async def query_rows(self, request: RetrievalRequest) -> list[CFRow]:
        return list(self.query_result[: request.limit])


class FakeWriter:
    def __init__(self, revision: int = 0) -> None:
        self.revision = revision
        self.deny_write = False
        self.revision_reads: list[int] = []
        self.writes: list[tuple[CFRow, int, float]] = []
        self.read_sequence: list[int] = []

    async def corpus_revision(self) -> int:
        if self.read_sequence:
            value = self.read_sequence.pop(0)
        else:
            value = self.revision
        self.revision_reads.append(value)
        return value

    async def write_metadata(
        self,
        row: CFRow,
        *,
        expected_revision: int,
        now: float,
    ) -> int | None:
        self.writes.append((row, expected_revision, now))
        if self.deny_write:
            return None
        self.revision = expected_revision + 1
        return self.revision


class FakeR2:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.puts: list[tuple[str, str]] = []

    async def put_text(self, key: str, value: str) -> None:
        self.values[key] = value
        self.puts.append((key, value))

    async def get_text(self, key: str) -> str | None:
        return self.values.get(key)


class FakeVectorize:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[float], dict[str, object]]] = []
        self.matches: list[tuple[str, float]] = []

    async def upsert(
        self,
        vector_id: str,
        values: object,
        *,
        metadata: object = None,
    ) -> object:
        assert isinstance(values, (list, tuple))
        assert isinstance(metadata, dict)
        self.upserts.append(
            (vector_id, [float(v) for v in values], metadata)
        )
        return {"mutationId": "m1"}

    async def query(self, values: object, *, top_k: int = 5) -> list[tuple[str, float]]:
        return list(self.matches[:top_k])


class FakeKv:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.fail_get = False
        self.fail_put = False
        self.puts: list[tuple[str, str, int | None]] = []

    async def get_text(self, key: str) -> str | None:
        if self.fail_get:
            raise RuntimeError("KV outage")
        return self.values.get(key)

    async def put_text(
        self,
        key: str,
        value: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        if self.fail_put:
            raise RuntimeError("KV outage")
        self.values[key] = value
        self.puts.append((key, value, ttl_seconds))


def _obj(
    *,
    knowledge_id: str = "ko_runtime_1",
    topic: str = "runtime",
    content: str = "verified body",
) -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId(knowledge_id),
        topic=topic,
        content=content,
        content_hash=sha256_hex(content),
        source_uri="https://example.test/source",
        retrieved_at=10.0,
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )


def _row(obj: KnowledgeObject, *, vectorized: bool = True) -> CFRow:
    return CFRow(
        knowledge_id=obj.knowledge_id,
        topic=obj.topic,
        content_hash=obj.content_hash,
        authority=obj.authority,
        status=obj.status,
        source_uri=obj.source_uri,
        retrieved_at=obj.retrieved_at,
        r2_blob_key=f"oai2-blobs/{obj.content_hash}" if obj.content else None,
        vectorize_id=str(obj.knowledge_id) if vectorized else None,
    )


def _runtime() -> tuple[
    AsyncCloudflareKnowledgeRuntime,
    FakeReader,
    FakeWriter,
    FakeR2,
    FakeVectorize,
    FakeKv,
]:
    reader = FakeReader()
    writer = FakeWriter(revision=4)
    r2 = FakeR2()
    vectorize = FakeVectorize()
    kv = FakeKv()
    runtime = AsyncCloudflareKnowledgeRuntime(
        reader=reader,  # type: ignore[arg-type]
        writer=writer,  # type: ignore[arg-type]
        r2=r2,  # type: ignore[arg-type]
        vectorize=vectorize,  # type: ignore[arg-type]
        kv=kv,  # type: ignore[arg-type]
        embedding_version="embed-v1",
        embedding_digest="embed-digest-v1",
    )
    return runtime, reader, writer, r2, vectorize, kv


@pytest.mark.asyncio
async def test_put_writes_r2_vector_then_authoritative_d1_metadata() -> None:
    runtime, _reader, writer, r2, vectorize, _kv = _runtime()
    obj = _obj()

    revision = await runtime.put(obj, vector=[1.0, 0.0], now=20.0)

    assert revision == 5
    assert r2.puts == [(f"oai2-blobs/{obj.content_hash}", obj.content)]
    assert vectorize.upserts[0][0] == str(obj.knowledge_id)
    assert vectorize.upserts[0][2]["content_hash"] == obj.content_hash
    row, expected_revision, now = writer.writes[0]
    assert row.knowledge_id == obj.knowledge_id
    assert row.vectorize_id == str(obj.knowledge_id)
    assert expected_revision == 4
    assert now == 20.0


@pytest.mark.asyncio
async def test_put_without_vector_does_not_claim_vectorize_id() -> None:
    runtime, _reader, writer, _r2, vectorize, _kv = _runtime()

    await runtime.put(_obj(), now=20.0)

    assert vectorize.upserts == []
    assert writer.writes[0][0].vectorize_id is None


@pytest.mark.asyncio
async def test_put_surfaces_revision_or_lease_conflict() -> None:
    runtime, _reader, writer, _r2, _vectorize, _kv = _runtime()
    writer.deny_write = True

    with pytest.raises(KnowledgeConflictError, match="revision or deletion-lease"):
        await runtime.put(_obj(), now=20.0)


@pytest.mark.asyncio
async def test_get_rehydrates_and_hash_checks_r2_body() -> None:
    runtime, reader, _writer, r2, _vectorize, _kv = _runtime()
    obj = _obj()
    row = _row(obj)
    reader.rows[str(obj.knowledge_id)] = row
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content

    got = await runtime.get(obj.knowledge_id)

    assert got is not None
    assert got.content == obj.content
    assert got.content_hash == obj.content_hash


@pytest.mark.asyncio
async def test_get_fails_explicitly_on_missing_or_tampered_r2_body() -> None:
    runtime, reader, _writer, r2, _vectorize, _kv = _runtime()
    obj = _obj()
    row = _row(obj)
    reader.rows[str(obj.knowledge_id)] = row

    with pytest.raises(KnowledgeIntegrityError, match="missing"):
        await runtime.get(obj.knowledge_id)

    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = "tampered"
    with pytest.raises(KnowledgeIntegrityError, match="hash mismatch"):
        await runtime.get(obj.knowledge_id)


@pytest.mark.asyncio
async def test_topic_retrieve_populates_revisioned_best_effort_cache() -> None:
    runtime, reader, writer, r2, _vectorize, kv = _runtime()
    obj = _obj()
    row = _row(obj, vectorized=False)
    reader.query_result = [row]
    reader.rows[str(obj.knowledge_id)] = row
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content

    result = await runtime.retrieve(RetrievalRequest(topic="runtime"))

    assert [item.knowledge_id for item in result.objects] == [obj.knowledge_id]
    assert len(kv.puts) == 1
    assert kv.puts[0][2] == 300
    assert writer.revision_reads == [4, 4]


@pytest.mark.asyncio
async def test_kv_outage_does_not_break_authoritative_retrieval() -> None:
    runtime, reader, _writer, r2, _vectorize, kv = _runtime()
    obj = _obj()
    row = _row(obj, vectorized=False)
    reader.query_result = [row]
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content
    kv.fail_get = True
    kv.fail_put = True

    result = await runtime.retrieve(RetrievalRequest(topic="runtime"))

    assert len(result.objects) == 1
    assert result.objects[0].content == obj.content


@pytest.mark.asyncio
async def test_semantic_retrieve_uses_vectorize_and_authoritative_d1_r2() -> None:
    runtime, reader, _writer, r2, vectorize, _kv = _runtime()
    obj = _obj()
    row = _row(obj)
    reader.rows[str(obj.knowledge_id)] = row
    vectorize.matches = [(str(obj.knowledge_id), 0.99)]
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0, 0.0],
    )

    assert [item.knowledge_id for item in result.objects] == [obj.knowledge_id]


@pytest.mark.asyncio
async def test_semantic_retrieve_surfaces_vectorize_d1_inconsistency() -> None:
    runtime, _reader, _writer, _r2, vectorize, _kv = _runtime()
    vectorize.matches = [("missing", 0.99)]

    with pytest.raises(KnowledgeIntegrityError, match="no authoritative D1 row"):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[1.0],
        )


@pytest.mark.asyncio
async def test_retrieve_rejects_mixed_corpus_revision_snapshot() -> None:
    runtime, reader, writer, r2, _vectorize, _kv = _runtime()
    obj = _obj()
    row = _row(obj, vectorized=False)
    reader.query_result = [row]
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content
    writer.read_sequence = [4, 5]

    with pytest.raises(KnowledgeConflictError, match="changed during retrieval"):
        await runtime.retrieve(RetrievalRequest(topic="runtime"))


@pytest.mark.asyncio
async def test_valid_cache_rehydrates_through_authoritative_d1_r2() -> None:
    runtime, reader, _writer, r2, _vectorize, kv = _runtime()
    obj = _obj()
    row = _row(obj, vectorized=False)
    reader.rows[str(obj.knowledge_id)] = row
    reader.query_result = [row]
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content

    request = RetrievalRequest(topic="runtime")
    first = await runtime.retrieve(request)
    assert len(first.objects) == 1
    reader.query_result = []

    second = await runtime.retrieve(request)
    assert len(second.objects) == 1
    assert second.objects[0].knowledge_id == obj.knowledge_id



@pytest.mark.asyncio
async def test_cached_retrieval_rejects_revision_change_during_rehydrate() -> None:
    runtime, reader, writer, r2, _vectorize, _kv = _runtime()
    obj = _obj()
    row = _row(obj, vectorized=False)
    reader.rows[str(obj.knowledge_id)] = row
    reader.query_result = [row]
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content

    request = RetrievalRequest(topic="runtime")
    first = await runtime.retrieve(request)
    assert len(first.objects) == 1

    writer.read_sequence = [4, 5]
    with pytest.raises(KnowledgeConflictError, match="cached retrieval"):
        await runtime.retrieve(request)
