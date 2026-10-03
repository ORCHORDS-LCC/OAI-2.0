from __future__ import annotations

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    CFRow,
    KnowledgeObject,
    RetrievalRequest,
    sha256_hex,
    vector_id_for,
)
from oai2.knowledge.cloudflare_bindings_runtime import CloudflareVectorizeStore
from oai2.knowledge.cloudflare_runtime import (
    AsyncCloudflareKnowledgeRuntime,
    KnowledgeConflictError,
    KnowledgeIntegrityError,
)
from oai2.knowledge.transport import VectorMatch

# The runtime instance under test embeds with this version. A different value is
# how the embedding-migration transition is exercised below.
_EMBEDDING_VERSION = "embed-v1"


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
        # Failure injection for the partial-dependency matrix (#205
        # REQ-CFOPS-014). Off by default; existing tests are unaffected.
        self.fail_put = False

    async def put_text(self, key: str, value: str) -> None:
        if self.fail_put:
            raise RuntimeError("R2 outage")
        self.values[key] = value
        self.puts.append((key, value))

    async def get_text(self, key: str) -> str | None:
        return self.values.get(key)


class FakeVectorize:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[float], dict[str, object]]] = []
        self.matches: list[VectorMatch] = []
        self.queries: list[int] = []
        # Failure injection for the partial-dependency matrix (#205
        # REQ-CFOPS-014). Off by default; existing tests are unaffected.
        self.fail_upsert = False
        self.fail_query = False

    async def upsert(
        self,
        vector_id: str,
        values: object,
        *,
        metadata: object = None,
    ) -> object:
        assert isinstance(values, (list, tuple))
        assert isinstance(metadata, dict)
        if self.fail_upsert:
            raise RuntimeError("Vectorize outage")
        self.upserts.append(
            (vector_id, [float(v) for v in values], metadata)
        )
        return {"mutationId": "m1"}

    async def query(self, values: object, *, top_k: int = 5) -> list[VectorMatch]:
        self.queries.append(top_k)
        if self.fail_query:
            raise RuntimeError("Vectorize outage")
        return list(self.matches[:top_k])

    def vector_ids(self) -> list[str]:
        return [u[0] for u in self.upserts]


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
    authority: float = 0.9,
    status: Status = Status.EXPERIMENTAL,
    source_uri: str | None = "https://example.test/source",
) -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId(knowledge_id),
        topic=topic,
        content=content,
        content_hash=sha256_hex(content),
        source_uri=source_uri,
        retrieved_at=10.0,
        authority=authority,
        status=status,
    )


def _row(
    obj: KnowledgeObject,
    *,
    vectorized: bool = True,
    embedding_version: str = _EMBEDDING_VERSION,
) -> CFRow:
    """An authoritative D1 row.

    ``vectorized=True`` is the post-fix shape: the row names the
    generation-specific vector id. ``vectorized="legacy"`` reproduces a row
    written before this change, where vectorize_id is the bare knowledge_id.
    """
    if vectorized is True:
        vectorize_id: str | None = vector_id_for(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            embedding_version=embedding_version,
        )
    elif vectorized == "legacy":
        vectorize_id = str(obj.knowledge_id)
    else:
        vectorize_id = None
    return CFRow(
        knowledge_id=obj.knowledge_id,
        topic=obj.topic,
        content_hash=obj.content_hash,
        authority=obj.authority,
        status=obj.status,
        source_uri=obj.source_uri,
        retrieved_at=obj.retrieved_at,
        r2_blob_key=f"oai2-blobs/{obj.content_hash}" if obj.content else None,
        vectorize_id=vectorize_id,
    )


def _match(
    obj: KnowledgeObject,
    score: float,
    *,
    embedding_version: str = _EMBEDDING_VERSION,
    vector_id: str | None = None,
    content_hash: str | None = None,
) -> VectorMatch:
    """A Vectorize match, carrying the attribution the read contract needs."""
    return VectorMatch(
        vector_id=vector_id
        if vector_id is not None
        else vector_id_for(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            embedding_version=embedding_version,
        ),
        score=score,
        knowledge_id=obj.knowledge_id,
        content_hash=content_hash if content_hash is not None else obj.content_hash,
        embedding_version=embedding_version,
    )


def _legacy_match(obj: KnowledgeObject, score: float) -> VectorMatch:
    """A match for a pre-migration vector, whose id is the bare knowledge_id."""
    return VectorMatch(
        vector_id=str(obj.knowledge_id),
        score=score,
        knowledge_id=obj.knowledge_id,
        content_hash=obj.content_hash,
        embedding_version=_EMBEDDING_VERSION,
    )


def _runtime(
    *,
    embedding_version: str = _EMBEDDING_VERSION,
) -> tuple[
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
        embedding_version=embedding_version,
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
    expected_vector = vector_id_for(
        knowledge_id=obj.knowledge_id,
        content_hash=obj.content_hash,
        embedding_version=_EMBEDDING_VERSION,
    )
    # The vector is identified by its generation, NOT by the claim id. This is
    # what makes the D1 write below an atomic switch rather than a mutation of
    # the vector an earlier committed row already points at.
    assert vectorize.upserts[0][0] == expected_vector
    assert vectorize.upserts[0][0] != str(obj.knowledge_id)
    assert vectorize.upserts[0][2]["content_hash"] == obj.content_hash
    row, expected_revision, now = writer.writes[0]
    assert row.knowledge_id == obj.knowledge_id
    assert row.vectorize_id == expected_vector
    assert len(expected_vector.encode()) <= 64, "Vectorize id limit"
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
    vectorize.matches = [_match(obj, 0.99)]
    assert row.r2_blob_key is not None
    r2.values[row.r2_blob_key] = obj.content

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0, 0.0],
    )

    assert [item.knowledge_id for item in result.objects] == [obj.knowledge_id]
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.knowledge_id == obj.knowledge_id
    assert candidate.score == pytest.approx(0.99)
    assert candidate.source_uri == obj.source_uri
    assert candidate.content_hash == obj.content_hash
    # Semantic queries bypass the topic-only KV cache until the query vector
    # is part of the cache key contract.
    assert _kv.puts == []


@pytest.mark.asyncio
async def test_semantic_retrieve_excludes_stale_vector_without_d1_metadata() -> None:
    runtime, _reader, _writer, _r2, vectorize, _kv = _runtime()
    vectorize.matches = [
        VectorMatch(
            vector_id="oai2v1-" + "0" * 56,
            score=0.99,
            knowledge_id=KnowledgeId("ko_missing"),
            content_hash="0" * 64,
            embedding_version=_EMBEDDING_VERSION,
        )
    ]

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )

    assert result.objects == []
    assert result.candidates == []
    assert bool(result) is False


# SUPERSEDED by test_a_match_naming_another_vector_is_excluded_not_raised.
#
# This used to assert that a match naming a vector D1 does not reference RAISES
# an integrity error. That was only correct while vector identity was the
# knowledge_id, which made an id mismatch a genuine contradiction. With
# generation-specific ids an id mismatch is the ordinary state of an orphaned
# vector, so raising here would report normal retry churn as corruption. The
# guarantee is preserved and is now stronger in the way that matters: such a
# match is never returned and never contributes a score.


@pytest.mark.asyncio
async def test_semantic_retrieve_overfetches_then_filters_authoritative_metadata() -> None:
    runtime, reader, _writer, r2, vectorize, _kv = _runtime()
    low = _obj(
        knowledge_id="ko_low",
        content="low body",
        authority=0.1,
    )
    retired = _obj(
        knowledge_id="ko_retired",
        content="retired body",
        status=Status.PROPOSED,
    )
    eligible = _obj(
        knowledge_id="ko_eligible",
        content="eligible body",
        authority=0.8,
        source_uri="https://example.test/eligible",
    )
    for obj in (low, retired, eligible):
        row = _row(obj)
        reader.rows[str(obj.knowledge_id)] = row
        assert row.r2_blob_key is not None
        r2.values[row.r2_blob_key] = obj.content

    vectorize.matches = [
        VectorMatch(
            vector_id="oai2v1-" + "0" * 56,
            score=0.99,
            knowledge_id=KnowledgeId("ko_missing"),
            content_hash="0" * 64,
            embedding_version=_EMBEDDING_VERSION,
        ),
        _match(low, 0.95),
        _match(retired, 0.90),
        _match(eligible, 0.80),
    ]

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic", limit=1, min_authority=0.5),
        query_vector=[0.5, 0.5],
    )

    assert vectorize.queries == [8]
    assert [obj.knowledge_id for obj in result.objects] == [eligible.knowledge_id]
    assert result.candidates[0].score == pytest.approx(0.80)
    assert result.candidates[0].source_uri == "https://example.test/eligible"
    assert result.candidates[0].content_hash == eligible.content_hash


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


def test_async_cloudflare_runtime_exports_from_knowledge_package() -> None:
    from oai2.knowledge import (
        AsyncCloudflareKnowledgeRuntime as ExportedRuntime,
    )
    from oai2.knowledge import KnowledgeConflictError as ExportedConflict
    from oai2.knowledge import KnowledgeIntegrityError as ExportedIntegrity
    from oai2.knowledge import KnowledgeRuntimeError as ExportedRuntimeError
    from oai2.knowledge import RetrievalCandidate as ExportedCandidate
    from oai2.knowledge.abstraction import RetrievalCandidate
    from oai2.knowledge.cloudflare_runtime import (
        AsyncCloudflareKnowledgeRuntime,
        KnowledgeConflictError,
        KnowledgeIntegrityError,
        KnowledgeRuntimeError,
    )

    assert ExportedRuntime is AsyncCloudflareKnowledgeRuntime
    assert ExportedConflict is KnowledgeConflictError
    assert ExportedIntegrity is KnowledgeIntegrityError
    assert ExportedRuntimeError is KnowledgeRuntimeError
    assert ExportedCandidate is RetrievalCandidate


# ---------------------------------------------------------------------------
# #205 WI-CFOPS-014 / 015 — partial dependency failure and idempotency.
#
# The async runtime writes R2 -> Vectorize -> D1 with no compensation step.
# These tests pin the state left behind at each failure point, so the
# consequences are documented behaviour rather than an accident, and so a
# future reordering cannot silently change which artifacts a failure orphans.
# They extend the existing fakes; no existing test is duplicated.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_r2_failure_writes_nothing_anywhere() -> None:
    """A is first in the order, so nothing should exist afterwards."""
    runtime, _reader, writer, r2, vectorize, _kv = _runtime()
    r2.fail_put = True

    with pytest.raises(RuntimeError, match="R2 outage"):
        await runtime.put(_obj(), vector=[1.0, 0.0], now=20.0)

    assert r2.puts == [], "a failed R2 write must not be recorded as a put"
    assert vectorize.upserts == [], "Vectorize must not be reached after R2 fails"
    assert writer.writes == [], "authoritative D1 must not be written"
    assert writer.revision == 4, "a failed put must not advance the revision"


@pytest.mark.asyncio
async def test_partial_vectorize_failure_leaves_an_r2_body_with_no_d1_row() -> None:
    """B. The orphaned body is a non-authoritative artifact, NOT a false success.

    The R2 key is content-addressed and may already be shared by a retained
    row, so it is deliberately kept. `gc.py` owns reconciliation of
    unreferenced bodies; this test records the orphan rather than deleting it,
    because deleting a possibly-shared body here would be unsafe.
    """
    runtime, reader, writer, r2, vectorize, _kv = _runtime()
    vectorize.fail_upsert = True
    obj = _obj()

    with pytest.raises(RuntimeError, match="Vectorize outage"):
        await runtime.put(obj, vector=[1.0, 0.0], now=20.0)

    assert r2.values == {f"oai2-blobs/{obj.content_hash}": obj.content}
    assert vectorize.upserts == []
    assert writer.writes == [], "authoritative D1 must not be written"
    assert writer.revision == 4
    assert await runtime.get(obj.knowledge_id) is None, (
        "a partially-written object must not be readable as authoritative"
    )
    assert reader.rows == {}


@pytest.mark.asyncio
async def test_partial_d1_failure_leaves_r2_and_vectorize_without_an_authoritative_row() -> None:
    """C. Neither non-authoritative store is authoritative; D1 is."""
    runtime, reader, writer, r2, vectorize, _kv = _runtime()
    writer.deny_write = True
    obj = _obj()

    with pytest.raises(KnowledgeConflictError):
        await runtime.put(obj, vector=[1.0, 0.0], now=20.0)

    assert f"oai2-blobs/{obj.content_hash}" in r2.values
    assert vectorize.vector_ids() == [
        vector_id_for(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            embedding_version=_EMBEDDING_VERSION,
        )
    ], "the denied generation is a NEW vector, not an overwrite of the committed one"
    assert writer.revision == 4, "a denied write must not advance the revision"
    assert await runtime.get(obj.knowledge_id) is None
    assert reader.rows == {}


@pytest.mark.asyncio
async def test_d1_write_exception_is_not_reported_as_success() -> None:
    """C'. A thrown D1 failure must not be mistaken for a committed write."""
    runtime, _reader, writer, r2, _vectorize, _kv = _runtime()
    obj = _obj()

    async def boom(*_args, **_kwargs):
        raise RuntimeError("D1 outage")

    writer.write_metadata = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="D1 outage"):
        await runtime.put(obj, vector=[1.0, 0.0], now=20.0)

    assert f"oai2-blobs/{obj.content_hash}" in r2.values
    assert writer.revision == 4


# ---------------------------------------------------------------------------
# REQ-CFOPS-015 — repeated PUT must not multiply effects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_put_targets_the_same_keys_or_ids() -> None:
    """FAKE ORCHESTRATION EVIDENCE ONLY.

    `FakeWriter` is a write RECORDER, not a keyed table. What this proves is
    that repeated PUTs target the same identity — the same R2 content key and
    the same Vectorize id. It does NOT prove the authoritative D1 table still
    holds exactly one row after repeated upserts; that is the D1 contract and
    needs a real keyed adapter/SQLite-backed test, which this issue does not
    have. Do not read this as persistence cardinality.

    Revision behaviour is asserted here and is NOT idempotent: every
    successful PUT advances it.
    """
    runtime, reader, writer, r2, vectorize, _kv = _runtime()
    obj = _obj()

    first = await runtime.put(obj, vector=[1.0, 0.0], now=20.0)
    second = await runtime.put(obj, vector=[1.0, 0.0], now=21.0)

    # Content-addressed R2 key: the same object, overwritten not duplicated.
    assert r2.puts[0][0] == r2.puts[1][0] == f"oai2-blobs/{obj.content_hash}"
    assert set(r2.values) == {f"oai2-blobs/{obj.content_hash}"}
    # Vectorize upserts by generation: identical inputs derive an identical id,
    # so a retried put re-upserts the same vector instead of accumulating one.
    assert {u[0] for u in vectorize.upserts} == {
        vector_id_for(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            embedding_version=_EMBEDDING_VERSION,
        )
    }
    # Same identity targeted on every write. Cardinality of the real
    # authoritative table is NOT established by a recorder.
    assert len({row.knowledge_id for row, _exp, _now in writer.writes}) == 1
    reader.rows[str(obj.knowledge_id)] = writer.writes[-1][0]
    assert len(reader.rows) == 1
    # NOT idempotent for revision: a successful put always advances it, so a
    # retry after an uncertain outcome costs a revision and invalidates every
    # KV cache entry. Idempotent for DATA, not for revision/cost.
    assert first == 5
    assert second == 6


@pytest.mark.asyncio
async def test_retry_after_partial_failure_creates_no_duplicate_authoritative_row() -> None:
    """E. A retry after a Vectorize failure must not double-write anything."""
    runtime, reader, writer, r2, vectorize, _kv = _runtime()
    obj = _obj()

    vectorize.fail_upsert = True
    with pytest.raises(RuntimeError, match="Vectorize outage"):
        await runtime.put(obj, vector=[1.0, 0.0], now=20.0)
    assert writer.writes == []

    vectorize.fail_upsert = False
    revision = await runtime.put(obj, vector=[1.0, 0.0], now=21.0)

    assert revision == 5, "the successful retry advances the revision exactly once"
    # Only the successful attempt reached the writer. This is orchestration
    # evidence; it is not a statement about the authoritative table.
    assert len(writer.writes) == 1
    reader.rows[str(obj.knowledge_id)] = writer.writes[0][0]
    assert len(reader.rows) == 1
    assert vectorize.vector_ids() == [
        vector_id_for(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            embedding_version=_EMBEDDING_VERSION,
        )
    ], "a retry of the same generation targets the same vector"
    assert set(r2.values) == {f"oai2-blobs/{obj.content_hash}"}


@pytest.mark.asyncio
async def test_content_change_under_stable_id_leaves_the_previous_body_orphaned() -> None:
    """Documented consequence of a content-addressed key under a stable id.

    The previous body is NOT deleted: its key may be shared by a retained row.
    `gc.py` classifies it UNREFERENCED_CANDIDATE and `sweep.py` owns deletion.
    """
    runtime, _reader, _writer, r2, _vectorize, _kv = _runtime()
    original = _obj(content="first body")
    updated = _obj(content="second body")
    assert original.knowledge_id == updated.knowledge_id
    assert original.content_hash != updated.content_hash

    await runtime.put(original, vector=[1.0, 0.0], now=20.0)
    await runtime.put(updated, vector=[1.0, 0.0], now=21.0)

    assert set(r2.values) == {
        f"oai2-blobs/{original.content_hash}",
        f"oai2-blobs/{updated.content_hash}",
    }, "the superseded body is retained for reference-safe GC, not deleted here"


@pytest.mark.asyncio
async def test_shared_content_across_rows_is_never_removed_by_a_failed_put() -> None:
    """F. A body shared by another row must survive a partial failure."""
    runtime, reader, _writer, r2, vectorize, _kv = _runtime()
    first = _obj(knowledge_id="ko_shared_a", content="shared body")
    second = _obj(knowledge_id="ko_shared_b", content="shared body")
    assert first.content_hash == second.content_hash

    await runtime.put(first, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(first.knowledge_id)] = _row(first)

    vectorize.fail_upsert = True
    with pytest.raises(RuntimeError, match="Vectorize outage"):
        await runtime.put(second, vector=[1.0, 0.0], now=21.0)

    assert f"oai2-blobs/{first.content_hash}" in r2.values, (
        "a shared content-addressed body must never be removed as rollback"
    )
    assert await runtime.get(first.knowledge_id) is not None

# ---------------------------------------------------------------------------
# #19 — D1/Vectorize split-brain, now an implemented invariant.
#
# THE DEFECT. The put order is R2 -> Vectorize -> D1 and the vector identity was
# the stable knowledge_id. A put that succeeded on Vectorize but was then denied
# by D1 therefore OVERWROTE the vector the last committed D1 row still pointed
# at, while D1 and R2 kept representing the previous content. Semantic retrieval
# then scored against the NEW embedding and returned the OLD object, because the
# only generation check was `row.vectorize_id != vector_id` and both were the
# same stable id. A caller could not tell which content the score belonged to.
#
# THE FIX. Vector identity is now the generation: vector_id_for(knowledge_id,
# content_hash, embedding_version). A new generation is a NEW vector id, so the
# vector a committed row references is never mutated by a put, and the D1 write
# is the atomic switch. A failed update therefore has nothing to roll back.
#
# These replaced an xfail(strict=True) and a characterisation test of the broken
# behaviour. Both are gone rather than relaxed: the invariant is now asserted
# directly, and the mixed-generation answer it used to produce is asserted to be
# impossible.
# ---------------------------------------------------------------------------


def _vid(obj: KnowledgeObject, *, version: str = _EMBEDDING_VERSION) -> str:
    return vector_id_for(
        knowledge_id=obj.knowledge_id,
        content_hash=obj.content_hash,
        embedding_version=version,
    )


@pytest.mark.asyncio
async def test_a_denied_update_never_overwrites_the_committed_vector() -> None:
    """The core invariant: a failed update cannot corrupt the live one.

    This is the test that was xfail. It now passes, and it passes for a
    structural reason rather than a defensive one: the denied generation was
    written under a DIFFERENT vector id, so there was never an overwrite to
    prevent.
    """
    runtime, reader, writer, _r2, vectorize, _kv = _runtime()

    old = _obj(knowledge_id="ko_split", content="OLD committed body")
    new = _obj(knowledge_id="ko_split", content="NEW uncommitted body")
    assert old.content_hash != new.content_hash

    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    committed = writer.writes[-1][0]
    reader.rows[str(old.knowledge_id)] = committed
    assert committed.vectorize_id == _vid(old)

    writer.deny_write = True
    with pytest.raises(KnowledgeConflictError):
        await runtime.put(new, vector=[0.0, 1.0], now=21.0)

    # Two distinct vectors now exist. The committed row still names the OLD
    # one, and the OLD vector was never rewritten.
    assert vectorize.vector_ids() == [_vid(old), _vid(new)]
    assert vectorize.upserts[0][1] == [1.0, 0.0], "the committed vector was overwritten"
    assert vectorize.upserts[1][1] == [0.0, 1.0], "the denied generation got its own vector"
    assert reader.rows[str(old.knowledge_id)].vectorize_id == _vid(old)
    assert reader.rows[str(old.knowledge_id)].content_hash == old.content_hash

    # Semantic retrieval resolves D1 through metadata knowledge_id and requires
    # the row to reference the matched vector. The denied generation's vector
    # matches nobody, so it can neither be returned nor lend a score.
    vectorize.matches = [_match(new, 0.97)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 1.0],
    )
    assert result.candidates == [], "an unreferenced vector produced a candidate"
    assert result.objects == []

    # The committed generation is still fully readable, with its own score.
    vectorize.matches = [_match(old, 0.91)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0, 0.0],
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].content_hash == old.content_hash
    assert result.candidates[0].score == 0.91


@pytest.mark.asyncio
async def test_a_successful_update_switches_d1_atomically() -> None:
    """The other half: a successful update DOES move authority, in one row."""
    runtime, reader, writer, _r2, vectorize, _kv = _runtime()

    old = _obj(knowledge_id="ko_switch", content="OLD committed body")
    new = _obj(knowledge_id="ko_switch", content="NEW committed body")

    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]
    await runtime.put(new, vector=[0.0, 1.0], now=21.0)

    assert vectorize.vector_ids() == [_vid(old), _vid(new)]
    # D1 is written once, and that single write is the switch.
    assert len(writer.writes) == 2
    assert writer.writes[-1][0].vectorize_id == _vid(new)
    reader.rows[str(new.knowledge_id)] = writer.writes[-1][0]

    # The new generation is now the only one that can contribute a score.
    vectorize.matches = [_match(old, 0.99), _match(new, 0.88)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 1.0],
    )
    assert [c.content_hash for c in result.candidates] == [new.content_hash]
    assert result.candidates[0].score == 0.88
    # And the superseded vector is unreachable, not merely lower-ranked.
    vectorize.matches = [_match(old, 0.99)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0, 0.0],
    )
    assert result.candidates == []


@pytest.mark.asyncio
async def test_repeated_failed_updates_never_accumulate_authoritative_state() -> None:
    """Many denied updates in a row: still exactly one authoritative vector."""
    runtime, reader, writer, _r2, vectorize, _kv = _runtime()

    old = _obj(knowledge_id="ko_repeat", content="generation 0")
    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]

    writer.deny_write = True
    generations = []
    for i in range(1, 6):
        obj = _obj(knowledge_id="ko_repeat", content=f"generation {i}")
        generations.append(obj)
        with pytest.raises(KnowledgeConflictError):
            await runtime.put(obj, vector=[0.0, float(i)], now=20.0 + i)

    # Five orphans were created and none of them is referenced.
    assert vectorize.vector_ids() == [_vid(old), *[_vid(g) for g in generations]]
    assert reader.rows[str(old.knowledge_id)].vectorize_id == _vid(old)

    # Every one of them is invisible to a semantic read.
    vectorize.matches = [_match(g, 0.99) for g in generations]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 1.0],
    )
    assert result.candidates == []
    assert result.objects == []

    # Retrying one of them successfully reuses the SAME vector id rather than
    # creating a second record for the same generation.
    writer.deny_write = False
    await runtime.put(generations[-1], vector=[0.0, 5.0], now=30.0)
    assert vectorize.vector_ids().count(_vid(generations[-1])) == 2, (
        "a retry must re-upsert the same vector, not create another"
    )
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]
    vectorize.matches = [_match(generations[-1], 0.77)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 5.0],
    )
    assert [c.content_hash for c in result.candidates] == [generations[-1].content_hash]


@pytest.mark.asyncio
async def test_a_vector_whose_content_hash_disagrees_with_d1_is_corruption() -> None:
    """Rule 4: same vector id, two contents. That is not an orphan, it is a bug.

    An id mismatch is an expected steady state while orphans exist, so it is an
    exclusion. A content mismatch on a MATCHED id means one vector asserts two
    different contents, which no legitimate state produces, and it raises.
    """
    runtime, reader, _writer, _r2, vectorize, _kv = _runtime()

    obj = _obj(knowledge_id="ko_corrupt")
    row = _row(obj)
    reader.rows[str(obj.knowledge_id)] = row
    # The match names the vector D1 references, but carries a different body.
    vectorize.matches = [_match(obj, 0.99, content_hash="f" * 64)]

    with pytest.raises(KnowledgeIntegrityError, match="carries content_hash"):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[1.0],
        )


@pytest.mark.asyncio
async def test_a_match_naming_another_vector_is_excluded_not_raised() -> None:
    """Rule 3, and the corrected form of the old 'id mismatch raises' test.

    The previous check raised whenever a match named a vector D1 did not
    reference. Under generation-specific ids that is the NORMAL case: a failed
    put leaves orphans behind, and eventually-consistent Vectorize can return
    one. Raising would turn ordinary retry churn into an integrity incident. The
    invariant is that such a match is never returned and never lends a score.
    """
    runtime, reader, _writer, _r2, vectorize, _kv = _runtime()
    from dataclasses import replace

    obj = _obj(knowledge_id="ko_idcheck")
    row = _row(obj)
    reader.rows[str(obj.knowledge_id)] = replace(row, vectorize_id="oai2v1-" + "1" * 56)
    vectorize.matches = [_match(obj, 0.99)]     # a vector D1 does not reference

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )
    assert result.candidates == [], "a score was attached to a row that never claimed it"
    assert result.objects == []


@pytest.mark.asyncio
async def test_a_row_without_a_vector_never_borrows_one() -> None:
    """A row with vectorize_id NULL has no semantic index entry at all."""
    runtime, reader, _writer, _r2, vectorize, _kv = _runtime()

    obj = _obj(knowledge_id="ko_novec")
    reader.rows[str(obj.knowledge_id)] = _row(obj, vectorized=False)
    vectorize.matches = [_match(obj, 0.99)]

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )
    assert result.candidates == []


@pytest.mark.asyncio
async def test_legacy_rows_stay_readable_and_migrate_on_the_next_update() -> None:
    """Rows written before this change store vectorize_id == knowledge_id.

    They are not rewritten here: that would mean rewriting history, and the
    schema does not require it. A legacy row stays readable through exactly the
    same integrity rules, because the id match and the content_hash agreement are
    what decide admission, not the id's shape. Its next successful update writes
    a generation-specific id and switches D1 atomically, which is the whole
    migration.
    """
    runtime, reader, writer, _r2, vectorize, _kv = _runtime()

    legacy = _obj(knowledge_id="ko_legacy", content="legacy body")
    row = _row(legacy, vectorized="legacy")
    assert row.vectorize_id == str(legacy.knowledge_id)
    reader.rows[str(legacy.knowledge_id)] = row
    _r2.values[row.r2_blob_key] = legacy.content

    # Still retrievable, and still scored, through the same rules.
    vectorize.matches = [_legacy_match(legacy, 0.75)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].content_hash == legacy.content_hash
    assert result.candidates[0].score == 0.75

    # A legacy row is still held to the corruption rule.
    vectorize.matches = [
        _legacy_match(legacy, 0.75).__class__(
            vector_id=str(legacy.knowledge_id),
            score=0.75,
            knowledge_id=legacy.knowledge_id,
            content_hash="f" * 64,
            embedding_version=_EMBEDDING_VERSION,
        )
    ]
    with pytest.raises(KnowledgeIntegrityError, match="carries content_hash"):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[1.0],
        )

    # The migration itself: one successful update, one atomic switch.
    updated = _obj(knowledge_id="ko_legacy", content="migrated body")
    await runtime.put(updated, vector=[0.0, 1.0], now=40.0)
    assert writer.writes[-1][0].vectorize_id == _vid(updated)
    assert _vid(updated) != str(updated.knowledge_id)
    reader.rows[str(updated.knowledge_id)] = writer.writes[-1][0]

    vectorize.matches = [_match(updated, 0.81), _legacy_match(legacy, 0.75)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 1.0],
    )
    assert [c.content_hash for c in result.candidates] == [updated.content_hash]
    assert result.candidates[0].score == 0.81


@pytest.mark.asyncio
async def test_a_match_from_another_embedding_version_is_excluded() -> None:
    """Section 9: the read contract carries the embedding version.

    A vector embedded by a different model has a score that is not comparable to
    this query's, so it is excluded rather than scored. This is the documented
    migration transition and is why the version is part of the match contract.
    """
    runtime, reader, _writer, _r2, vectorize, _kv = _runtime()

    obj = _obj(knowledge_id="ko_embedver")
    row = _row(obj, embedding_version="embed-v0")
    reader.rows[str(obj.knowledge_id)] = row
    _r2.values[row.r2_blob_key] = obj.content

    # The row legitimately references a v0 vector; the runtime is on v1.
    vectorize.matches = [_match(obj, 0.99, embedding_version="embed-v0")]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )
    assert result.candidates == [], "a cross-embedding-space score was accepted"

    # A v1 runtime pointed at a v1 vector is admitted as normal, so the
    # exclusion is about the version and not about the id shape.
    v1_row = _row(obj, embedding_version=_EMBEDDING_VERSION)
    reader.rows[str(obj.knowledge_id)] = v1_row
    vectorize.matches = [_match(obj, 0.99, embedding_version=_EMBEDDING_VERSION)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )
    assert [c.score for c in result.candidates] == [0.99]


@pytest.mark.asyncio
async def test_an_embedding_version_change_writes_a_new_vector_id() -> None:
    """Re-embedding a corpus under a new version cannot collide with the old.

    Two runtimes, same knowledge_id and same content, different embedding
    version. Their vector ids must differ, so a re-embed never overwrites the
    vector the current corpus still reads from.
    """
    old_runtime, _r1, old_writer, _r2a, old_vec, _k1 = _runtime(
        embedding_version="embed-v1"
    )
    new_runtime, _r2, new_writer, _r2b, new_vec, _k2 = _runtime(
        embedding_version="embed-v2"
    )
    obj = _obj(knowledge_id="ko_reembed", content="same body")

    await old_runtime.put(obj, vector=[1.0, 0.0], now=20.0)
    await new_runtime.put(obj, vector=[1.0, 0.0], now=21.0)

    assert old_vec.vector_ids() == [_vid(obj, version="embed-v1")]
    assert new_vec.vector_ids() == [_vid(obj, version="embed-v2")]
    assert _vid(obj, version="embed-v1") != _vid(obj, version="embed-v2")
    assert old_writer.writes[-1][0].vectorize_id != new_writer.writes[-1][0].vectorize_id


@pytest.mark.asyncio
async def test_a_concurrent_writer_with_a_stale_expected_revision_is_refused() -> None:
    """The CAS path: another writer took the revision first. Nothing is corrupted."""
    runtime, reader, writer, r2, vectorize, _kv = _runtime()

    old = _obj(knowledge_id="ko_cas", content="first writer")
    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]

    # Another writer commits first, so the revision this runtime read is stale
    # and the CAS denies the write.
    writer.revision = 9
    writer.deny_write = True

    other = _obj(knowledge_id="ko_cas", content="second writer")
    with pytest.raises(KnowledgeConflictError):
        await runtime.put(other, vector=[0.0, 1.0], now=21.0)

    # The first writer's vector is intact and still authoritative.
    assert vectorize.upserts[0][1] == [1.0, 0.0]
    assert reader.rows[str(old.knowledge_id)].vectorize_id == _vid(old)
    assert reader.rows[str(old.knowledge_id)].content_hash == old.content_hash
    assert r2.values[f"oai2-blobs/{old.content_hash}"] == old.content

    vectorize.matches = [_match(other, 0.99)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 1.0],
    )
    assert result.candidates == []


@pytest.mark.asyncio
async def test_a_vectorize_outage_during_a_semantic_read_is_a_failure_not_an_empty_result() -> None:
    """A broken dependency must never read as 'nothing matched'."""
    runtime, _reader, _writer, _r2, vectorize, _kv = _runtime()
    vectorize.fail_query = True

    with pytest.raises(RuntimeError, match="Vectorize outage"):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[1.0],
        )


# ---------------------------------------------------------------------------
# #19 — the same acceptance, but through the REAL adapter over a binding fake
# that models the documented Vectorize response shape.
#
# The tests above inject a store-level fake, so they prove the runtime's
# integrity rules but not that the adapter can actually obtain the attribution
# those rules depend on. A nearest-neighbour query does not return metadata
# unless it asks for it (documented default returnMetadata:"none"), so the
# whole stack has to work when the attribution arrives via getByIds instead.
# ---------------------------------------------------------------------------


class FakeVectorizeBinding:
    """A binding-level fake that models Cloudflare's documented behaviour."""

    def __init__(self) -> None:
        self.vectors: dict[str, dict[str, object]] = {}
        self.upserts: list[list[dict[str, object]]] = []
        self.queries: list[tuple[list[float], dict[str, object] | None]] = []
        self.get_by_ids_calls: list[list[str]] = []
        # Scripted ranking, as (vector_id, score) in rank order.
        self.ranking: list[tuple[str, float]] = []

    async def upsert(self, vectors: object) -> object:
        assert isinstance(vectors, list)
        for v in vectors:
            assert isinstance(v, dict)
            vid = v["id"]
            assert isinstance(vid, str)
            self.vectors[vid] = {
                "values": list(v["values"]),  # type: ignore[arg-type]
                "metadata": dict(v.get("metadata") or {}),  # type: ignore[arg-type]
            }
        self.upserts.append(vectors)
        return {"mutationId": "m1"}

    async def query(self, vector: object, options: object = None) -> object:
        assert isinstance(vector, list)
        assert options is None or isinstance(options, dict)
        self.queries.append((vector, options))
        top_k = (options or {}).get("topK", 5)
        assert isinstance(top_k, int)
        # Documented default: id and score only. No metadata is attached.
        return {
            "matches": [
                {"id": vid, "score": score} for vid, score in self.ranking[:top_k]
            ]
        }

    async def getByIds(self, ids: object) -> object:
        assert isinstance(ids, list)
        self.get_by_ids_calls.append(list(ids))
        out = []
        for vid in ids:
            rec = self.vectors.get(vid)
            if rec is not None:
                out.append(
                    {
                        "id": vid,
                        "values": rec["values"],
                        "metadata": rec["metadata"],
                    }
                )
        return out


def _runtime_through_binding() -> tuple[
    AsyncCloudflareKnowledgeRuntime,
    FakeReader,
    FakeWriter,
    FakeR2,
    FakeVectorizeBinding,
    FakeKv,
]:
    """Runtime wired through the REAL CloudflareVectorizeStore, not a fake store."""
    reader = FakeReader()
    writer = FakeWriter(revision=4)
    r2 = FakeR2()
    binding = FakeVectorizeBinding()
    kv = FakeKv()
    runtime = AsyncCloudflareKnowledgeRuntime(
        reader=reader,  # type: ignore[arg-type]
        writer=writer,  # type: ignore[arg-type]
        r2=r2,  # type: ignore[arg-type]
        vectorize=CloudflareVectorizeStore(binding),  # type: ignore[arg-type]
        kv=kv,  # type: ignore[arg-type]
        embedding_version=_EMBEDDING_VERSION,
        embedding_digest="embed-digest-v1",
    )
    return runtime, reader, writer, r2, binding, kv


@pytest.mark.asyncio
async def test_real_binding_path_denied_update_never_serves_a_mixed_generation() -> None:
    """DENIED UPDATE, through query -> getByIds -> VectorMatch -> D1 -> R2."""
    runtime, reader, writer, _r2, binding, _kv = _runtime_through_binding()

    old = _obj(knowledge_id="ko_real", content="OLD committed body")
    new = _obj(knowledge_id="ko_real", content="NEW uncommitted body")
    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]

    writer.deny_write = True
    with pytest.raises(KnowledgeConflictError):
        await runtime.put(new, vector=[0.0, 1.0], now=21.0)

    old_vid, new_vid = _vid(old), _vid(new)
    assert set(binding.vectors) == {old_vid, new_vid}

    # The raw query ranks ONLY the orphan. Its score is computed from the
    # denied generation, so it must not attach to the old object.
    binding.ranking = [(new_vid, 0.97)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"), query_vector=[0.0, 1.0],
    )
    assert result.candidates == [], "NEW score attached to OLD object"
    assert result.objects == []

    # If the OLD vector also ranks, its own score and object come back together.
    binding.ranking = [(new_vid, 0.97), (old_vid, 0.88)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"), query_vector=[1.0, 0.0],
    )
    assert [c.content_hash for c in result.candidates] == [old.content_hash]
    assert result.candidates[0].score == 0.88

    # The whole path really went through the binding.
    assert binding.queries, "no ranking query was issued"
    assert binding.get_by_ids_calls, "no attribution call was issued"
    assert all("returnMetadata" not in (o or {}) for _v, o in binding.queries)


@pytest.mark.asyncio
async def test_real_binding_path_successful_update_returns_new_content_and_score() -> None:
    """SUCCESSFUL UPDATE: new vector, D1 points at it, both agree."""
    runtime, reader, writer, _r2, binding, _kv = _runtime_through_binding()

    old = _obj(knowledge_id="ko_real2", content="OLD committed body")
    new = _obj(knowledge_id="ko_real2", content="NEW committed body")
    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]
    await runtime.put(new, vector=[0.0, 1.0], now=21.0)
    reader.rows[str(new.knowledge_id)] = writer.writes[-1][0]

    binding.ranking = [(_vid(new), 0.93)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"), query_vector=[0.0, 1.0],
    )
    assert [c.content_hash for c in result.candidates] == [new.content_hash]
    assert result.candidates[0].score == 0.93
    assert [o.content for o in result.objects] == [new.content]

    # The superseded generation cannot contribute.
    binding.ranking = [(_vid(old), 0.99)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"), query_vector=[1.0, 0.0],
    )
    assert result.candidates == []


@pytest.mark.asyncio
async def test_real_binding_path_unattributable_match_is_a_dependency_failure() -> None:
    """MALFORMED ATTRIBUTION: a failure, not an empty semantic result."""
    runtime, reader, writer, _r2, binding, _kv = _runtime_through_binding()

    obj = _obj(knowledge_id="ko_real3", content="body")
    await runtime.put(obj, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(obj.knowledge_id)] = writer.writes[-1][0]
    binding.ranking = [(_vid(obj), 0.9)]

    # The binding returns an id and a score, but the stored record has no
    # usable attribution.
    binding.vectors[_vid(obj)]["metadata"] = {"knowledge_id": "ko_real3"}
    with pytest.raises(RuntimeError):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"), query_vector=[1.0],
        )

    # And a record the index no longer holds is equally a failure.
    binding.vectors.pop(_vid(obj))
    with pytest.raises(RuntimeError):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"), query_vector=[1.0],
        )


@pytest.mark.asyncio
async def test_real_binding_path_honours_query_rank_over_attribution_order() -> None:
    """Ranking is the query's; getByIds order must not be able to reorder it."""
    runtime, reader, writer, _r2, binding, _kv = _runtime_through_binding()

    a = _obj(knowledge_id="ko_rank_a", content="alpha", authority=0.9)
    b = _obj(knowledge_id="ko_rank_b", content="bravo", authority=0.9)
    for obj in (a, b):
        await runtime.put(obj, vector=[1.0], now=20.0)
        reader.rows[str(obj.knowledge_id)] = writer.writes[-1][0]

    binding.ranking = [(_vid(a), 0.99), (_vid(b), 0.80)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"), query_vector=[1.0],
    )
    assert [c.knowledge_id for c in result.candidates] == [a.knowledge_id, b.knowledge_id]
    assert [c.score for c in result.candidates] == [0.99, 0.80]


# ---------------------------------------------------------------------------
# #19 / #73 — MIXED EMBEDDING VERSIONS: integrity is protected, RECALL IS NOT.
#
# Excluding a VectorMatch whose embedding_version differs from the query
# runtime's is correct and necessary: its score was computed in a different
# vector space and is not comparable. But the exclusion happens AFTER the raw
# top-K is fixed, so every old-version vector in that top-K consumes a slot
# before being discarded. During a partial re-embed that is a recall loss with
# no correctness cost and no visible signal.
#
# This is a MEASUREMENT, not a fix. #73 owns the migration. Nothing here claims
# a recall figure, and a mock cannot establish one — the numbers below are
# exact for the scenario constructed, and say nothing about a real index.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_version_recall_loss_is_measured_not_hidden() -> None:
    """Quantify the slot consumption. Integrity holds; recall can be truncated."""
    # A runtime already migrated to embed-v2, over an index still half-populated
    # with embed-v1 vectors.
    runtime, reader, writer, _r2, binding, _kv = _runtime_through_binding()
    runtime = AsyncCloudflareKnowledgeRuntime(
        reader=reader,  # type: ignore[arg-type]
        writer=writer,  # type: ignore[arg-type]
        r2=_r2,  # type: ignore[arg-type]
        vectorize=CloudflareVectorizeStore(binding),  # type: ignore[arg-type]
        kv=_kv,  # type: ignore[arg-type]
        embedding_version="embed-v2",
        embedding_digest="embed-digest-v2",
    )

    # A realistic partial migration: the corpus is large and still mostly
    # legacy, with only a few objects already re-embedded. limit=8 over-fetches
    # to top_k=32, so the legacy population must exceed that to fill the
    # ceiling — which is exactly what happens in a real half-migrated index.
    limit = 8
    top_k = min(max(limit * 4, 8), 100)
    legacy_count = top_k + 8          # comfortably more than the ceiling
    v1_objs = [
        _obj(knowledge_id=f"ko_v1_{i}", content=f"legacy body {i}")
        for i in range(legacy_count)
    ]
    v2_objs = [
        _obj(knowledge_id=f"ko_v2_{i}", content=f"current body {i}")
        for i in range(2)
    ]
    # Index them the way a partial migration actually leaves things: every
    # vector present, each carrying the version it was embedded with.
    for obj in v1_objs:
        await runtime.put(obj, vector=[1.0, 0.0], now=20.0)
        reader.rows[str(obj.knowledge_id)] = writer.writes[-1][0]
    for obj in v2_objs:
        await runtime.put(obj, vector=[0.0, 1.0], now=21.0)
        reader.rows[str(obj.knowledge_id)] = writer.writes[-1][0]

    # Re-stamp: the v1 objects' rows and vectors are the pre-migration ones.
    for obj in v1_objs:
        legacy_row = _row(obj, embedding_version="embed-v1")
        reader.rows[str(obj.knowledge_id)] = legacy_row
        binding.vectors[_vid(obj, version="embed-v1")] = {
            "values": [1.0, 0.0],
            "metadata": {
                "knowledge_id": str(obj.knowledge_id),
                "content_hash": obj.content_hash,
                "embedding_version": "embed-v1",
                "status": obj.status.value,
                "authority": obj.authority,
            },
        }
    for obj in v2_objs:
        binding.vectors[_vid(obj, version="embed-v2")] = {
            "values": [0.0, 1.0],
            "metadata": {
                "knowledge_id": str(obj.knowledge_id),
                "content_hash": obj.content_hash,
                "embedding_version": "embed-v2",
                "status": obj.status.value,
                "authority": obj.authority,
            },
        }

    # The raw ranking: the legacy population outranks every current-version
    # object, and top_k is what the runtime actually requests.
    binding.ranking = [(_vid(o, version="embed-v1"), 1.0 - i * 0.001) for i, o in
                       enumerate(v1_objs)]
    binding.ranking += [(_vid(o, version="embed-v2"), 0.1 + i * 0.01) for i, o in
                        enumerate(v2_objs)]

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic", limit=limit), query_vector=[0.0, 1.0],
    )

    raw_returned = min(len(binding.ranking), top_k)
    excluded_old_version = raw_returned - len(result.candidates)
    current_version_returned = len(result.candidates)

    # The integrity rule did exactly what it must: nothing from the other
    # embedding space leaked into the answer.
    assert all(
        c.knowledge_id in {o.knowledge_id for o in v2_objs}
        for c in result.candidates
    ), "a cross-version score reached the result"

    # But the current-version objects were ranked below the raw top-K ceiling
    # and never even got attributed. This is the recall cost, stated as a
    # count so it is visible rather than theoretical.
    assert raw_returned == top_k
    assert excluded_old_version == top_k, (
        f"expected every ranked legacy vector to consume a slot, "
        f"saw {excluded_old_version} excluded of {raw_returned} raw"
    )
    assert current_version_returned == 0, (
        "with the legacy vectors saturating the ceiling, no current-version "
        "object survives — this is the recall failure the version filter causes"
    )
    # The binding was asked for attribution only for what ranked, so the
    # current-version vectors were never even looked up.
    attributed = {vid for call in binding.get_by_ids_calls for vid in call}
    assert not any(_vid(o, version="embed-v2") in attributed for o in v2_objs)
