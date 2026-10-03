from __future__ import annotations

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
        self.matches: list[tuple[str, float]] = []
        self.queries: list[int] = []
        # Failure injection for the partial-dependency matrix (#205
        # REQ-CFOPS-014). Off by default; existing tests are unaffected.
        self.fail_upsert = False

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

    async def query(self, values: object, *, top_k: int = 5) -> list[tuple[str, float]]:
        self.queries.append(top_k)
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
    vectorize.matches = [("missing", 0.99)]

    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[1.0],
    )

    assert result.objects == []
    assert result.candidates == []
    assert bool(result) is False


@pytest.mark.asyncio
async def test_semantic_retrieve_surfaces_vectorize_id_mismatch() -> None:
    runtime, reader, _writer, _r2, vectorize, _kv = _runtime()
    obj = _obj()
    row = _row(obj)
    from dataclasses import replace

    reader.rows[str(obj.knowledge_id)] = replace(row, vectorize_id="different")
    vectorize.matches = [(str(obj.knowledge_id), 0.99)]

    with pytest.raises(KnowledgeIntegrityError, match="disagrees with D1 vectorize_id"):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[1.0],
        )


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
        ("ko_missing", 0.99),
        (str(low.knowledge_id), 0.95),
        (str(retired.knowledge_id), 0.90),
        (str(eligible.knowledge_id), 0.80),
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
    assert [u[0] for u in vectorize.upserts] == [str(obj.knowledge_id)]
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
    # Vectorize upserts by knowledge_id: same id, no second record.
    assert {u[0] for u in vectorize.upserts} == {str(obj.knowledge_id)}
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
    assert [u[0] for u in vectorize.upserts] == [str(obj.knowledge_id)]
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
# #205 section 3 — D1/Vectorize split-brain reproduction.
#
# The put order is R2 -> Vectorize -> D1, and the vector identity is the
# stable knowledge_id. A put that succeeds on Vectorize but is then denied by
# D1 therefore overwrites the vector that the LAST COMMITTED D1 row still
# points at, while D1 and R2 keep representing the previous content.
#
# Semantic retrieval then scores against the NEW embedding and returns the OLD
# object, because the only generation check is `row.vectorize_id != vector_id`
# and both are the same stable id. That is a mixed-generation answer, not an
# orphan: the caller cannot tell which content the score belongs to.
#
# REPRODUCED, NOT FIXED HERE. This is a multi-store architectural invariant
# owned by the knowledge D1/R2/Vectorize integrity work (#19), with the
# version/retirement lifecycle in #73 and R2 GC in #209. Fixing it in this
# issue would duplicate that owner. The test is marked xfail(strict=True) so it
# documents the current behaviour and FAILS once the behaviour is corrected,
# forcing this marker to be removed rather than silently going stale.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "D1/Vectorize split-brain. A put that succeeds on Vectorize but is "
        "denied by D1 overwrites the embedding the last committed D1 row still "
        "points at, because the vector identity is the stable knowledge_id. "
        "Semantic retrieval then scores against the NEW embedding and returns "
        "the OLD object, and the only generation check -- "
        "`row.vectorize_id != vector_id` -- passes because both are the same "
        "stable id. Desired behaviour: refuse to return a candidate whose "
        "embedding generation disagrees with its authoritative content. "
        "Owner: #19 (integrity), #73 (version lifecycle), #209 (R2 GC)."
    ),
)
async def test_split_brain_semantic_read_must_not_mix_generations() -> None:
    """The invariant that SHOULD hold, and currently does not.

    A candidate is only returned when the embedding that produced its score
    belongs to the same committed content generation as the object returned.
    """
    runtime, reader, writer, r2, vectorize, _kv = _runtime()

    old = _obj(knowledge_id="ko_split", content="OLD committed body")
    new = _obj(knowledge_id="ko_split", content="NEW uncommitted body")
    assert old.content_hash != new.content_hash

    # 1. The authoritative object: OLD content, D1 pointing at its body.
    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]
    assert writer.writes[-1][0].content_hash == old.content_hash

    # 2-5. An update to the SAME knowledge_id: R2 and Vectorize succeed, the
    #      authoritative D1 write is denied.
    writer.deny_write = True
    with pytest.raises(KnowledgeConflictError):
        await runtime.put(new, vector=[0.0, 1.0], now=21.0)

    # Precondition: the vector store now holds the NEW embedding, D1 still
    # describes the OLD generation, and the NEW body is orphaned.
    assert [u[1] for u in vectorize.upserts] == [[1.0, 0.0], [0.0, 1.0]]
    assert reader.rows[str(old.knowledge_id)].content_hash == old.content_hash
    assert f"oai2-blobs/{new.content_hash}" in r2.values

    # 6. Semantic retrieval whose match was computed from the NEW embedding.
    vectorize.matches = [(str(old.knowledge_id), 0.97)]

    with pytest.raises(KnowledgeIntegrityError):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[0.0, 1.0],
        )


@pytest.mark.asyncio
async def test_split_brain_is_currently_observable_as_a_mixed_generation_answer() -> None:
    """Characterisation of the defect, so the report is not just an assertion.

    This is what actually happens today, pinned so the xfail above cannot be
    "fixed" by accident or by a test that never exercised the real path. It
    documents a defect and must be replaced when the invariant is implemented.
    """
    runtime, reader, writer, _r2, vectorize, _kv = _runtime()

    old = _obj(knowledge_id="ko_split", content="OLD committed body")
    new = _obj(knowledge_id="ko_split", content="NEW uncommitted body")

    await runtime.put(old, vector=[1.0, 0.0], now=20.0)
    reader.rows[str(old.knowledge_id)] = writer.writes[-1][0]

    writer.deny_write = True
    with pytest.raises(KnowledgeConflictError):
        await runtime.put(new, vector=[0.0, 1.0], now=21.0)

    vectorize.matches = [(str(old.knowledge_id), 0.97)]
    result = await runtime.retrieve(
        RetrievalRequest(topic="semantic"),
        query_vector=[0.0, 1.0],
    )

    # No error. A normal candidate is returned: score from the NEW embedding,
    # object from the OLD generation, and the candidate carries no generation
    # or embedding provenance of its own.
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.content_hash == old.content_hash
    assert candidate.content_hash != new.content_hash
    assert candidate.score == 0.97
    assert not hasattr(candidate, "embedding_version")


@pytest.mark.asyncio
async def test_the_existing_id_check_still_catches_a_genuine_id_disagreement() -> None:
    # Control: the guard that DOES exist is not dead code. It fires when the
    # ids genuinely differ, which is the only case the current design detects.
    runtime, reader, _writer, _r2, vectorize, _kv = _runtime()
    from dataclasses import replace

    obj = _obj(knowledge_id="ko_idcheck")
    row = _row(obj)
    reader.rows[str(obj.knowledge_id)] = replace(row, vectorize_id="stale-vector-id")
    vectorize.matches = [(str(obj.knowledge_id), 0.99)]

    with pytest.raises(KnowledgeIntegrityError, match="disagrees with D1 vectorize_id"):
        await runtime.retrieve(
            RetrievalRequest(topic="semantic"),
            query_vector=[1.0],
        )
