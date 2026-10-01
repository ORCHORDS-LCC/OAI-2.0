"""Tests for the Cloudflare knowledge adapter (mock bindings only)."""

from __future__ import annotations

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    CFPlan,
    CloudflareKnowledgeStore,
    IngestionJob,
    IngestionPipeline,
    MockCloudflareBindings,
    cache_key_for,
    object_to_row,
    r2_blob_key_for,
    sha256_hex,
)


def _make_obj(
    topic: str,
    content: str,
    authority: float = 0.5,
    kid_suffix: str = "",
    source_uri: str | None = None,
) -> object:
    from oai2.knowledge import KnowledgeObject

    return KnowledgeObject(
        knowledge_id=KnowledgeId(f"ko_{sha256_hex(topic + kid_suffix)[:12]}"),
        topic=topic,
        content=content,
        content_hash=sha256_hex(content),
        source_uri=source_uri,
        retrieved_at=0.0,
        authority=authority,
        status=Status.EXPERIMENTAL,
    )


def test_plan_is_public_safe() -> None:
    plan = CFPlan()
    # Public-safe defaults — no real IDs.
    assert "<" in plan.account_id
    assert "<" in plan.r2_bucket
    assert "<" in plan.d1_database_id
    assert "<" in plan.kv_namespace
    # Vectorize name is a stable logical id, not a secret.
    assert plan.vectorize_index == "oai2-knowledge-v1"


def test_row_to_d1_round_trip() -> None:
    obj = _make_obj("a/b", "hello", authority=0.6)
    row = object_to_row(obj)
    assert row.r2_blob_key == r2_blob_key_for(obj.content_hash)
    assert row.vectorize_id == obj.knowledge_id

    b = MockCloudflareBindings()
    b.d1_upsert(row)
    got = b.d1_get(obj.knowledge_id)
    assert got is not None
    assert got.topic == "a/b"
    assert got.authority == 0.6
    assert got.status == Status.EXPERIMENTAL


def test_store_put_get_round_trip() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("agent-loop", "see SYSTEM_ARCHITECTURE.md")
    store.put(obj)

    got = store.get(obj.knowledge_id)
    assert got is not None
    assert got.content == "see SYSTEM_ARCHITECTURE.md"
    assert got.content_hash == obj.content_hash
    # Body lives in R2 under the canonical key.
    assert b.r2_get(r2_blob_key_for(obj.content_hash)) == obj.content.encode("utf-8")


def test_store_retrieve_filters_by_topic_and_authority() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    high = _make_obj("router", "router authority body", authority=0.9, kid_suffix="h")
    low = _make_obj("router", "low authority body", authority=0.1, kid_suffix="l")
    other = _make_obj("planner", "planner body", authority=0.9, kid_suffix="o")
    for o in (high, low, other):
        store.put(o)

    from oai2.knowledge import RetrievalRequest

    res = store.retrieve(
        RetrievalRequest(topic="router", limit=10, min_authority=0.5)
    )
    ids = [o.knowledge_id for o in res.objects]
    assert high.knowledge_id in ids
    assert low.knowledge_id not in ids
    assert other.knowledge_id not in ids


def test_cache_key_is_stable_for_same_request() -> None:
    from oai2.knowledge import RetrievalRequest

    r1 = RetrievalRequest(topic="x", limit=8, min_authority=0.0)
    r2 = RetrievalRequest(topic="x", limit=8, min_authority=0.0)
    assert cache_key_for(r1, "v1", 7) == cache_key_for(r2, "v1", 7)


def test_cache_key_changes_when_status_set_changes() -> None:
    from oai2.knowledge import RetrievalRequest

    r1 = RetrievalRequest(topic="x")
    r2 = RetrievalRequest(
        topic="x",
        include_status=(Status.IMPLEMENTED,),
    )
    assert cache_key_for(r1, "v1", 7) != cache_key_for(r2, "v1", 7)


def test_cache_revision_invalidates_after_write() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("cache-me", "x", kid_suffix="one")
    store.put(obj)

    from oai2.knowledge import RetrievalRequest

    req = RetrievalRequest(topic="cache-me")
    r1 = store.retrieve(req)
    assert len(r1.objects) == 1
    revision_before = b.cache_revision()

    other = _make_obj("cache-me", "now-appears", kid_suffix="two")
    store.put(other)
    assert b.cache_revision() == revision_before + 1

    r2 = store.retrieve(req)
    assert {o.knowledge_id for o in r2.objects} == {
        obj.knowledge_id,
        other.knowledge_id,
    }


def test_ingestion_pipeline_populates_cloudflare_store() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    pipeline = IngestionPipeline(store)
    jobs = [
        IngestionJob(
            source_uri="local://test", topic="t1", content="c1", authority=0.8
        ),
        IngestionJob(
            source_uri="local://test", topic="t2", content="c2", authority=0.7
        ),
    ]
    results = pipeline.ingest(jobs)
    assert len(results) == 2
    assert all(r.value == "ok" for r in results)
    assert len(b.rows) == 2


def test_vectorize_query_returns_top_k() -> None:
    b = MockCloudflareBindings()
    b.vectorize_upsert("a", [1.0, 0.0, 0.0])
    b.vectorize_upsert("b", [0.0, 1.0, 0.0])
    b.vectorize_upsert("c", [0.9, 0.1, 0.0])
    out = b.vectorize_query([1.0, 0.0, 0.0], top_k=2)
    assert [vid for vid, _ in out] == ["a", "c"]


def test_cfrrow_is_frozen() -> None:
    obj = _make_obj("t", "c")
    row = object_to_row(obj)
    import dataclasses

    try:
        dataclasses.replace(row, topic="other")
    except Exception:
        pass
    # The point is just: it is a frozen dataclass; mutation should fail.
    try:
        row.topic = "x"  # type: ignore[misc]
    except (AttributeError, dataclasses.FrozenInstanceError):
        return
    raise AssertionError("CFRow should be frozen")


def test_all_uses_adapter_contract_not_mock_internals() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    a = _make_obj("all-a", "a", kid_suffix="a")
    b_obj = _make_obj("all-b", "b", kid_suffix="b")
    store.put(a)
    store.put(b_obj)
    assert {o.knowledge_id for o in store.all()} == {a.knowledge_id, b_obj.knowledge_id}


def test_cache_key_changes_with_corpus_revision() -> None:
    from oai2.knowledge import RetrievalRequest

    req = RetrievalRequest(topic="x")
    assert cache_key_for(req, "v1", 1) != cache_key_for(req, "v1", 2)


def test_store_put_rejects_mismatched_content_hash() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("integrity", "trusted body")
    obj.content_hash = sha256_hex("different body")
    with pytest.raises(ValueError, match="content_hash"):
        store.put(obj)


def test_store_get_rejects_missing_blob() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("integrity", "trusted body", kid_suffix="missing")
    store.put(obj)
    del b.blobs[r2_blob_key_for(obj.content_hash)]
    with pytest.raises(ValueError, match="missing"):
        store.get(obj.knowledge_id)


def test_store_get_rejects_mismatched_blob() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("integrity", "trusted body", kid_suffix="tampered")
    store.put(obj)
    b.blobs[r2_blob_key_for(obj.content_hash)] = b"tampered body"
    with pytest.raises(ValueError, match="hash"):
        store.get(obj.knowledge_id)


def test_store_put_writes_body_before_index() -> None:
    """put() must write the R2 body before the D1 index so the index never
    references a missing blob (issue #3)."""

    class Recording(MockCloudflareBindings):
        def __init__(self) -> None:
            super().__init__()
            self.writes: list[str] = []

        def r2_put(self, key: str, body: bytes) -> None:
            self.writes.append(("r2_put", key))
            super().r2_put(key, body)

        def d1_upsert(self, row: object) -> None:
            self.writes.append(("d1_upsert", row.knowledge_id))
            super().d1_upsert(row)

    b = Recording()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("partial-write/order", "body-comes-first")
    store.put(obj)

    r2_events = [e for e in b.writes if e[0] == "r2_put"]
    d1_events = [e for e in b.writes if e[0] == "d1_upsert"]
    assert len(r2_events) == 1, b.writes
    assert len(d1_events) == 1, b.writes
    # r2_put must precede d1_upsert in the write log.
    first_r2 = b.writes.index(r2_events[0])
    first_d1 = b.writes.index(d1_events[0])
    assert first_r2 < first_d1, b.writes


def test_store_put_rolls_back_when_r2_fails() -> None:
    """If r2_put fails, the D1 index must not be written (issue #3)."""

    class R2Fails(MockCloudflareBindings):
        def r2_put(self, key: str, body: bytes) -> None:
            raise RuntimeError("simulated R2 outage")

    b = R2Fails()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("partial-write/r2-fails", "will-be-rolled-back")

    with pytest.raises(RuntimeError, match="R2 outage"):
        store.put(obj)

    assert b.d1_get(obj.knowledge_id) is None
    assert b.r2_get(r2_blob_key_for(obj.content_hash)) is None


def test_store_put_preserves_shared_body_when_index_fails() -> None:
    """A failed D1 write must not delete a body retained by another row (#209)."""

    class D1FailsOnDemand(MockCloudflareBindings):
        def __init__(self) -> None:
            super().__init__()
            self.fail_writes = False

        def d1_upsert(self, row: object) -> None:
            if self.fail_writes:
                raise RuntimeError("simulated D1 outage")
            super().d1_upsert(row)

    b = D1FailsOnDemand()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    retained = _make_obj("shared/original", "shared-body", kid_suffix="retained")
    failing = _make_obj("shared/new", "shared-body", kid_suffix="failing")
    blob_key = r2_blob_key_for(retained.content_hash)

    store.put(retained)
    b.fail_writes = True

    with pytest.raises(RuntimeError, match="D1 outage"):
        store.put(failing)

    assert b.d1_get(failing.knowledge_id) is None
    got = store.get(retained.knowledge_id)
    assert got is not None
    assert got.content == "shared-body"
    assert b.r2_get(blob_key) == b"shared-body"


def test_store_preserves_source_uri() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("provenance", "source body", source_uri="https://example.test/source")
    store.put(obj)
    got = store.get(obj.knowledge_id)
    assert got is not None
    assert got.source_uri == "https://example.test/source"


def test_retrieve_ignores_corrupted_cache() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="stub-v0")
    obj = _make_obj("cache-integrity", "trusted body")
    store.put(obj)

    from oai2.knowledge import RetrievalRequest

    req = RetrievalRequest(topic="cache-integrity")
    key = cache_key_for(req, "stub-v0", b.cache_revision())
    b.kv_put(key, "{not-valid-json")

    result = store.retrieve(req)
    assert [item.content for item in result.objects] == ["trusted body"]


def test_retrieve_ignores_cache_ref_with_wrong_hash() -> None:
    from oai2.knowledge import RetrievalRequest
    from oai2.knowledge.transport import KnowledgeCacheRef, QueryCacheEnvelope

    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="stub-v0")
    obj = _make_obj("cache-ref-integrity", "trusted body")
    store.put(obj)

    req = RetrievalRequest(topic="cache-ref-integrity")
    key = cache_key_for(req, "stub-v0", b.cache_revision())
    bad = QueryCacheEnvelope(
        corpus_revision=b.cache_revision(),
        embedding_digest="stub-v0",
        request_fingerprint=key,
        refs=[
            KnowledgeCacheRef(
                knowledge_id=obj.knowledge_id,
                content_hash=sha256_hex("wrong"),
            )
        ],
    )
    b.kv_put(key, bad.model_dump_json())

    result = store.retrieve(req)
    assert [item.content for item in result.objects] == ["trusted body"]


def test_retrieve_survives_kv_get_failure() -> None:
    from oai2.knowledge import RetrievalRequest

    class KVGetFails(MockCloudflareBindings):
        def kv_get(self, key: str) -> str | None:
            raise RuntimeError("simulated KV read outage")

    b = KVGetFails()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("kv-read-outage", "authoritative body")
    store.put(obj)

    result = store.retrieve(RetrievalRequest(topic="kv-read-outage"))
    assert [item.content for item in result.objects] == ["authoritative body"]


def test_retrieve_survives_kv_put_failure() -> None:
    from oai2.knowledge import RetrievalRequest

    class KVPutFails(MockCloudflareBindings):
        def kv_put(self, key: str, value: str, ttl: int | None = None) -> None:
            raise RuntimeError("simulated KV write outage")

    b = KVPutFails()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("kv-write-outage", "authoritative body")
    store.put(obj)

    result = store.retrieve(RetrievalRequest(topic="kv-write-outage"))
    assert [item.content for item in result.objects] == ["authoritative body"]


def test_put_uses_atomic_index_and_revision_contract() -> None:
    class Recording(MockCloudflareBindings):
        def __init__(self) -> None:
            super().__init__()
            self.atomic_calls = 0

        def d1_upsert_and_bump_revision(self, row: object) -> int:
            self.atomic_calls += 1
            return super().d1_upsert_and_bump_revision(row)

    b = Recording()
    store = CloudflareKnowledgeStore(b, embedding_digest="test-embed-digest-v1")
    obj = _make_obj("atomic-revision", "body")
    before = b.cache_revision()

    store.put(obj)

    assert b.atomic_calls == 1
    assert b.cache_revision() == before + 1


def test_cache_contains_refs_not_full_bodies() -> None:
    from oai2.knowledge import RetrievalRequest
    from oai2.knowledge.transport import QueryCacheEnvelope

    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b, embedding_digest="stub-v0")
    obj = _make_obj("compact-cache", "body should stay in R2")
    store.put(obj)
    req = RetrievalRequest(topic="compact-cache")

    store.retrieve(req)

    key = cache_key_for(req, "stub-v0", b.cache_revision())
    raw = b.kv_get(key)
    assert raw is not None
    envelope = QueryCacheEnvelope.model_validate_json(raw)
    assert len(envelope.refs) == 1
    assert envelope.refs[0].knowledge_id == obj.knowledge_id
    assert "body should stay in R2" not in raw


def test_store_requires_explicit_embedding_digest() -> None:
    """A silently-shared default embedding_digest would let two distinct
    deployments collide in the same KV namespace. The constructor must
    fail closed instead of accepting an implicit default."""
    b = MockCloudflareBindings()
    with pytest.raises(TypeError, match="embedding_digest"):
        CloudflareKnowledgeStore(b)  # type: ignore[call-arg]


def test_store_rejects_empty_embedding_digest() -> None:
    """An empty digest would silently collapse every cache key into the
    same namespace and produce the same silent-collision hazard as the
    prior default. The constructor must refuse empty strings."""
    b = MockCloudflareBindings()
    with pytest.raises(ValueError, match="embedding_digest must be a non-empty string"):
        CloudflareKnowledgeStore(b, embedding_digest="")


def test_cache_key_namespace_isolated_by_embedding_digest() -> None:
    """Two digests backed by the same bindings must NOT collide in the
    shared KV namespace. The cache key must change when the embedding
    digest changes, so a misconfigured deployment cannot silently read
    another deployment's cached results."""
    from oai2.knowledge import RetrievalRequest

    b = MockCloudflareBindings()
    store_a = CloudflareKnowledgeStore(b, embedding_digest="embed-A")

    obj = _make_obj("isolated-cache", "body")
    store_a.put(obj)
    req = RetrievalRequest(topic="isolated-cache")

    store_a.retrieve(req)

    # store_a wrote a cache envelope under its own digest's key.
    key_a = cache_key_for(req, "embed-A", b.cache_revision())
    assert b.kv_get(key_a) is not None
    # A different digest hashes to a different key and must observe an
    # empty KV — not the cache envelope written by store_a.
    key_b = cache_key_for(req, "embed-B", b.cache_revision())
    assert key_a != key_b
    assert b.kv_get(key_b) is None
