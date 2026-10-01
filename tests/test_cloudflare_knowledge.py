"""Tests for the Cloudflare knowledge adapter (mock bindings only)."""

from __future__ import annotations

import json

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
    store = CloudflareKnowledgeStore(b)
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
    store = CloudflareKnowledgeStore(b)
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
    store = CloudflareKnowledgeStore(b)
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
    store = CloudflareKnowledgeStore(b)
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
    store = CloudflareKnowledgeStore(b)
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
    store = CloudflareKnowledgeStore(b)
    obj = _make_obj("integrity", "trusted body")
    obj.content_hash = sha256_hex("different body")
    with pytest.raises(ValueError, match="content_hash"):
        store.put(obj)


def test_store_get_rejects_missing_blob() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b)
    obj = _make_obj("integrity", "trusted body", kid_suffix="missing")
    store.put(obj)
    del b.blobs[r2_blob_key_for(obj.content_hash)]
    with pytest.raises(ValueError, match="missing"):
        store.get(obj.knowledge_id)


def test_store_get_rejects_mismatched_blob() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b)
    obj = _make_obj("integrity", "trusted body", kid_suffix="tampered")
    store.put(obj)
    b.blobs[r2_blob_key_for(obj.content_hash)] = b"tampered body"
    with pytest.raises(ValueError, match="hash"):
        store.get(obj.knowledge_id)


def test_store_preserves_source_uri() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b)
    obj = _make_obj("provenance", "source body", source_uri="https://example.test/source")
    store.put(obj)
    got = store.get(obj.knowledge_id)
    assert got is not None
    assert got.source_uri == "https://example.test/source"


def test_retrieve_ignores_corrupted_cache() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b)
    obj = _make_obj("cache-integrity", "trusted body")
    store.put(obj)

    from oai2.knowledge import RetrievalRequest

    req = RetrievalRequest(topic="cache-integrity")
    key = cache_key_for(req, "stub-v0", b.cache_revision())
    cached = obj.model_dump()
    cached["content"] = "tampered body"
    b.kv_put(key, json.dumps({"objects": [cached]}))

    result = store.retrieve(req)
    assert [item.content for item in result.objects] == ["trusted body"]
