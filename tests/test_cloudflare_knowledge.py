"""Tests for the Cloudflare knowledge adapter (mock bindings only)."""

from __future__ import annotations

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
    topic: str, content: str, authority: float = 0.5, kid_suffix: str = ""
) -> object:
    from oai2.knowledge import KnowledgeObject

    return KnowledgeObject(
        knowledge_id=KnowledgeId(f"ko_{sha256_hex(topic + kid_suffix)[:12]}"),
        topic=topic,
        content=content,
        content_hash=sha256_hex(content),
        source_uri=None,
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
    assert cache_key_for(r1, "v1") == cache_key_for(r2, "v1")


def test_cache_key_changes_when_status_set_changes() -> None:
    from oai2.knowledge import RetrievalRequest

    r1 = RetrievalRequest(topic="x")
    r2 = RetrievalRequest(
        topic="x",
        include_status=(Status.IMPLEMENTED,),
    )
    assert cache_key_for(r1, "v1") != cache_key_for(r2, "v1")


def test_cache_hit_returns_same_objects() -> None:
    b = MockCloudflareBindings()
    store = CloudflareKnowledgeStore(b)
    obj = _make_obj("cache-me", "x")
    store.put(obj)

    from oai2.knowledge import RetrievalRequest

    req = RetrievalRequest(topic="cache-me")
    r1 = store.retrieve(req)
    assert len(r1.objects) == 1

    # Mutate the underlying store; cached retrieval should still match.
    other = _make_obj("cache-me", "should-not-appear")
    store.put(other)

    r2 = store.retrieve(req)
    assert [o.knowledge_id for o in r2.objects] == [obj.knowledge_id]


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
