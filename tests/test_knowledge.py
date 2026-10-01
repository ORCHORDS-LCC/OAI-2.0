"""Knowledge store + ingestion tests."""

from __future__ import annotations

from oai2.core import Status
from oai2.knowledge import (
    InMemoryKnowledgeStore,
    IngestionJob,
    IngestionPipeline,
    IngestionStatus,
    KnowledgeObject,
    RetrievalRequest,
    sha256_hex,
)


def test_sha256_is_deterministic() -> None:
    assert sha256_hex("hi") == sha256_hex("hi")
    assert sha256_hex("hi") != sha256_hex("ho")


def test_in_memory_store_put_and_retrieve() -> None:
    store = InMemoryKnowledgeStore()
    obj = KnowledgeObject(
        knowledge_id="ko_1",  # type: ignore[arg-type]
        topic="apple silicon",
        content="M5 Max GPU has 40 cores",
        content_hash=sha256_hex("M5 Max GPU has 40 cores"),
        source_uri="local://test",
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )
    store.put(obj)
    assert store.get("ko_1") == obj  # type: ignore[arg-type]
    res = store.retrieve(RetrievalRequest(topic="apple"))
    assert res.objects and res.objects[0].knowledge_id == "ko_1"


def test_ingestion_pipeline_emits_objects() -> None:
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)
    jobs = [
        IngestionJob(
            source_uri="local://a",
            topic="mlx",
            content="mlx is a numpy-like framework",
            authority=0.8,
            status=Status.EXPERIMENTAL,
        ),
        IngestionJob(
            source_uri="local://b",
            topic="mlx",
            content="mlx supports unified memory",
            authority=0.7,
            status=Status.EXPERIMENTAL,
        ),
    ]
    statuses = pipeline.ingest(jobs)
    assert statuses == [IngestionStatus.OK, IngestionStatus.OK]
    res = store.retrieve(RetrievalRequest(topic="mlx"))
    assert len(res.objects) == 2
    # Higher authority wins.
    assert res.objects[0].authority == 0.8
