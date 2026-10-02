"""Pin the input-validation surface of ``oai2.knowledge.abstraction``.

The behavioural tests in ``tests/test_knowledge.py`` cover the happy path
(``InMemoryKnowledgeStore`` put/get/retrieve/all + ``IngestionPipeline``).
This file pins the outer guard rails that are also public contracts:

* ``KnowledgeObject`` Pydantic field validation -- ``topic`` length [1, 256],
  ``content_hash`` length [8, 128], ``authority`` in [0.0, 1.0], and the
  ``extra="forbid"`` model config.
* ``InMemoryKnowledgeStore.retrieve`` filtering semantics -- ``topic`` is a
  case-insensitive substring match, ``min_authority`` excludes below-th
  objects, ``include_status`` excludes any object whose ``status`` is not
  in the allowed tuple, results are sorted by ``(authority, retrieved_at)``
  DESCENDING, and ``limit`` truncates after the sort.
* ``RetrievalResult.__bool__`` semantics -- falsy when ``objects`` is empty,
  truthy otherwise. ``__bool__`` is declared with ``# pragma: no cover`` in
  the source because the branch is trivial, but the contract still matters
  for callers that use ``if result:`` to gate downstream work.

A refactor that widens any of these contracts (e.g. drops ``extra="forbid"``,
flips the sort to ASCENDING, makes ``retrieve`` ignore ``min_authority``)
would silently admit or discard results in a way that propagates to
``evidence_package`` and the truth-evaluation scoring.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    InMemoryKnowledgeStore,
    KnowledgeObject,
    RetrievalRequest,
    RetrievalResult,
    sha256_hex,
)


def _obj(
    knowledge_id: str = "ko_1",
    *,
    content: str = "claim text",
    topic: str = "retrieval",
    authority: float = 0.9,
    status: Status = Status.EXPERIMENTAL,
    content_hash: str | None = None,
    source_uri: str | None = "https://example.test/source",
    retrieved_at: float = 0.0,
) -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId(knowledge_id),  # type: ignore[arg-type]
        topic=topic,
        content=content,
        content_hash=content_hash if content_hash is not None else sha256_hex(content),
        source_uri=source_uri,
        authority=authority,
        status=status,
        retrieved_at=retrieved_at,
    )


# ---------------------------------------------------------------------------
# KnowledgeObject Pydantic field validation (negative + boundary positive)
# ---------------------------------------------------------------------------


def test_knowledge_object_rejects_empty_topic() -> None:
    with pytest.raises(ValidationError):
        _obj(topic="")


def test_knowledge_object_accepts_topic_at_min_length() -> None:
    obj = _obj(topic="t")
    assert obj.topic == "t"


def test_knowledge_object_rejects_topic_over_256_chars() -> None:
    with pytest.raises(ValidationError):
        _obj(topic="t" * 257)


def test_knowledge_object_accepts_topic_at_max_length() -> None:
    boundary = "t" * 256
    obj = _obj(topic=boundary)
    assert len(obj.topic) == 256


@pytest.mark.parametrize("bad_hash", ["a" * 7, "a" * 129, "short"])
def test_knowledge_object_rejects_out_of_range_content_hash_length(bad_hash: str) -> None:
    with pytest.raises(ValidationError):
        _obj(content_hash=bad_hash)


def test_knowledge_object_accepts_content_hash_at_min_length() -> None:
    boundary = "a" * 8
    obj = _obj(content_hash=boundary)
    assert obj.content_hash == boundary


def test_knowledge_object_accepts_content_hash_at_max_length() -> None:
    boundary = "a" * 128
    obj = _obj(content_hash=boundary)
    assert obj.content_hash == boundary


@pytest.mark.parametrize("bad_authority", [-0.01, -1.0, 1.01, 2.0])
def test_knowledge_object_rejects_out_of_range_authority(bad_authority: float) -> None:
    with pytest.raises(ValidationError):
        _obj(authority=bad_authority)


def test_knowledge_object_accepts_authority_at_zero_boundary() -> None:
    obj = _obj(authority=0.0)
    assert obj.authority == 0.0


def test_knowledge_object_accepts_authority_at_one_boundary() -> None:
    obj = _obj(authority=1.0)
    assert obj.authority == 1.0


def test_knowledge_object_rejects_unknown_field_due_to_extra_forbid() -> None:
    """``model_config = ConfigDict(extra="forbid")`` means a kwarg the model
    does not declare raises ``ValidationError`` instead of being silently
    dropped. A future refactor that flips this to ``extra="ignore"`` would
    silently widen the input domain."""
    with pytest.raises(ValidationError):
        KnowledgeObject(
            knowledge_id=KnowledgeId("ko_1"),  # type: ignore[arg-type]
            topic="retrieval",
            content="claim",
            content_hash=sha256_hex("claim"),
            unknown_field="bogus",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# RetrievalResult.__bool__ semantics
# ---------------------------------------------------------------------------


def test_retrieval_result_is_falsy_when_objects_empty() -> None:
    result = RetrievalResult(topic="x", objects=[])
    assert bool(result) is False
    assert (not result) is True


def test_retrieval_result_is_truthy_when_objects_present() -> None:
    result = RetrievalResult(topic="x", objects=[_obj()])
    assert bool(result) is True


# ---------------------------------------------------------------------------
# InMemoryKnowledgeStore.retrieve filtering semantics
# ---------------------------------------------------------------------------


def test_retrieve_returns_empty_result_for_empty_store() -> None:
    store = InMemoryKnowledgeStore()
    result = store.retrieve(RetrievalRequest(topic="anything"))
    assert result.objects == []
    assert result.candidates == []
    assert not result


def test_retrieve_excludes_objects_below_min_authority() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_low", authority=0.2))
    store.put(_obj("ko_high", authority=0.8))
    result = store.retrieve(RetrievalRequest(topic="retrieval", min_authority=0.5))
    assert [obj.knowledge_id for obj in result.objects] == [KnowledgeId("ko_high")]


def test_retrieve_topic_match_is_case_insensitive_substring() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_mlx_gpu", topic="MLX-GPU"))
    for query in ("mlx", "MLX", "Mlx-gpu", "gpu"):
        result = store.retrieve(RetrievalRequest(topic=query))
        assert len(result.objects) == 1, f"failed for query '{query}'"
        assert result.objects[0].knowledge_id == KnowledgeId("ko_mlx_gpu")


def test_retrieve_excludes_status_not_in_include_tuple() -> None:
    """Default ``include_status`` is ``(Status.IMPLEMENTED, Status.EXPERIMENTAL)``.
    ``Status.PROPOSED`` is excluded."""
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_impl", status=Status.IMPLEMENTED))
    store.put(_obj("ko_exp", status=Status.EXPERIMENTAL))
    store.put(_obj("ko_prop", status=Status.PROPOSED))
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    ids = {obj.knowledge_id for obj in result.objects}
    assert ids == {KnowledgeId("ko_impl"), KnowledgeId("ko_exp")}


def test_retrieve_respects_explicit_include_status_filter() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_impl", status=Status.IMPLEMENTED))
    store.put(_obj("ko_prop", status=Status.PROPOSED))
    result = store.retrieve(
        RetrievalRequest(topic="retrieval", include_status=(Status.PROPOSED,))
    )
    ids = {obj.knowledge_id for obj in result.objects}
    assert ids == {KnowledgeId("ko_prop")}


def test_retrieve_sorts_by_authority_descending_then_retrieved_at_descending() -> None:
    """Sort key is ``(authority, retrieved_at)`` with ``reverse=True``: highest
    authority first; ties broken by most recent ``retrieved_at`` first."""
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_a_low_old", authority=0.5, retrieved_at=100.0))
    store.put(_obj("ko_b_low_new", authority=0.5, retrieved_at=200.0))
    store.put(_obj("ko_c_high", authority=0.9))
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    assert [obj.knowledge_id for obj in result.objects] == [
        KnowledgeId("ko_c_high"),
        KnowledgeId("ko_b_low_new"),
        KnowledgeId("ko_a_low_old"),
    ]


def test_retrieve_respects_limit_truncation() -> None:
    store = InMemoryKnowledgeStore()
    for i in range(5):
        store.put(_obj(f"ko_{i}", authority=0.5 + i * 0.1))
    result = store.retrieve(RetrievalRequest(topic="retrieval", limit=2))
    assert len(result.objects) == 2
    # Highest authority two retained.
    assert result.objects[0].knowledge_id == KnowledgeId("ko_4")
    assert result.objects[1].knowledge_id == KnowledgeId("ko_3")


def test_retrieve_candidates_mirror_object_subset_one_to_one() -> None:
    """For each retained object, exactly one ``RetrievalCandidate`` is emitted
    carrying the same ``knowledge_id`` and ``content_hash``. The candidate
    list is the per-row evidence that pins the object back to its provenance."""
    store = InMemoryKnowledgeStore()
    obj = _obj("ko_a", content="x", authority=0.8)
    store.put(obj)
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    assert len(result.candidates) == len(result.objects) == 1
    cand = result.candidates[0]
    assert cand.knowledge_id == obj.knowledge_id
    assert cand.content_hash == obj.content_hash
    assert cand.source_uri == obj.source_uri
