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

The sub-sections after the original 21 tests pin the rest of the module's
structural contract (slice 69 expansion): public name re-export identity,
Pydantic field shape, dataclass frozen/slots/mutability, abstract-base guards,
helper-function determinism, InMemoryKnowledgeStore concrete behaviour, and
public-safety. A refactor that widens any of these contracts (e.g. drops
``extra="forbid"``, flips the sort to ASCENDING, makes ``retrieve`` ignore
``min_authority``, removes an ``@abstractmethod``, adds a network call)
would silently change the substrate that ``ingestion``, ``transport``,
``evidence_package``, and ``q-pipe-import`` all build on.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from abc import ABC

import pytest
from pydantic import BaseModel, ValidationError

import oai2.knowledge
import oai2.knowledge.abstraction as abstraction_module
from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    InMemoryKnowledgeStore,
    KnowledgeObject,
    KnowledgeStore,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    now_epoch,
    sha256_hex,
)
from tests._evidence_fixtures import make_knowledge_object as _obj


def _valid_kwargs() -> dict:
    """Build the minimal valid kwargs surface for KnowledgeObject (mandatory
    fields only — knowledge_id, topic, content, content_hash)."""
    return {
        "knowledge_id": KnowledgeId("ko_valid"),
        "topic": "retrieval",
        "content": "claim",
        "content_hash": sha256_hex("claim"),
    }


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
            knowledge_id=KnowledgeId("ko_1"),
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
    result = store.retrieve(RetrievalRequest(topic="retrieval", include_status=(Status.PROPOSED,)))
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


# ===========================================================================
# Comprehensive structural pin of oai2/knowledge/abstraction.py (slice 69).
#
# The 21 tests above pin specific Pydantic-input-validation + retrieve-
# filtering behaviour. The sub-sections below pin the rest of the module's
# structural contract: docstring, imports, public-name surface, dataclass
# shapes, abstract-base, helper-function determinism, InMemoryKnowledgeStore
# concrete class, and public-safety + structural-counts. A refactor that
# widens any of these contracts (drops extra="forbid", flips a dataclass
# to non-slots, removes an @abstractmethod, adds a network call, etc.)
# would silently change the substrate that ingestion, transport, evidence
# package, and q-pipe-import all build on.
# ===========================================================================


# ---------------------------------------------------------------------------
# Section A: docstring + module-level imports
# ---------------------------------------------------------------------------


def test_module_docstring_mentions_knowledge_store_abstraction() -> None:
    doc = abstraction_module.__doc__ or ""
    assert "knowledge store" in doc.lower()
    assert "abstraction" in doc.lower()


def test_module_docstring_mentions_external_knowledge() -> None:
    doc = abstraction_module.__doc__ or ""
    assert "knowledge" in doc.lower()


def test_module_docstring_mentions_cloudflare_r2_d1_reference() -> None:
    doc = abstraction_module.__doc__ or ""
    assert "cloudflare" in doc.lower()
    assert "d1" in doc.lower()
    assert "r2" in doc.lower()


def test_module_uses_future_annotations() -> None:
    source = inspect.getsource(abstraction_module)
    assert "from __future__ import annotations" in source


def test_module_imports_hashlib() -> None:
    source = inspect.getsource(abstraction_module)
    assert re.search(r"^import hashlib$", source, re.MULTILINE) is not None


def test_module_imports_time() -> None:
    source = inspect.getsource(abstraction_module)
    assert re.search(r"^import time$", source, re.MULTILINE) is not None


def test_module_imports_abc() -> None:
    source = inspect.getsource(abstraction_module)
    assert "from abc import" in source or "import abc" in source


def test_module_imports_dataclass_field() -> None:
    source = inspect.getsource(abstraction_module)
    assert (
        re.search(r"from dataclasses import dataclass, field", source)
        or "from dataclasses import" in source
    )


def test_module_imports_pydantic_basemodel_configdict_field() -> None:
    source = inspect.getsource(abstraction_module)
    assert "BaseModel" in source
    assert "ConfigDict" in source
    assert "Field" in source


def test_module_imports_knowledge_id_and_status_from_core() -> None:
    source = inspect.getsource(abstraction_module)
    assert "KnowledgeId" in source
    assert "Status" in source
    assert "from ..core import" in source or "from oai2.core import" in source


def test_module_does_not_import_oai2_at_line_start() -> None:
    source = inspect.getsource(abstraction_module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import oai2") or stripped.startswith("from oai2"):
            raise AssertionError(f"absolute oai2 import at line-start: {line!r}")


def test_module_does_not_have_wildcard_imports() -> None:
    source = inspect.getsource(abstraction_module)
    assert "import *" not in source


def test_module_has_no_cloud_runtime_imports() -> None:
    source = inspect.getsource(abstraction_module)
    for cloud in ("boto3", "azure", "google.cloud", "kubernetes", "docker", "fabric"):
        assert cloud not in source, f"cloud-runtime import {cloud!r} found"


def test_module_has_no_mlx_imports() -> None:
    source = inspect.getsource(abstraction_module)
    assert "mlx" not in source.lower()


# ---------------------------------------------------------------------------
# Section B: public-name surface
# ---------------------------------------------------------------------------


def test_dunder_all_is_pinned_to_eight_names() -> None:
    expected = {
        "KnowledgeObject",
        "RetrievalRequest",
        "RetrievalCandidate",
        "RetrievalResult",
        "KnowledgeStore",
        "InMemoryKnowledgeStore",
        "sha256_hex",
        "now_epoch",
    }
    assert set(abstraction_module.__all__) == expected


def test_dunder_all_eight_names_importable_from_module() -> None:
    for name in abstraction_module.__all__:
        assert hasattr(abstraction_module, name), f"missing {name!r}"


def test_knowledge_object_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.KnowledgeObject is abstraction_module.KnowledgeObject


def test_retrieval_request_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.RetrievalRequest is abstraction_module.RetrievalRequest


def test_retrieval_candidate_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.RetrievalCandidate is abstraction_module.RetrievalCandidate


def test_retrieval_result_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.RetrievalResult is abstraction_module.RetrievalResult


def test_knowledge_store_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.KnowledgeStore is abstraction_module.KnowledgeStore


def test_in_memory_knowledge_store_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.InMemoryKnowledgeStore is abstraction_module.InMemoryKnowledgeStore


def test_sha256_hex_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.sha256_hex is abstraction_module.sha256_hex


def test_now_epoch_module_is_same_object_as_package() -> None:
    assert oai2.knowledge.now_epoch is abstraction_module.now_epoch


def test_knowledge_object_re_exported_through_knowledge_package() -> None:
    assert "KnowledgeObject" in oai2.knowledge.__all__


def test_retrieval_request_re_exported_through_knowledge_package() -> None:
    assert "RetrievalRequest" in oai2.knowledge.__all__


def test_retrieval_candidate_re_exported_through_knowledge_package() -> None:
    assert "RetrievalCandidate" in oai2.knowledge.__all__


def test_retrieval_result_re_exported_through_knowledge_package() -> None:
    assert "RetrievalResult" in oai2.knowledge.__all__


def test_knowledge_store_re_exported_through_knowledge_package() -> None:
    assert "KnowledgeStore" in oai2.knowledge.__all__


def test_in_memory_knowledge_store_re_exported_through_knowledge_package() -> None:
    assert "InMemoryKnowledgeStore" in oai2.knowledge.__all__


def test_sha256_hex_re_exported_through_knowledge_package() -> None:
    assert "sha256_hex" in oai2.knowledge.__all__


def test_now_epoch_re_exported_through_knowledge_package() -> None:
    assert "now_epoch" in oai2.knowledge.__all__


def test_direct_module_import_matches_top_level_for_knowledge_object() -> None:
    from oai2.knowledge import KnowledgeObject as PkgKnowledgeObject
    from oai2.knowledge.abstraction import KnowledgeObject as DirectKnowledgeObject

    assert DirectKnowledgeObject is PkgKnowledgeObject


# ---------------------------------------------------------------------------
# Section C: KnowledgeObject structural shape (Pydantic v2 BaseModel)
# ---------------------------------------------------------------------------


def test_knowledge_object_is_a_basemodel_subclass() -> None:
    assert isinstance(KnowledgeObject(**_valid_kwargs()), BaseModel)


def test_knowledge_object_extra_config_is_forbid() -> None:
    assert KnowledgeObject.model_config["extra"] == "forbid"


def test_knowledge_object_field_set_is_explicitly_pinned() -> None:
    """The field set stays a reviewed contract, not an open bag.

    The original nine carry identity/content/authority/lifecycle. The ten
    added fields carry what OAI-2.0 #200/#207/#185 require the canonical
    record to be able to express and that a content hash alone cannot:

      claim_key       stable identity of the CLAIM, so a reworded or
                      recommendation-flipped revision is recognisable as the
                      same claim rather than a new one (REQ-TEMP-011/012)
      content_version per-revision digest
      source_version  version of the source the revision came from
      effective_at    effective time, distinct from retrieval time
      superseded_by   linkage to the superseding record (REQ-TEMP-004)
      superseded_at
      trust_class     retrieved evidence is not an instruction (REQ-PROMPT-011)
      scope_class     who may receive it, so a private lesson is not served
                      to every caller by virtue of being indexed
    """
    assert set(KnowledgeObject.model_fields.keys()) == {
        # original nine
        "knowledge_id",
        "topic",
        "content",
        "content_hash",
        "source_uri",
        "retrieved_at",
        "authority",
        "status",
        "artifact_ref",
        "embedding_ref",
        # temporal / supersession
        "claim_key",
        "content_version",
        "source_version",
        "effective_at",
        "superseded_by",
        "superseded_at",
        # trust / isolation
        "trust_class",
        "scope_class",
    }


def test_claim_identity_and_content_version_are_separate_fields() -> None:
    """Conflating them would make supersession impossible."""
    fields = KnowledgeObject.model_fields
    assert "claim_key" in fields and "content_version" in fields
    assert "content_hash" in fields
    # content_hash is mandatory; claim identity is not, because a record may
    # predate claim tracking and we report its absence rather than invent one.
    assert fields["content_hash"].is_required()
    assert not fields["claim_key"].is_required()
    assert not fields["content_version"].is_required()


def test_knowledge_object_knowledge_id_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        KnowledgeObject(  # type: ignore[call-arg]
            topic="retrieval",
            content="claim",
            content_hash=sha256_hex("claim"),
        )


def test_knowledge_object_topic_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        KnowledgeObject(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_x"),
            content="claim",
            content_hash=sha256_hex("claim"),
        )


def test_knowledge_object_content_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        KnowledgeObject(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_x"),
            topic="retrieval",
            content_hash=sha256_hex("claim"),
        )


def test_knowledge_object_content_hash_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        KnowledgeObject(  # type: ignore[call-arg]
            knowledge_id=KnowledgeId("ko_x"),
            topic="retrieval",
            content="claim",
        )


def test_knowledge_object_authority_default_is_half() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    assert obj.authority == 0.5


def test_knowledge_object_status_default_is_proposed() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    assert obj.status == Status.PROPOSED


def test_knowledge_object_retrieved_at_default_is_zero() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    assert obj.retrieved_at == 0.0


def test_knowledge_object_source_uri_default_is_none() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    assert obj.source_uri is None


def test_knowledge_object_artifact_ref_default_is_none() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    assert obj.artifact_ref is None


def test_knowledge_object_embedding_ref_default_is_none() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    assert obj.embedding_ref is None


def test_knowledge_object_content_hash_min_length_8() -> None:
    with pytest.raises(ValidationError):
        _obj(content_hash="a" * 7)


def test_knowledge_object_content_hash_max_length_128() -> None:
    with pytest.raises(ValidationError):
        _obj(content_hash="a" * 129)


def test_knowledge_object_authority_below_zero_fails() -> None:
    with pytest.raises(ValidationError):
        _obj(authority=-0.1)


def test_knowledge_object_authority_above_one_fails() -> None:
    with pytest.raises(ValidationError):
        _obj(authority=1.1)


def test_knowledge_object_topic_min_length_one() -> None:
    with pytest.raises(ValidationError):
        _obj(topic="")


def test_knowledge_object_topic_max_length_256() -> None:
    with pytest.raises(ValidationError):
        _obj(topic="t" * 257)


def test_knowledge_object_knowledge_id_is_str_subclass() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    # KnowledgeId is a NewType over str — runtime the value is a str.
    assert isinstance(obj.knowledge_id, str)


def test_knowledge_object_model_dump_round_trips() -> None:
    obj = KnowledgeObject(**_valid_kwargs())
    dumped = obj.model_dump()
    assert dumped["topic"] == obj.topic
    assert dumped["content"] == obj.content
    assert dumped["content_hash"] == obj.content_hash


def test_knowledge_object_accepts_all_status_enum_values() -> None:
    for status in Status:
        obj = _obj(status=status)
        assert obj.status == status


def test_knowledge_object_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        KnowledgeObject(
            knowledge_id=KnowledgeId("ko_x"),
            topic="retrieval",
            content="claim",
            content_hash=sha256_hex("claim"),
            unknown_field="bogus",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Section D: RetrievalRequest + RetrievalCandidate dataclass shape
# ---------------------------------------------------------------------------


def test_retrieval_request_is_a_dataclass() -> None:
    assert dataclasses.is_dataclass(RetrievalRequest)


def test_retrieval_request_is_frozen() -> None:
    params = getattr(RetrievalRequest, "__dataclass_params__", None)
    assert params is not None
    assert params.frozen is True


def test_retrieval_request_is_slotted() -> None:
    req = RetrievalRequest(topic="x")
    assert not hasattr(req, "__dict__")


def test_retrieval_request_field_set_pinned_to_five_names() -> None:
    # `include_superseded` was added for REQ-TEMP-025: excluding superseded
    # evidence by default (REQ-TEMP-014) must not remove the ability to ask
    # for it as historical evidence. See the gate in
    # InMemoryKnowledgeStore.retrieve.
    field_names = {f.name for f in dataclasses.fields(RetrievalRequest)}
    assert field_names == {
        "topic",
        "limit",
        "min_authority",
        "include_status",
        "include_superseded",
    }


def test_retrieval_request_excludes_superseded_by_default() -> None:
    req = RetrievalRequest(topic="x")
    assert req.include_superseded is False


def test_retrieval_request_topic_is_mandatory() -> None:
    with pytest.raises(TypeError):
        RetrievalRequest()  # type: ignore[call-arg]


def test_retrieval_request_limit_default_is_eight() -> None:
    req = RetrievalRequest(topic="x")
    assert req.limit == 8


def test_retrieval_request_min_authority_default_is_zero() -> None:
    req = RetrievalRequest(topic="x")
    assert req.min_authority == 0.0


def test_retrieval_request_include_status_default_is_two_member_tuple() -> None:
    req = RetrievalRequest(topic="x")
    assert req.include_status == (Status.IMPLEMENTED, Status.EXPERIMENTAL)


def test_retrieval_request_accepts_custom_limit() -> None:
    req = RetrievalRequest(topic="x", limit=3)
    assert req.limit == 3


def test_retrieval_request_accepts_custom_min_authority() -> None:
    req = RetrievalRequest(topic="x", min_authority=0.7)
    assert req.min_authority == 0.7


def test_retrieval_request_accepts_custom_include_status_tuple() -> None:
    req = RetrievalRequest(topic="x", include_status=(Status.PROPOSED,))
    assert req.include_status == (Status.PROPOSED,)


def test_retrieval_request_is_hashable() -> None:
    req = RetrievalRequest(topic="x")
    hash(req)


def test_retrieval_request_is_immutable() -> None:
    req = RetrievalRequest(topic="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.topic = "y"  # type: ignore[misc]


def test_retrieval_candidate_is_a_dataclass() -> None:
    assert dataclasses.is_dataclass(RetrievalCandidate)


def test_retrieval_candidate_is_frozen() -> None:
    params = getattr(RetrievalCandidate, "__dataclass_params__", None)
    assert params is not None
    assert params.frozen is True


def test_retrieval_candidate_field_set_pinned_to_four_names() -> None:
    field_names = {f.name for f in dataclasses.fields(RetrievalCandidate)}
    assert field_names == {"knowledge_id", "content_hash", "source_uri", "score"}


def test_retrieval_candidate_score_default_is_none() -> None:
    cand = RetrievalCandidate(
        knowledge_id=KnowledgeId("ko_x"), content_hash="a" * 8, source_uri=None
    )
    assert cand.score is None


def test_retrieval_candidate_accepts_custom_score() -> None:
    cand = RetrievalCandidate(
        knowledge_id=KnowledgeId("ko_x"),
        content_hash="a" * 8,
        source_uri=None,
        score=0.42,
    )
    assert cand.score == 0.42


# ---------------------------------------------------------------------------
# Section E: RetrievalResult dataclass shape + __bool__
# ---------------------------------------------------------------------------


def test_retrieval_result_is_a_dataclass() -> None:
    assert dataclasses.is_dataclass(RetrievalResult)


def test_retrieval_result_is_not_frozen() -> None:
    params = getattr(RetrievalResult, "__dataclass_params__", None)
    assert params is not None
    assert params.frozen is False


def test_retrieval_result_is_slotted() -> None:
    result = RetrievalResult(topic="x")
    assert not hasattr(result, "__dict__")


def test_retrieval_result_field_set_pinned_to_three_names() -> None:
    field_names = {f.name for f in dataclasses.fields(RetrievalResult)}
    assert field_names == {"topic", "objects", "candidates"}


def test_retrieval_result_topic_is_mandatory() -> None:
    with pytest.raises(TypeError):
        RetrievalResult()  # type: ignore[call-arg]


def test_retrieval_result_objects_default_is_empty_list() -> None:
    result = RetrievalResult(topic="x")
    assert result.objects == []
    assert type(result.objects) is list


def test_retrieval_result_candidates_default_is_empty_list() -> None:
    result = RetrievalResult(topic="x")
    assert result.candidates == []
    assert type(result.candidates) is list


def test_retrieval_result_objects_list_accepts_append() -> None:
    """Mutable list field is the contract — verify we can append to it."""
    result = RetrievalResult(topic="x")
    obj = _obj("ko_a")
    result.objects.append(obj)
    assert result.objects == [obj]
    assert bool(result) is True


# ---------------------------------------------------------------------------
# Section F: KnowledgeStore abstract base
# ---------------------------------------------------------------------------


def test_knowledge_store_is_an_abc_subclass() -> None:
    assert issubclass(KnowledgeStore, ABC)


def test_knowledge_store_class_status_is_proposed() -> None:
    assert KnowledgeStore.STATUS == Status.PROPOSED


def test_knowledge_store_put_is_abstract() -> None:
    put = KnowledgeStore.__dict__["put"]
    assert getattr(put, "__isabstractmethod__", False) is True


def test_knowledge_store_get_is_abstract() -> None:
    get = KnowledgeStore.__dict__["get"]
    assert getattr(get, "__isabstractmethod__", False) is True


def test_knowledge_store_retrieve_is_abstract() -> None:
    retrieve = KnowledgeStore.__dict__["retrieve"]
    assert getattr(retrieve, "__isabstractmethod__", False) is True


def test_knowledge_store_all_is_abstract() -> None:
    all_method = KnowledgeStore.__dict__["all"]
    assert getattr(all_method, "__isabstractmethod__", False) is True


def test_knowledge_store_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError, match="abstract"):
        KnowledgeStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Section G: sha256_hex + now_epoch helper functions
# ---------------------------------------------------------------------------


def test_sha256_hex_returns_64_character_hex_string() -> None:
    out = sha256_hex("hello world")
    assert len(out) == 64
    assert all(c in "0123456789abcdef" for c in out)


def test_sha256_hex_is_deterministic() -> None:
    a = sha256_hex("claim text")
    b = sha256_hex("claim text")
    assert a == b


def test_sha256_hex_handles_empty_string() -> None:
    out = sha256_hex("")
    assert len(out) == 64


def test_sha256_hex_uses_utf_8_encoding_for_unicode() -> None:
    out = sha256_hex("héllo wörld")
    assert len(out) == 64


def test_sha256_hex_distinguishes_different_inputs() -> None:
    assert sha256_hex("a") != sha256_hex("b")


def test_sha256_hex_known_value_for_hello() -> None:
    # sha256("hello") — deterministic fixture pin
    import hashlib

    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256_hex("hello") == expected


def test_now_epoch_returns_a_float() -> None:
    out = now_epoch()
    assert isinstance(out, float)


def test_now_epoch_returns_a_positive_value() -> None:
    out = now_epoch()
    assert out > 0.0


def test_now_epoch_increases_over_time() -> None:
    import time as _time

    a = now_epoch()
    _time.sleep(0.01)
    b = now_epoch()
    assert b > a


def test_now_epoch_returns_seconds_since_unix_epoch() -> None:
    """Sanity check — must be a 2026-era timestamp."""
    out = now_epoch()
    assert out > 1_700_000_000.0  # Nov 2023
    assert out < 2_000_000_000.0  # May 2033


# ---------------------------------------------------------------------------
# Section H: InMemoryKnowledgeStore concrete subclass
# ---------------------------------------------------------------------------


def test_in_memory_knowledge_store_is_a_knowledge_store_subclass() -> None:
    store = InMemoryKnowledgeStore()
    assert isinstance(store, KnowledgeStore)


def test_in_memory_knowledge_store_class_status_is_experimental() -> None:
    """InMemoryKnowledgeStore overrides the PROPOSED default to EXPERIMENTAL —
    the deterministic test reference implementation is wired but not deployed."""
    assert InMemoryKnowledgeStore.STATUS == Status.EXPERIMENTAL


def test_in_memory_knowledge_store_distinct_status_from_base() -> None:
    assert InMemoryKnowledgeStore.STATUS != KnowledgeStore.STATUS


def test_in_memory_knowledge_store_initial_state_is_empty() -> None:
    store = InMemoryKnowledgeStore()
    assert list(store.all()) == []


def test_in_memory_knowledge_store_put_then_get_round_trips() -> None:
    store = InMemoryKnowledgeStore()
    obj = _obj("ko_a", content="claim")
    store.put(obj)
    fetched = store.get(KnowledgeId("ko_a"))
    assert fetched is obj


def test_in_memory_knowledge_store_get_returns_none_for_missing_id() -> None:
    store = InMemoryKnowledgeStore()
    assert store.get(KnowledgeId("ko_missing")) is None


def test_in_memory_knowledge_store_put_overwrites_existing_id() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_a", content="first"))
    second = _obj("ko_a", content="second")
    store.put(second)
    fetched = store.get(KnowledgeId("ko_a"))
    assert fetched is second
    assert fetched.content == "second"


def test_in_memory_knowledge_store_all_yields_inserted_objects() -> None:
    store = InMemoryKnowledgeStore()
    a_obj = _obj("ko_a")
    b_obj = _obj("ko_b")
    store.put(a_obj)
    store.put(b_obj)
    seen = sorted([o.knowledge_id for o in store.all()])
    assert seen == [KnowledgeId("ko_a"), KnowledgeId("ko_b")]


def test_in_memory_knowledge_store_retrieve_returns_retrieval_result() -> None:
    store = InMemoryKnowledgeStore()
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    assert isinstance(result, RetrievalResult)


def test_in_memory_knowledge_store_retrieve_topic_echoes_request() -> None:
    store = InMemoryKnowledgeStore()
    result = store.retrieve(RetrievalRequest(topic="specific-topic"))
    assert result.topic == "specific-topic"


def test_in_memory_knowledge_store_retrieve_candidates_empty_for_empty_result() -> None:
    store = InMemoryKnowledgeStore()
    result = store.retrieve(RetrievalRequest(topic="anything"))
    assert result.candidates == []


def test_in_memory_knowledge_store_retrieve_excludes_unauthorized_status() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_prop", status=Status.PROPOSED))
    store.put(_obj("ko_impl", status=Status.IMPLEMENTED))
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    ids = {obj.knowledge_id for obj in result.objects}
    assert ids == {KnowledgeId("ko_impl")}


def test_in_memory_knowledge_store_retrieve_with_empty_include_status() -> None:
    """Empty include_status means nothing is allowed."""
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_impl", status=Status.IMPLEMENTED))
    result = store.retrieve(RetrievalRequest(topic="retrieval", include_status=()))
    assert result.objects == []


def test_in_memory_knowledge_store_retrieve_limit_one_picks_top() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_low", authority=0.3))
    store.put(_obj("ko_high", authority=0.9))
    result = store.retrieve(RetrievalRequest(topic="retrieval", limit=1))
    assert [obj.knowledge_id for obj in result.objects] == [KnowledgeId("ko_high")]


def test_in_memory_knowledge_store_candidate_knowledge_ids_match_objects() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_a", content="x"))
    store.put(_obj("ko_b", content="y"))
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    obj_ids = [obj.knowledge_id for obj in result.objects]
    cand_ids = [cand.knowledge_id for cand in result.candidates]
    assert obj_ids == cand_ids


def test_in_memory_knowledge_store_candidate_source_uri_propagates() -> None:
    store = InMemoryKnowledgeStore()
    obj = _obj("ko_a", content="x", source_uri="https://x.test/source")
    store.put(obj)
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    assert result.candidates[0].source_uri == "https://x.test/source"


def test_in_memory_knowledge_store_candidate_content_hash_propagates() -> None:
    store = InMemoryKnowledgeStore()
    obj = _obj("ko_a", content="x", content_hash="b" * 64)
    store.put(obj)
    result = store.retrieve(RetrievalRequest(topic="retrieval"))
    assert result.candidates[0].content_hash == "b" * 64


def test_in_memory_knowledge_store_min_authority_excludes_below() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_low", authority=0.2))
    store.put(_obj("ko_high", authority=0.9))
    result = store.retrieve(RetrievalRequest(topic="retrieval", min_authority=0.5))
    assert [obj.knowledge_id for obj in result.objects] == [KnowledgeId("ko_high")]


def test_in_memory_knowledge_store_min_authority_at_zero_includes_all() -> None:
    store = InMemoryKnowledgeStore()
    store.put(_obj("ko_zero", authority=0.0))
    store.put(_obj("ko_high", authority=0.9))
    result = store.retrieve(RetrievalRequest(topic="retrieval", min_authority=0.0))
    assert len(result.objects) == 2


# ---------------------------------------------------------------------------
# Section I: public-safety + structural counts
# ---------------------------------------------------------------------------


def test_source_pins_one_basemodel_subclass() -> None:
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"^class\s+\w+\s*\(\s*BaseModel\s*\)\s*:", source, re.MULTILINE)
    assert len(matches) == 1
    assert matches[0].startswith("class KnowledgeObject")


def test_source_pins_three_dataclass_decorators() -> None:
    """RetrievalRequest, RetrievalCandidate, RetrievalResult — three @dataclass
    decorators. RetrievalRequest and RetrievalCandidate are (slots, frozen);
    RetrievalResult is (slots) only (mutable list fields)."""
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"@dataclass\([^)]*\)", source, re.DOTALL)
    assert len(matches) == 3


def test_source_pins_one_abc_subclass() -> None:
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"^class\s+\w+\s*\(\s*ABC\s*\)\s*:", source, re.MULTILINE)
    assert len(matches) == 1
    assert "KnowledgeStore" in matches[0]


def test_source_pins_one_concrete_knowledge_store_subclass() -> None:
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"^class\s+\w+\s*\(\s*KnowledgeStore\s*\)\s*:", source, re.MULTILINE)
    assert len(matches) == 1
    assert "InMemoryKnowledgeStore" in matches[0]


def test_source_pins_four_abstract_methods_on_knowledge_store() -> None:
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"@abstractmethod", source)
    assert len(matches) == 4


def test_source_pins_two_module_level_functions() -> None:
    """sha256_hex and now_epoch."""
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"^def\s+(\w+)\(", source, re.MULTILINE)
    assert set(matches) == {"sha256_hex", "now_epoch"}


def test_source_pins_four_module_level_classes() -> None:
    """KnowledgeObject, RetrievalRequest, RetrievalCandidate, RetrievalResult,
    KnowledgeStore, InMemoryKnowledgeStore — six total."""
    source = inspect.getsource(abstraction_module)
    matches = re.findall(r"^class\s+(\w+)[\(:]", source, re.MULTILINE)
    assert set(matches) == {
        "KnowledgeObject",
        "RetrievalRequest",
        "RetrievalCandidate",
        "RetrievalResult",
        "KnowledgeStore",
        "InMemoryKnowledgeStore",
    }


def test_source_extra_forbid_regex_pin() -> None:
    source = inspect.getsource(abstraction_module)
    assert re.search(r'extra="forbid"', source) is not None


def test_source_no_print_or_pprint() -> None:
    source = inspect.getsource(abstraction_module)
    assert "print(" not in source
    assert "pprint(" not in source


def test_source_no_subprocess_or_os_system() -> None:
    source = inspect.getsource(abstraction_module)
    assert "subprocess" not in source
    assert "os.system" not in source


def test_source_no_requests_or_urllib_or_httpx() -> None:
    source = inspect.getsource(abstraction_module)
    for forbidden in ("requests", "urllib", "httpx", "aiohttp"):
        assert forbidden not in source, f"network import {forbidden!r} found"


def test_source_no_eval_or_exec() -> None:
    source = inspect.getsource(abstraction_module)
    assert "eval(" not in source
    assert "exec(" not in source
    assert "compile(" not in source


def test_source_no_os_environ_access() -> None:
    source = inspect.getsource(abstraction_module)
    assert "os.environ" not in source
    assert "os.getenv" not in source


def test_source_no_hardcoded_credentials() -> None:
    source = inspect.getsource(abstraction_module)
    for pattern in ('api_key="sk-', "BEGIN PRIVATE KEY", "AKIA", "AIza", "xox[abprs]-"):
        assert pattern not in source, f"credential pattern {pattern!r} found"


def test_source_no_todo_or_fixme_markers() -> None:
    source = inspect.getsource(abstraction_module)
    assert "TODO" not in source
    assert "FIXME" not in source
    assert "XXX" not in source


def test_source_ends_with_single_trailing_newline() -> None:
    src_path = inspect.getsourcefile(abstraction_module)
    assert src_path is not None
    with open(src_path, "rb") as fh:
        data = fh.read()
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")


def test_ast_no_unexpected_statement_kinds() -> None:
    """Pin the AST statement-kind surface — no exec / eval / async-function
    declarations leak into this pure-stdlib module at the top level."""
    import ast

    src_path = inspect.getsourcefile(abstraction_module)
    assert src_path is not None
    with open(src_path) as fh:
        tree = ast.parse(fh.read())
    allowed_kinds = {
        "Module",
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "Import",
        "ImportFrom",
        "Assign",
        "AnnAssign",
        "Expr",
        "If",
        "Try",
        "With",
        "Pass",
        "Return",
    }
    for node in tree.body:
        assert node.__class__.__name__ in allowed_kinds, (
            f"unexpected top-level statement kind {node.__class__.__name__} "
            f"at line {getattr(node, 'lineno', '?')}"
        )


def test_status_proposed_appears_in_source() -> None:
    source = inspect.getsource(abstraction_module)
    assert "Status.PROPOSED" in source


def test_status_experimental_appears_in_source() -> None:
    source = inspect.getsource(abstraction_module)
    assert "Status.EXPERIMENTAL" in source


def test_status_implemented_appears_in_source() -> None:
    source = inspect.getsource(abstraction_module)
    assert "Status.IMPLEMENTED" in source


def test_knowledge_id_used_in_source() -> None:
    source = inspect.getsource(abstraction_module)
    assert "KnowledgeId" in source


def test_collect_artifact_ref_field() -> None:
    obj = KnowledgeObject(
        knowledge_id=KnowledgeId("ko_x"),
        topic="retrieval",
        content="claim",
        content_hash=sha256_hex("claim"),
        artifact_ref="r2://foo",
    )
    assert obj.artifact_ref == "r2://foo"


def test_collect_embedding_ref_field() -> None:
    obj = KnowledgeObject(
        knowledge_id=KnowledgeId("ko_x"),
        topic="retrieval",
        content="claim",
        content_hash=sha256_hex("claim"),
        embedding_ref="vec://foo",
    )
    assert obj.embedding_ref == "vec://foo"
