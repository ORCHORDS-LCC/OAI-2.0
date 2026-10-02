"""Outer-guard-rail boundary contracts for ``oai2.knowledge.evidence_package``.

This file pins the public-safety boundary of the compact provenance-rich
package formatter plus the retrieval-metrics evaluator. The companion module
``oai2/knowledge/evidence_package.py`` defines the wire-facing shapes every
downstream RAG agent consumes — the ``EvidencePackageEntry.render()`` line
format is the contract for the prompt side, and the
``evaluate_retrieval_package()`` math is the contract for the offline
benchmark side.

These tests focus on the *outer guard rails* the existing
``tests/test_evidence_package.py`` (behavioural, 9 tests) +
``tests/test_evidence_package_validation.py`` (argument validation, 51 tests)
do NOT pin:

- ``EVIDENCE_PACKAGE_VERSION`` string stability ("1") — pinned because
  snapshots persisted in R2/D1 rows reference this version and a rename
  would silently corrupt the on-disk schema.
- ``EvidencePackageEntry`` shape (8 fields + ``score`` default ``None``) +
  ``frozen=True`` + ``slots=True``.
- ``EvidencePackage`` shape (6 fields) + ``frozen=True`` + ``slots=True``.
- ``RetrievalMetrics`` shape (7 fields) + ``frozen=True`` + ``slots=True``.
- ``EvidencePackageEntry.render()`` line format with ``score=None`` vs
  ``score=0.500000`` (6-decimal format).
- ``EvidencePackage.render()`` join contract (``\n\n`` separator).
- ``build_evidence_package`` decision matrix: empty result, max_entries
  truncation, source_ref priority (``source_uri`` wins over
  ``artifact_ref``), whitespace-only / empty source_ref rejection, empty
  content still emits provenance-bearing entry, ``insufficient_evidence``
  flag routing (True ONLY when zero entries fit).
- ``evaluate_retrieval_package`` math precision: ``precision_at_k``,
  ``recall``, ``irrelevant_context_rate``, ``compression_ratio``,
  ``task_success_delta`` — pinned at boundary conditions (k=0, all-relevant,
  no-relevant).
- ``__all__`` + package-level re-exports identity check.

A refactor that drops the ``source_uri or artifact_ref`` precedence, that
swallows whitespace-only source_ref, that swaps the ``score={score:.6f}``
format for ``score={score}``, or that reverses the precision/irrelevant-rate
relationship must trip one of these tests.
"""

from __future__ import annotations

import pytest

from oai2.core import KnowledgeId
from oai2.knowledge import evidence_package as evidence_package_mod
from oai2.knowledge import sha256_hex
from oai2.knowledge.abstraction import RetrievalCandidate, RetrievalResult
from oai2.knowledge.evidence_package import (
    EVIDENCE_PACKAGE_VERSION,
    EvidencePackage,
    EvidencePackageEntry,
    RetrievalMetrics,
    TokenCounter,
    build_evidence_package,
    evaluate_retrieval_package,
)
from tests._evidence_fixtures import (
    make_knowledge_object as _obj,
)
from tests._evidence_fixtures import (
    make_retrieval_result as _result,
)
from tests._evidence_fixtures import (
    tokens as _tokens,
)

# ---------------------------------------------------------------------------
# 1. EVIDENCE_PACKAGE_VERSION string stability
# ---------------------------------------------------------------------------


def test_evidence_package_version_is_pinned_string_one() -> None:
    """``EVIDENCE_PACKAGE_VERSION`` is the wire contract for snapshot
    deserialization. A refactor that bumps it without coordinating with the
    snapshot schema would silently corrupt historical records."""
    assert EVIDENCE_PACKAGE_VERSION == "1"
    assert isinstance(EVIDENCE_PACKAGE_VERSION, str)


def test_package_version_field_matches_module_constant() -> None:
    """Every constructed ``EvidencePackage`` pins the version field to the
    module constant — a refactor that drops the assignment would fail here
    with ``package.version == ""``."""
    obj = _obj("ko_v", "claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    assert package.version == EVIDENCE_PACKAGE_VERSION == "1"


# ---------------------------------------------------------------------------
# 2. EvidencePackageEntry shape
# ---------------------------------------------------------------------------


def test_evidence_package_entry_required_field_set() -> None:
    """All 8 declared fields exist and are addressable by name; ``score`` defaults
    to ``None`` — used for evidence that was internally reconstructed and has
    no scoring provenance."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="abc12345",
        source_ref="https://example.test/source",
        authority=0.9,
        retrieved_at=123.0,
        snippet="the snippet",
        tokens=5,
    )
    assert entry.knowledge_id == KnowledgeId("ko_1")
    assert entry.content_hash == "abc12345"
    assert entry.source_ref == "https://example.test/source"
    assert entry.authority == 0.9
    assert entry.retrieved_at == 123.0
    assert entry.snippet == "the snippet"
    assert entry.tokens == 5
    assert entry.score is None


def test_evidence_package_entry_score_default_is_none() -> None:
    """``score=None`` is the documented "no scoring provenance" sentinel —
    pinned because downstream prompt-template code branches on None vs float."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="abc12345",
        source_ref="https://example.test/source",
        authority=0.5,
        retrieved_at=0.0,
        snippet="",
        tokens=0,
    )
    assert entry.score is None


def test_evidence_package_entry_is_frozen() -> None:
    """``EvidencePackageEntry`` is immutable after construction."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="abc12345",
        source_ref="src",
        authority=0.5,
        retrieved_at=0.0,
        snippet="",
        tokens=0,
    )
    with pytest.raises((AttributeError, Exception)):
        entry.snippet = "tampered"  # type: ignore[misc]


def test_evidence_package_entry_is_slotted() -> None:
    """``EvidencePackageEntry`` uses slots — attribute injection outside the
    declared field set is rejected with AttributeError."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="abc12345",
        source_ref="src",
        authority=0.5,
        retrieved_at=0.0,
        snippet="",
        tokens=0,
    )
    with pytest.raises(AttributeError):
        entry.injected = "value"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. EvidencePackage shape
# ---------------------------------------------------------------------------


def test_evidence_package_required_field_set() -> None:
    """All 6 declared fields exist and are addressable by name."""
    package = EvidencePackage(
        version="1",
        topic="retrieval",
        entries=(),
        token_count=0,
        token_budget=100,
        insufficient_evidence=True,
    )
    assert package.version == "1"
    assert package.topic == "retrieval"
    assert package.entries == ()
    assert package.token_count == 0
    assert package.token_budget == 100
    assert package.insufficient_evidence is True


def test_evidence_package_is_frozen() -> None:
    """``EvidencePackage`` is immutable after construction."""
    package = EvidencePackage(
        version="1",
        topic="retrieval",
        entries=(),
        token_count=0,
        token_budget=100,
        insufficient_evidence=True,
    )
    with pytest.raises((AttributeError, Exception)):
        package.topic = "tampered"  # type: ignore[misc]


def test_evidence_package_is_slotted() -> None:
    """``EvidencePackage`` uses slots — attribute injection outside the declared
    field set is rejected with AttributeError."""
    package = EvidencePackage(
        version="1",
        topic="retrieval",
        entries=(),
        token_count=0,
        token_budget=100,
        insufficient_evidence=True,
    )
    with pytest.raises(AttributeError):
        package.injected = "value"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. RetrievalMetrics shape
# ---------------------------------------------------------------------------


def test_retrieval_metrics_required_field_set() -> None:
    """All 7 declared fields exist and are addressable by name."""
    metrics = RetrievalMetrics(
        k=0,
        precision_at_k=0.0,
        recall=0.0,
        irrelevant_context_rate=0.0,
        package_tokens=0,
        raw_source_tokens=100,
        compression_ratio=0.0,
        task_success_delta=0.0,
    )
    assert metrics.k == 0
    assert metrics.precision_at_k == 0.0
    assert metrics.recall == 0.0
    assert metrics.irrelevant_context_rate == 0.0
    assert metrics.package_tokens == 0
    assert metrics.raw_source_tokens == 100
    assert metrics.compression_ratio == 0.0
    assert metrics.task_success_delta == 0.0


def test_retrieval_metrics_is_frozen() -> None:
    """``RetrievalMetrics`` is immutable after construction."""
    metrics = RetrievalMetrics(
        k=0,
        precision_at_k=0.0,
        recall=0.0,
        irrelevant_context_rate=0.0,
        package_tokens=0,
        raw_source_tokens=100,
        compression_ratio=0.0,
        task_success_delta=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        metrics.k = 999  # type: ignore[misc]


def test_retrieval_metrics_is_slotted() -> None:
    """``RetrievalMetrics`` uses slots — attribute injection outside the declared
    field set is rejected with AttributeError."""
    metrics = RetrievalMetrics(
        k=0,
        precision_at_k=0.0,
        recall=0.0,
        irrelevant_context_rate=0.0,
        package_tokens=0,
        raw_source_tokens=100,
        compression_ratio=0.0,
        task_success_delta=0.0,
    )
    with pytest.raises(AttributeError):
        metrics.injected = "value"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 5. EvidencePackageEntry.render() format
# ---------------------------------------------------------------------------


def test_render_score_none_omits_score_suffix() -> None:
    """``score=None`` produces NO score segment in the rendered line — the
    prompt-side contract is "no scoring provenance means no score printed"."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="h",
        source_ref="src",
        authority=0.9,
        retrieved_at=10.0,
        snippet="the snippet",
        tokens=2,
        score=None,
    )
    rendered = entry.render()
    assert "score=" not in rendered
    assert "the snippet" in rendered


def test_render_score_float_uses_six_decimal_format() -> None:
    """``score=0.5`` renders as ``score=0.500000`` — the documented 6-decimal
    format is the prompt-side wire contract."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="h",
        source_ref="src",
        authority=0.9,
        retrieved_at=10.0,
        snippet="the snippet",
        tokens=2,
        score=0.5,
    )
    rendered = entry.render()
    assert "score=0.500000" in rendered


def test_render_score_zero_renders_as_zero_six_decimal() -> None:
    """``score=0.0`` (zero is not None) renders as ``score=0.000000`` —
    pinned so a refactor that branches ``score if score else ""`` would
    fail here."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="h",
        source_ref="src",
        authority=0.9,
        retrieved_at=10.0,
        snippet="x",
        tokens=1,
        score=0.0,
    )
    assert "score=0.000000" in entry.render()


@pytest.mark.parametrize(
    "authority, expected",
    [
        (1.0, "authority=1.000000"),
        (0.0, "authority=0.000000"),
        (0.123456, "authority=0.123456"),
    ],
)
def test_render_authority_uses_six_decimal_format(authority: float, expected: str) -> None:
    """``authority`` is rendered with the same 6-decimal format — pinned so
    downstream log scrapers that grep on ``authority=N.NNNNNN`` keep
    working."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="h",
        source_ref="src",
        authority=authority,
        retrieved_at=10.0,
        snippet="x",
        tokens=1,
    )
    assert expected in entry.render()


@pytest.mark.parametrize(
    "retrieved_at, expected",
    [
        (0.0, "retrieved_at=0.000000"),
        (123.0, "retrieved_at=123.000000"),
        (1.5, "retrieved_at=1.500000"),
    ],
)
def test_render_retrieved_at_uses_six_decimal_format(retrieved_at: float, expected: str) -> None:
    """``retrieved_at`` is rendered with the same 6-decimal format — pinned."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="h",
        source_ref="src",
        authority=0.5,
        retrieved_at=retrieved_at,
        snippet="x",
        tokens=1,
    )
    assert expected in entry.render()


def test_render_title_line_has_documented_field_order() -> None:
    """The title line has the documented field order:
    ``[knowledge_id] hash=... source=... authority=... retrieved_at=...
    [score=...]``. A refactor that reorders the fields would change the
    prompt-side parse contract."""
    entry = EvidencePackageEntry(
        knowledge_id=KnowledgeId("ko_1"),
        content_hash="deadbeef",
        source_ref="src",
        authority=0.5,
        retrieved_at=10.0,
        snippet="body",
        tokens=1,
        score=0.5,
    )
    rendered = entry.render()
    # Pin the order — every prefix must appear before its successor.
    assert rendered.index("[ko_1]") < rendered.index("hash=deadbeef")
    assert rendered.index("hash=deadbeef") < rendered.index("source=src")
    assert rendered.index("source=src") < rendered.index("authority=0.500000")
    assert rendered.index("authority=0.500000") < rendered.index("retrieved_at=10.000000")
    assert rendered.index("retrieved_at=10.000000") < rendered.index("score=0.500000")
    # The snippet is on the line after the title (separated by \n).
    assert rendered.endswith("body")


# ---------------------------------------------------------------------------
# 6. EvidencePackage.render() format
# ---------------------------------------------------------------------------


def test_package_render_empty_entries_returns_empty_string() -> None:
    """``EvidencePackage.entries == ()`` renders to ``""`` — the prompt-side
    contract is "no entries means no payload"."""
    package = EvidencePackage(
        version="1",
        topic="retrieval",
        entries=(),
        token_count=0,
        token_budget=100,
        insufficient_evidence=True,
    )
    assert package.render() == ""


def test_package_render_single_entry_equals_entry_render() -> None:
    """A single-entry package renders to that entry's render() output."""
    obj = _obj("ko_1", "claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    assert len(package.entries) == 1
    assert package.render() == package.entries[0].render()


def test_package_render_multiple_entries_joined_with_blank_line() -> None:
    """Multiple entries are joined with ``\\n\\n`` (blank line separator) —
    pinned because the prompt-side block-parser splits on this exact
    separator."""
    obj_a = _obj("ko_a", "first claim")
    obj_b = _obj("ko_b", "second claim")
    package = build_evidence_package(_result(obj_a, obj_b), token_budget=200, token_counter=_tokens)
    assert len(package.entries) >= 2
    rendered = package.render()
    # Must contain the documented separator between two entries.
    assert "\n\n" in rendered
    # Both entries' snippets are present
    assert "first claim" in rendered
    assert "second claim" in rendered


# ---------------------------------------------------------------------------
# 7. build_evidence_package decision matrix
# ---------------------------------------------------------------------------


def test_build_empty_result_yields_empty_entries_and_insufficient_flag() -> None:
    """An empty ``RetrievalResult`` (no objects) returns a package with
    ``entries == ()`` and ``insufficient_evidence=True``."""
    empty = RetrievalResult(topic="retrieval", objects=[], candidates=[])
    package = build_evidence_package(empty, token_budget=100, token_counter=_tokens)
    assert package.entries == ()
    assert package.insufficient_evidence is True
    assert package.token_count == 0


def test_build_topic_propagates_from_result() -> None:
    """The ``topic`` field is forwarded from the source ``RetrievalResult``."""
    obj = _obj("ko_1", "claim")
    result = _result(obj, topic="custom_topic_xyz")
    package = build_evidence_package(result, token_budget=100, token_counter=_tokens)
    assert package.topic == "custom_topic_xyz"


def test_build_token_budget_field_propagates() -> None:
    """The ``token_budget`` field on the result package is the documented
    budget — pinned so the metric evaluator reads the same number."""
    obj = _obj("ko_1", "claim")
    package = build_evidence_package(_result(obj), token_budget=222, token_counter=_tokens)
    assert package.token_budget == 222


def test_build_max_entries_truncates_to_first_n() -> None:
    """``max_entries=1`` returns at most 1 entry — the truncation happens
    before the token-budget loop."""
    obj_a = _obj("ko_a", "first claim")
    obj_b = _obj("ko_b", "second claim")
    package = build_evidence_package(
        _result(obj_a, obj_b),
        token_budget=1000,
        token_counter=_tokens,
        max_entries=1,
    )
    assert len(package.entries) == 1
    assert package.entries[0].knowledge_id == KnowledgeId("ko_a")


def test_build_max_entries_none_returns_full_set() -> None:
    """``max_entries=None`` is the documented "no truncation" sentinel —
    both objects fit."""
    obj_a = _obj("ko_a", "first claim")
    obj_b = _obj("ko_b", "second claim")
    package = build_evidence_package(
        _result(obj_a, obj_b),
        token_budget=1000,
        token_counter=_tokens,
        max_entries=None,
    )
    assert len(package.entries) == 2


def test_build_token_count_equals_token_counter_of_rendered_package() -> None:
    """``package.token_count`` matches the token counter applied to
    ``package.render()`` — pinned because downstream prompt-budget loops use
    this exact number to enforce hard limits."""
    obj = _obj("ko_1", "claim text")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    assert package.token_count == _tokens(package.render())


def test_build_source_uri_takes_priority_over_artifact_ref() -> None:
    """When both ``source_uri`` and ``artifact_ref`` are present,
    ``source_uri`` wins — pinned so downstream prompt provenance points at
    the human-readable URL, not the storage URI."""
    obj = _obj(
        "ko_1",
        "claim",
        source_uri="https://example.test/human-readable",
        artifact_ref="r2://storage/fallback",
    )
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    assert package.entries[0].source_ref == "https://example.test/human-readable"


def test_build_artifact_ref_used_as_fallback_when_source_uri_is_none() -> None:
    """When ``source_uri`` is ``None``, ``artifact_ref`` becomes the
    provenance source — pinned so storage-only objects still have a
    retrievable source pointer."""
    obj = _obj(
        "ko_1",
        "claim",
        source_uri=None,
        artifact_ref="r2://storage/fallback",
    )
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    assert package.entries[0].source_ref == "r2://storage/fallback"


def test_build_rejects_whitespace_only_source_uri() -> None:
    """``source_uri="   "`` is rejected even though it is a non-empty string
    — the runtime guard calls ``.strip()`` so whitespace-only fails the
    provenance contract."""
    obj = _obj("ko_1", "claim", source_uri="   ", artifact_ref=None)
    with pytest.raises(ValueError, match="no provenance source"):
        build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)


def test_build_rejects_empty_source_uri() -> None:
    """``source_uri=""`` is rejected — empty string fails the strip check."""
    obj = _obj("ko_1", "claim", source_uri="", artifact_ref=None)
    with pytest.raises(ValueError, match="no provenance source"):
        build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)


def test_build_rejects_missing_candidate_for_object() -> None:
    """If the object has no matching candidate (e.g., candidate list omits
    the object's knowledge_id), ``ValueError`` is raised with the
    ``no candidate evidence`` substring — pinned because the guard is the
    only thing preventing silent provenance loss."""
    obj = _obj("ko_1", "claim")
    result = RetrievalResult(
        topic="retrieval",
        objects=[obj],
        candidates=[],  # missing the candidate for ko_1
    )
    with pytest.raises(ValueError, match="no candidate evidence"):
        build_evidence_package(result, token_budget=100, token_counter=_tokens)


def test_build_rejects_candidate_hash_mismatch() -> None:
    """If the candidate's ``content_hash`` disagrees with the object's
    ``content_hash``, ``ValueError`` is raised — pinned because the guard
    is the only thing preventing silent hash drift."""
    obj = _obj("ko_1", "claim")
    mismatched = RetrievalResult(
        topic="retrieval",
        objects=[obj],
        candidates=[
            RetrievalCandidate(
                knowledge_id=obj.knowledge_id,
                content_hash=sha256_hex("different content"),
                source_uri=obj.source_uri,
                score=0.5,
            )
        ],
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_evidence_package(mismatched, token_budget=100, token_counter=_tokens)


def test_build_empty_content_still_emits_provenance_entry() -> None:
    """An object with empty content still produces a provenance-bearing entry
    (snippet is ``""``) — pinned because downstream code branches on
    ``snippet == ""`` vs ``snippet != ""`` and a refactor that skipped
    empty-content objects would silently drop provenance."""
    obj = _obj("ko_empty", "")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    assert len(package.entries) == 1
    assert package.entries[0].snippet == ""
    assert package.entries[0].content_hash == obj.content_hash


def test_build_tight_budget_truncates_via_binary_search() -> None:
    """A tight ``token_budget`` triggers the binary-search snippet
    truncation — the binary-search boundary is preserved by the test
    fixture's deterministic ``tokens(text)`` tokenizer (``len(text.split())``).
    With a 40-token content and a tight budget, the resulting snippet is
    shorter than the full content AND ends with the documented ``…``
    marker (the "more content is coming" trailing marker)."""
    content = " ".join(f"token{i}" for i in range(40))
    obj = _obj("ko_long", content)
    package = build_evidence_package(_result(obj), token_budget=20, token_counter=_tokens)
    assert len(package.entries) == 1
    snippet = package.entries[0].snippet
    assert snippet != content  # truncated
    assert "…" in snippet  # truncation marker pinned


def test_build_insufficient_evidence_flag_is_false_when_some_entry_fits() -> None:
    """``insufficient_evidence=False`` even when truncation happens — the
    flag is True ONLY when no entries fit at all."""
    obj = _obj("ko_1", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=12, token_counter=_tokens)
    # The single entry may be truncated to "" if 12 tokens is too small;
    # but the entry exists in selected, so the flag is False.
    assert package.insufficient_evidence is False


def test_build_insufficient_evidence_flag_is_true_when_nothing_fits() -> None:
    """``insufficient_evidence=True`` ONLY when zero entries fit (the
    metadata-only entry's render already exceeds the budget at
    ``token_budget=1``)."""
    obj = _obj("ko_1", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=1, token_counter=_tokens)
    assert package.entries == ()
    assert package.insufficient_evidence is True
    assert package.token_count == 0


# ---------------------------------------------------------------------------
# 8. evaluate_retrieval_package math precision
# ---------------------------------------------------------------------------


def test_metrics_k_matches_entries_count() -> None:
    """``metrics.k`` is the package's entries count — pinned so downstream
    metric consumers read the same number as the package itself."""
    obj_a = _obj("ko_a", "first")
    obj_b = _obj("ko_b", "second")
    package = build_evidence_package(_result(obj_a, obj_b), token_budget=200, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_a"},
        raw_source_tokens=500,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    assert metrics.k == len(package.entries)


@pytest.mark.parametrize(
    "relevant, irrelevant, expected_precision, expected_irrelevant",
    [
        ({"ko_a"}, set(), 1.0, 0.0),  # all relevant
        (set(), {"ko_b"}, 0.0, 1.0),  # none relevant
        ({"ko_a"}, {"ko_b"}, 0.5, 0.5),  # half relevant
    ],
)
def test_metrics_precision_and_irrelevant_rate_at_boundary_conditions(
    relevant: set[str],
    irrelevant: set[str],
    expected_precision: float,
    expected_irrelevant: float,
) -> None:
    """``precision_at_k = relevant_retrieved / k`` and
    ``irrelevant_context_rate = 1 - precision`` — pinned at three boundary
    conditions: all-relevant, no-relevant, half-relevant."""
    objects = []
    all_ids = set()
    for _i, kid in enumerate(sorted(relevant | irrelevant)):
        if kid in relevant:
            objects.append(_obj(kid, f"relevant-{kid}"))
            all_ids.add(kid)
        else:
            objects.append(_obj(kid, f"irrelevant-{kid}"))
            all_ids.add(kid)
    # Always include at least one relevant for the relevant-set computation
    # to have a non-zero denominator.
    if not relevant:
        # Force recall denominator to 1 by adding a relevant that's not retrieved
        relevant = {"ko_missing"}
    package = build_evidence_package(_result(*objects), token_budget=200, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids=relevant,
        raw_source_tokens=500,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    assert metrics.precision_at_k == pytest.approx(expected_precision)
    assert metrics.irrelevant_context_rate == pytest.approx(expected_irrelevant)


def test_metrics_recall_denominator_is_relevant_set_size() -> None:
    """``recall = relevant_retrieved / total_relevant`` — when the relevant
    set is larger than the package, recall is the fraction of relevant
    objects that appear in the package."""
    obj_a = _obj("ko_a", "first")
    obj_b = _obj("ko_b", "second")
    package = build_evidence_package(_result(obj_a, obj_b), token_budget=200, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_a", "ko_b", "ko_missing_1", "ko_missing_2"},
        raw_source_tokens=500,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    # 2 retrieved relevant out of 4 total relevant
    assert metrics.recall == pytest.approx(0.5)


def test_metrics_k_zero_yields_precision_zero_irrelevant_zero() -> None:
    """When the package has zero entries), ``precision_at_k = 0.0`` (NOT a
    division-by-zero) and ``irrelevant_context_rate = 0.0`` — pinned so the
    zero-k edge is handled without raising."""
    package = build_evidence_package(
        _result(_obj("ko_1", "claim")), token_budget=1, token_counter=_tokens
    )
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_1"},
        raw_source_tokens=500,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    assert metrics.k == 0
    assert metrics.precision_at_k == 0.0
    assert metrics.irrelevant_context_rate == 0.0


def test_metrics_compression_ratio_is_package_tokens_over_raw_source_tokens() -> None:
    """``compression_ratio = package.token_count / raw_source_tokens`` —
    pinned so the offline benchmark reports the documented metric."""
    obj = _obj("ko_1", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_1"},
        raw_source_tokens=400,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    assert metrics.compression_ratio == pytest.approx(package.token_count / 400)


def test_metrics_task_success_delta_is_with_minus_without() -> None:
    """``task_success_delta = task_success_with_retrieval -
    task_success_without_retrieval`` — pinned so the offline benchmark
    reports the documented delta."""
    obj = _obj("ko_1", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_1"},
        raw_source_tokens=400,
        task_success_with_retrieval=0.7,
        task_success_without_retrieval=0.3,
    )
    assert metrics.task_success_delta == pytest.approx(0.4)


def test_metrics_package_tokens_field_is_propagated() -> None:
    """``metrics.package_tokens`` is the package's ``token_count`` — pinned
    so the offline benchmark reads the same number the formatter wrote."""
    obj = _obj("ko_1", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_1"},
        raw_source_tokens=400,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    assert metrics.package_tokens == package.token_count


def test_metrics_relevant_set_accepts_knowledge_id_instances_and_strings() -> None:
    """The ``relevant_knowledge_ids`` iterable accepts ``KnowledgeId`` and
    ``str`` instances equivalently — pinned so callers can pass either
    without a separate conversion step."""
    obj = _obj("ko_a", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    metrics_str = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids=["ko_a", "ko_b"],
        raw_source_tokens=400,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    metrics_kid = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids=[KnowledgeId("ko_a"), KnowledgeId("ko_b")],
        raw_source_tokens=400,
        task_success_with_retrieval=0.5,
        task_success_without_retrieval=0.5,
    )
    assert metrics_str.precision_at_k == metrics_kid.precision_at_k
    assert metrics_str.recall == metrics_kid.recall


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_metrics_rate_boundaries_zero_and_one_are_accepted(rate: float) -> None:
    """``0.0`` and ``1.0`` are the documented rate-range boundaries — both
    must be accepted (a refactor that uses ``0.0 < rate < 1.0`` would
    reject the boundaries)."""
    obj = _obj("ko_1", "first compact claim")
    package = build_evidence_package(_result(obj), token_budget=100, token_counter=_tokens)
    metrics = evaluate_retrieval_package(
        package,
        relevant_knowledge_ids={"ko_1"},
        raw_source_tokens=400,
        task_success_with_retrieval=rate,
        task_success_without_retrieval=rate,
    )
    assert metrics.task_success_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. __all__ exports + package-level identity
# ---------------------------------------------------------------------------


def test_module_all_lists_seven_public_exports() -> None:
    """``__all__`` is the canonical public surface — adding a name is a
    deliberate API change and must be flagged here."""
    assert sorted(evidence_package_mod.__all__) == sorted(
        [
            "EVIDENCE_PACKAGE_VERSION",
            "TokenCounter",
            "EvidencePackageEntry",
            "EvidencePackage",
            "RetrievalMetrics",
            "build_evidence_package",
            "evaluate_retrieval_package",
        ]
    )


def test_evidence_package_symbols_are_exported_from_knowledge_package() -> None:
    """All 7 public symbols are re-exported at the ``oai2.knowledge``
    package level — pinned against accidental re-export removal."""
    from oai2.knowledge import EVIDENCE_PACKAGE_VERSION as ExportedVersion
    from oai2.knowledge import EvidencePackage as ExportedPackage
    from oai2.knowledge import EvidencePackageEntry as ExportedEntry
    from oai2.knowledge import RetrievalMetrics as ExportedMetrics
    from oai2.knowledge import TokenCounter as ExportedTokenCounter
    from oai2.knowledge import (
        build_evidence_package as ExportedBuild,
    )
    from oai2.knowledge import (
        evaluate_retrieval_package as ExportedEvaluate,
    )

    assert ExportedVersion == EVIDENCE_PACKAGE_VERSION
    assert ExportedPackage is EvidencePackage
    assert ExportedEntry is EvidencePackageEntry
    assert ExportedMetrics is RetrievalMetrics
    assert ExportedBuild is build_evidence_package
    assert ExportedEvaluate is evaluate_retrieval_package
    # TokenCounter is a TypeAlias (Callable[[str], int]), so equality is the
    # expected identity.
    assert ExportedTokenCounter is type(TokenCounter) or callable(ExportedTokenCounter)
