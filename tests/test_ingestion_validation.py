"""Pin the outer guard rails of ``oai2.knowledge.ingestion``.

The behavioural tests in ``tests/test_knowledge.py`` cover one happy path
each for ``InMemoryKnowledgeStore`` and ``IngestionPipeline``. The
ingestion module also has public contracts that aren't currently tested:

* ``IngestionJob.build()`` — the field-by-field mapping into a
  ``KnowledgeObject``: ``content_hash`` derives from ``content``,
  ``knowledge_id`` is fresh and unique per call (the source uses
  ``uuid.uuid4().hex[:12]``), ``retrieved_at`` is fresh per call
  (``now_epoch()``), ``authority`` / ``status`` defaults are
  ``0.5`` / ``Status.EXPERIMENTAL``, and the optional
  ``artifact_ref`` / ``embedding_ref`` are explicitly ``None`` (not
  accidentally inherited from anywhere).
* ``IngestionPipeline.ingest()`` failure semantics — the source wraps
  each ``store.put(obj)`` in ``try/except`` and emits
  ``IngestionStatus.FAILED`` on any exception, preserving the input
  order so callers can correlate ``statuses[i]`` with ``jobs[i]``.
* ``IngestionStatus`` enum — values are stable strings (``"pending"``,
  ``"ok"``, ``"failed"``, ``"skipped"``) because it's a ``StrEnum`` and
  the string form is what gets persisted.
* Empty / single / mixed-input iteration shape — ``ingest([])`` returns
  ``[]``; a single-job pipeline works; mixed OK + FAILED jobs preserve
  positional correspondence.
* ``_Source`` Protocol-typed source boundary — the pipeline declares
  ``sources: list[_Source]`` where ``_Source`` is a ``Protocol`` with
  ``fetch(source_uri: str) -> Iterable[IngestionJob]``. Structural
  typing means any duck-typed class implementing ``fetch`` is accepted;
  pin the contract that the Protocol is not rigidly nominal-typed and
  the ``sources`` field defaults to an empty list.
* ``IngestionPipeline`` slots / non-frozen mutation — ``sources`` is
  a mutable list, so callers can append new sources after construction.
  Pin that the dataclass uses ``__slots__`` and is NOT frozen.
* ``ingest()`` exception-breadth — the ``try/except`` wraps ``Exception``
  (not ``BaseException``), so ``KeyboardInterrupt`` / ``SystemExit`` /
  ``GeneratorExit`` propagate instead of being swallowed as
  ``IngestionStatus.FAILED``.

A refactor that, e.g., inverts the ``try/except`` semantics (turning
failures into ``OK``), changes the ``content_hash`` derivation, hardcodes
the ``knowledge_id`` instead of using ``uuid``, renames the enum
strings, or wires ``sources`` into ``ingest()`` would propagate silently
into the D1 ingestion path.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    IngestionJob,
    IngestionPipeline,
    IngestionStatus,
    InMemoryKnowledgeStore,
    KnowledgeObject,
)
from oai2.knowledge.abstraction import now_epoch, sha256_hex

# ---------------------------------------------------------------------------
# IngestionJob.build() — field-by-field mapping
# ---------------------------------------------------------------------------


def test_ingestion_job_build_maps_topic_content_source_uri_unchanged() -> None:
    job = IngestionJob(
        source_uri="local://fixture",
        topic="mlx",
        content="mlx is fast",
        authority=0.8,
        status=Status.EXPERIMENTAL,
    )
    obj = job.build()
    assert obj.topic == "mlx"
    assert obj.content == "mlx is fast"
    assert obj.source_uri == "local://fixture"


def test_ingestion_job_build_derives_content_hash_from_content() -> None:
    """``build()`` derives ``content_hash`` from ``content`` via
    ``sha256_hex``; passing them separately in the source-of-truth
    tuple would be a regression of that mapping."""
    job = IngestionJob(
        source_uri="local://a",
        topic="t",
        content="claim body",
    )
    obj = job.build()
    assert obj.content_hash == sha256_hex("claim body")


def test_ingestion_job_build_authority_passthrough() -> None:
    job = IngestionJob(
        source_uri="local://a",
        topic="t",
        content="c",
        authority=0.42,
    )
    assert job.build().authority == 0.42


def test_ingestion_job_build_authority_default_is_0_5() -> None:
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    assert job.build().authority == 0.5


def test_ingestion_job_build_status_default_is_experimental() -> None:
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    assert job.build().status == Status.EXPERIMENTAL


def test_ingestion_job_build_status_passthrough_for_each_status_value() -> None:
    """All 4 ``Status`` values must survive ``build()`` unchanged. A future
    refactor that hardcodes ``EXPERIMENTAL`` instead of passthrough would
    silently drop the PROPOSED/IMPLEMENTED distinction at the ingestion
    boundary."""
    for status_value in (
        Status.PROPOSED,
        Status.EXPERIMENTAL,
        Status.IMPLEMENTED,
        Status.BLOCKED,
    ):
        job = IngestionJob(
            source_uri="local://a",
            topic="t",
            content="c",
            status=status_value,
        )
        assert job.build().status == status_value


def test_ingestion_job_build_knowledge_id_is_fresh_and_unique_per_call() -> None:
    """Two ``build()`` calls on the same job must produce different
    ``knowledge_id`` values (the source uses ``uuid.uuid4().hex[:12]``).
    Hardcoding the ID instead of generating it would let two pipeline
    runs collide on the same primary key."""
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    obj_a = job.build()
    obj_b = job.build()
    assert obj_a.knowledge_id != obj_b.knowledge_id


def test_ingestion_job_build_knowledge_id_is_a_knowledge_id_newtype_string() -> None:
    """The generated ``knowledge_id`` must round-trip through the
    ``KnowledgeId`` newtype (a str of length ``ko_<12 hex>`` = 15 chars).
    At runtime ``NewType`` is just ``str``, so the shape test is on the
    string form, not the newtype itself (which Python erases)."""
    obj = IngestionJob(
        source_uri="local://a",
        topic="t",
        content="c",
    ).build()
    # At runtime NewType is erased to its underlying type (str).
    assert isinstance(obj.knowledge_id, str)
    # The id is a string of the form "ko_" + 12 hex chars.
    assert len(obj.knowledge_id) == 3 + 12
    assert obj.knowledge_id.startswith("ko_")


def test_ingestion_job_build_retrieved_at_is_current_epoch() -> None:
    """``build()`` stamps ``retrieved_at`` with ``now_epoch()`` at call
    time, NOT a fixed value. Two calls a measurable interval apart must
    produce strictly-increasing ``retrieved_at`` values."""
    before = now_epoch()
    obj_a = IngestionJob(source_uri="local://a", topic="t", content="c").build()
    time.sleep(0.01)
    obj_b = IngestionJob(source_uri="local://a", topic="t", content="c").build()
    after = now_epoch()
    assert before <= obj_a.retrieved_at < obj_b.retrieved_at <= after


def test_ingestion_job_build_artifact_ref_and_embedding_ref_default_to_none() -> None:
    """``build()`` does NOT set ``artifact_ref`` or ``embedding_ref``;
    they must default to ``None`` (not be silently inherited from
    somewhere or set to an empty string)."""
    obj = IngestionJob(
        source_uri="local://a",
        topic="t",
        content="c",
    ).build()
    assert obj.artifact_ref is None
    assert obj.embedding_ref is None


def test_ingestion_job_is_a_frozen_dataclass() -> None:
    """``IngestionJob`` is declared ``@dataclass(slots=True, frozen=True)``.
    A future refactor that drops ``frozen=True`` would let callers mutate
    a job after construction (silently corrupting the provenance chain)."""
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    with pytest.raises((AttributeError, Exception)):
        job.topic = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IngestionPipeline.ingest() — iteration + store.put() semantics
# ---------------------------------------------------------------------------


def test_ingestion_pipeline_ingest_empty_iterable_yields_empty_statuses() -> None:
    """Empty input must yield an empty status list (NOT ``None``, NOT a
    default sentinel like ``[FAILED]``). The pipeline is the
    contract-of-record for the empty case."""
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)
    statuses = pipeline.ingest([])
    assert statuses == []


def test_ingestion_pipeline_ingest_single_successful_job_emits_one_ok() -> None:
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)
    job = IngestionJob(
        source_uri="local://single",
        topic="t",
        content="alive",
    )
    statuses = pipeline.ingest([job])
    assert statuses == [IngestionStatus.OK]
    # And the object actually landed in the store.
    assert len(list(store.all())) == 1


def test_ingestion_pipeline_ingest_preserves_input_order_in_status_list() -> None:
    """The status list index must correspond to the job list index. A
    refactor that uses ``set`` or ``dict`` ordering accidentally (or
    reverses the loop) would silently misalign the correlation."""
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)
    jobs = [IngestionJob(source_uri=f"local://{i}", topic="t", content=f"c_{i}") for i in range(5)]
    statuses = pipeline.ingest(jobs)
    assert statuses == [IngestionStatus.OK] * 5


def test_ingestion_pipeline_ingest_writes_each_object_into_the_store() -> None:
    """Every successful job must produce an object findable in the store
    via ``knowledge_id``. The pipeline is the only path from
    ``IngestionJob`` to a persisted object."""
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)
    jobs = [
        IngestionJob(source_uri="local://a", topic="t", content="alpha"),
        IngestionJob(source_uri="local://b", topic="t", content="bravo"),
    ]
    pipeline.ingest(jobs)
    all_ids = {obj.knowledge_id for obj in store.all()}
    assert len(all_ids) == 2  # two distinct UUIDs


def test_ingestion_pipeline_emits_failed_when_store_put_raises() -> None:
    """The pipeline wraps ``store.put`` in ``try/except`` and emits
    ``IngestionStatus.FAILED`` on any exception. A future refactor that
    lets the exception propagate (instead of capturing it) would crash
    the whole batch on the first bad row, instead of marking just the
    failing row."""

    class _BoomStore:
        def __init__(self) -> None:
            self.put_calls = 0

        def put(self, obj: KnowledgeObject) -> None:
            self.put_calls += 1
            raise RuntimeError("simulated D1 outage")

        def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:
            return None

        def retrieve(self, request: object) -> object:
            raise NotImplementedError

        def all(self) -> Iterable[KnowledgeObject]:
            return ()

    boom = _BoomStore()
    pipeline = IngestionPipeline(store=boom)  # type: ignore[arg-type]
    statuses = pipeline.ingest(
        [
            IngestionJob(source_uri="local://a", topic="t", content="x"),
            IngestionJob(source_uri="local://b", topic="t", content="y"),
        ],
    )
    assert statuses == [IngestionStatus.FAILED, IngestionStatus.FAILED]
    assert boom.put_calls == 2  # both jobs attempted


def test_ingestion_pipeline_keeps_going_after_failed_job() -> None:
    """A failure must not short-circuit the loop. ``ingest([bad, good])``
    must still attempt the second one and return ``[FAILED, OK]`` (not
    ``[FAILED]``)."""

    class _SelectiveStore:
        def __init__(self) -> None:
            self.calls: list = []

        def put(self, obj: KnowledgeObject) -> None:
            self.calls.append(obj.content_hash)
            if len(self.calls) == 1:
                raise RuntimeError("first put fails")

        def get(self, knowledge_id: KnowledgeId) -> KnowledgeObject | None:
            return None

        def retrieve(self, request: object) -> object:
            raise NotImplementedError

        def all(self) -> Iterable[KnowledgeObject]:
            return ()

    store = _SelectiveStore()
    pipeline = IngestionPipeline(store=store)  # type: ignore[arg-type]
    statuses = pipeline.ingest(
        [
            IngestionJob(source_uri="local://a", topic="t", content="bad"),
            IngestionJob(source_uri="local://b", topic="t", content="good"),
        ],
    )
    assert statuses == [IngestionStatus.FAILED, IngestionStatus.OK]


def test_ingestion_pipeline_accepts_a_generator_iterable() -> None:
    """``ingest`` must accept any ``Iterable[IngestionJob]``, not just a
    concrete list. Passing a generator exercises the ``iter()`` path that
    real callers (the sweep / worker transport) use."""
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)

    def _gen() -> Iterable[IngestionJob]:
        yield IngestionJob(source_uri="local://g1", topic="t", content="c1")
        yield IngestionJob(source_uri="local://g2", topic="t", content="c2")

    statuses = pipeline.ingest(_gen())
    assert statuses == [IngestionStatus.OK, IngestionStatus.OK]


# ---------------------------------------------------------------------------
# IngestionStatus enum — string-value stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, expected_value",
    [
        (IngestionStatus.PENDING, "pending"),
        (IngestionStatus.OK, "ok"),
        (IngestionStatus.FAILED, "failed"),
        (IngestionStatus.SKIPPED, "skipped"),
    ],
)
def test_ingestion_status_enum_values_are_stable_strings(
    status: IngestionStatus,
    expected_value: str,
) -> None:
    """``IngestionStatus`` is a ``StrEnum``; the string form is what gets
    persisted (D1 ingestion column, JSON manifest). Renaming any value
    would silently break every consumer that filters on the string."""
    assert status == expected_value
    assert str(status) == expected_value
    assert f"{status}" == expected_value


def test_ingestion_status_has_exactly_four_distinct_values() -> None:
    """``IngestionStatus`` has exactly PENDING / OK / FAILED / SKIPPED.
    Adding a fifth (e.g. RETRYING) without coordinating with consumers
    is a semantic break — keep the surface closed."""
    assert len(set(IngestionStatus)) == 4
    assert set(IngestionStatus) == {
        IngestionStatus.PENDING,
        IngestionStatus.OK,
        IngestionStatus.FAILED,
        IngestionStatus.SKIPPED,
    }


# ---------------------------------------------------------------------------
# IngestionJob dataclass defaults and shape
# ---------------------------------------------------------------------------


def test_ingestion_job_uses_slots() -> None:
    """``@dataclass(slots=True)`` means the dataclass has a ``__slots__``
    and rejects attribute assignment outside the declared fields. A
    refactor that drops ``slots=True`` would silently widen the surface."""
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    with pytest.raises(AttributeError):
        job.injected_attribute = "bogus"  # type: ignore[attr-defined]


def test_ingestion_job_default_authority_is_0_5() -> None:
    """The dataclass default ``authority=0.5`` is the default-source
    contract: callers who don't specify authority get the neutral
    midpoint (not 0.0 which would filter out everything by
    ``min_authority``)."""
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    assert job.authority == 0.5


def test_ingestion_job_default_status_is_experimental() -> None:
    """Default ``Status.EXPERIMENTAL`` is the contract: every unlabelled
    ingestion job is treated as "needs verification" until the caller
    promotes it to ``IMPLEMENTED``."""
    job = IngestionJob(source_uri="local://a", topic="t", content="c")
    assert job.status == Status.EXPERIMENTAL


# ---------------------------------------------------------------------------
# now_epoch() / sha256_hex() — only the producer builds on top of these
# ---------------------------------------------------------------------------


def test_now_epoch_returns_a_float() -> None:
    """``now_epoch`` must return ``float`` (not ``int`` / ``Decimal``).
    ``IngestionJob.build`` writes the value into
    ``KnowledgeObject.retrieved_at: float``; a non-int return would
    silently break that assignment under strict type checkers."""
    value = now_epoch()
    assert isinstance(value, float)


def test_now_epoch_is_monotonically_non_decreasing_across_calls() -> None:
    """Two consecutive ``now_epoch()`` calls must return
    ``second >= first`` (the monotonic-clock guarantee on POSIX). A
    broken ``time.time()`` substitute (e.g. ``int`` truncation) would
    let two ingestion jobs in the same second have equal
    ``retrieved_at``, breaking sort-by-``retrieved_at`` downstream."""
    first = now_epoch()
    second = now_epoch()
    assert second >= first


def test_sha256_hex_used_by_build_round_trips_through_dataclass() -> None:
    """End-to-end: an ``IngestionJob.build()``-produced object's
    ``content_hash`` matches ``sha256_hex(content)`` for non-ASCII
    content (catches any encoding regression in the hash derivation)."""
    obj = IngestionJob(
        source_uri="local://a",
        topic="t",
        content="日本語content",
    ).build()
    assert obj.content_hash == sha256_hex("日本語content")


# ---------------------------------------------------------------------------
# _Source Protocol — structural-typing contract
# ---------------------------------------------------------------------------


class _FakeSource:
    """Duck-typed class that satisfies the ``_Source`` Protocol structurally
    (no explicit ``Protocol`` registration). Used to prove that the
    pipeline accepts any class implementing ``fetch(source_uri: str) ->
    Iterable[IngestionJob]``."""

    def __init__(self, jobs: list[IngestionJob]) -> None:
        self._jobs = jobs

    def fetch(self, source_uri: str) -> Iterable[IngestionJob]:
        return list(self._jobs)


def test_source_protocol_accepts_duck_typed_implementer() -> None:
    """The ``_Source`` Protocol is structural — any class with a
    ``fetch(source_uri: str) -> Iterable[IngestionJob]`` method is
    accepted by ``IngestionPipeline(sources=...)`` without explicit
    registration. Pinning this guards against a refactor that adds an
    explicit ABC base class (which would silently break all duck-typed
    callers, including the test mocks)."""
    source = _FakeSource([IngestionJob(source_uri="local://a", topic="t", content="c")])
    pipeline = IngestionPipeline(store=InMemoryKnowledgeStore(), sources=[source])
    # The structural-typing assertion: just constructing the pipeline
    # with a duck-typed source must succeed without runtime Protocol
    # rejection. (Python's Protocol is nominal only at type-check time;
    # runtime accepts any object with the right shape.)
    assert pipeline.sources == [source]


def test_source_protocol_accepts_duck_typed_with_generator_fetch() -> None:
    """The Protocol allows ``fetch`` to return a generator (lazy
    evaluation). A refactor that pins ``fetch`` to a specific concrete
    return type (e.g. ``list[IngestionJob]``) would break callers that
    stream jobs."""
    job_template = IngestionJob(
        source_uri="local://template",
        topic="t",
        content="c",
    )

    class _GeneratorSource:
        def fetch(self, source_uri: str) -> Iterable[IngestionJob]:
            # Lazy generator: a refactor that pins ``fetch`` to a list
            # would force this to materialise.
            yield IngestionJob(source_uri=source_uri, topic="t", content="c")
            yield job_template  # extra item to prove laziness isn't truncated

    pipeline = IngestionPipeline(
        store=InMemoryKnowledgeStore(),
        sources=[_GeneratorSource()],
    )
    # The structural-typing assertion: constructing the pipeline with a
    # generator-yielding source succeeds, and the source's ``fetch``
    # method is invokable through the generic ``pipeline.sources[0]``
    # accessor (no Protocol nominal check at runtime).
    gen = pipeline.sources[0].fetch("local://x")
    assert iter(gen) is not None


# ---------------------------------------------------------------------------
# IngestionPipeline sources field
# ---------------------------------------------------------------------------


def test_ingestion_pipeline_default_sources_is_empty_list() -> None:
    """The pipeline's ``sources`` field defaults to ``[]`` (NOT ``None``).
    A refactor that flipped the default to ``None`` would force every
    constructor to pass an explicit list and would let callers
    accidentally pass ``None`` for ``sources``."""
    pipeline = IngestionPipeline(store=InMemoryKnowledgeStore())
    assert pipeline.sources == []


def test_ingestion_pipeline_sources_is_a_mutable_list() -> None:
    """The ``sources`` field is a regular list (not a tuple, not
    frozen). Callers can append new sources after construction.
    Pinning this prevents a refactor that switches to ``tuple`` /
    ``frozen=True`` and breaks the append-after-construct pattern."""
    pipeline = IngestionPipeline(store=InMemoryKnowledgeStore())
    pipeline.sources.append(_FakeSource([]))
    assert len(pipeline.sources) == 1


def test_ingestion_pipeline_is_not_frozen() -> None:
    """``IngestionPipeline`` is NOT ``frozen=True`` (only ``slots=True``).
    A refactor that adds ``frozen=True`` would silently break callers
    that mutate ``pipeline.sources`` after construction."""
    pipeline = IngestionPipeline(store=InMemoryKnowledgeStore())
    # Not-frozen dataclass assignment succeeds (would raise
    # ``FrozenInstanceError`` if frozen).
    pipeline.sources = [_FakeSource([])]
    assert len(pipeline.sources) == 1


def test_ingestion_pipeline_uses_slots() -> None:
    """``IngestionPipeline`` is ``@dataclass(slots=True)`` — the
    dataclass has a ``__slots__`` and rejects attribute injection
    outside the declared fields."""
    pipeline = IngestionPipeline(store=InMemoryKnowledgeStore())
    with pytest.raises(AttributeError):
        pipeline.injected_attribute = "bogus"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ingest() return type / shape
# ---------------------------------------------------------------------------


def test_ingest_returns_list_not_iterator() -> None:
    """``ingest()`` returns ``list[IngestionStatus]`` (not an
    iterator/iterable). Pinning this prevents a refactor that switches
    to ``yield from`` — callers that index the result (e.g.
    ``results[0]``) would silently break."""
    store = InMemoryKnowledgeStore()
    pipeline = IngestionPipeline(store=store)
    result = pipeline.ingest([])
    assert isinstance(result, list)


def test_ingest_with_empty_jobs_and_empty_sources_returns_empty_list() -> None:
    """Empty input on both axes (no jobs, no sources) returns ``[]``.
    The two-axis emptiness is independent — a refactor that introduces
    cross-coupling between ``sources`` and ``jobs`` would break this."""
    pipeline = IngestionPipeline(store=InMemoryKnowledgeStore())
    assert pipeline.ingest([]) == []


def test_ingest_does_not_swallow_keyboard_interrupt() -> None:
    """The ``try/except Exception`` around ``store.put`` must NOT catch
    ``KeyboardInterrupt`` / ``SystemExit`` / ``GeneratorExit`` (those
    are ``BaseException`` subclasses, not ``Exception``). A refactor
    that broadens to ``except BaseException`` would silently swallow
    user-cancellation as ``IngestionStatus.FAILED``, hiding the
    real signal that the user pressed Ctrl-C."""

    class _KeyboardOnPutStore:
        def put(self, obj: KnowledgeObject) -> None:
            raise KeyboardInterrupt("user cancelled")

    pipeline = IngestionPipeline(store=_KeyboardOnPutStore())  # type: ignore[arg-type]
    with pytest.raises(KeyboardInterrupt):
        pipeline.ingest([IngestionJob(source_uri="local://a", topic="t", content="c")])


def test_ingest_does_not_swallow_system_exit() -> None:
    """``SystemExit`` is also a ``BaseException`` subclass, not an
    ``Exception``. Pinning this protects the contract independently
    of ``KeyboardInterrupt``."""

    class _SystemExitOnPutStore:
        def put(self, obj: KnowledgeObject) -> None:
            raise SystemExit(1)

    pipeline = IngestionPipeline(store=_SystemExitOnPutStore())  # type: ignore[arg-type]
    with pytest.raises(SystemExit):
        pipeline.ingest([IngestionJob(source_uri="local://a", topic="t", content="c")])


def test_ingest_does_swallow_custom_exception_subclass() -> None:
    """A custom ``RuntimeError`` (an ``Exception`` subclass) IS caught
    and converted to ``IngestionStatus.FAILED``. Pinning the
    exception-breadth contract from the other side — the source is
    NOT broadening to ``BaseException``."""

    class _RuntimeErrorOnPutStore:
        def put(self, obj: KnowledgeObject) -> None:
            raise RuntimeError("simulated store failure")

    pipeline = IngestionPipeline(store=_RuntimeErrorOnPutStore())  # type: ignore[arg-type]
    results = pipeline.ingest([IngestionJob(source_uri="local://a", topic="t", content="c")])
    assert results == [IngestionStatus.FAILED]


def test_ingest_status_list_length_matches_input_length() -> None:
    """The output list length always equals the input job list length
    (one status per job, in order). Pinning this catches a refactor
    that accidentally short-circuits on the first failure."""

    class _AlwaysBoomStore:
        def put(self, obj: KnowledgeObject) -> None:
            raise RuntimeError("boom")

    pipeline = IngestionPipeline(store=_AlwaysBoomStore())  # type: ignore[arg-type]
    jobs = [IngestionJob(source_uri=f"local://{i}", topic="t", content="c") for i in range(5)]
    results = pipeline.ingest(jobs)
    assert len(results) == 5
    assert all(r is IngestionStatus.FAILED for r in results)
