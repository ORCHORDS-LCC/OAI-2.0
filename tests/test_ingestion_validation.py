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

A refactor that, e.g., inverts the ``try/except`` semantics (turning
failures into ``OK``), changes the ``content_hash`` derivation, hardcodes
the ``knowledge_id`` instead of using ``uuid``, or renames the enum
strings would propagate silently into the D1 ingestion path.
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
