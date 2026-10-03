"""#57 stage latency, and a cache metric that does not lie.

Two claims are under test here, and both are claims about NOT over-reporting.

LATENCY
    Count, sum, min, max, aggregated from the duration the lifecycle owner
    already emits from a monotonic clock. Four numbers, constant memory, no
    per-trace history. A percentile computed from an unbounded sample list is
    not a latency metric, it is a memory leak that renders a chart, and the
    "exact window" it reports silently stops being true the moment the window
    moves.

CACHE
    The runtime consults the cache, and there are THREE ways that can go, not
    two:

    * the lookup returned nothing            -> a miss;
    * the lookup returned something usable   -> a hit;
    * the lookup returned something REFUSED  -> neither, and calling it a hit
      is exactly the over-report the brief forbids.

    The third case is not hypothetical. ``cache_key_for`` embeds the corpus
    revision, so under any real write load every cached envelope is stale by
    construction and the cache can only ever report 0% hits while working
    perfectly. Folding that into "miss" makes a busy corpus indistinguishable
    from an empty KV namespace — two opposite diagnoses with opposite fixes —
    so it gets its own bucket.

    And a KV read that RAISED is not a miss. A miss is the cache working. An
    exception is the cache not answering, which is ``kv.degraded`` and
    contributes to no hit rate at all.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from oai2.core import KnowledgeId, Status
from oai2.knowledge import (
    CFRow,
    KnowledgeObject,
    RetrievalRequest,
    sha256_hex,
    vector_id_for,
)
from oai2.knowledge.cloudflare import cache_key_for
from oai2.knowledge.cloudflare_runtime import (
    AsyncCloudflareKnowledgeRuntime,
    KnowledgeConflictError,
)
from oai2.knowledge.transport import (
    TRANSPORT_VERSION,
    KnowledgeCacheRef,
    QueryCacheEnvelope,
)
from oai2.observability import (
    CollectingEventSink,
    EventCategory,
    EventType,
    MetricsAggregator,
    Trace,
    TraceRecorder,
    new_request_recorder,
)

_EMBEDDING_VERSION = "embed-v1"
_EMBEDDING_DIGEST = "embed-digest-v1"
_CANARY = "CANARY_DO_NOT_LOG_9f3a2b1c4d"

CACHE_EVENTS = (
    "knowledge.cache.hit",
    "knowledge.cache.miss",
    "knowledge.cache.stale",
)


def _completed(
    sink: Any,
    *,
    duration_ms: float | None,
    outcome: str = "ok",
    operation: str = "retrieve",
) -> TraceRecorder:
    """Record one completion with a duration we control exactly.

    The duration is set on the event rather than slept for, so the assertions
    are about the aggregator's arithmetic and not about how fast this machine
    is on a Tuesday.
    """
    recorder = new_request_recorder(
        sink=sink, request_id="req-lat", operation=operation
    )
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_COMPLETED,
        timestamp=1000.0,
        outcome=outcome,
        duration_ms=duration_ms,
    )
    return recorder


# ---------------------------------------------------------------------------
# §9 — request latency, aggregated into four bounded numbers
# ---------------------------------------------------------------------------


def test_latency_is_aggregated_and_not_stored_per_trace() -> None:
    """RED at d0e271a: the aggregator had no latency block at all."""
    agg = MetricsAggregator()
    for ms in (10.0, 20.0, 30.0):
        _completed(agg, duration_ms=ms)

    latency = agg.snapshot()["request_latency"]
    assert latency == {
        "count": 3,
        "sum_ms": 60.0,
        "min_ms": 10.0,
        "max_ms": 30.0,
    }


def test_latency_count_tracks_completions_that_actually_had_a_clock() -> None:
    """A completion with no duration must not dilute the distribution.

    Diluting it with an implicit zero would drag the mean toward 0 and make a
    fast service look broken, which is the one direction an operator cannot
    afford to be wrong in.
    """
    agg = MetricsAggregator()
    _completed(agg, duration_ms=5.0)
    _completed(agg, duration_ms=None)

    latency = agg.snapshot()["request_latency"]
    assert latency["count"] == 1
    assert latency["sum_ms"] == 5.0
    assert latency["min_ms"] == 5.0
    assert latency["max_ms"] == 5.0


def test_an_untouched_aggregator_reports_no_latency_rather_than_zeroes() -> None:
    """count=0 with min=max=0 would read as "a request took 0ms"."""
    latency = MetricsAggregator().snapshot()["request_latency"]
    assert latency["count"] == 0
    assert latency["min_ms"] is None
    assert latency["max_ms"] is None
    assert latency["sum_ms"] == 0.0


def test_latency_aggregation_is_bounded_by_a_constant() -> None:
    """No per-trace history, so memory cannot grow with request count.

    A dict keyed by trace_id is the obvious way to compute latency and the
    wrong one: it is unbounded, it is keyed by an identifier, and it turns
    into a percentile the moment somebody wants one.
    """
    agg = MetricsAggregator(max_recent_events=8)
    for index in range(500):
        _completed(agg, duration_ms=float(index))

    latency = agg.snapshot()["request_latency"]
    assert latency["count"] == 500
    assert latency["sum_ms"] == sum(float(i) for i in range(500))
    assert latency["min_ms"] == 0.0
    assert latency["max_ms"] == 499.0
    assert agg.snapshot()["retained_event_count"] == 8


def test_reset_clears_latency_so_a_second_run_starts_from_nothing() -> None:
    agg = MetricsAggregator()
    _completed(agg, duration_ms=99.0)
    agg.reset()
    assert agg.snapshot()["request_latency"]["count"] == 0


def test_a_negative_duration_is_refused_by_the_schema() -> None:
    """A backwards wall clock must not become a negative latency."""
    trace = Trace()
    with pytest.raises(ValueError):
        trace.record(
            category=EventCategory.KNOWLEDGE,
            event_type=EventType.KNOWLEDGE_REQUEST_COMPLETED,
            timestamp=1.0,
            duration_ms=-1.0,
        )


def test_duration_comes_from_a_monotonic_clock_not_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wall clock may step backwards mid-request; a duration may not.

    This is the reason ``duration_ms`` is measured with ``time.monotonic``.
    Deriving it by subtracting two ``time.time()`` readings is correct on a
    machine whose clock never moves, and wrong by an unbounded amount on one
    where NTP steps the clock during the request.
    """
    recorder = new_request_recorder(sink=NullSink(), request_id="req-clock")
    recorder.start_clock()

    # The wall clock jumps backwards by an hour. The monotonic clock does not.
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    elapsed = recorder.elapsed_ms()

    assert elapsed is not None
    assert elapsed >= 0.0, "a backwards wall clock produced a negative duration"
    assert elapsed < 60_000, "the duration tracked the wall clock, not the clock"


class NullSink:
    def emit(self, event: object) -> None:
        return None


# ---------------------------------------------------------------------------
# §10 — the cache reports what it actually did
# ---------------------------------------------------------------------------


class _Reader:
    def __init__(self) -> None:
        self.rows: dict[str, CFRow] = {}
        self.query_result: list[CFRow] = []

    async def get_row(self, knowledge_id: KnowledgeId) -> CFRow | None:
        return self.rows.get(str(knowledge_id))

    async def query_rows(self, request: RetrievalRequest) -> list[CFRow]:
        return list(self.query_result[: request.limit])


class _Writer:
    def __init__(self, revision: int = 4) -> None:
        self.revision = revision
        self.read_sequence: list[int] = []

    async def corpus_revision(self) -> int:
        if self.read_sequence:
            return self.read_sequence.pop(0)
        return self.revision


class _R2:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get_text(self, key: str) -> str | None:
        return self.values.get(key)

    async def put_text(self, key: str, value: str) -> None:
        self.values[key] = value


class _Vectorize:
    async def query(self, values: object, *, top_k: int = 5) -> list[Any]:
        return []


class _Kv:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.get_error: Exception | None = None
        self.put_error: Exception | None = None

    async def get_text(self, key: str) -> str | None:
        if self.get_error is not None:
            raise self.get_error
        return self.values.get(key)

    async def put_text(
        self, key: str, value: str, *, ttl_seconds: int | None = None
    ) -> None:
        if self.put_error is not None:
            raise self.put_error
        self.values[key] = value


def _object() -> KnowledgeObject:
    return KnowledgeObject(
        knowledge_id=KnowledgeId("ko_cache_1"),
        topic="cache-topic",
        content="cached body",
        content_hash=sha256_hex("cached body"),
        source_uri="https://example.test/src",
        retrieved_at=10.0,
        authority=0.9,
        status=Status.EXPERIMENTAL,
    )


def _row_for(obj: KnowledgeObject) -> CFRow:
    return CFRow(
        knowledge_id=obj.knowledge_id,
        topic=obj.topic,
        content_hash=obj.content_hash,
        authority=obj.authority,
        status=obj.status,
        source_uri=obj.source_uri,
        retrieved_at=obj.retrieved_at,
        r2_blob_key=f"oai2-blobs/{obj.content_hash}",
        vectorize_id=vector_id_for(
            knowledge_id=obj.knowledge_id,
            content_hash=obj.content_hash,
            embedding_version=_EMBEDDING_VERSION,
        ),
    )


def _runtime(kv: _Kv) -> tuple[AsyncCloudflareKnowledgeRuntime, _Writer, _Reader]:
    reader = _Reader()
    writer = _Writer(revision=4)
    r2 = _R2()
    obj = _object()
    reader.rows[str(obj.knowledge_id)] = _row_for(obj)
    reader.query_result = [reader.rows[str(obj.knowledge_id)]]
    r2.values[f"oai2-blobs/{obj.content_hash}"] = obj.content
    runtime = AsyncCloudflareKnowledgeRuntime(
        reader=reader,  # type: ignore[arg-type]
        writer=writer,  # type: ignore[arg-type]
        r2=r2,  # type: ignore[arg-type]
        vectorize=_Vectorize(),  # type: ignore[arg-type]
        kv=kv,  # type: ignore[arg-type]
        embedding_version=_EMBEDDING_VERSION,
        embedding_digest=_EMBEDDING_DIGEST,
    )
    return runtime, writer, reader


def _seed_usable_envelope(kv: _Kv, request: RetrievalRequest, revision: int) -> str:
    """Store an envelope that ``_try_cached_result`` will accept verbatim."""
    key = cache_key_for(request, _EMBEDDING_DIGEST, revision)
    obj = _object()
    envelope = QueryCacheEnvelope(
        version=TRANSPORT_VERSION,
        corpus_revision=revision,
        embedding_digest=_EMBEDDING_DIGEST,
        request_fingerprint=key,
        refs=[
            KnowledgeCacheRef(
                knowledge_id=obj.knowledge_id, content_hash=obj.content_hash
            )
        ],
    )
    kv.values[key] = envelope.model_dump_json()
    return key


def _cache_events(sink: CollectingEventSink) -> tuple[str, ...]:
    return tuple(e.event_type for e in sink.events if e.event_type in CACHE_EVENTS)


def test_a_served_retrieval_is_reported_as_a_cache_hit() -> None:
    kv = _Kv()
    runtime, _writer, _reader = _runtime(kv)
    request = RetrievalRequest(topic="cache-topic")
    _seed_usable_envelope(kv, request, 4)
    sink = CollectingEventSink()

    result = asyncio.run(
        runtime.retrieve(
            request,
            trace=new_request_recorder(
                sink=sink, request_id="r", operation="retrieve"
            ),
        )
    )

    assert result.objects, "the retrieval must actually be served"
    assert _cache_events(sink) == ("knowledge.cache.hit",)
    assert sink.events[0].outcome == "cache_hit", (
        "a cache outcome must be unmistakably not a request outcome"
    )
    assert sink.events[0].operation == "retrieve"


def test_an_empty_cache_is_reported_as_a_miss() -> None:
    kv = _Kv()
    runtime, _writer, _reader = _runtime(kv)
    sink = CollectingEventSink()

    asyncio.run(
        runtime.retrieve(
            RetrievalRequest(topic="cache-topic"),
            trace=new_request_recorder(sink=sink, request_id="r"),
        )
    )

    assert _cache_events(sink) == ("knowledge.cache.miss",)


def test_an_envelope_for_another_revision_is_invisible_to_this_lookup() -> None:
    """The corpus revision really is part of the key, not decoration.

    An envelope written against a different revision is not merely refused — it
    is never even read, because the lookup key differs. That is why "stale" is
    a distinct state from "miss" at all: a corpus that has moved on produces
    misses for the old key, and the runtime never sees the old value.
    """
    kv = _Kv()
    runtime, writer, _reader = _runtime(kv)
    request = RetrievalRequest(topic="cache-topic")
    _seed_usable_envelope(kv, request, 999)
    writer.revision = 4
    sink = CollectingEventSink()

    result = asyncio.run(
        runtime.retrieve(
            request, trace=new_request_recorder(sink=sink, request_id="r")
        )
    )

    assert result.objects, "the authoritative path still served the request"
    assert _cache_events(sink) == ("knowledge.cache.miss",)


def test_an_unusable_envelope_that_was_actually_read_is_reported_as_stale() -> None:
    """Same key, but the envelope describes a different corpus revision.

    This is the steady state on a corpus under write load: the key was looked
    up, something came back, and it was refused. Reporting that as a miss
    would make a working cache look permanently empty.
    """
    kv = _Kv()
    runtime, writer, _reader = _runtime(kv)
    request = RetrievalRequest(topic="cache-topic")
    key = cache_key_for(request, _EMBEDDING_DIGEST, 4)
    obj = _object()
    kv.values[key] = QueryCacheEnvelope(
        version=TRANSPORT_VERSION,
        corpus_revision=3,  # the revision this envelope was written for
        embedding_digest=_EMBEDDING_DIGEST,
        request_fingerprint=key,
        refs=[
            KnowledgeCacheRef(
                knowledge_id=obj.knowledge_id, content_hash=obj.content_hash
            )
        ],
    ).model_dump_json()
    writer.revision = 4
    sink = CollectingEventSink()

    result = asyncio.run(
        runtime.retrieve(
            request, trace=new_request_recorder(sink=sink, request_id="r")
        )
    )

    assert result.objects, "the authoritative path still served the request"
    assert _cache_events(sink) == ("knowledge.cache.stale",)


def test_a_corrupt_envelope_is_stale_not_a_hit() -> None:
    kv = _Kv()
    runtime, _writer, _reader = _runtime(kv)
    request = RetrievalRequest(topic="cache-topic")
    key = cache_key_for(request, _EMBEDDING_DIGEST, 4)
    kv.values[key] = "{not json at all"
    sink = CollectingEventSink()

    asyncio.run(
        runtime.retrieve(
            request, trace=new_request_recorder(sink=sink, request_id="r")
        )
    )

    assert _cache_events(sink) == ("knowledge.cache.stale",)


def test_a_kv_read_that_raises_is_degraded_and_never_a_miss() -> None:
    """The distinction that lets a silent cache outage be noticed at all.

    A miss is the cache working. If an exception were reported as a miss, a
    permanently broken KV namespace would look like a permanently cold cache:
    nothing would error, every request would just be slower, and nothing would
    ever page anyone.
    """
    kv = _Kv()
    kv.get_error = RuntimeError(f"namespace {_CANARY} unavailable")
    runtime, _writer, _reader = _runtime(kv)
    sink = CollectingEventSink()

    result = asyncio.run(
        runtime.retrieve(
            RetrievalRequest(topic="cache-topic"),
            trace=new_request_recorder(sink=sink, request_id="r"),
        )
    )

    assert result.objects, "a degraded cache must not fail the request"
    assert "knowledge.kv.degraded" in [e.event_type for e in sink.events]
    assert _cache_events(sink) == (), (
        "a degraded read contributes to no hit rate"
    )


def test_the_cache_outcomes_partition_every_consulted_lookup() -> None:
    """hit + miss + stale accounts for every lookup that was answered.

    A cache metric whose parts do not sum to the total cannot be read as a
    rate, and a rate that does not sum to 100% is how a third state goes
    permanently unnoticed.
    """
    agg = MetricsAggregator()
    request = RetrievalRequest(topic="cache-topic")

    # miss
    kv = _Kv()
    runtime, _w, _r = _runtime(kv)
    recorder = new_request_recorder(sink=agg, request_id="r1", operation="retrieve")
    asyncio.run(runtime.retrieve(request, trace=recorder))

    # hit
    kv = _Kv()
    runtime, _w, _r = _runtime(kv)
    _seed_usable_envelope(kv, request, 4)
    recorder = new_request_recorder(sink=agg, request_id="r2", operation="retrieve")
    asyncio.run(runtime.retrieve(request, trace=recorder))

    # stale
    kv = _Kv()
    runtime, _w, _r = _runtime(kv)
    key = cache_key_for(request, _EMBEDDING_DIGEST, 4)
    kv.values[key] = "{not json"
    recorder = new_request_recorder(sink=agg, request_id="r3", operation="retrieve")
    asyncio.run(runtime.retrieve(request, trace=recorder))

    snapshot = agg.snapshot()["counters"]
    assert snapshot["cache_hit_count"] == 1
    assert snapshot["cache_miss_count"] == 1
    assert snapshot["cache_stale_count"] == 1
    assert (
        snapshot["cache_hit_count"]
        + snapshot["cache_miss_count"]
        + snapshot["cache_stale_count"]
    ) == 3


def test_a_concurrent_write_during_a_cached_read_is_a_conflict_not_a_hit() -> None:
    """The cache produced a candidate, then a writer invalidated it.

    The request failed. Counting it as a hit would put a success in the
    numerator of a hit rate for a retrieval that returned nothing.
    """
    kv = _Kv()
    runtime, writer, _reader = _runtime(kv)
    request = RetrievalRequest(topic="cache-topic")
    _seed_usable_envelope(kv, request, 4)
    # Start the lookup at 4, then report 5 on the re-read: the corpus moved
    # under us between the cache read and the hand-back.
    writer.read_sequence = [4, 5]
    sink = CollectingEventSink()

    with pytest.raises(KnowledgeConflictError):
        asyncio.run(
            runtime.retrieve(
                request, trace=new_request_recorder(sink=sink, request_id="r")
            )
        )

    assert _cache_events(sink) == ()


def test_a_semantic_retrieval_consults_no_cache_and_reports_none() -> None:
    """The vector path never reads the cache, so it has no cache outcome.

    Emitting a miss here would invent a cache interaction that did not happen
    and depress the reported hit rate for requests that never asked.
    """
    kv = _Kv()
    runtime, _writer, _reader = _runtime(kv)
    sink = CollectingEventSink()

    asyncio.run(
        runtime.retrieve(
            RetrievalRequest(topic="cache-topic"),
            query_vector=[0.1, 0.2],
            trace=new_request_recorder(sink=sink, request_id="r"),
        )
    )

    assert _cache_events(sink) == ()


def test_no_cache_event_carries_a_cache_key_or_the_topic() -> None:
    """The key is a hash of the topic, but the topic must not ride along."""
    kv = _Kv()
    runtime, _writer, _reader = _runtime(kv)
    sink = CollectingEventSink()

    asyncio.run(
        runtime.retrieve(
            RetrievalRequest(topic=_CANARY),
            trace=new_request_recorder(sink=sink, request_id="r"),
        )
    )

    assert sink.events, "the retrieval must still report something"
    for event in sink.events:
        blob = repr(event.model_dump())
        assert _CANARY not in blob, f"cache event leaked caller content: {event}"


def test_the_cache_vocabulary_is_part_of_the_known_event_set() -> None:
    """A new type is added to the AUTHORING vocabulary, not smuggled in.

    The open-string design means a forgotten entry would still parse, and would
    be counted as ``unknown_event_types`` forever. That is the failure mode
    this test exists to prevent.
    """
    for name in CACHE_EVENTS:
        assert name in {member.value for member in EventType}, (
            f"{name} is emitted but is not in the authoring vocabulary"
        )
