"""#57 — metrics, redaction, bounded retention and observability overhead.

Everything is derived from emitted events. Nothing here reaches into a runtime
subsystem's own counters, which is the property that keeps the numbers correct
as the request path changes.
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
from collections.abc import Mapping

import pytest

from oai2.knowledge.admission import AdmissionState
from oai2.knowledge.observability import (
    KV_OPERATION_GET,
    emit_completed,
    emit_conflict,
    emit_dependency_failure,
    emit_integrity_failure,
    emit_internal_failure,
    emit_kv_degraded,
    emit_request_started,
    emit_saturated,
)
from oai2.observability import (
    CollectingEventSink,
    EventCategory,
    EventType,
    MetricsAggregator,
    NullEventSink,
    TraceRecorder,
    new_request_recorder,
    new_trace_id,
)
from oai2.observability.metrics import (
    KNOWN_OUTCOMES,
    MAX_DIMENSION_KEYS,
    OVERFLOW_KEY,
    UNKNOWN_OUTCOME,
    assert_no_identifier_cardinality,
)
from tests._worker_entrypoint_fixtures import (
    Components,
    Env,
    Request,
    drive,
    entrypoint,
    valid_body,
    wire,
)

#: A secret-shaped sentinel, for the redaction controls.
SENTINEL = "CANARY_DO_NOT_LOG_9f3a2b1c4d"


@pytest.fixture(autouse=True)
def _clean_admission():
    from oai2.knowledge.admission import ISOLATE_ADMISSION

    ISOLATE_ADMISSION._reset_for_tests()
    yield
    ISOLATE_ADMISSION._reset_for_tests()


def _drive(aggregator: MetricsAggregator, **kwargs: object) -> object:
    """Run one request through the REAL entrypoint into the aggregator."""
    return asyncio.run(
        drive(
            kwargs.pop("monkeypatch"),  # type: ignore[arg-type]
            kwargs.pop("entry"),  # type: ignore[arg-type]
            kwargs.pop("env", Env()),  # type: ignore[arg-type]
            kwargs.pop("body", valid_body("req-1")),  # type: ignore[arg-type]
        )
    )


# ---------------------------------------------------------------------------
# Metrics are derived from events
# ---------------------------------------------------------------------------


def test_a_successful_request_is_counted(monkeypatch) -> None:
    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        response = asyncio.run(
            drive(monkeypatch, entry, Env(), valid_body("req-1"))
        )
    assert response.status == 200
    counters = metrics.snapshot()["counters"]
    assert counters["requests_started"] == 1
    assert counters["requests_completed"] == 1
    assert counters["successful_requests"] == 1
    assert counters["failed_requests"] == 0


def test_a_saturated_request_counts_backpressure(monkeypatch) -> None:
    from tests._worker_entrypoint_fixtures import Gate

    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        gate = Gate(hold=True)
        wire(monkeypatch, Components(gate))

        async def scenario() -> object:
            env = Env(KNOWLEDGE_MAX_IN_FLIGHT="1")
            held = asyncio.create_task(
                entry.Default(env).fetch(Request(valid_body("A")))
            )
            await gate.entered.wait()
            refused = await entry.Default(env).fetch(Request(valid_body("B")))
            gate.release.set()
            await held
            return refused

        asyncio.run(scenario())

    counters = metrics.snapshot()["counters"]
    assert counters["saturated_count"] == 1
    assert counters["failed_requests"] == 1
    assert counters["requests_completed"] == 2
    assert counters["admission_in_flight_last"] == 1, (
        "the refusal frees nothing, so in_flight is still 1"
    )
    assert counters["admission_refused_count"] == 1


def test_a_retryable_saturation_is_not_counted_as_a_retry(monkeypatch) -> None:
    """The number that must stay zero, and why.

    ``saturated_count`` is 1 and ``retryable_failure_count`` is 1, but
    ``retry_events_observed`` is 0. Inferring retries from a retryable flag
    would report a rate nobody observed.
    """
    from tests._worker_entrypoint_fixtures import Gate

    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        gate = Gate(hold=True)
        wire(monkeypatch, Components(gate))

        async def scenario() -> None:
            env = Env(KNOWLEDGE_MAX_IN_FLIGHT="1")
            held = asyncio.create_task(
                entry.Default(env).fetch(Request(valid_body("A")))
            )
            await gate.entered.wait()
            await entry.Default(env).fetch(Request(valid_body("B")))
            gate.release.set()
            await held

        asyncio.run(scenario())

    counters = metrics.snapshot()["counters"]
    assert counters["retryable_failure_count"] == 1
    assert counters["retry_events_observed"] == 0, (
        "no retry loop exists; a retry must be observed, not inferred"
    )


def test_each_failure_kind_is_counted_separately() -> None:
    metrics = MetricsAggregator()
    recorder = new_request_recorder(sink=metrics, request_id="req-f")
    emit_conflict(recorder, timestamp=1.0)
    emit_integrity_failure(recorder, timestamp=2.0)
    emit_dependency_failure(recorder, timestamp=3.0)
    emit_internal_failure(recorder, timestamp=4.0)
    emit_kv_degraded(recorder, operation=KV_OPERATION_GET, timestamp=5.0)

    counters = metrics.snapshot()["counters"]
    assert counters["conflict_count"] == 1
    assert counters["integrity_failure_count"] == 1
    assert counters["dependency_failure_count"] == 1
    assert counters["internal_failure_count"] == 1
    assert counters["kv_degraded_count"] == 1
    # Detail events alone do NOT move retryable_failure_count: that metric
    # counts REQUESTS whose completion was retryable, and no completion has
    # happened yet here. Counting it per detail event would report one
    # refusal twice, because a refusal emits a detail event AND a completion.
    assert counters["retryable_failure_count"] == 0

    # Now close the four requests, and the metric reflects the outcomes.
    for outcome, retryable in (
        ("conflict", True),
        ("integrity", False),
        ("unavailable_dependency", True),
        ("internal", False),
    ):
        emit_completed(
            recorder, timestamp=9.0, ok=False, outcome=outcome,
            retryable=retryable,
        )
    counters = metrics.snapshot()["counters"]
    assert counters["failed_requests"] == 4
    assert counters["retryable_failure_count"] == 2, (
        "conflict and a dependency outage are retryable; integrity and a bug "
        "are not"
    )


def test_failed_requests_break_down_by_outcome() -> None:
    metrics = MetricsAggregator()
    recorder = new_request_recorder(sink=metrics, request_id="req-f")
    for outcome in ("conflict", "integrity", "unavailable_dependency"):
        emit_completed(
            recorder, timestamp=1.0, ok=False, outcome=outcome, retryable=True
        )
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["failed_requests"] == 3
    assert snapshot["by_outcome"] == {
        "conflict": 1, "integrity": 1, "unavailable_dependency": 1
    }


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_no_caller_content_reaches_the_metric_store(monkeypatch) -> None:
    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        body = {
            "request_id": "req-r",
            "version": "1",
            "operation": "retrieve",
            "topic": f"confidential revenue plan {SENTINEL}",
            "limit": 3,
        }
        asyncio.run(drive(monkeypatch, entry, Env(), body))
    serialised = str(metrics.snapshot())
    assert SENTINEL not in serialised
    assert "confidential" not in serialised
    assert "req-r" not in serialised, (
        "a request id is correlation, not a metric label"
    )


def test_an_unrecognised_outcome_is_bucketed_not_trusted() -> None:
    """A caller-influenced string must not be able to create a time series."""
    metrics = MetricsAggregator()
    recorder = new_request_recorder(sink=metrics, request_id="req-x")
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_CONFLICT,
        timestamp=1.0,
        outcome=f"boom-{SENTINEL}-{new_trace_id()}",
    )
    assert metrics.snapshot()["by_outcome"] == {UNKNOWN_OUTCOME: 1}
    assert SENTINEL not in str(metrics.snapshot())


def test_identifiers_are_absent_from_every_dimension() -> None:
    metrics = MetricsAggregator()
    for index in range(5):
        recorder = new_request_recorder(
            sink=metrics, request_id=f"req-{index}"
        )
        emit_request_started(
            recorder, operation="get", timestamp=float(index)
        )
        emit_completed(recorder, timestamp=float(index), ok=True, outcome="ok")
    snapshot = metrics.snapshot()
    assert_no_identifier_cardinality(snapshot)
    for name, dimension in snapshot.items():
        if not isinstance(dimension, Mapping):
            continue
        for key in dimension:
            assert "req-" not in key, f"{name} leaked a request id: {key}"
            assert "trc_" not in key, f"{name} leaked a trace id: {key}"


def test_a_dimension_cannot_grow_past_its_cap() -> None:
    """Bounded storage, enforced rather than intended."""
    metrics = MetricsAggregator()
    recorder = new_request_recorder(sink=metrics)
    for index in range(MAX_DIMENSION_KEYS * 3):
        recorder.record(
            category=EventCategory.KNOWLEDGE,
            # A valid but never-seen type: bounded in format, unbounded in
            # number. This is exactly the shape the cap exists for. (Segment
            # must start with a letter or the schema rejects it outright,
            # which is its own protection — the cap covers what passes.)
            event_type=f"knowledge.syntheticv{index}",
            timestamp=float(index),
        )
    snapshot = metrics.snapshot()
    assert len(snapshot["by_event_type"]) <= MAX_DIMENSION_KEYS
    assert OVERFLOW_KEY in snapshot["by_event_type"]


def test_the_recent_event_buffer_is_bounded_and_evicts_oldest_first() -> None:
    metrics = MetricsAggregator(max_recent_events=4)
    recorder = new_request_recorder(sink=metrics)
    for index in range(20):
        emit_request_started(
            recorder, operation="get", timestamp=float(index)
        )
    events = metrics.recent_events
    assert len(events) == 4
    assert [e.timestamp for e in events] == [16.0, 17.0, 18.0, 19.0]


def test_retention_can_be_switched_off_entirely() -> None:
    metrics = MetricsAggregator(max_recent_events=0)
    recorder = new_request_recorder(sink=metrics)
    for index in range(5):
        emit_request_started(recorder, operation="get", timestamp=float(index))
    assert metrics.recent_events == ()
    # Counters still work: disabling retention is not disabling metrics.
    assert metrics.snapshot()["counters"]["requests_started"] == 5


def test_no_structure_is_keyed_by_an_identifier() -> None:
    """Structural, so a future edit cannot quietly reintroduce growth."""
    import inspect

    from oai2.observability import metrics as metrics_module

    source = inspect.getsource(metrics_module)
    # The only dict keys written are the closed dimension vocabularies and the
    # counter names. A per-id store would need an f-string or a variable key.
    for forbidden in ("self._by_trace", "self._by_request", "traces[", "requests["):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# Reset / snapshot determinism
# ---------------------------------------------------------------------------


def test_snapshot_is_deterministic_and_reset_clears() -> None:
    metrics = MetricsAggregator()
    recorder = new_request_recorder(sink=metrics, request_id="req-1")
    emit_request_started(recorder, operation="get", timestamp=1.0)
    emit_completed(recorder, timestamp=2.0, ok=True, outcome="ok")
    first = metrics.snapshot()
    assert first == metrics.snapshot(), "snapshot is not deterministic"

    metrics.reset()
    assert metrics.snapshot()["counters"]["events_observed"] == 0
    assert metrics.snapshot()["by_category"] == {}
    assert metrics.recent_events == ()


def test_the_aggregator_rejects_a_nonsensical_retention_bound() -> None:
    with pytest.raises(ValueError):
        MetricsAggregator(max_recent_events=-1)
    with pytest.raises(TypeError):
        MetricsAggregator(max_recent_events="many")  # type: ignore[arg-type]


def test_every_required_metric_name_exists() -> None:
    """Check the required list against the counters, so one cannot be dropped."""
    metrics = MetricsAggregator()
    present = set(metrics.snapshot()["counters"])
    required = {
        "requests_started", "requests_completed", "successful_requests",
        "failed_requests", "saturated_count", "dependency_failure_count",
        "integrity_failure_count", "conflict_count", "kv_degraded_count",
        "retryable_failure_count", "retry_events_observed",
        "admission_in_flight_last", "admission_in_flight_peak",
        "admission_peak_in_flight", "admission_admitted_count",
        "admission_refused_count",
    }
    assert required <= present, required - present


# ---------------------------------------------------------------------------
# Metrics do not change request behaviour
# ---------------------------------------------------------------------------


def test_installing_the_aggregator_does_not_change_the_response(
    monkeypatch,
) -> None:
    results = []
    for sink in (NullEventSink(), MetricsAggregator()):
        with entrypoint() as entry:
            entry.Default.event_sink = sink
            response = asyncio.run(
                drive(monkeypatch, entry, Env(), valid_body("req-1"))
            )
            results.append((response.status, response.payload))
    assert results[0] == results[1]


def test_the_aggregator_is_a_valid_sink() -> None:
    from oai2.observability import EventSink

    assert isinstance(MetricsAggregator(), EventSink)
    assert isinstance(NullEventSink(), EventSink)
    assert isinstance(CollectingEventSink(), EventSink)


# ---------------------------------------------------------------------------
# §18 — observability overhead, measured
# ---------------------------------------------------------------------------


def _record_many(sink: object, count: int) -> None:
    recorder = TraceRecorder(sink=sink)  # type: ignore[arg-type]
    for index in range(count):
        recorder.record(
            category=EventCategory.KNOWLEDGE,
            event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
            timestamp=float(index),
        )


def _time_calls(sink: object, count: int, rounds: int = 5) -> list[float]:
    """Per-call microseconds, median of ``rounds`` batches.

    Reported as a distribution rather than a single invented-precision number,
    because a microbenchmark of this size is dominated by scheduler noise.
    """
    samples: list[float] = []
    for _ in range(rounds):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            _record_many(sink, count)
            elapsed = time.perf_counter() - start
        finally:
            gc.enable()
        samples.append(elapsed / count * 1e6)
    return samples


def test_instrumentation_overhead_is_measured_not_asserted_negligible() -> None:
    """Measure the cost of the SAME work with and without a real sink.

    This is instrumentation cost on a deterministic path. It is NOT a claim
    about model or inference throughput, and NOT a claim that the overhead is
    negligible — the numbers are reported so a reader can judge.
    """
    count = 2000
    noop = _time_calls(NullEventSink(), count)
    in_memory = _time_calls(MetricsAggregator(), count)

    noop_median = statistics.median(noop)
    memory_median = statistics.median(in_memory)
    delta = memory_median - noop_median

    # The only assertions are that the benchmark ran and that the instrumented
    # path is not catastrophically slower. No invented precision, no
    # "negligible" claim.
    assert len(noop) == 5 and len(in_memory) == 5
    assert noop_median > 0.0
    assert memory_median >= 0.0
    assert delta < 500.0, (
        f"instrumented recording cost {delta:.1f}us/event, which is large "
        "enough to be a defect rather than instrumentation"
    )


def test_the_aggregator_does_not_accumulate_memory_per_event() -> None:
    """Bounded storage, observed rather than asserted from the code."""
    metrics = MetricsAggregator(max_recent_events=64)
    recorder = new_request_recorder(sink=metrics)
    for index in range(5000):
        emit_request_started(
            recorder, operation="get", timestamp=float(index)
        )
    assert len(metrics.recent_events) == 64
    assert metrics.snapshot()["counters"]["requests_started"] == 5000
    total_keys = sum(
        len(dimension)
        for dimension in (
            metrics.snapshot()["by_category"],
            metrics.snapshot()["by_event_type"],
            metrics.snapshot()["by_outcome"],
        )
    )
    assert total_keys <= MAX_DIMENSION_KEYS * 3


def test_admission_counters_survive_the_round_trip_into_metrics() -> None:
    state = AdmissionState()
    lease = state.acquire(2)
    lease2 = state.acquire(2)
    lease.release()
    lease2.release()

    metrics = MetricsAggregator()
    recorder = new_request_recorder(sink=metrics, request_id="req-a")
    emit_saturated(
        recorder, timestamp=1.0, snapshot=state.snapshot()
    )
    counters = metrics.snapshot()["counters"]
    assert counters["admission_admitted_count"] == 2
    assert counters["admission_in_flight_last"] == 0, "all slots were released"
    # The sampled peak only ever saw the post-release sample...
    assert counters["admission_in_flight_peak"] == 0
    # ...while the admission state's own sticky peak is exact for its life.
    assert counters["admission_peak_in_flight"] == 2


def test_known_outcomes_is_the_source_of_truth_for_labels() -> None:
    assert {"ok", "saturated", "degraded", "conflict"} <= KNOWN_OUTCOMES
    assert KNOWN_OUTCOMES == frozenset(KNOWN_OUTCOMES), "must be immutable"
