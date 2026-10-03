"""§15: a sink that breaks must be COUNTED without being REPORTED THROUGH.

THE PROBLEM
-----------
``TraceRecorder.sink_errors`` and ``trace_errors`` were per-instance counters
on an object that dies with the request. Nothing read them, nothing outlived
them, and the only way to see one was to hold the recorder — which a test can
do and a Worker never can. Describing them as "operationally observable"
because ``snapshot()`` exists was not true.

THE CONSTRAINT THAT SHAPES THE ANSWER
-------------------------------------
A sink failure must NOT be reported through the sink that just failed. That is
not a style preference: it is a loop. A sink that raises on every call would
be handed an event about its own failure, raise again, be handed another, and
the request would either spin or be taken down by a fault in a subsystem whose
entire job is to be disposable.

So the counters live in an ISOLATE-SCOPED accumulator that no sink owns, the
recorder writes to it directly, and the aggregator READS it when someone asks
for a snapshot. Reading is not emitting: nothing is ever handed back to a sink
to report a delivery failure.
"""

from __future__ import annotations

import pytest

from oai2.observability import (
    CollectingEventSink,
    EventCategory,
    EventType,
    MetricsAggregator,
    NullEventSink,
    new_request_recorder,
)
from oai2.observability.health import ISOLATE_RECORDER_HEALTH, RecorderHealth


@pytest.fixture(autouse=True)
def _health_reset() -> None:
    ISOLATE_RECORDER_HEALTH._reset_for_tests()


class ExplodingSink:
    """Raises on every delivery, and records how often it was asked."""

    def __init__(self) -> None:
        self.attempts = 0

    def emit(self, event: object) -> None:
        self.attempts += 1
        raise RuntimeError("sink is down")


class BrokenAggregator(MetricsAggregator):
    """A metrics sink that is itself broken.

    This is the case the design has to survive: the thing that would normally
    report the fault is the thing that is faulty.
    """

    def emit(self, event: object) -> None:
        raise RuntimeError("aggregator is down")


def _record(recorder: object, count: int = 1) -> None:
    for _ in range(count):
        recorder.record(  # type: ignore[attr-defined]
            category=EventCategory.KNOWLEDGE,
            event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
            timestamp=1.0,
        )


def test_sink_errors_survive_the_request_that_caused_them() -> None:
    """RED at d0e271a: the only counter died with the recorder."""
    health = RecorderHealth()
    recorder = new_request_recorder(sink=ExplodingSink(), request_id="r")
    _record(recorder, 3)

    assert recorder.sink_errors == 3, "the per-instance count is still useful"
    assert health.snapshot() == {"sink_errors": 0, "trace_errors": 0}


def test_an_isolate_scoped_health_accumulator_receives_the_fault() -> None:
    recorder = new_request_recorder(
        sink=ExplodingSink(), request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(recorder, 3)

    assert ISOLATE_RECORDER_HEALTH.snapshot() == {
        "sink_errors": 3,
        "trace_errors": 0,
    }


def test_a_sink_failure_is_never_reported_through_that_sink() -> None:
    """The loop guard.

    If the failure were routed back through the sink, this sink would have
    been asked to deliver more than the three real events.
    """
    sink = ExplodingSink()
    recorder = new_request_recorder(
        sink=sink, request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(recorder, 3)

    assert sink.attempts == 3, (
        "the sink was handed events about its own failure; that is a loop"
    )
    assert ISOLATE_RECORDER_HEALTH.snapshot()["sink_errors"] == 3


def test_a_broken_aggregator_still_gets_its_faults_counted() -> None:
    """The sink that is failing is a MetricsAggregator.

    Its own ``emit`` is unusable, so if the count were delivered through the
    normal path it would be lost exactly when it mattered most.
    """
    broken = BrokenAggregator()
    recorder = new_request_recorder(
        sink=broken, request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(recorder, 2)

    assert ISOLATE_RECORDER_HEALTH.snapshot()["sink_errors"] == 2
    assert broken.counter("events_observed") == 0, (
        "nothing reached the aggregator, which is exactly the symptom an "
        "operator would otherwise see with no explanation"
    )


def test_a_recorder_without_health_still_counts_per_instance() -> None:
    """Health is opt-in wiring, not a new requirement on every call site."""
    recorder = new_request_recorder(sink=ExplodingSink(), request_id="r")
    _record(recorder, 1)
    assert recorder.sink_errors == 1
    assert ISOLATE_RECORDER_HEALTH.snapshot()["sink_errors"] == 0


def test_the_isolate_aggregator_surfaces_health_in_its_snapshot() -> None:
    """Otherwise the count is a number nobody can read, which is the old bug.

    The aggregator READS the accumulator when a snapshot is taken. It never
    emits into it, and nothing hands it an event about its own failure.
    """
    from oai2.observability.isolate import ISOLATE_METRICS

    ISOLATE_METRICS._reset_for_tests()
    sink = ISOLATE_METRICS.sink_for(enabled=True, max_recent_events=8)
    assert isinstance(sink, MetricsAggregator)

    recorder = new_request_recorder(
        sink=ExplodingSink(), request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(recorder, 4)

    assert sink.snapshot()["recorder_health"] == {  # type: ignore[attr-defined]
        "sink_errors": 4,
        "trace_errors": 0,
        # True distinguishes "faults are being watched and there are none" from
        # "nothing is watching". Collapsing those two is how a dead telemetry
        # path looks identical to a healthy one.
        "observed": True,
    }
    ISOLATE_METRICS._reset_for_tests()


def test_an_unwired_aggregator_says_so_rather_than_reporting_zero() -> None:
    """A bare aggregator claims nothing is watching, which is the truth.

    Reporting ``observed: False`` beats reporting zeros, because zeros read as
    a measurement and would let a deployment with no health wiring look exactly
    like one that has checked and found nothing wrong.
    """
    health = MetricsAggregator().snapshot()["recorder_health"]
    assert health == {
        "sink_errors": 0,
        "trace_errors": 0,
        "observed": False,
    }


def test_a_null_sink_is_not_a_fault() -> None:
    """Telemetry being off is a configuration, not an error.

    Counting deliberate no-ops as sink errors would make the "0 requested"
    case indistinguishable from "everything is broken", which is the one
    distinction that matters when reading the number.
    """
    recorder = new_request_recorder(
        sink=NullEventSink(), request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(recorder, 2)

    assert ISOLATE_RECORDER_HEALTH.snapshot()["sink_errors"] == 0
    assert recorder.sink_errors == 0


def test_health_is_bounded_and_resettable() -> None:
    recorder = new_request_recorder(
        sink=ExplodingSink(), request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(recorder, 1000)
    snapshot = ISOLATE_RECORDER_HEALTH.snapshot()
    assert snapshot == {"sink_errors": 1000, "trace_errors": 0}

    ISOLATE_RECORDER_HEALTH._reset_for_tests()
    assert ISOLATE_RECORDER_HEALTH.snapshot() == {
        "sink_errors": 0,
        "trace_errors": 0,
    }


def test_an_exploding_sink_still_cannot_change_request_semantics() -> None:
    """REQ-OBS-025, re-proven with the health wiring in place.

    Adding an accumulator must not have introduced a path where telemetry
    trouble can fail a request.
    """
    good = CollectingEventSink()
    bad = ExplodingSink()

    good_recorder = new_request_recorder(sink=good, request_id="r")
    bad_recorder = new_request_recorder(
        sink=bad, request_id="r", health=ISOLATE_RECORDER_HEALTH
    )
    _record(good_recorder, 2)
    _record(bad_recorder, 2)

    assert len(good.events) == 2
    assert good.events[0].event_type == bad_recorder.events[0].event_type
    assert good.events[0].seq == bad_recorder.events[0].seq
    assert good.events[0].trace_id == good_recorder.trace_id
    assert ISOLATE_RECORDER_HEALTH.snapshot()["sink_errors"] == 2


# ---------------------------------------------------------------------------
# Containment is only one level deep, or it is not containment.
# ---------------------------------------------------------------------------


class _ExplodingHealth(RecorderHealth):
    """A health reporter that fails while reporting a failure."""

    def note_sink_error(self) -> None:
        raise RuntimeError("health reporter is down")

    def note_trace_error(self) -> None:
        raise RuntimeError("health reporter is down")


class _ExplodingSink:
    """A sink that always fails, to drive the health path."""

    def emit(self, event: object) -> None:
        raise RuntimeError("sink is down")


class _WorkingSink:
    def __init__(self) -> None:
        self.events: list[object] = []

    def emit(self, event: object) -> None:
        self.events.append(event)


def test_a_failing_health_reporter_cannot_break_the_request() -> None:
    """REQ-OBS-025: telemetry must not change request semantics.

    `record` is documented as never raising, and the module states the
    property exists so that "a broken one must not turn a successful D1 /
    R2 / Vectorize / KV operation into a failed knowledge request".

    Reporting a fault is itself a call into a collaborator, so without a
    guard around it the containment was one level deep: a health reporter
    that raised converted a *telemetry* failure into a *request* failure.
    The observer broke the operation it observes.
    """
    recorder = new_request_recorder(
        sink=_ExplodingSink(), request_id="r", health=_ExplodingHealth()
    )
    # Must not raise.
    _record(recorder, 2)
    # And the fault is still counted on the recorder's own accumulator, which
    # no collaborator owns, so the failure is not lost -- only the report of
    # it is contained.
    assert recorder.snapshot()["sink_errors"] == 2


def test_a_failing_health_reporter_does_not_break_a_healthy_sink() -> None:
    """The counterpart: a working sink still receives its events.

    A containment fix that swallowed the emit path would pass the test above
    while silently dropping every event, so the healthy path is pinned
    explicitly.
    """
    sink = _WorkingSink()
    recorder = new_request_recorder(
        sink=sink, request_id="r", health=_ExplodingHealth()
    )
    _record(recorder, 3)
    assert len(sink.events) == 3
    assert recorder.snapshot()["sink_errors"] == 0


def test_a_working_health_reporter_is_still_notified() -> None:
    """The opposite direction: the guard must not silence a healthy reporter.

    `_note_quietly` swallows everything, so a regression that stopped calling
    the reporter entirely would be invisible. Health notification is a real
    side effect and is pinned.
    """

    class _CountingHealth(RecorderHealth):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def note_sink_error(self) -> None:
            self.calls.append("sink")
            super().note_sink_error()

    health = _CountingHealth()
    recorder = new_request_recorder(
        sink=_ExplodingSink(), request_id="r", health=health
    )
    _record(recorder)
    assert health.calls == ["sink"]
    assert health.snapshot()["sink_errors"] == 1
    assert recorder.snapshot()["sink_errors"] == 1


def test_a_health_reporter_is_optional() -> None:
    """No health reporter must remain a supported configuration."""
    recorder = new_request_recorder(sink=_ExplodingSink(), request_id="r")
    _record(recorder)
    assert recorder.snapshot()["sink_errors"] == 1
