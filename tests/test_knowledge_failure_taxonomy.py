"""RED reproductions for the correctness defects found in ``d0e271a``.

Two defects, both in the Worker entrypoint's failure classification:

1. The admission refusal is caught with ``except Exception``, so ANY exception
   raised by ``acquire_admission`` is reported to the caller as a SATURATED
   503 with a retryable body. A ``RuntimeError`` from a broken admission
   primitive would be advertised to callers as routine backpressure.
2. Component construction and transport execution share one ``except
   Exception`` that reports UNAVAILABLE_DEPENDENCY with the message "dependency
   initialization failed" — while a source comment claims the transport was
   never reached. When the exception actually came FROM the transport, that
   comment is false and the classification is wrong.

Each test here is written to FAIL against the shipped code and pass after the
fix. They are kept in the suite afterwards, because a regression in failure
taxonomy is exactly the kind of defect that hides until a customer sees a 503
that should have been a 500.
"""

from __future__ import annotations

import asyncio

import pytest

from oai2.knowledge.admission import KnowledgeSaturatedError
from oai2.knowledge.observability import (
    KV_OPERATION_GET,
    emit_completed,
    emit_kv_degraded,
    emit_request_started,
    emit_saturated,
)
from oai2.knowledge.transport import (
    KnowledgeTransportRequest,
    KnowledgeTransportResponse,
    TransportAuthContext,
    TransportError,
    TransportErrorCode,
    TransportOperation,
)
from oai2.observability import CollectingEventSink, MetricsAggregator
from tests._worker_entrypoint_fixtures import (
    Components,
    Env,
    Gate,
    Request,
    drive,
    entrypoint,
    reset_admission,
    valid_body,
    wire,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_admission()
    yield
    reset_admission()


class _Admitting:
    """Replaces ``acquire_admission`` in the entrypoint module namespace."""

    def __init__(self, error: Exception | None) -> None:
        self.error = error

    def __call__(self, limit: int) -> object:
        if self.error is not None:
            raise self.error
        raise AssertionError("admission should have refused")  # pragma: no cover


def _patch_admission(entry: object, error: Exception | None) -> None:
    entry.acquire_admission = _Admitting(error)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Defect 1 — only KnowledgeSaturatedError is saturation
# ---------------------------------------------------------------------------


def test_a_saturated_error_is_reported_as_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        _patch_admission(entry, KnowledgeSaturatedError(limit=1, in_flight=1))
        response = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )

    assert response.status == 503
    assert response.payload["error"]["code"] == "saturated"
    assert response.payload["error"]["retryable"] is True
    assert metrics.counter("saturated_count") == 1
    assert metrics.counter("saturated_count") != 0


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("admission primitive is broken"),
        TypeError("admission was called with the wrong shape"),
    ],
)
def test_an_unexpected_admission_error_is_not_reported_as_saturation(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """A bug in the admission primitive is NOT backpressure.

    Advertising it to callers as a retryable 503 says "try again shortly" about
    a condition retrying cannot fix.
    """
    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        _patch_admission(entry, error)
        response = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )

    assert response.status == 500, (
        f"{type(error).__name__} was reported as "
        f"{response.payload['error']['code']}"
    )
    assert response.payload["error"]["code"] == "internal"
    assert response.payload["error"]["retryable"] is False
    assert metrics.counter("saturated_count") == 0, (
        "a non-saturation error produced a saturation event"
    )
    assert metrics.counter("internal_failure_count") == 1


def test_a_custom_exception_from_admission_is_not_saturation() -> None:
    class _Weird(Exception):
        pass

    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        _patch_admission(entry, _Weird("something entirely unexpected"))
        response = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )
    assert response.status == 500
    assert response.payload["error"]["code"] == "internal"
    assert metrics.counter("saturated_count") == 0


def test_a_bad_configured_limit_is_internal_not_saturation() -> None:
    """Distinct from the above: a ValueError is an OPERATOR error."""
    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        response = asyncio.run(
            entry.Default(Env(KNOWLEDGE_MAX_IN_FLIGHT="0")).fetch(
                Request(valid_body("req-1"))
            )
        )
    assert response.status == 500
    assert response.payload["error"]["code"] == "internal"
    assert metrics.counter("saturated_count") == 0
    assert metrics.counter("internal_failure_count") == 0, (
        "a misconfigured limit is an operator error, not a code fault; it "
        "completes as internal but is not a runtime internal failure"
    )


# ---------------------------------------------------------------------------
# Defect 2 — build is not execution
# ---------------------------------------------------------------------------


def test_a_component_construction_failure_is_internal_not_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_cloudflare_knowledge_components(ensure_schema=False)`` performs
    local object assembly and validation. It makes no remote dependency call,
    so an exception from it is not evidence that a dependency is unavailable.

    With ``ensure_schema=False`` there is no DDL, no binding round trip, and
    nothing to be "unavailable" about.
    """
    metrics = MetricsAggregator()
    with entrypoint() as entry:
        entry.Default.event_sink = metrics

        async def exploding_build(**_kwargs: object) -> Components:
            raise ValueError("embedding_version must be a non-empty string")

        response = asyncio.run(
            drive(
                monkeypatch, entry, Env(), valid_body("req-1"),
                build=exploding_build,
            )
        )

    assert response.status == 500
    assert response.payload["error"]["code"] == "internal"
    assert metrics.counter("dependency_failure_count") == 0, (
        "a local construction failure was reported as a dependency outage"
    )
    assert metrics.counter("internal_failure_count") == 1


def test_an_exception_escaping_the_transport_is_internal_not_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KnowledgeWorkerTransport.handle`` converts KNOWN dependency, conflict
    and integrity errors into typed responses itself.

    So an exception ESCAPING it is by definition not one of those: it is an
    unexpected local failure. Labelling it UNAVAILABLE_DEPENDENCY tells the
    caller a dependency is down, and tells an operator to look at D1 when the
    bug is here.
    """
    for error in (ValueError("a bug"), RuntimeError("also a bug")):
        reset_admission()
        metrics = MetricsAggregator()
        with entrypoint() as entry:
            entry.Default.event_sink = metrics
            gate = Gate(hold=False)
            wire(monkeypatch, Components(gate, error=error))
            response = asyncio.run(
                entry.Default(Env()).fetch(Request(valid_body("req-1")))
            )
            assert response.status == 500, f"{type(error).__name__}"
            assert response.payload["error"]["code"] == "internal", (
                f"{type(error).__name__} was misreported"
            )
            assert response.payload["error"]["retryable"] is False
        assert metrics.counter("dependency_failure_count") == 0
        assert metrics.counter("internal_failure_count") == 1


def test_the_two_internal_boundaries_report_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build failure and transport failure must not collapse into one message.

    Both are INTERNAL and both are 500, so their EVENT is deliberately the
    same. Their MESSAGE is the only thing that tells an operator which
    half of the request path broke, and re-merging the two boundaries would
    quietly destroy that distinction while leaving every status code and
    every counter exactly as green as before.

    This is the control that a taxonomy test alone cannot provide: the
    classification can be right on both sides and the regression still
    happen.
    """
    messages: list[str] = []

    with entrypoint() as entry:
        entry.Default.event_sink = MetricsAggregator()

        async def exploding_build(**_kwargs: object) -> Components:
            raise ValueError("embedding_version must be a non-empty string")

        build_failure = asyncio.run(
            drive(
                monkeypatch, entry, Env(), valid_body("req-1"),
                build=exploding_build,
            )
        )
        messages.append(str(build_failure.payload["error"]["message"]))

    reset_admission()

    with entrypoint() as entry:
        entry.Default.event_sink = MetricsAggregator()
        wire(monkeypatch, Components(Gate(hold=False), error=RuntimeError("boom")))
        transport_failure = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )
        messages.append(str(transport_failure.payload["error"]["message"]))

    assert build_failure.status == transport_failure.status == 500
    assert messages[0] != messages[1], (
        "the two internal boundaries report the same text, so a caller and an "
        "operator cannot tell which one failed"
    )
    assert "initialization" in messages[0]
    assert "transport" in messages[1]


def test_a_typed_dependency_failure_is_not_double_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed UNAVAILABLE_DEPENDENCY response is already correct.

    The entrypoint must not add a second failure event on top of it, or one
    outage counts twice.
    """
    metrics = MetricsAggregator()
    canned = KnowledgeTransportResponse(
        request_id="req-1",
        ok=False,
        error=TransportError(
            code=TransportErrorCode.UNAVAILABLE_DEPENDENCY,
            message="d1 refused the connection",
            retryable=True,
        ),
    )
    with entrypoint() as entry:
        entry.Default.event_sink = metrics
        wire(monkeypatch, Components(Gate(hold=False), result=canned))
        response = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )

    assert response.status == 503
    assert response.payload["error"]["code"] == "unavailable_dependency"
    # The fake transport emits no events, so the entrypoint is the only
    # possible source: it must emit none.
    assert metrics.counter("dependency_failure_count") == 0, (
        "the entrypoint added a failure event for an already-typed response"
    )
    assert metrics.counter("internal_failure_count") == 0
    assert metrics.counter("failed_requests") == 1


# ---------------------------------------------------------------------------
# Defect 4 — schema versioning for the additive `operation` field
# ---------------------------------------------------------------------------


def test_a_stored_one_zero_event_still_parses() -> None:
    """Forward compatibility: a trace written before `operation` existed."""
    from oai2.observability import TraceEvent

    legacy = {
        "schema_version": "1.0",
        "trace_id": "trc_0123456789abcdef",
        "seq": 0,
        "category": "knowledge",
        "event_type": "knowledge.request.started",
        "timestamp": 1.0,
        "request_id": "req-legacy",
        "detail": "get",
    }
    event = TraceEvent.model_validate(legacy)
    assert event.operation is None, (
        "a 1.0 event has no operation and must still load"
    )
    assert event.detail == "get", "its existing data is preserved"


def test_an_additive_field_is_a_minor_bump_not_a_major() -> None:
    from oai2.observability import OBSERVABILITY_SCHEMA_VERSION

    major, _, minor = OBSERVABILITY_SCHEMA_VERSION.partition(".")
    assert major == "1", "adding a field must not invent a new major"
    assert minor == "1", (
        f"an additive field must bump the minor to 1.1, got {minor!r}"
    )


# ---------------------------------------------------------------------------
# Defect 3 — structured operation, not a `detail` string
# ---------------------------------------------------------------------------


def test_every_knowledge_event_carries_the_structured_operation() -> None:
    """`detail` is prose; a metric needs a field it can group by."""
    from oai2.knowledge.observability import (
        emit_conflict,
        emit_dependency_failure,
        emit_integrity_failure,
        emit_internal_failure,
    )
    from oai2.observability import (
        EventCategory,
        new_request_recorder,
    )

    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-1", operation="put")
    emit_request_started(recorder, timestamp=1.0)
    emit_conflict(recorder, timestamp=2.0)
    emit_integrity_failure(recorder, timestamp=3.0)
    emit_dependency_failure(recorder, timestamp=4.0)
    emit_internal_failure(recorder, timestamp=5.0)
    emit_kv_degraded(recorder, operation=KV_OPERATION_GET, timestamp=6.0)
    emit_saturated(
        recorder,
        timestamp=7.0,
        # A SNAPSHOT, not AdmissionCounters: the conversion is the documented
        # seam, and passing the already-built model would bypass it.
        snapshot={
            "in_flight": 1, "peak_in_flight": 1,
            "admitted_count": 1, "refused_count": 1, "last_limit": 1,
        },
    )
    emit_completed(recorder, timestamp=8.0, ok=True, outcome="ok")

    assert len(sink) == 8
    for event in sink.events:
        if event.event_type == "knowledge.kv.degraded":
            # Documented exception: kv.degraded carries the CACHE operation,
            # not the request operation, because the useful question is which
            # cache call failed. On a `retrieve` request the request operation
            # is "retrieve" and the cache operation is "get".
            assert event.operation == KV_OPERATION_GET
            continue
        assert event.operation == "put", (
            f"{event.event_type} has no structured operation"
        )
        assert event.category is EventCategory.KNOWLEDGE
    # And the request operation is still available on the events that did not
    # override it, so a per-operation metric is possible.
    assert {e.operation for e in sink.events if e.event_type !=
            "knowledge.kv.degraded"} == {"put"}


def test_the_operation_is_bounded_and_not_free_text() -> None:
    from pydantic import ValidationError

    from oai2.observability import EventCategory, EventType, TraceEvent

    base = {
        "trace_id": "trc_0123456789abcdef",
        "seq": 0,
        "category": EventCategory.MODEL,
        "event_type": EventType.MODEL_REQUEST,
        "timestamp": 1.0,
    }
    assert TraceEvent(**base, operation="get").operation == "get"
    for bad in ("x" * 65, "get;drop", "a b", ""):
        with pytest.raises(ValidationError):
            TraceEvent(**base, operation=bad)


def test_operation_survives_a_json_round_trip() -> None:
    from oai2.observability import EventCategory, EventType, TraceEvent

    event = TraceEvent(
        trace_id="trc_0123456789abcdef",
        seq=0,
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
        timestamp=1.0,
        operation="retrieve",
    )
    reloaded = TraceEvent.model_validate(event.model_dump(mode="json"))
    assert reloaded.operation == "retrieve"
    assert reloaded.schema_version == "1.1"


# ---------------------------------------------------------------------------
# Defect 6 — metrics retention must not keep raw events
# ---------------------------------------------------------------------------


def test_metrics_retention_excludes_identifiers_and_free_text() -> None:
    """A future producer must not be able to make the metrics store an archive
    of request ids, topics and evidence ids just by emitting an event."""
    metrics = MetricsAggregator(max_recent_events=16)
    from oai2.observability import (
        EventCategory,
        EventType,
        new_request_recorder,
    )

    recorder = new_request_recorder(
        sink=metrics, request_id="req-secret", operation="get"
    )
    recorder.record(
        category=EventCategory.VERIFIER,
        event_type=EventType.VERIFIER_ASSESSMENT,
        timestamp=1.0,
        detail="a canary that must not be retained: CANARY_XYZ",
        evidence_ids=("EVID-SECRET-1",),
    )
    blob = str(metrics.snapshot()) + str(metrics.recent_events)
    assert "req-secret" not in blob
    assert "trc_" not in blob
    assert "EVID-SECRET-1" not in blob
    assert "CANARY_XYZ" not in blob


def test_the_recorder_can_explicitly_opt_out_of_inheriting_the_operation() -> None:
    from oai2.observability import EventCategory, EventType, new_request_recorder

    sink = CollectingEventSink()
    recorder = new_request_recorder(
        sink=sink, request_id="req-1", operation="get"
    )
    recorder.record(
        category=EventCategory.MODEL,
        event_type=EventType.MODEL_REQUEST,
        timestamp=1.0,
        operation=None,
    )
    assert sink.events[0].operation is None


def test_a_transport_request_still_validates() -> None:
    """Guard the helper the other tests lean on."""
    request = KnowledgeTransportRequest(
        request_id="req-1",
        version="1",
        auth=TransportAuthContext(
            subject="s", capabilities=("knowledge.read",)
        ),
        operation=TransportOperation.GET,
        knowledge_id="ko_x",
    )
    assert request.operation.value == "get"
