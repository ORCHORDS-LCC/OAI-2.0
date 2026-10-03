"""#56 emission: the schema as a REAL production-path event stream.

Everything here drives the actual Worker entrypoint, transport or runtime, with
only the bound resources faked. The claim under test is not "the schema is
well-formed" — that is ``test_observability_events.py`` — but "the real request
path produces these events, in this order, and a broken observer cannot change
what the request does".

Section headings name the requirement from the work item.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

from oai2.knowledge.admission import (
    ISOLATE_ADMISSION,
    PUBLIC_SATURATION_MESSAGE,
    AdmissionState,
    KnowledgeSaturatedError,
)
from oai2.knowledge.cloudflare_runtime import (
    AsyncCloudflareKnowledgeRuntime,
    KnowledgeConflictError,
    KnowledgeIntegrityError,
)
from oai2.knowledge.observability import (
    KV_OPERATION_GET,
    KV_OPERATION_PUT,
    admission_counters,
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
from oai2.knowledge.worker_transport import KnowledgeWorkerTransport
from oai2.observability import (
    CollectingEventSink,
    EventCategory,
    EventType,
    NullEventSink,
    TraceEvent,
    TraceRecorder,
    new_request_recorder,
)
from oai2.observability.isolate import ISOLATE_METRICS
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

#: A canary that stands in for caller content. If this string ever reaches an
#: event or a metric, something is copying data it must not.
#:
#: Deliberately NOT shaped like a credential. An earlier revision used a
#: ``ghp_``-prefixed value, and ``scripts/verify.py``'s public-safety scan
#: correctly flagged the TEST FILE for a hard-coded token — which is the scan
#: working as designed. A canary has to be recognisable to the assertion and
#: inert to every other tool.
SENTINEL = "CANARY_DO_NOT_LOG_9f3a2b1c4d"


class _ExplodingSink:
    def emit(self, event: object) -> None:
        raise RuntimeError("observability is down")


@pytest.fixture(autouse=True)
def _clean_admission():
    reset_admission()
    yield
    reset_admission()


def _emit_one(sink: object) -> None:
    recorder = TraceRecorder(sink=sink)  # type: ignore[arg-type]
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
        timestamp=1.0,
    )


# ---------------------------------------------------------------------------
# §2 — the sink contract
# ---------------------------------------------------------------------------


def test_the_null_sink_records_without_error() -> None:
    _emit_one(NullEventSink())


def test_a_sink_that_raises_is_contained_and_counted() -> None:
    """A broken observer must not become a broken request.

    The event is still recorded locally, so ordering and in-process inspection
    are unaffected, and the failure is visible rather than swallowed forever.
    """
    recorder = TraceRecorder(sink=_ExplodingSink(), request_id="req-1")
    event = recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
        timestamp=1.0,
    )
    assert event is not None, "the event must still be recorded locally"
    assert len(recorder.events) == 1
    assert recorder.sink_errors == 1
    assert recorder.trace_errors == 0, (
        "a sink fault is not our bug; the two counters must stay separate"
    )


def test_a_sink_failure_does_not_break_the_sequence() -> None:
    recorder = TraceRecorder(sink=_ExplodingSink())
    for index in range(3):
        recorder.record(
            category=EventCategory.MODEL,
            event_type=EventType.MODEL_REQUEST,
            timestamp=float(index),
        )
    assert [e.seq for e in recorder.events] == [0, 1, 2]
    assert recorder.sink_errors == 3


def test_the_collecting_sink_is_bounded_and_evicts_oldest_first() -> None:
    sink = CollectingEventSink(max_events=3)
    recorder = TraceRecorder(sink=sink)
    for index in range(10):
        recorder.record(
            category=EventCategory.MODEL,
            event_type=EventType.MODEL_REQUEST,
            timestamp=float(index),
        )
    assert len(sink) == 3, "an unbounded sink is a memory leak in any long process"
    assert [e.timestamp for e in sink.events] == [7.0, 8.0, 9.0]
    assert sink.event_types() == ("model.request",) * 3


def test_the_collecting_sink_rejects_a_nonsensical_bound() -> None:
    with pytest.raises(ValueError):
        CollectingEventSink(max_events=0)
    with pytest.raises(TypeError):
        CollectingEventSink(max_events="ten")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §4 — one trace per accepted request, correlated across layers
# ---------------------------------------------------------------------------


def test_one_request_produces_one_trace_across_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """entrypoint -> admission -> transport -> completion, one trace id.

    The failure mode this pins is a trace minted per LAYER: several
    correlated-looking groups that share nothing. The transport is handed the
    entrypoint's own recorder object, so identity cannot have diverged.
    """
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        components = Components(Gate(hold=False))
        wire(monkeypatch, components)
        response = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )

    assert response.status == 200
    assert components.traces and components.traces[0] is not None
    assert len({e.trace_id for e in sink.events}) == 1, (
        "a request must not produce more than one trace identity"
    )
    assert sink.events[0].trace_id == components.traces[0].trace_id


def test_request_id_is_the_wire_correlation_inside_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        asyncio.run(drive(monkeypatch, entry, Env(), valid_body("req-xyz")))

    assert {e.request_id for e in sink.events} == {"req-xyz"}
    # trace_id must NOT have replaced request_id, nor been added to the wire.
    assert "trace_id" not in KnowledgeTransportResponse.model_fields, (
        "adding trace_id to the public response would be an unauthorised wire "
        "change"
    )


def test_only_validated_requests_are_traced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scope is narrow on purpose.

    A malformed body or a failed auth is a transport-boundary reject, not a
    knowledge operation. Counting it would make "requests started" disagree
    with what the runtime actually attempted.
    """
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        instance = entry.Default(Env())
        for request in (
            Request(body={"request_id": "x"}),
            Request(method="GET"),
            Request(
                body=valid_body(), headers={"Authorization": "Bearer wrong"}
            ),
        ):
            asyncio.run(instance.fetch(request))
    assert sink.events == (), "boundary rejects must not appear as knowledge ops"


# ---------------------------------------------------------------------------
# §5 / §8 — entrypoint lifecycle, exactly one completion
# ---------------------------------------------------------------------------


def test_a_successful_request_emits_started_then_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        response = asyncio.run(
            drive(monkeypatch, entry, Env(), valid_body("req-1"))
        )

    assert response.status == 200
    assert sink.event_types() == (
        "knowledge.request.started",
        "knowledge.request.completed",
    )
    assert sink.events[0].operation == "get", (
        "the operation is a structured field, not prose in `detail`"
    )
    assert sink.events[1].outcome == "ok"
    assert sink.events[1].retryable is False


def test_a_saturated_request_emits_started_saturated_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
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

        refused = asyncio.run(scenario())

    assert refused.status == 503
    assert refused.payload["error"]["code"] == "saturated"

    # The sink saw BOTH requests. A is held open while B completes, so the
    # two traces interleave; the required shape is per request, and the two
    # must also be distinguishable by trace.
    assert len({e.trace_id for e in sink.events}) == 2, (
        "two requests must produce two distinct traces"
    )
    for_refused = [e for e in sink.events if e.request_id == "B"]
    assert tuple(e.event_type for e in for_refused) == (
        "knowledge.request.started",
        "knowledge.saturated",
        "knowledge.request.completed",
    )
    assert [e.seq for e in for_refused] == [0, 1, 2]

    saturated = for_refused[1]
    assert saturated.admission is not None
    assert saturated.retryable is True
    completed = for_refused[2]
    assert completed.outcome == "saturated"
    assert completed.retryable is True


def test_a_refused_request_never_reaches_any_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must not build components, embed, or touch a binding.

    A saturated request that still constructed the runtime would do the very
    expensive work admission exists to avoid.
    """
    builds: list[int] = []
    with entrypoint() as entry:
        entry.Default.event_sink = CollectingEventSink()
        gate = Gate(hold=True)
        wire(monkeypatch, Components(gate))

        async def scenario() -> object:
            env = Env(KNOWLEDGE_MAX_IN_FLIGHT="1")
            held = asyncio.create_task(
                entry.Default(env).fetch(Request(valid_body("A")))
            )
            await gate.entered.wait()

            async def counting_build(**_kwargs: object) -> Components:
                builds.append(1)
                return Components(Gate(hold=False))

            entry.build_cloudflare_knowledge_components = counting_build
            refused = await entry.Default(env).fetch(Request(valid_body("B")))
            gate.release.set()
            await held
            return refused

        response = asyncio.run(scenario())

    assert response.status == 503
    assert builds == [], "the refused request built components"


def test_component_initialisation_failure_emits_internal_then_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_components()`` runs with ensure_schema=False: local assembly, no
    remote call. An exception from it is not a dependency outage.

    The earlier version reported UNAVAILABLE_DEPENDENCY here, which told a
    caller a dependency was down and an operator to go and look at D1 when the
    fault was local construction.
    """
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink

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
    assert response.payload["error"]["retryable"] is False
    assert sink.event_types() == (
        "knowledge.request.started",
        "knowledge.internal.failure",
        "knowledge.request.completed",
    )


def test_a_misconfigured_limit_still_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle ownership does not stop at the admission door."""
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        response = asyncio.run(
            drive(
                monkeypatch, entry, Env(KNOWLEDGE_MAX_IN_FLIGHT="0"),
                valid_body("req-1"),
            )
        )
    assert response.status == 500
    assert sink.event_types() == (
        "knowledge.request.started",
        "knowledge.request.completed",
    )
    assert sink.events[1].outcome == "internal"


@pytest.mark.parametrize(
    "code,expected_status",
    [
        (TransportErrorCode.NOT_FOUND, 404),
        (TransportErrorCode.INTERNAL, 500),
        (TransportErrorCode.CONFLICT, 409),
    ],
)
def test_every_admitted_request_completes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    code: TransportErrorCode,
    expected_status: int,
) -> None:
    """Not once per layer, and not zero times on an error path."""
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        canned = KnowledgeTransportResponse(
            request_id="req-1",
            ok=False,
            error=TransportError(code=code, message="m", retryable=False),
        )
        wire(
            monkeypatch,
            Components(Gate(hold=False), result=canned),
        )
        response = asyncio.run(
            entry.Default(Env()).fetch(Request(valid_body("req-1")))
        )

    assert response.status == expected_status
    completions = [
        e for e in sink.events if e.event_type == "knowledge.request.completed"
    ]
    assert len(completions) == 1, f"{len(completions)} completions emitted"
    assert [e.seq for e in sink.events] == list(range(len(sink.events)))
    assert completions[0].outcome == code.value


# ---------------------------------------------------------------------------
# §6 — the admission snapshot conversion
# ---------------------------------------------------------------------------


def test_saturation_counters_report_state_after_the_refusal() -> None:
    """Documented semantics, pinned.

    ``in_flight`` is the slots still held when the refusal was counted (a
    refusal frees nothing) and ``refused_count`` INCLUDES this refusal.
    """
    state = AdmissionState()
    lease = state.acquire(1)
    with pytest.raises(KnowledgeSaturatedError):
        state.acquire(1)
    counters = admission_counters(state.snapshot())
    assert counters.in_flight == 1, "a refusal does not free a slot"
    assert counters.refused_count == 1, "the current refusal is counted"
    lease.release()


def test_the_conversion_drops_last_limit() -> None:
    """Configuration is not observed load, and must not ride along."""
    snapshot = AdmissionState().snapshot()
    assert "last_limit" in snapshot
    dumped = admission_counters(snapshot).model_dump()
    assert "last_limit" not in dumped
    assert set(dumped) == {
        "in_flight", "peak_in_flight", "admitted_count", "refused_count"
    }


def test_the_conversion_refuses_a_malformed_snapshot() -> None:
    base = {
        "in_flight": 0, "peak_in_flight": 0,
        "admitted_count": 0, "refused_count": 0,
    }
    for key, bad in (("in_flight", None), ("in_flight", -1), ("in_flight", True)):
        with pytest.raises(ValueError):
            admission_counters({**base, key: bad})  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        admission_counters({})  # type: ignore[arg-type]


def test_counters_never_reach_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry carries capacity; the response must not."""
    with entrypoint() as entry:
        entry.Default.event_sink = CollectingEventSink()
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

        response = asyncio.run(scenario())

    assert response.payload["error"]["message"] == PUBLIC_SATURATION_MESSAGE
    assert "in_flight" not in str(response.payload)
    assert "1" not in response.payload["error"]["message"]


# ---------------------------------------------------------------------------
# §7 — transport failure detail
# ---------------------------------------------------------------------------


def _request() -> KnowledgeTransportRequest:
    return KnowledgeTransportRequest(
        request_id="req-t",
        version="1",
        auth=TransportAuthContext(
            subject="s", capabilities=("knowledge.read", "knowledge.write")
        ),
        operation=TransportOperation.GET,
        knowledge_id="ko_x",
    )


class _ExplodingRuntime:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def put(self, obj: object, *, vector: object = None,
                  now: float | None = None, trace: object = None) -> int:
        raise self.error

    async def get(self, knowledge_id: str, *, trace: object = None) -> object:
        raise self.error

    async def retrieve(self, request: object, *, query_vector: object = None,
                       trace: object = None) -> object:
        raise self.error


@pytest.mark.parametrize(
    "error,expected_type,expected_outcome,expected_retryable",
    [
        (KnowledgeConflictError("revision moved"), "knowledge.conflict",
         "conflict", True),
        (KnowledgeIntegrityError("body hash mismatch"),
         "knowledge.integrity.failure", "integrity", False),
        (RuntimeError("d1 refused the connection"),
         "knowledge.dependency.failure", "unavailable_dependency", True),
        (ValueError("a bug in our own code"), "knowledge.internal.failure",
         "internal", False),
    ],
)
def test_each_failure_class_emits_its_own_event(
    error: Exception,
    expected_type: str,
    expected_outcome: str,
    expected_retryable: bool,
) -> None:
    """§7. An unexpected exception must NOT be labelled a dependency failure.

    Conflating them would page whoever owns D1 for a bug in this repository,
    and would make failure-by-outcome describe infrastructure health that does
    not exist.
    """
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-t")
    transport = KnowledgeWorkerTransport(
        _ExplodingRuntime(error)  # type: ignore[arg-type]
    )

    response = asyncio.run(transport.handle(_request(), now=1.0, trace=recorder))

    assert response.ok is False
    assert sink.event_types() == (expected_type,)
    event = sink.events[0]
    assert event.outcome == expected_outcome
    assert event.retryable is expected_retryable
    assert event.request_id == "req-t"
    assert event.trace_id == recorder.trace_id, (
        "the transport must not mint a second trace"
    )


def test_the_transport_never_emits_a_completion() -> None:
    """Lifecycle ownership is the entrypoint's; a second one would duplicate."""
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-t")
    transport = KnowledgeWorkerTransport(
        _ExplodingRuntime(KnowledgeConflictError("x"))  # type: ignore[arg-type]
    )
    asyncio.run(transport.handle(_request(), now=1.0, trace=recorder))
    assert "knowledge.request.completed" not in sink.event_types()


def test_a_transport_call_without_a_recorder_emits_nothing() -> None:
    """No recorder means no trace to correlate into, so nothing is invented."""
    transport = KnowledgeWorkerTransport(
        _ExplodingRuntime(KnowledgeConflictError("x"))  # type: ignore[arg-type]
    )
    assert asyncio.run(transport.handle(_request(), now=1.0)).ok is False


# ---------------------------------------------------------------------------
# §9 — KV degradation, where the exception exists
# ---------------------------------------------------------------------------


class _Kv:
    def __init__(self, *, get_error: Exception | None = None) -> None:
        self.get_error = get_error
        self.gets = 0

    async def get_text(self, key: str) -> str | None:
        self.gets += 1
        if self.get_error is not None:
            raise self.get_error
        return None

    async def put_text(self, key: str, value: str, *,
                       ttl_seconds: int | None = None) -> None:
        return None


def _runtime(kv: _Kv) -> AsyncCloudflareKnowledgeRuntime:
    class _Revision:
        async def corpus_revision(self) -> int:
            return 0

    return AsyncCloudflareKnowledgeRuntime(
        reader=_Revision(),  # type: ignore[arg-type]
        writer=_Revision(),  # type: ignore[arg-type]
        r2=None,  # type: ignore[arg-type]
        vectorize=None,  # type: ignore[arg-type]
        kv=kv,  # type: ignore[arg-type]
        embedding_version="embed-v1",
        embedding_digest="digest-v1",
    )


def test_a_kv_get_that_raises_emits_degraded() -> None:
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-kv")
    runtime = _runtime(_Kv(get_error=RuntimeError("kv namespace gone")))

    lookup = asyncio.run(runtime._cache_get("key", trace=recorder))

    assert lookup.degraded is True, (
        "an unavailable cache and an empty one are different operational "
        "problems and must not reach the caller as the same value"
    )
    assert lookup.raw is None
    assert sink.event_types() == ("knowledge.kv.degraded",)
    event = sink.events[0]
    assert event.operation == KV_OPERATION_GET, (
        "the cache operation is a structured label, not prose"
    )
    assert event.retryable is True
    assert event.trace_id == recorder.trace_id


def test_a_kv_get_that_raises_is_not_reported_as_a_miss() -> None:
    """A miss is the cache working; an exception is the cache failing to answer.

    Reporting the second as the first is how a permanently broken KV namespace
    survives unnoticed: nothing errors and nothing alerts, every request is
    just slower, and the hit rate looks like a cold cache rather than a fault.
    """
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-kv")
    runtime = _runtime(_Kv(get_error=RuntimeError("kv namespace gone")))

    asyncio.run(runtime._cache_get("key", trace=recorder))

    assert sink.event_types() == ("knowledge.kv.degraded",)
    assert "knowledge.cache.miss" not in sink.event_types()


def test_the_kv_detail_is_a_fixed_vocabulary() -> None:
    """Exception text can carry endpoint, namespace and account detail."""
    recorder = new_request_recorder(sink=CollectingEventSink())
    with pytest.raises(ValueError, match="fixed vocabulary"):
        emit_kv_degraded(recorder, operation=f"get {SENTINEL}", timestamp=1.0)
    for good in (KV_OPERATION_GET, KV_OPERATION_PUT):
        emit_kv_degraded(recorder, operation=good, timestamp=1.0)


# ---------------------------------------------------------------------------
# §11 — knowledge.retry has no emitter, and that must stay true
# ---------------------------------------------------------------------------


def test_no_source_module_emits_knowledge_retry() -> None:
    """A retryable flag means a retry MAY happen. Only an actor retrying emits.

    Emitting from a ``retryable=True`` response would make metrics count
    intentions as facts, and would over-report retries by exactly the rate at
    which clients honour the hint.
    """
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for directory in ("oai2", "deploy"):
        for path in (root / directory).rglob("*.py"):
            source = path.read_text()
            # Defining the type or the constructor helper is allowed; calling
            # the emitter from a request path is not.
            calls = source.count("emit_knowledge_retry(")
            calls += source.count("knowledge_retry(") - source.count(
                "def knowledge_retry("
            )
            if calls > 0:
                offenders.append(f"{path.relative_to(root)}({calls})")
    assert offenders == [], f"knowledge.retry is emitted from {offenders}"


def test_a_retryable_response_does_not_produce_a_retry_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
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

    assert "knowledge.saturated" in sink.event_types()
    assert "knowledge.retry" not in sink.event_types()


# ---------------------------------------------------------------------------
# §12 — required trace order
# ---------------------------------------------------------------------------


def test_the_documented_order_holds_for_the_degraded_retrieval_shape() -> None:
    """0 started, 1 kv.degraded, 2 completed — through the real Trace."""
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-r")

    emit_request_started(recorder, operation="retrieve", timestamp=1.0)
    emit_kv_degraded(recorder, operation=KV_OPERATION_GET, timestamp=2.0)
    emit_completed(recorder, timestamp=3.0, ok=True, outcome="ok")

    assert sink.event_types() == (
        "knowledge.request.started",
        "knowledge.kv.degraded",
        "knowledge.request.completed",
    )
    assert [e.seq for e in sink.events] == [0, 1, 2]


def test_the_saturated_shape_has_no_gap_and_no_duplicate() -> None:
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-s")
    emit_request_started(recorder, operation="get", timestamp=1.0)
    emit_saturated(
        recorder, timestamp=2.0, snapshot=AdmissionState().snapshot()
    )
    emit_completed(
        recorder, timestamp=3.0, ok=False, outcome="saturated", retryable=True
    )
    assert [e.seq for e in sink.events] == [0, 1, 2]
    assert len(sink.event_types()) == 3


# ---------------------------------------------------------------------------
# §13 — public safety
# ---------------------------------------------------------------------------


def test_no_event_carries_a_secret_from_a_raised_dependency() -> None:
    """The transport's failure message may contain it; telemetry must not.

    A driver exception is the classic leak vector, and exactly the string a
    naive implementation would copy into ``detail``.
    """
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-t")
    transport = KnowledgeWorkerTransport(
        _ExplodingRuntime(  # type: ignore[arg-type]
            RuntimeError(f"connect failed token={SENTINEL} bucket=my-bucket")
        )
    )
    asyncio.run(transport.handle(_request(), now=1.0, trace=recorder))

    assert sink.events, "the failure must still be reported"
    for event in sink.events:
        dumped = str(event.model_dump(mode="json"))
        assert SENTINEL not in dumped
        assert "my-bucket" not in dumped


def test_a_sentinel_in_a_kv_exception_never_reaches_an_event() -> None:
    sink = CollectingEventSink()
    recorder = new_request_recorder(sink=sink, request_id="req-kv")
    runtime = _runtime(_Kv(get_error=RuntimeError(f"ns={SENTINEL} failed")))
    asyncio.run(runtime._cache_get("secret-cache-key", trace=recorder))
    for event in sink.events:
        dumped = str(event.model_dump(mode="json"))
        assert SENTINEL not in dumped
        assert "secret-cache-key" not in dumped


def test_the_started_event_records_the_operation_not_the_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The topic is caller content. Only the operation name is recorded."""
    sink = CollectingEventSink()
    with entrypoint() as entry:
        entry.Default.event_sink = sink
        body = {
            "request_id": "req-r",
            "version": "1",
            "operation": "retrieve",
            "topic": f"quarterly revenue {SENTINEL}",
            "limit": 5,
        }
        asyncio.run(drive(monkeypatch, entry, Env(), body))

    assert sink.events[0].operation == "retrieve"
    for event in sink.events:
        assert SENTINEL not in str(event.model_dump(mode="json"))


def test_the_event_model_has_exactly_the_declared_fields() -> None:
    """A new field is a review event; this pins the surface."""
    assert set(TraceEvent.model_fields) == {
        "schema_version", "trace_id", "seq", "category", "event_type",
        "timestamp", "operation", "duration_ms", "request_id", "evidence_ids",
        "admission", "outcome", "retryable", "attempt", "detail",
    }


def test_no_admission_counter_field_names_a_resource() -> None:
    from oai2.observability import AdmissionCounters

    assert set(AdmissionCounters.model_fields) == {
        "in_flight", "peak_in_flight", "admitted_count", "refused_count"
    }


# ---------------------------------------------------------------------------
# §19 — telemetry changes nothing about the request
# ---------------------------------------------------------------------------


def test_disabling_telemetry_produces_an_identical_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same response, same code, same retryable — only artifacts differ."""
    results = []
    for sink in (NullEventSink(), CollectingEventSink()):
        reset_admission()
        with entrypoint() as entry:
            entry.Default.event_sink = sink
            response = asyncio.run(
                drive(monkeypatch, entry, Env(), valid_body("req-1"))
            )
            results.append((response.status, response.payload))
    assert results[0] == results[1]
    assert results[0][0] == 200


def test_a_throwing_sink_does_not_change_a_successful_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headline guarantee of §3, end to end through the entrypoint."""
    results = []
    for sink in (NullEventSink(), _ExplodingSink()):
        reset_admission()
        with entrypoint() as entry:
            entry.Default.event_sink = sink
            response = asyncio.run(
                drive(monkeypatch, entry, Env(), valid_body("req-1"))
            )
            results.append((response.status, response.payload))
    assert results[0] == results[1], (
        "a broken observer changed what the request did"
    )
    assert results[0][0] == 200


def test_a_throwing_sink_does_not_change_a_failure_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original failure class must remain original."""
    results = []
    for sink in (NullEventSink(), _ExplodingSink()):
        reset_admission()
        with entrypoint() as entry:
            entry.Default.event_sink = sink

            async def exploding_build(**_kwargs: object) -> Components:
                raise RuntimeError("d1 binding unavailable")

            response = asyncio.run(
                drive(
                    monkeypatch, entry, Env(), valid_body("req-1"),
                    build=exploding_build,
                )
            )
            results.append((response.status, response.payload))
    assert results[0] == results[1]
    assert results[0][0] == 500
    assert results[0][1]["error"]["code"] == "internal"


@pytest.fixture(autouse=True)
def _isolate_entry_module():
    """Drop the cached entrypoint module after every test.

    ``Default.event_sink`` is a CLASS attribute, so a test that substitutes a
    sink would otherwise leave it installed on the cached module and change
    the starting state of whichever test runs next. That is a real property of
    the production design — a sink is process configuration, set once — and it
    has to be undone between tests rather than ignored.
    """
    yield
    sys.modules.pop("deploy.cloudflare.knowledge_worker.entry", None)


def test_the_entrypoint_default_sink_is_the_no_op() -> None:
    """Instrumentation must never be a prerequisite for serving.

    Asserted on the RESOLVED sink rather than on the class attribute, because
    the attribute stopped being the answer when metrics became configurable.
    What matters is that an UNCONFIGURED Worker still drops every event — not
    what the default happens to be spelled as.
    """
    with entrypoint() as entry:
        ISOLATE_METRICS._reset_for_tests()
        assert entry.Default.event_sink is None, (
            "nothing is installed by default; the sink is resolved from "
            "configuration per request"
        )
        assert isinstance(
            entry._resolve_sink(Env(), entry.Default), NullEventSink
        )


def test_no_domain_model_gained_a_trace_field() -> None:
    """Trace identity is request execution context, not stored identity.

    Persisting it would make two otherwise-identical authoritative rows differ
    for a reason that has nothing to do with their content.
    """
    import dataclasses

    from oai2.knowledge.abstraction import KnowledgeObject, RetrievalResult
    from oai2.knowledge.cloudflare import CFRow
    from oai2.knowledge.transport import VectorizeMetadata

    for model in (KnowledgeObject, RetrievalResult, VectorizeMetadata, CFRow):
        if dataclasses.is_dataclass(model):
            names = {f.name for f in dataclasses.fields(model)}
        else:
            names = set(model.model_fields)
        assert "trace_id" not in names, model.__name__
        assert "seq" not in names, f"{model.__name__} gained a sequence field"


# ---------------------------------------------------------------------------
# §10 — explicit plumbing, no ambient state
# ---------------------------------------------------------------------------


def test_the_runtime_does_not_store_a_recorder_on_self() -> None:
    """A parameter cannot outlive the call it was given to.

    ``self`` is rebuilt per request today, so a stored recorder would appear
    to work — which is exactly the shape that leaks a previous request's trace
    the moment anything caches the runtime.
    """
    runtime = _runtime(_Kv())
    # No recorder is STORED. The runtime is a plain class, so this is about
    # what exists, not about what assignment would allow.
    for name in ("recorder", "trace", "_trace", "_recorder", "current_trace"):
        assert not hasattr(runtime, name), name
    for name in ("put", "get", "retrieve", "_cache_get"):
        signature = inspect.signature(getattr(runtime, name))
        assert "trace" in signature.parameters, name
    # And a completed call leaves nothing behind.
    asyncio.run(
        runtime._cache_get("k", trace=new_request_recorder(request_id="r"))
    )
    for name in ("recorder", "trace", "_trace"):
        assert not hasattr(runtime, name), f"{name} persisted after a call"


def test_the_recorder_is_request_scoped_not_global() -> None:
    """No ambient 'current trace' a second request could inherit."""
    import oai2.observability as obs

    first = new_request_recorder(request_id="a")
    second = new_request_recorder(request_id="b")
    assert first.trace_id != second.trace_id
    assert not hasattr(obs, "current_trace")
    assert not hasattr(obs, "set_current_trace")


def test_the_entrypoint_sink_is_a_class_attribute_not_per_request_state() -> None:
    """A sink is configuration; admission state is what must be shared."""
    with entrypoint() as entry:
        # It lives on the class (configuration), and a per-invocation instance
        # does not carry its own copy (that would be per-request state).
        assert "event_sink" in vars(entry.Default)
        instance = entry.Default(Env())
        assert "event_sink" not in vars(instance)
        assert instance.event_sink is entry.Default.event_sink


def test_admission_state_is_still_module_scope_and_shared() -> None:
    """Instrumentation must not have moved admission onto the entrypoint."""
    from oai2.knowledge.admission import ISOLATE_ADMISSION as shared

    with entrypoint() as entry:
        assert entry.ISOLATE_ADMISSION is shared
        assert not hasattr(entry.Default, "ISOLATE_ADMISSION")


def test_the_real_isolate_admission_is_untouched_by_this_file() -> None:
    assert ISOLATE_ADMISSION.in_flight == 0
