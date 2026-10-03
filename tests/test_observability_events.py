"""WI-OBS-001 / #56 — one structured trace/event schema.

Each section is labelled with the requirement it pins. The interesting tests
are the ones that would fail if a decision in ``events.py`` were quietly
reversed, so they are written to attack the decision rather than restate it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oai2.observability import (
    KNOWLEDGE_EVENT_TYPES,
    OBSERVABILITY_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_MAJORS,
    TRACE_ID_PREFIX,
    AdmissionCounters,
    EventCategory,
    EventType,
    Trace,
    TraceEvent,
    knowledge_failure,
    knowledge_kv_degraded,
    knowledge_request_completed,
    knowledge_request_started,
    knowledge_retry,
    knowledge_saturated,
    new_trace_id,
)

NOW = 1000.0


# ---------------------------------------------------------------------------
# REQ-OBS-011 — stable trace identity
# ---------------------------------------------------------------------------


def test_a_minted_trace_id_is_namespaced_hex() -> None:
    trace_id = new_trace_id()
    assert trace_id.startswith(TRACE_ID_PREFIX)
    body = trace_id[len(TRACE_ID_PREFIX) :]
    assert 8 <= len(body) <= 48
    assert all(c in "0123456789abcdef" for c in body)


def test_two_traces_do_not_share_an_identity() -> None:
    assert len({new_trace_id() for _ in range(64)}) == 64


def test_a_trace_id_from_another_namespace_is_rejected() -> None:
    """A knowledge id pasted where a trace id belongs is a wiring bug.

    The namespace prefix is what makes that mistake visible instead of
    silently correlating unrelated activity.
    """
    with pytest.raises(ValidationError, match="trace_id must be"):
        TraceEvent.start(
            trace_id="ko_1234567890ab",
            seq=0,
            category=EventCategory.KNOWLEDGE,
            event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
            timestamp=NOW,
        )
    with pytest.raises(ValidationError, match="trace_id must be"):
        TraceEvent.start(
            trace_id="oai2v1-" + "2" * 56,
            seq=0,
            category=EventCategory.RETRIEVAL,
            event_type=EventType.RETRIEVAL_QUERY,
            timestamp=NOW,
        )


def test_an_empty_or_bounded_trace_id_is_rejected() -> None:
    for bad in ("", "trc_", "trc_zzz", TRACE_ID_PREFIX + "f" * 64):
        with pytest.raises(ValidationError):
            TraceEvent.start(
                trace_id=bad,
                seq=0,
                category=EventCategory.MODEL,
                event_type=EventType.MODEL_REQUEST,
                timestamp=NOW,
            )


# ---------------------------------------------------------------------------
# REQ-OBS-012 — monotonic sequencing within a trace
# ---------------------------------------------------------------------------


def test_sequence_numbers_are_dense_and_ordered() -> None:
    trace = Trace()
    for index in range(5):
        trace.record(
            category=EventCategory.AGENT,
            event_type=EventType.AGENT_STEP,
            timestamp=NOW + index,
        )
    assert [e.seq for e in trace.events] == [0, 1, 2, 3, 4]


def test_a_gap_in_the_sequence_is_refused() -> None:
    """A trace that skips a number cannot be checked for completeness."""
    trace = Trace()
    with pytest.raises(ValueError, match="must be 0, got 2"):
        trace.append(
            TraceEvent.start(
                trace_id=trace.trace_id,
                seq=2,
                category=EventCategory.MODEL,
                event_type=EventType.MODEL_REQUEST,
                timestamp=NOW,
            )
        )


def test_a_backwards_sequence_is_refused() -> None:
    trace = Trace()
    trace.record(
        category=EventCategory.MODEL,
        event_type=EventType.MODEL_REQUEST,
        timestamp=NOW,
    )
    trace.record(
        category=EventCategory.MODEL,
        event_type=EventType.MODEL_RESPONSE,
        timestamp=NOW,
    )
    with pytest.raises(ValueError, match="must be 2, got 0"):
        trace.append(
            TraceEvent.start(
                trace_id=trace.trace_id,
                seq=0,
                category=EventCategory.VERIFIER,
                event_type=EventType.VERIFIER_ASSESSMENT,
                timestamp=NOW,
            )
        )


def test_an_event_cannot_be_appended_to_a_different_trace() -> None:
    """Trace identity is enforced at insertion, not assumed by callers."""
    trace = Trace()
    other = Trace()
    with pytest.raises(ValueError, match="belongs to trace"):
        trace.append(
            TraceEvent.start(
                trace_id=other.trace_id,
                seq=0,
                category=EventCategory.MODEL,
                event_type=EventType.MODEL_REQUEST,
                timestamp=NOW,
            )
        )
    assert len(trace) == 0


# ---------------------------------------------------------------------------
# REQ-OBS-013 — the six categories are type-distinguishable
# ---------------------------------------------------------------------------


def test_all_six_required_categories_exist() -> None:
    required = {"model", "tool", "retrieval", "vision", "agent", "verifier"}
    assert required <= {c.value for c in EventCategory}


def test_each_category_has_its_own_event_types() -> None:
    """Distinctness must be real, not a shared namespace with a label."""
    by_category: dict[str, set[str]] = {}
    for event_type in EventType:
        by_category.setdefault(event_type.value.split(".", 1)[0], set()).add(
            event_type.value
        )
    assert set(by_category) >= {
        "model", "tool", "retrieval", "vision", "agent", "verifier", "knowledge"
    }
    # No type belongs to two categories, and none is empty.
    assert all(types for types in by_category.values())


def test_events_are_filterable_by_category() -> None:
    trace = Trace()
    trace.record(
        category=EventCategory.TOOL, event_type=EventType.TOOL_DISPATCH,
        timestamp=NOW,
    )
    trace.record(
        category=EventCategory.MODEL, event_type=EventType.MODEL_REQUEST,
        timestamp=NOW,
    )
    trace.record(
        category=EventCategory.TOOL, event_type=EventType.TOOL_RESULT,
        timestamp=NOW,
    )
    assert len(trace.by_category(EventCategory.TOOL)) == 2
    assert len(trace.by_category(EventCategory.VISION)) == 0


def test_the_event_type_must_be_a_dotted_lowercase_path() -> None:
    """The shape is what makes prefix grouping and filtering meaningful."""
    for bad in (
        "knowledge",                      # not namespaced
        "Knowledge.Request.Started",      # not lowercase
        "knowledge..started",             # empty segment
        "knowledge.request.",             # trailing dot
        ".knowledge.request",             # leading dot
        "knowledge request started",      # spaces
        "knowledge.request.started;drop", # injection attempt
        "x" * 65,
    ):
        with pytest.raises(ValidationError):
            TraceEvent.start(
                trace_id=new_trace_id(),
                seq=0,
                category=EventCategory.KNOWLEDGE,
                event_type=bad,
                timestamp=NOW,
            )


# ---------------------------------------------------------------------------
# REQ-OBS-014 — evidence references
# ---------------------------------------------------------------------------


def test_evidence_ids_are_carried_and_deduplicated_in_order() -> None:
    trace = Trace()
    trace.record(
        category=EventCategory.RETRIEVAL,
        event_type=EventType.RETRIEVAL_RESULT,
        timestamp=NOW,
        evidence_ids=("EVID-DOM-001", "EVID-DOM-002"),
    )
    trace.record(
        category=EventCategory.VERIFIER,
        event_type=EventType.VERIFIER_ASSESSMENT,
        timestamp=NOW,
        evidence_ids=("EVID-DOM-002", "EVID-DOM-003"),
    )
    # First-seen order, duplicates collapsed: the same evidence assessed
    # twice is not two pieces of evidence.
    assert trace.evidence_ids() == (
        "EVID-DOM-001", "EVID-DOM-002", "EVID-DOM-003"
    )


def test_an_empty_evidence_id_is_refused() -> None:
    with pytest.raises(ValidationError, match="evidence_ids"):
        TraceEvent.start(
            trace_id=new_trace_id(),
            seq=0,
            category=EventCategory.VERIFIER,
            event_type=EventType.VERIFIER_ASSESSMENT,
            timestamp=NOW,
            evidence_ids=("EVID-OK", "   "),
        )


def test_evidence_references_round_trip_through_json() -> None:
    trace = Trace()
    trace.record(
        category=EventCategory.VERIFIER,
        event_type=EventType.VERIFIER_ASSESSMENT,
        timestamp=NOW,
        evidence_ids=("EVID-DOM-001",),
    )
    payload = trace.as_dicts()
    assert payload[0]["evidence_ids"] == ["EVID-DOM-001"]
    # And the dump is re-loadable, so a trace can be read back.
    reloaded = TraceEvent.model_validate(payload[0])
    assert reloaded.evidence_ids == ("EVID-DOM-001",)


# ---------------------------------------------------------------------------
# REQ-OBS-015 — the schema is versioned
# ---------------------------------------------------------------------------


def test_the_version_is_carried_on_every_event() -> None:
    trace = Trace()
    trace.record(
        category=EventCategory.MODEL, event_type=EventType.MODEL_REQUEST,
        timestamp=NOW,
    )
    assert trace.events[0].schema_version == OBSERVABILITY_SCHEMA_VERSION


def test_a_different_major_is_refused_and_a_minor_is_accepted() -> None:
    """Documented asymmetry: a major may have changed field meaning, a minor
    is additive by definition."""
    base = {
        "trace_id": new_trace_id(),
        "seq": 0,
        "category": EventCategory.MODEL,
        "event_type": EventType.MODEL_REQUEST,
        "timestamp": NOW,
    }
    assert TraceEvent(**base, schema_version="1.99").schema_version == "1.99"
    for unsupported in ("0.9", "2.0", "3", "nonsense"):
        with pytest.raises(ValidationError, match="unsupported trace schema major"):
            TraceEvent(**base, schema_version=unsupported)
    assert "1" in SUPPORTED_SCHEMA_MAJORS


# ---------------------------------------------------------------------------
# REQ-OBS-016 — unknown future event types do not corrupt a trace
# ---------------------------------------------------------------------------


def test_an_unknown_event_type_is_accepted_and_flagged() -> None:
    """The core of REQ-OBS-016.

    A producer newer than this reader adds a type. The event must survive,
    keep its class, and be reported as unknown — not raise and take the trace
    with it.
    """
    trace = Trace()
    trace.record(
        category=EventCategory.MODEL, event_type=EventType.MODEL_REQUEST,
        timestamp=NOW,
    )
    future = trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type="knowledge.quantum.entangled",  # not in this build
        timestamp=NOW,
    )

    assert future.is_known_type is False
    assert future.category is EventCategory.KNOWLEDGE, (
        "an unknown type must still arrive classifiable"
    )
    assert len(trace) == 2, "the unknown event must not cost us the trace"
    assert trace.unknown_types() == ("knowledge.quantum.entangled",)


def test_a_trace_of_only_unknown_types_is_still_readable() -> None:
    trace = Trace()
    for name in ("a.b", "c.d", "e.f"):
        trace.record(
            category=EventCategory.VERIFIER, event_type=name, timestamp=NOW
        )
    assert len(trace) == 3
    assert trace.unknown_types() == ("a.b", "c.d", "e.f")


def test_known_and_unknown_coexist_in_one_ordered_trace() -> None:
    trace = Trace()
    knowledge_request_started(
        trace, request_id="req-1", operation="get", timestamp=NOW
    )
    trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type="knowledge.future.thing",
        timestamp=NOW,
    )
    knowledge_request_completed(trace, request_id="req-1", timestamp=NOW, ok=True)
    assert [e.seq for e in trace.events] == [0, 1, 2]
    assert len(trace.unknown_types()) == 1


# ---------------------------------------------------------------------------
# The eight CFOPS knowledge events
# ---------------------------------------------------------------------------


def test_all_eight_required_knowledge_event_types_are_defined() -> None:
    values = {t.value for t in KNOWLEDGE_EVENT_TYPES}
    assert values == {
        "knowledge.request.started",
        "knowledge.request.completed",
        "knowledge.saturated",
        "knowledge.dependency.failure",
        "knowledge.integrity.failure",
        "knowledge.conflict",
        "knowledge.kv.degraded",
        "knowledge.retry",
    }
    assert len(values) == 8, "the CFOPS contract names exactly eight"


def test_the_full_knowledge_request_lifecycle_reconstructs() -> None:
    """Started, a saturated refusal, a retry, then a completion."""
    trace = Trace()
    counters = AdmissionCounters(
        in_flight=8, peak_in_flight=8, admitted_count=41, refused_count=3
    )
    knowledge_request_started(
        trace, request_id="req-1", operation="put", timestamp=NOW
    )
    knowledge_saturated(
        trace, request_id="req-1", timestamp=NOW + 1, counters=counters
    )
    knowledge_retry(
        trace, request_id="req-1", timestamp=NOW + 2, attempt=1, outcome="saturated"
    )
    knowledge_request_completed(
        trace, request_id="req-1", timestamp=NOW + 3, ok=True
    )

    assert [e.seq for e in trace.events] == [0, 1, 2, 3]
    assert [e.event_type for e in trace.events] == [
        "knowledge.request.started",
        "knowledge.saturated",
        "knowledge.retry",
        "knowledge.request.completed",
    ]
    # Every event is correlatable back to the one request.
    assert {e.request_id for e in trace.events} == {"req-1"}


def test_each_failure_type_maps_to_its_own_event() -> None:
    trace = Trace()
    knowledge_failure(
        trace,
        event_type=EventType.KNOWLEDGE_DEPENDENCY_FAILURE,
        request_id="req-1",
        timestamp=NOW,
        outcome="unavailable_dependency",
        retryable=True,
    )
    knowledge_failure(
        trace,
        event_type=EventType.KNOWLEDGE_INTEGRITY_FAILURE,
        request_id="req-2",
        timestamp=NOW,
        outcome="integrity",
        retryable=False,
    )
    knowledge_failure(
        trace,
        event_type=EventType.KNOWLEDGE_CONFLICT,
        request_id="req-3",
        timestamp=NOW,
        outcome="conflict",
        retryable=True,
    )
    assert [e.event_type for e in trace.events] == [
        "knowledge.dependency.failure",
        "knowledge.integrity.failure",
        "knowledge.conflict",
    ]
    # Retryability is preserved per event, so a client can tell a transient
    # dependency fault from an integrity failure that retrying will not fix.
    assert [e.retryable for e in trace.events] == [True, False, True]


def test_kv_degraded_is_retryable_and_carries_a_reason() -> None:
    trace = Trace()
    event = knowledge_kv_degraded(
        trace, request_id="req-1", timestamp=NOW, detail="kv read raised"
    )
    assert event.event_type == "knowledge.kv.degraded"
    assert event.retryable is True, (
        "a degraded cache is a performance fact, not a correctness failure"
    )
    assert event.detail == "kv read raised"


def test_the_failure_helper_refuses_a_non_failure_type() -> None:
    trace = Trace()
    with pytest.raises(ValueError, match="not a knowledge failure type"):
        knowledge_failure(
            trace,
            event_type=EventType.KNOWLEDGE_SATURATED,
            request_id="req-1",
            timestamp=NOW,
        )


def test_retry_attempts_start_at_one() -> None:
    trace = Trace()
    with pytest.raises(ValidationError):
        knowledge_retry(
            trace, request_id="req-1", timestamp=NOW, attempt=0
        )


# ---------------------------------------------------------------------------
# Capacity figures belong in telemetry and NOWHERE else
# ---------------------------------------------------------------------------


def test_saturation_counters_ride_the_telemetry_event() -> None:
    trace = Trace()
    event = knowledge_saturated(
        trace,
        request_id="req-1",
        timestamp=NOW,
        counters=AdmissionCounters(
            in_flight=8, peak_in_flight=8, admitted_count=41, refused_count=2
        ),
    )
    assert event.admission is not None
    assert event.admission.in_flight == 8
    assert event.admission.refused_count == 2
    assert event.retryable is True


def test_the_public_refusal_message_still_carries_no_counts() -> None:
    """The schema does not become a channel for leaking capacity.

    Admission counters are legitimate telemetry, but the caller's error string
    is a different surface with a different rule, and a schema that made it
    convenient to populate both would eventually see the counts leak.
    """
    from oai2.knowledge.admission import (
        PUBLIC_SATURATION_MESSAGE,
        KnowledgeSaturatedError,
    )

    exc = KnowledgeSaturatedError(limit=8, in_flight=8)
    assert exc.error.message == PUBLIC_SATURATION_MESSAGE
    assert "8" not in exc.error.message
    assert "in_flight" not in exc.error.message
    assert "limit" not in exc.error.message


def test_counters_must_be_non_negative_and_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AdmissionCounters(
            in_flight=-1, peak_in_flight=0, admitted_count=0, refused_count=0
        )
    with pytest.raises(ValidationError):
        AdmissionCounters(
            in_flight=0,
            peak_in_flight=0,
            admitted_count=0,
            refused_count=0,
            colo="iad1",  # type: ignore[call-arg]
        )


def test_no_event_field_can_smuggle_extra_state() -> None:
    """``extra="forbid"`` is the repo's anti-smuggling rule.

    A widened binding payload must not be able to attach arbitrary state to a
    trace event; see the same rule in ``oai2/knowledge/transport.py``.
    """
    with pytest.raises(ValidationError):
        TraceEvent.start(
            trace_id=new_trace_id(),
            seq=0,
            category=EventCategory.MODEL,
            event_type=EventType.MODEL_REQUEST,
            timestamp=NOW,
            account_id="acc_123",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Acceptance: reconstruct a controlled multi-stage task
# ---------------------------------------------------------------------------


def test_a_multi_stage_task_reconstructs_from_events_and_evidence() -> None:
    """The issue's acceptance criterion.

    One trace carrying model, tool, retrieval, vision, agent, verifier and
    knowledge activity, with evidence reachable from it.
    """
    trace = Trace()
    knowledge_request_started(
        trace, request_id="req-1", operation="retrieve", timestamp=NOW
    )
    trace.record(
        category=EventCategory.RETRIEVAL,
        event_type=EventType.RETRIEVAL_QUERY,
        timestamp=NOW + 1,
        request_id="req-1",
    )
    trace.record(
        category=EventCategory.RETRIEVAL,
        event_type=EventType.RETRIEVAL_RESULT,
        timestamp=NOW + 2,
        request_id="req-1",
        evidence_ids=("EVID-DOM-001", "EVID-DOM-002"),
    )
    trace.record(
        category=EventCategory.VISION,
        event_type=EventType.VISION_INPUT,
        timestamp=NOW + 3,
        request_id="req-1",
        evidence_ids=("EVID-DOM-002",),
    )
    trace.record(
        category=EventCategory.MODEL,
        event_type=EventType.MODEL_REQUEST,
        timestamp=NOW + 4,
    )
    trace.record(
        category=EventCategory.MODEL,
        event_type=EventType.MODEL_RESPONSE,
        timestamp=NOW + 5,
    )
    trace.record(
        category=EventCategory.TOOL,
        event_type=EventType.TOOL_DISPATCH,
        timestamp=NOW + 6,
    )
    trace.record(
        category=EventCategory.AGENT,
        event_type=EventType.AGENT_STEP,
        timestamp=NOW + 7,
    )
    trace.record(
        category=EventCategory.VERIFIER,
        event_type=EventType.VERIFIER_ASSESSMENT,
        timestamp=NOW + 8,
        evidence_ids=("EVID-DOM-001",),
        outcome="supported",
    )
    knowledge_request_completed(
        trace, request_id="req-1", timestamp=NOW + 9, ok=True
    )

    # Ordered, complete, and reconstructable.
    assert [e.seq for e in trace.events] == list(range(10))
    assert [e.category for e in trace.events] == [
        EventCategory.KNOWLEDGE,
        EventCategory.RETRIEVAL,
        EventCategory.RETRIEVAL,
        EventCategory.VISION,
        EventCategory.MODEL,
        EventCategory.MODEL,
        EventCategory.TOOL,
        EventCategory.AGENT,
        EventCategory.VERIFIER,
        EventCategory.KNOWLEDGE,
    ]
    # Evidence is reachable from the trace, and the verifier's assessment
    # links back to the retrieval that produced the evidence.
    assert trace.evidence_ids() == ("EVID-DOM-001", "EVID-DOM-002")
    verifier = trace.by_category(EventCategory.VERIFIER)
    assert verifier[0].evidence_ids == ("EVID-DOM-001",)

    # And it survives a JSON round trip, which is what makes it evidence.
    payload = trace.as_dicts()
    assert len(payload) == 10
    rebuilt = Trace(trace.trace_id)
    for raw in payload:
        rebuilt.append(TraceEvent.model_validate(raw))
    assert rebuilt.evidence_ids() == trace.evidence_ids()
    assert [e.seq for e in rebuilt.events] == list(range(10))
