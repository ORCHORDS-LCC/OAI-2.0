"""Knowledge-side event emission: the adapter between the runtime and #56.

This is where the knowledge subsystem MEETS the trace schema. It exists so the
knowledge layers emit through named, testable operations instead of each call
site assembling a :class:`TraceEvent` by hand and getting an identifier, a
category or a counter conversion subtly wrong.

It also holds the one genuinely knowledge-specific conversion:
``admission.snapshot()`` -> :class:`AdmissionCounters`. That conversion is
explicit and named rather than ``AdmissionCounters(**snapshot)``, because the
snapshot carries ``last_limit`` and the counters model deliberately does not —
a spread would either raise or, worse, teach someone to widen the model to
make a call site work.

EMITTER OWNERSHIP
-----------------
Fixed, so lifecycle events cannot be duplicated:

============================  ==================================
``knowledge.request.started`` entrypoint, after validation
``knowledge.saturated``       entrypoint, on admission refusal
``knowledge.conflict``        transport
``knowledge.integrity.failure`` transport
``knowledge.dependency.failure`` transport, and entrypoint for a
                               pre-transport component-init failure
``knowledge.internal.failure`` transport, unexpected exception
``knowledge.kv.degraded``     runtime, where the KV exception exists
``knowledge.request.completed`` entrypoint, ALWAYS
============================  ==================================

The entrypoint owns the lifecycle, so COMPLETED is emitted exactly once no
matter which layer observed the failure. The transport emits detail only, and
never a completion. ``knowledge.retry`` has NO emitter: there is no retry loop
in this codebase, and a retryable flag means a retry MAY happen, not that one
did. Emitting it from a ``retryable=True`` response would make metrics count
intentions as facts.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..observability import (
    AdmissionCounters,
    EventCategory,
    EventType,
    TraceRecorder,
)

#: Fixed vocabulary for the KV operation that degraded. NOT the exception
#: text: a driver message can carry endpoint, account or namespace detail, and
#: this string ends up in telemetry that is retained. A bounded enum of the two
#: operations that exist is the whole information content here.
KV_OPERATION_GET = "get"
KV_OPERATION_PUT = "put"


def _counter(snapshot: Mapping[str, int | None], key: str) -> int:
    """Read one counter, rejecting absent or non-integer values.

    The snapshot type is ``int | None`` only because ``last_limit`` is optional;
    the four counters this consumes are always present. Narrowing here keeps
    the strictness the module documents without a cast that would hide a
    genuine ``None`` appearing where a count should be.
    """
    value = snapshot[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"admission counter {key!r} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"admission counter {key!r} must be non-negative")
    return value


def admission_counters(snapshot: Mapping[str, int | None]) -> AdmissionCounters:
    """Convert an admission snapshot into telemetry counters.

    SEMANTICS, stated because they are the kind of thing that gets misread
    later:

    * ``in_flight`` is the state **after** the current operation's admission
      bookkeeping. For a saturation event that means the slots still held when
      the refusal was counted, not a post-decrement figure — a refusal frees
      nothing.
    * ``refused_count`` **includes the refusal being reported**, because
      ``AdmissionState.acquire`` increments it before raising.
    * ``last_limit`` is dropped. The configured limit is deployment
      configuration, and ``AdmissionCounters`` models observed load, not
      configuration. An operator who needs the limit has it in the Worker
      configuration; a retained counter that mixes the two invites a wrong
      reading during an incident.

    A missing key is an error rather than a silent default. This runs on the
    request path, and a telemetry conversion that fails must not fail the
    request — so the caller wraps it. Making the conversion strict keeps the
    failure at the boundary where it can still be seen.
    """
    return AdmissionCounters(
        in_flight=_counter(snapshot, "in_flight"),
        peak_in_flight=_counter(snapshot, "peak_in_flight"),
        admitted_count=_counter(snapshot, "admitted_count"),
        refused_count=_counter(snapshot, "refused_count"),
    )


# ---------------------------------------------------------------------------
# Lifecycle — entrypoint
# ---------------------------------------------------------------------------


def emit_request_started(
    recorder: TraceRecorder | None,
    *,
    operation: str,
    timestamp: float,
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
        timestamp=timestamp,
        detail=operation,
    )


def emit_saturated(
    recorder: TraceRecorder | None,
    *,
    timestamp: float,
    snapshot: Mapping[str, int | None],
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_SATURATED,
        timestamp=timestamp,
        outcome="saturated",
        retryable=True,
        admission=admission_counters(snapshot),
    )


def emit_completed(
    recorder: TraceRecorder | None,
    *,
    timestamp: float,
    ok: bool,
    outcome: str,
    retryable: bool | None = None,
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_COMPLETED,
        timestamp=timestamp,
        outcome=outcome,
        retryable=retryable,
    )


# ---------------------------------------------------------------------------
# Failure detail — transport, and the entrypoint for a pre-transport failure
# ---------------------------------------------------------------------------


def emit_dependency_failure(
    recorder: TraceRecorder | None,
    *,
    timestamp: float,
    request_id: str | None = None,
    outcome: str = "unavailable_dependency",
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_DEPENDENCY_FAILURE,
        timestamp=timestamp,
        request_id=request_id,
        outcome=outcome,
        retryable=True,
    )


def emit_integrity_failure(
    recorder: TraceRecorder | None,
    *,
    timestamp: float,
    request_id: str | None = None,
    outcome: str = "integrity",
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_INTEGRITY_FAILURE,
        timestamp=timestamp,
        request_id=request_id,
        outcome=outcome,
        retryable=False,
    )


def emit_conflict(
    recorder: TraceRecorder | None,
    *,
    timestamp: float,
    request_id: str | None = None,
    outcome: str = "conflict",
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_CONFLICT,
        timestamp=timestamp,
        request_id=request_id,
        outcome=outcome,
        retryable=True,
    )


def emit_internal_failure(
    recorder: TraceRecorder | None,
    *,
    timestamp: float,
    request_id: str | None = None,
    outcome: str = "internal",
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_INTERNAL_FAILURE,
        timestamp=timestamp,
        request_id=request_id,
        outcome=outcome,
        retryable=False,
    )


# ---------------------------------------------------------------------------
# Degradation — runtime
# ---------------------------------------------------------------------------


def emit_kv_degraded(
    recorder: TraceRecorder | None,
    *,
    operation: str,
    timestamp: float,
) -> None:
    """A KV read or write raised.

    ``operation`` is the fixed vocabulary ``"get"`` / ``"put"`` and nothing
    else. The exception is deliberately NOT described: driver messages carry
    endpoint, namespace and account detail, and this value is retained
    telemetry. The dependency class is ``kv`` and the operation is the whole
    fact worth keeping.

    A cache MISS emits nothing. A miss is the cache working.
    """
    if recorder is None:
        return
    if operation not in (KV_OPERATION_GET, KV_OPERATION_PUT):
        raise ValueError(
            f"kv degraded operation must be a fixed vocabulary value, got {operation!r}"
        )
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_KV_DEGRADED,
        timestamp=timestamp,
        outcome="degraded",
        retryable=True,
        detail=operation,
    )


__all__ = [
    "KV_OPERATION_GET",
    "KV_OPERATION_PUT",
    "admission_counters",
    "emit_completed",
    "emit_conflict",
    "emit_dependency_failure",
    "emit_integrity_failure",
    "emit_internal_failure",
    "emit_kv_degraded",
    "emit_request_started",
    "emit_saturated",
]
