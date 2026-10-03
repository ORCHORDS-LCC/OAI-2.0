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

===================================  ==================================
``knowledge.request.started``     entrypoint, after validation
``knowledge.saturated``           entrypoint, on admission refusal
``knowledge.conflict``            transport
``knowledge.integrity.failure``   transport
``knowledge.dependency.failure``  transport, and entrypoint for a
                                  pre-transport component-init failure
``knowledge.internal.failure``    transport, unexpected exception
``knowledge.kv.degraded``         runtime, where the KV exception exists
``knowledge.cache.hit``           runtime, retrieval served from cache
``knowledge.cache.miss``          runtime, cache consulted and empty
``knowledge.cache.stale``         runtime, cache answered unusably
``knowledge.request.completed``   entrypoint, ALWAYS
===================================  ==================================

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

#: Outcome vocabulary for the three cache events. Namespaced with a ``cache_``
#: prefix on purpose: these land in the same bounded ``by_outcome`` dimension
#: as request outcomes, and a bare ``miss`` there would be indistinguishable
#: from a request that failed to find something. A reader who sees
#: ``cache_hit: 41`` knows immediately it is not a request outcome.
CACHE_OUTCOME_HIT = "cache_hit"
CACHE_OUTCOME_MISS = "cache_miss"
CACHE_OUTCOME_STALE = "cache_stale"


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
    timestamp: float,
    operation: str | None = None,
) -> None:
    """The operation name normally comes from the RECORDER.

    ``operation`` is optional and exists so the STARTED event can state it
    explicitly if a caller wants the event correct in isolation. Omitting it
    inherits the recorder's, which is the normal path and the reason every
    later event in the trace is groupable without being told.
    """
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
        timestamp=timestamp,
        **({} if operation is None else {"operation": operation}),
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
    """End of the lifecycle, and the only event that carries a duration.

    The duration comes from the recorder's MONOTONIC clock. It is never
    derived from two wall-clock timestamps, which can go backwards.
    """
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_COMPLETED,
        timestamp=timestamp,
        outcome=outcome,
        retryable=retryable,
        duration_ms=recorder.elapsed_ms(),
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

    It OVERRIDES the recorder's inherited operation deliberately. On a
    ``retrieve`` request the knowledge operation is ``retrieve`` and the cache
    operation is ``get``; for this event the useful question is which cache
    call failed, so the cache operation is what the label says. Every other
    knowledge event in the same trace still carries ``retrieve``.

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
        operation=operation,
    )


# ---------------------------------------------------------------------------
# Cache outcome — runtime, where the lookup actually happens
# ---------------------------------------------------------------------------


def _emit_cache(
    recorder: TraceRecorder | None,
    *,
    event_type: EventType,
    outcome: str,
    timestamp: float,
) -> None:
    if recorder is None:
        return
    recorder.record(
        category=EventCategory.KNOWLEDGE,
        event_type=event_type,
        timestamp=timestamp,
        outcome=outcome,
        # A cache outcome is not a request failure, so it carries no retryable
        # flag. The request's own completion event carries the outcome that
        # actually decides whether a caller should retry.
        retryable=None,
    )


def emit_cache_hit(
    recorder: TraceRecorder | None, *, timestamp: float
) -> None:
    """The retrieval was SERVED from the cache.

    Emitted only when the cached result was actually returned to the caller.
    A lookup that produced a candidate and was then invalidated by a
    concurrent write is a conflict, not a hit — the caller got nothing, and
    putting it in the numerator of a hit rate would credit a success that did
    not happen.
    """
    _emit_cache(
        recorder,
        event_type=EventType.KNOWLEDGE_CACHE_HIT,
        outcome=CACHE_OUTCOME_HIT,
        timestamp=timestamp,
    )


def emit_cache_miss(
    recorder: TraceRecorder | None, *, timestamp: float
) -> None:
    """The cache was consulted and held nothing. This is the cache working.

    A MISS is never reported for a lookup that RAISED. That is
    ``knowledge.kv.degraded``: an exception is the cache failing to answer, not
    answering "no", and reporting it as a miss is how a permanently broken KV
    namespace survives unnoticed — nothing errors, every request is just
    slower.
    """
    _emit_cache(
        recorder,
        event_type=EventType.KNOWLEDGE_CACHE_MISS,
        outcome=CACHE_OUTCOME_MISS,
        timestamp=timestamp,
    )


def emit_cache_stale(
    recorder: TraceRecorder | None, *, timestamp: float
) -> None:
    """The cache returned an envelope the runtime REFUSED to use.

    Corrupt, the wrong corpus revision, the wrong embedding digest, or naming
    a row that no longer exists. This is a third state, not a miss and not a
    hit: it means the cache is being read and its contents are no longer
    usable, which is a different operational problem from an empty cache and
    calls for a different fix.
    """
    _emit_cache(
        recorder,
        event_type=EventType.KNOWLEDGE_CACHE_STALE,
        outcome=CACHE_OUTCOME_STALE,
        timestamp=timestamp,
    )


__all__ = [
    "CACHE_OUTCOME_HIT",
    "CACHE_OUTCOME_MISS",
    "CACHE_OUTCOME_STALE",
    "KV_OPERATION_GET",
    "KV_OPERATION_PUT",
    "admission_counters",
    "emit_cache_hit",
    "emit_cache_miss",
    "emit_cache_stale",
    "emit_completed",
    "emit_conflict",
    "emit_dependency_failure",
    "emit_integrity_failure",
    "emit_internal_failure",
    "emit_kv_degraded",
    "emit_request_started",
    "emit_saturated",
]
