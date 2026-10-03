"""Bounded metrics derived from the event stream (WI-OBS-001 / #57).

DERIVED, NOT PUSHED
-------------------
Every counter here is computed from a :class:`TraceEvent` that some layer
already decided to emit. No runtime subsystem maintains its own counter, and
nothing calls ``increment("saturated")`` from inside a request handler.

That is the whole design reason this file exists. A subsystem that maintains
its own counter has to be touched, remembered, and kept consistent as the
request path changes; a metric derived from an event stays correct when the
event stream changes and wrong only if the event stream itself is wrong — which
is visible.

CARDINALITY IS THE REAL CONSTRAINT (REQ-OBS-025 redaction)
-----------------------------------------------------------
A metric store whose dimensions include ``request_id`` or ``trace_id`` creates
one time series per request, which is unbounded growth and no metric at all.
Identifiers are used for CORRELATION and never as labels.

So every dimension here is drawn from a closed vocabulary:

* ``category`` and ``event_type`` — bounded by the schema's own format and by
  an explicit cap on distinct values, with an overflow bucket.
* ``outcome`` — accepted only if it is a known value; anything else becomes
  ``"unknown"``. An attacker-influenced string must not be able to create
  time series.
* ``retryable`` — a bool.

BOUNDED STORAGE (REQ-OBS-023)
-----------------------------
Two structures, both bounded:

* counters keyed by the bounded dimensions above, with hard caps;
* an optional recent-event ring buffer, oldest-first, capped.

There is no dict keyed by trace or request id, anywhere. ``snapshot()`` is a
flat, serialisable dict suitable for a readiness probe or a test.

A DELIBERATE OMISSION: ``retry_count``
--------------------------------------
``knowledge.retry`` has no emitter, because there is no retry loop. This
aggregator therefore counts retry events as ``retry_events_observed``, which
is currently always ``0``, and NEVER infers a retry from a ``retryable=True``
response. A retryable flag means a retry MAY occur; counting it as a retry
would over-report by exactly the rate at which clients honour the hint, and
would make the number wrong in the one direction nobody checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .events import AdmissionCounters, EventType, TraceEvent
from .sink import EventSink

#: Cap on distinct values per dimension. Beyond this, new keys land in the
#: overflow bucket rather than growing the store.
MAX_DIMENSION_KEYS: Final[int] = 64

OVERFLOW_KEY: Final[str] = "other"
UNKNOWN_OUTCOME: Final[str] = "unknown"

#: Outcomes this build recognises. A value outside this set is bucketed, not
#: trusted, so a caller-influenced string cannot create a time series.
KNOWN_OUTCOMES: Final[frozenset[str]] = frozenset({
    "ok", "saturated", "degraded",
    "not_found", "conflict", "integrity",
    "unavailable_dependency", "internal",
    "authorization", "validation",
})


class _Dimension:
    """A capped, counted mapping. The only label store in this module.

    The cap is on the TOTAL number of keys, overflow bucket included, so
    ``len(dimension) <= max_keys`` always holds. One slot is reserved for the
    overflow bucket, which is why the effective limit on distinct real labels is
    ``max_keys - 1``: without that reservation the first overflow would be the
    ``max_keys + 1``-th key.
    """

    __slots__ = ("_counts", "_max_keys", "_max_real_keys")

    def __init__(self, max_keys: int = MAX_DIMENSION_KEYS) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be positive")
        self._max_keys = max_keys
        # With a single slot there is no room for both a real label and the
        # overflow bucket, so the single slot IS the overflow bucket.
        self._max_real_keys = max_keys - 1 if max_keys > 1 else 0
        self._counts: dict[str, int] = {}

    def increment(self, key: str) -> None:
        if key in self._counts:
            self._counts[key] += 1
            return
        if len(self._counts) >= self._max_real_keys:
            key = OVERFLOW_KEY
        self._counts[key] = self._counts.get(key, 0) + 1
        # The invariant is driven past its limit by the test that overflows the
        # cap, rather than assumed here.
        assert len(self._counts) <= self._max_keys, (
            f"dimension exceeded its cap: {sorted(self._counts)}"
        )

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)

    def reset(self) -> None:
        self._counts.clear()

    def __len__(self) -> int:
        return len(self._counts)


class MetricsAggregator(EventSink):
    """Consumes the event stream and maintains bounded counters.

    Implements :class:`EventSink`, so it can be installed wherever a sink goes.
    Being a sink is what makes it impossible for it to be bypassed: if it is
    not installed, nothing counts, and that is visible as a zero rather than
    as a silently wrong number.
    """

    __slots__ = (
        "_by_category",
        "_by_event_type",
        "_by_outcome",
        "_counters",
        "_max_events",
        "_recent",
    )

    #: Single scalar counters. Named to match the required metric list so a
    #: reviewer can check the list against this tuple.
    _COUNTER_NAMES: Final[tuple[str, ...]] = (
        "events_observed",
        "unknown_event_types",
        "requests_started",
        "requests_completed",
        "successful_requests",
        "failed_requests",
        "saturated_count",
        "dependency_failure_count",
        "integrity_failure_count",
        "conflict_count",
        "internal_failure_count",
        "kv_degraded_count",
        "retryable_failure_count",
        "retry_events_observed",
        "admission_in_flight_last",
        "admission_in_flight_peak",
        "admission_peak_in_flight",
        "admission_admitted_count",
        "admission_refused_count",
    )

    def __init__(self, *, max_recent_events: int = 256) -> None:
        if isinstance(max_recent_events, bool) or not isinstance(
            max_recent_events, int
        ):
            raise TypeError("max_recent_events must be an int")
        if max_recent_events < 0:
            raise ValueError("max_recent_events must be non-negative")
        self._counters: dict[str, int] = dict.fromkeys(self._COUNTER_NAMES, 0)
        self._by_category = _Dimension()
        self._by_event_type = _Dimension()
        self._by_outcome = _Dimension()
        self._max_events = max_recent_events
        self._recent: list[TraceEvent] = []

    # -- EventSink ------------------------------------------------------

    def emit(self, event: TraceEvent) -> None:
        self._counters["events_observed"] += 1
        self._by_category.increment(event.category.value)
        self._by_event_type.increment(event.event_type)
        if not event.is_known_type:
            self._counters["unknown_event_types"] += 1
        self._count_outcome(event)
        self._count_admission(event)
        self._retain(event)

    # -- internals ------------------------------------------------------

    def _count_outcome(self, event: TraceEvent) -> None:
        outcome = event.outcome
        if outcome is not None:
            # Bucket anything unrecognised rather than trusting it as a label.
            self._by_outcome.increment(
                outcome if outcome in KNOWN_OUTCOMES else UNKNOWN_OUTCOME
            )
        kind = event.event_type
        if kind == EventType.KNOWLEDGE_REQUEST_STARTED.value:
            self._counters["requests_started"] += 1
        elif kind == EventType.KNOWLEDGE_REQUEST_COMPLETED.value:
            self._counters["requests_completed"] += 1
            if event.outcome == "ok":
                self._counters["successful_requests"] += 1
            else:
                self._counters["failed_requests"] += 1
                # Counted HERE, on the completion, and not on every event that
                # happens to carry the flag. A single refusal emits both a
                # `saturated` detail event and a `completed` event, both
                # retryable, and counting both would report twice the
                # requests that actually failed retryably.
                if event.retryable is True:
                    self._counters["retryable_failure_count"] += 1
        elif kind == EventType.KNOWLEDGE_SATURATED.value:
            self._counters["saturated_count"] += 1
        elif kind == EventType.KNOWLEDGE_DEPENDENCY_FAILURE.value:
            self._counters["dependency_failure_count"] += 1
        elif kind == EventType.KNOWLEDGE_INTEGRITY_FAILURE.value:
            self._counters["integrity_failure_count"] += 1
        elif kind == EventType.KNOWLEDGE_CONFLICT.value:
            self._counters["conflict_count"] += 1
        elif kind == EventType.KNOWLEDGE_INTERNAL_FAILURE.value:
            self._counters["internal_failure_count"] += 1
        elif kind == EventType.KNOWLEDGE_KV_DEGRADED.value:
            self._counters["kv_degraded_count"] += 1
        elif kind == EventType.KNOWLEDGE_RETRY.value:
            # Only an actual retry event counts. Never a retryable flag.
            self._counters["retry_events_observed"] += 1

    def _count_admission(self, event: TraceEvent) -> None:
        counters: AdmissionCounters | None = event.admission
        if counters is None:
            return
        # TWO different peaks, deliberately not conflated:
        #
        # * `admission_in_flight_peak` is this aggregator's running max of the
        #   `in_flight` it has actually SAMPLED. Sampling only happens on
        #   events that carry counters, so it under-reports load that occurred
        #   between events.
        # * `admission_peak_in_flight` is the admission state's own sticky
        #   peak, which is exact for the isolate's whole life.
        #
        # Collapsing them would produce a number that is either a sampling
        # artifact or a lifetime figure, depending on which one won, and an
        # operator could not tell which they were reading.
        self._counters["admission_in_flight_last"] = counters.in_flight
        if counters.in_flight > self._counters["admission_in_flight_peak"]:
            self._counters["admission_in_flight_peak"] = counters.in_flight
        if counters.peak_in_flight > self._counters["admission_peak_in_flight"]:
            self._counters["admission_peak_in_flight"] = counters.peak_in_flight
        self._counters["admission_admitted_count"] = counters.admitted_count
        self._counters["admission_refused_count"] = counters.refused_count

    def _retain(self, event: TraceEvent) -> None:
        if self._max_events == 0:
            return
        self._recent.append(event)
        if len(self._recent) > self._max_events:
            del self._recent[: len(self._recent) - self._max_events]

    # -- reading --------------------------------------------------------

    @property
    def recent_events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._recent)

    def counter(self, name: str) -> int:
        return self._counters[name]

    def snapshot(self) -> dict[str, Any]:
        """Flat, bounded, serialisable. Suitable for a probe or a diff."""
        return {
            "counters": dict(self._counters),
            "by_category": self._by_category.snapshot(),
            "by_event_type": self._by_event_type.snapshot(),
            "by_outcome": self._by_outcome.snapshot(),
            "retained_event_count": len(self._recent),
            "retained_event_capacity": self._max_events,
        }

    def reset(self) -> None:
        self._counters = dict.fromkeys(self._COUNTER_NAMES, 0)
        self._by_category.reset()
        self._by_event_type.reset()
        self._by_outcome.reset()
        self._recent.clear()


def assert_no_identifier_cardinality(snapshot: Mapping[str, Any]) -> None:
    """Guard against a dimension that is actually a per-request id.

    Cheap and deliberately blunt: any dimension key or value that looks like an
    identifier this module has no business labelling by fails. Called by the
    tests, and available to a readiness probe.
    """
    import re

    suspicious = re.compile(r"(?:^|_)(id|ids|trace|request|topic|query)(?:$|_)")
    for name, value in snapshot.items():
        if name in ("counters", "retained_event_count", "retained_event_capacity"):
            continue
        if not isinstance(value, Mapping):
            continue
        for key in value:
            assert not suspicious.search(key), (
                f"metric dimension {name!r} is keyed by what looks like an "
                f"identifier: {key!r}"
            )


__all__ = [
    "KNOWN_OUTCOMES",
    "MAX_DIMENSION_KEYS",
    "OVERFLOW_KEY",
    "UNKNOWN_OUTCOME",
    "MetricsAggregator",
    "assert_no_identifier_cardinality",
]
