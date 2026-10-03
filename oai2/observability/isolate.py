"""Isolate-scoped local metrics: the supported way to turn observability on.

WHY THIS EXISTS
---------------
Until now the only way to get a metrics aggregator out of a Worker was for a
test to assign it onto a class attribute. That is not a configuration seam, it
is a way for the test suite to be the only consumer of the instrumentation,
and it is precisely why the CFOPS observability consumer could not honestly
claim operational metrics were exposed: the shipped configuration discarded
every event before it left the process.

WHAT THIS IS, EXACTLY
---------------------
A single bounded, in-process aggregator whose SCOPE IS ONE ISOLATE — one
process, one aggregate, gone when the isolate recycles. It is deliberately not
a metrics backend, not a fleet-wide view, and not persistent. An operator
reading it sees what THIS isolate saw since it started. Two isolates serving
the same Worker are two unrelated numbers, and a dashboard that averaged them
would be averaging populations that were never comparable.

This is stated up front because the failure mode for local metrics is not
someone misreading the API — it is someone building a fleet-wide alerting rule
on a number that only ever described one process.

THE ISOLATE REUSE PROBLEM
-------------------------
Cloudflare can reuse an isolate across a binding-only deployment change. A
sink resolved once and pinned to the class would therefore keep running
forever under whatever the first request in that isolate happened to see: turn
metrics off, and the isolate carries on retaining request activity for the rest
of its life, and turning telemetry off appears not to work.

So configuration is consulted on EVERY request, and a change rebuilds rather
than being pinned. Rebuilding discards accumulated counters, which is the
honest outcome — numbers collected under a previous retention bound must not be
reported as though they were collected under the current one — and the
rebuild is COUNTED (:attr:`IsolateMetrics.rebuilds`) so it is visible rather
than silent.

NO BINDINGS ARE HELD
--------------------
Nothing here captures a binding, a client, an endpoint or a namespace. The
aggregator is a pure counter over events that have already been produced, so
there is nothing that can go stale when the isolate outlives a configuration
change. That is a design constraint, not an accident of the current wiring.
"""

from __future__ import annotations

from .health import ISOLATE_RECORDER_HEALTH
from .metrics import MetricsAggregator
from .sink import EventSink, NullEventSink

#: Retained recent events when nothing is configured. Small on purpose: the
#: default exists so an operator can turn metrics on and see something, not so
#: an operator who never asked for them inherits a growing buffer.
DEFAULT_METRICS_RECENT_EVENTS = 256

#: Hard ceiling on the retention a configuration can request.
#:
#: Bounded twice on purpose. The environment is operator-controlled and the
#: clamp is the code's own promise: no configuration value, however large,
#: turns a bounded store into an unbounded one. Ten million is a number someone
#: will eventually type.
MAX_METRICS_RECENT_EVENTS = 4096


def clamp_recent_events(value: int) -> int:
    """Clamp a requested retention into the range this build supports."""
    return max(0, min(int(value), MAX_METRICS_RECENT_EVENTS))


class IsolateMetrics:
    """One aggregate for one isolate. Not a singleton by design.

    A class is used so a host can hold a second instance for a second process
    — which is what "not fleet-wide" has to mean if it is to mean anything.
    """

    __slots__ = ("_config", "_aggregator", "_rebuilds")

    def __init__(self) -> None:
        self._config: tuple[bool, int] | None = None
        self._aggregator: MetricsAggregator | None = None
        self._rebuilds = 0

    @property
    def rebuilds(self) -> int:
        """How many times accumulated state was DISCARDED to follow config.

        Not a counter of builds. A value of 1 means one aggregate was thrown
        away because the operator changed the configuration — which is the
        single most confusing thing to happen to a dashboard that appears to
        have lost its history, and the one an operator needs to be able to see.
        """
        return self._rebuilds

    @property
    def config(self) -> tuple[bool, int] | None:
        """The configuration currently in force, or None before first use."""
        return self._config

    def sink_for(self, *, enabled: bool, max_recent_events: int) -> EventSink:
        """Resolve the sink for THIS request from THIS request's configuration.

        Returns a shared :class:`MetricsAggregator` when metrics are on and its
        configuration is unchanged, so counters accumulate across requests.
        Rebuilds when the configuration changes, and returns a
        :class:`NullEventSink` when metrics are off.
        """
        config = (bool(enabled), clamp_recent_events(max_recent_events))
        if not config[0]:
            if self._config != config or self._aggregator is not None:
                # Drop the aggregate rather than parking it: an operator who
                # disabled metrics gets no retained buffer, not a dormant one.
                self._discard()
            self._config = config
            return NullEventSink()
        if self._aggregator is None:
            self._aggregator = self._build(config[1])
        elif self._config != config:
            self._discard()
            self._aggregator = self._build(config[1])
        self._config = config
        return self._aggregator

    def _build(self, max_recent_events: int) -> MetricsAggregator:
        """Build the aggregate wired to the isolate health accumulator.

        This is the one place that wiring exists, so an operator reading a
        snapshot gets the fault counts without any call site having to remember
        to fold them in.
        """
        return MetricsAggregator(
            max_recent_events=max_recent_events,
            health=ISOLATE_RECORDER_HEALTH,
        )

    def aggregator(self) -> MetricsAggregator | None:
        """The live aggregate, or None when metrics are not enabled."""
        return self._aggregator

    def _discard(self) -> None:
        if self._aggregator is not None:
            self._rebuilds += 1
        self._aggregator = None

    def _reset_for_tests(self) -> None:
        self._config = None
        self._aggregator = None
        self._rebuilds = 0


#: Module-scope instance. One per isolate, exactly like
#: ``oai2.knowledge.admission.ISOLATE_ADMISSION`` and for the same reason: the
#: Worker constructs a new entrypoint object per invocation, so anything that
#: must span requests cannot live on ``self``.
ISOLATE_METRICS = IsolateMetrics()


__all__ = [
    "DEFAULT_METRICS_RECENT_EVENTS",
    "ISOLATE_METRICS",
    "MAX_METRICS_RECENT_EVENTS",
    "IsolateMetrics",
    "clamp_recent_events",
]
