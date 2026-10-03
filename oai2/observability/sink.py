"""The event sink seam (WI-OBS-001, #56).

THE SMALLEST CONTRACT THAT WORKS
--------------------------------
A sink has exactly one required operation: accept one :class:`TraceEvent`. That
is deliberate. Every extra obligation — ``flush``, ``close``, ``configure``,
an async ``__aemit__`` — is a way for a telemetry backend to become a
prerequisite for the thing it observes, and the entire point of
:mod:`oai2.observability.recorder` is that telemetry cannot change request
semantics.

SYNCHRONOUS ON PURPOSE
---------------------
``emit`` is sync, not async. On the Worker request path an ``await`` in the
observability path would put a network call on the critical path of a request
whose correctness does not depend on it. Synchronous, local emission cannot
add awaited latency.

If emission ever DOES have to be async — a real remote sink, a batching
forwarder — that is a deliberate design step, not an accident, and it must be
decoupled by queueing behind a background task with an explicit overflow
policy. Do not simply change the signature to ``async def``.

WHAT THIS IS NOT
----------------
* Not a logging API. There is no knowledge-specific logging surface here; the
  sink sees events, not subsystems.
* Not a metrics backend. Counters are derived FROM the event stream by a
  consumer (see #57), never pushed here.
* Not storage. Nothing in this module retains events; retention is a sink's
  own concern and is bounded by whoever implements it.
* Not a network client. No filesystem, socket, or credentials, now or later,
  unless a sink is implemented that genuinely needs them.

A :class:`NullEventSink` exists so that "no observability configured" is a
normal, zero-cost state rather than a special case every call site branches on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .events import TraceEvent


@runtime_checkable
class EventSink(Protocol):
    """Accepts one event. Synchronous. Non-blocking by contract.

    An implementation MUST NOT raise; the recorder defends against it anyway
    (see :mod:`oai2.observability.recorder`), but a sink that raises is a bug,
    and a sink that blocks is a latency bug the Worker would feel directly.
    """

    def emit(self, event: TraceEvent) -> None: ...


class NullEventSink:
    """Drops everything. The default, and a real implementation of the protocol.

    Exists so instrumentation is never a precondition. With this installed the
    recorder still builds and orders events (so a trace can be inspected
    in-process), and the emit itself costs one attribute lookup and a return.
    """

    __slots__ = ()

    def emit(self, event: TraceEvent) -> None:  # noqa: ARG002 - protocol shape
        return None


class CollectingEventSink:
    """Keeps events in memory. For tests, benchmarks and local diagnostics.

    BOUNDED, deliberately. An unbounded list of events is an unbounded memory
    growth curve in any long-lived process, and a metrics consumer must not be
    able to leak. Oldest-first eviction keeps memory flat under sustained load
    and makes the retained window deterministic.

    Not a production store: this is in-process, lost on restart, and sized for
    a recent-events window rather than for history.
    """

    __slots__ = ("_events", "max_events")

    def __init__(self, *, max_events: int = 1024) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be an int")
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            # Oldest-first, deterministic, and O(1) amortized: delete the
            # whole excess in one slice rather than popping per event.
            del self._events[: len(self._events) - self.max_events]

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def event_types(self) -> tuple[str, ...]:
        return tuple(e.event_type for e in self._events)

    def reset(self) -> None:
        self._events.clear()


__all__ = [
    "CollectingEventSink",
    "EventSink",
    "NullEventSink",
]
