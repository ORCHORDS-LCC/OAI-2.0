"""Request-scoped trace recording, and the containment boundary around it.

WHY A RECORDER AND NOT "PASS THE TRACE"
---------------------------------------
The schema has ordering rules (dense monotonic ``seq``) and an output rule
(hand the event to a sink). Those two belong together, because getting either
one wrong at a call site means an out-of-order or undelivered event. So layers
do not call ``Trace.record`` and ``sink.emit`` themselves; they call
``TraceRecorder.record`` and neither can be forgotten or done twice.

It is also the request-scoped observability context required to reach the
runtime: it carries the ``trace_id`` minted once per accepted request and the
``request_id`` reused as wire correlation inside that trace, so a deep layer
emits a correctly correlated event without having to be handed two identifiers
separately. Explicit plumbing, no ``contextvars`` and no mutable "current
trace" global — a global would be wrong here for the same reason admission
state cannot live on ``self``, and it would leak across requests in an isolate
that reuses module scope.

TELEMETRY CANNOT CHANGE REQUEST SEMANTICS
------------------------------------------
This is the property the whole design exists to guarantee (REQ-OBS-025). A
recorder is non-authoritative, so a broken one must not turn a successful D1 /
R2 / Vectorize / KV operation into a failed knowledge request.

:attr:`TraceRecorder.record` therefore never raises. It swallows BOTH:

* **sink failures** — the observer misbehaving. Counted as ``sink_errors``.
* **trace failures** — our own ordering invariant breaking. Counted as
  ``trace_errors`` and, deliberately, as a *separate* counter, because that one
  is our bug and #57 needs to be able to distinguish "telemetry is broken" from
  "we built a corrupt trace".

Neither is silently discarded: both are counted, and :meth:`snapshot` exposes
them. A sink that always throws is a real operational fact, and quietly
swallowing it forever would make observability failures invisible — which is
the opposite of what observability is for.

The original request exception is never swallowed to make room for logging.
The recorder does not wrap request code; request code calls it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .events import EventCategory, EventType, Trace, TraceEvent, new_trace_id
from .sink import EventSink, NullEventSink


class TraceRecorder:
    """One recorder per accepted request. Owns ordering and delivery."""

    __slots__ = (
        "_request_id",
        "_sink",
        "_sink_errors",
        "_trace",
        "_trace_errors",
    )

    def __init__(
        self,
        *,
        sink: EventSink | None = None,
        trace_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self._trace = Trace(trace_id)
        self._sink: EventSink = sink if sink is not None else NullEventSink()
        self._request_id = request_id
        self._sink_errors = 0
        self._trace_errors = 0

    @property
    def trace_id(self) -> str:
        return self._trace.trace_id

    @property
    def request_id(self) -> str | None:
        return self._request_id

    @property
    def trace(self) -> Trace:
        return self._trace

    @property
    def sink_errors(self) -> int:
        """Emissions a sink refused. The observer misbehaved."""
        return self._sink_errors

    @property
    def trace_errors(self) -> int:
        """Events that could not be appended to the trace. Our bug."""
        return self._trace_errors

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return self._trace.events

    def next_seq(self) -> int:
        return self._trace.next_seq

    def record(
        self,
        *,
        category: EventCategory,
        event_type: EventType | str,
        timestamp: float,
        request_id: str | None = None,
        **fields: Any,
    ) -> TraceEvent | None:
        """Append and deliver one event. Never raises.

        ``request_id`` defaults to the recorder's, so a deep layer emits a
        correlated event without threading the wire identifier down to it.
        Pass an explicit value only to override.

        Returns the event, or ``None`` if the trace rejected it — the counters
        carry the reason.
        """
        try:
            event = self._trace.record(
                category=category,
                event_type=event_type,
                timestamp=timestamp,
                request_id=(
                    self._request_id if request_id is None else request_id
                ),
                **fields,
            )
        except Exception:
            # Our own invariant broke. Counted separately from a sink fault so
            # #57 can tell an observer problem from a corrupt-trace problem.
            self._trace_errors += 1
            return None

        try:
            self._sink.emit(event)
        except Exception:
            # The observer misbehaved. The event is still in the local trace,
            # so ordering and in-process inspection are unaffected.
            self._sink_errors += 1
        return event

    def snapshot(self) -> Mapping[str, object]:
        """Operator-facing recorder state. Candidate input for #57."""
        return {
            "trace_id": self._trace.trace_id,
            "request_id": self._request_id,
            "event_count": len(self._trace),
            "sink_errors": self._sink_errors,
            "trace_errors": self._trace_errors,
        }


def new_request_recorder(
    *,
    sink: EventSink | None = None,
    request_id: str | None = None,
) -> TraceRecorder:
    """Mint the single trace identity for one accepted request.

    Called ONCE, at the transport boundary after validation. Every layer below
    receives this recorder; none of them mints a trace. Minting per layer would
    produce one trace per layer, which is the opposite of correlation.
    """
    return TraceRecorder(sink=sink, trace_id=new_trace_id(), request_id=request_id)


__all__ = [
    "TraceRecorder",
    "new_request_recorder",
]
