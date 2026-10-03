"""Isolate-scoped health for the observability path itself.

THE FACT THIS EXISTS TO KEEP
----------------------------
Telemetry is non-authoritative, so a recorder swallows every failure a sink
throws. That containment is the whole point — a broken observer must not turn
a successful D1 / R2 / Vectorize / KV operation into a failed request — but it
has a cost that is easy to forget: the swallowed failure has to be counted
somewhere, or "we swallow observer faults" quietly becomes "we lose them".

Before this module the only counters were on the recorder itself, which dies
with the request. A test could hold one and read it. A Worker never can. So
"we have a counter" was true and "you can see a counter" was not.

WHAT IT IS NOT
--------------
* Not a second telemetry stack. Two integers and a read.
* Not reported THROUGH a sink. See below.
* Not per-request. Isolate scope, exactly like
  ``oai2.knowledge.admission.ISOLATE_ADMISSION``, for the same reason: the
  Worker builds a new entrypoint object per invocation, so anything that has
  to outlive a request cannot live on ``self``.

WHY NO SINK EVER SEES THIS
--------------------------
Reporting a sink's own failure back to that sink is a loop, not a feature. A
sink that raises on every call would be handed an event describing its own
failure, raise again, be handed another, and the request would either spin or
be brought down by a fault in the subsystem whose entire job is to be
disposable.

So nothing here is ever emitted. The counters are incremented by the
recorder's containment boundary and READ when a consumer asks for a snapshot.
A metrics sink reads them; it is never handed them.
"""

from __future__ import annotations

from typing import Any


class RecorderHealth:
    """Two counters: events a sink refused, events the trace rejected.

    Kept separate because they mean opposite things to whoever is on call.
    ``sink_errors`` is the observer misbehaving — a deployment problem.
    ``trace_errors`` is OUR ordering invariant breaking — a bug in this
    repository. Collapsing them would make "the metrics backend is down" and
    "we built a corrupt trace" the same number, and those want opposite
    responses.
    """

    __slots__ = ("_sink_errors", "_trace_errors")

    def __init__(self) -> None:
        self._sink_errors = 0
        self._trace_errors = 0

    @property
    def sink_errors(self) -> int:
        """Deliveries a sink refused. The observer misbehaved."""
        return self._sink_errors

    @property
    def trace_errors(self) -> int:
        """Events the trace rejected. Our bug."""
        return self._trace_errors

    def note_sink_error(self) -> None:
        self._sink_errors += 1

    def note_trace_error(self) -> None:
        self._trace_errors += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "sink_errors": self._sink_errors,
            "trace_errors": self._trace_errors,
        }

    def reset(self) -> None:
        self._sink_errors = 0
        self._trace_errors = 0

    def _reset_for_tests(self) -> None:
        self.reset()


#: One per isolate. Deliberately a plain module-scope object and deliberately
#: NOT something a recorder reaches for on its own: a hidden global is a
#: hidden global, and the recorder already argues against ambient state for
#: the trace. The health accumulator is passed in explicitly, at the same place
#: the sink is, so the wiring is visible at the one call site that owns it.
ISOLATE_RECORDER_HEALTH = RecorderHealth()


def health_snapshot(health: RecorderHealth | None) -> dict[str, Any]:
    """Read a health accumulator for a consumer's snapshot. Never emits.

    The ``None`` case is the common one — a process that installed no health
    wiring still needs a well-formed snapshot, and reporting zeros would be a
    claim that nothing is wrong rather than a statement that nothing is
    watching.
    """
    if health is None:
        return {"sink_errors": 0, "trace_errors": 0, "observed": False}
    snapshot = health.snapshot()
    snapshot["observed"] = True
    return snapshot


__all__ = [
    "ISOLATE_RECORDER_HEALTH",
    "RecorderHealth",
    "health_snapshot",
]
