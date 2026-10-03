"""Isolate-local admission control for the knowledge path (REQ-CFOPS-013).

WHAT THIS IS, PRECISELY
-----------------------
A bounded in-flight counter for a single Cloudflare Worker isolate. It refuses
work rather than queueing it, and the refusal is an explicit, retryable
``SATURATED`` response.

WHAT THIS IS NOT, AND THE WORDING MATTERS
-----------------------------------------
This does NOT enforce Cloudflare's capacity, and nothing here should be
described as if it did. A Cloudflare Worker runs many isolates across many
machines; a counter held in one isolate's memory observes one isolate. The total
number of concurrent knowledge operations across the deployment is
unobservable from here, and two isolates can each be at their own limit while
the fleet as a whole is nowhere near saturated, or the reverse.

So the honest statement is: this bounds what ONE isolate will start at once, and
it makes the refusal visible and retryable instead of letting work pile up
inside a runtime that can be evicted at any moment. Global or fleet-wide
admission needs shared durable state and belongs to #199, which owns live
capacity operations. Do not read a passing test here as evidence about
deployment-wide behaviour.

WHY FAIL FAST AND NOT QUEUE
---------------------------
An isolate can be evicted without warning. A queue held in isolate memory
converts a clean, immediate, retryable refusal into a request that hangs until
eviction and then fails for a different reason, with the caller's deadline
already spent. Refusing now keeps the failure attributable and cheap.

WHY NOT THE INFERENCE SCHEDULER
-------------------------------
``oai2/runtime/admission.py`` is a real policy core, but it is shaped for
inference work: aggregate memory in GB, tool slots, reasoning-mode deadlines,
and a caller-supplied capacity snapshot. Knowledge reads and writes are not
those quantities, and importing that contract would give this path a model of
capacity that does not describe it. The mechanism below is the smallest one
that can honestly say what it bounds.
"""

from __future__ import annotations

import threading

from .transport import TransportError, TransportErrorCode


class KnowledgeSaturatedError(Exception):
    """Raised when this isolate is already at its knowledge admission limit.

    Retryable by construction: no work was started, nothing was partially
    applied, and a later attempt is expected to succeed once in-flight work
    drains. ``TransportError`` is a model rather than an exception, so the
    wire-shaped failure travels as ``error`` and the transport raises this and
    translates it.
    """

    def __init__(self, *, limit: int, in_flight: int) -> None:
        super().__init__(
            f"knowledge admission limit reached for this isolate "
            f"({in_flight}/{limit} in flight); retry shortly"
        )
        self.limit = limit
        self.in_flight = in_flight
        self.error = TransportError(
            code=TransportErrorCode.SATURATED,
            message=str(self),
            retryable=True,
        )

    @property
    def code(self) -> TransportErrorCode:
        return self.error.code

    @property
    def retryable(self) -> bool:
        return self.error.retryable


class IsolateAdmissionGuard:
    """A bounded, fail-fast in-flight guard scoped to one isolate.

    Not a context manager: the guard is acquired for the duration of one
    operation and must be released on every exit path including an exception,
    which an ``async with`` on the awaiting code guarantees and a bare
    acquire/release pair does not.
    """

    def __init__(self, *, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        self._limit = limit
        self._in_flight = 0
        # An isolate is single-threaded at the event-loop level, but a
        # thread-safe counter costs nothing and removes a whole class of
        # question about whether the limit can be raced past.
        self._lock = threading.Lock()
        self._peak_in_flight = 0
        self._refused_count = 0
        self._admitted_count = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def peak_in_flight(self) -> int:
        return self._peak_in_flight

    @property
    def refused_count(self) -> int:
        return self._refused_count

    @property
    def admitted_count(self) -> int:
        return self._admitted_count

    def try_acquire(self) -> None:
        """Admit one operation, or raise :class:`KnowledgeSaturatedError`.

        The limit is never exceeded, and the check and the increment are a
        single critical section so concurrent attempts cannot both observe a
        free slot.
        """
        with self._lock:
            if self._in_flight >= self._limit:
                self._refused_count += 1
                raise KnowledgeSaturatedError(
                    limit=self._limit, in_flight=self._in_flight
                )
            self._in_flight += 1
            self._admitted_count += 1
            if self._in_flight > self._peak_in_flight:
                self._peak_in_flight = self._in_flight

    def release(self) -> None:
        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("knowledge admission released without being held")
            self._in_flight -= 1

    def __enter__(self) -> IsolateAdmissionGuard:
        self.try_acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def is_retryable_saturation(error: TransportError) -> bool:
    """Whether a caller may retry a refusal without changing the request.

    Narrow on purpose: a CONFLICT or a non-retryable INTEGRITY failure is not
    resolved by retrying sooner, and treating every failure as retryable is how
    a backoff loop turns into a load amplifier.
    """
    return error.code is TransportErrorCode.SATURATED and error.retryable


def is_retryable_saturation_error(exc: BaseException) -> bool:
    """Same question, for the exception a caller may actually be holding."""
    return isinstance(exc, KnowledgeSaturatedError) and exc.retryable


__all__ = [
    "IsolateAdmissionGuard",
    "KnowledgeSaturatedError",
    "is_retryable_saturation",
    "is_retryable_saturation_error",
]
