"""Isolate-local admission control for the knowledge path (REQ-CFOPS-013).

SCOPE — READ THIS BEFORE REUSING ANYTHING HERE
----------------------------------------------
This bounds work **within one Cloudflare Worker isolate**. That is the whole
claim, and it must not be restated as anything larger.

A Worker runs many isolates across many machines. The state in this module is
module-scope Python state, which persists for the life of ONE isolate. It
therefore observes one isolate and nothing else. Two isolates can each be at
their own limit while the fleet is nowhere near saturated, and the reverse. No
account-wide, Worker-wide-across-isolates, or Cloudflare-global capacity is
enforced or observable here. Fleet-wide admission needs shared durable state
and belongs to #199, which owns live capacity operations. A passing test in
this module is not evidence about deployment-wide behaviour.

WHY THERE IS NO LOCK
--------------------
An earlier revision of this used ``threading.Lock``. That was wrong for the
deployment runtime: Cloudflare's Python Workers run under WebAssembly/Pyodide,
where ``threading`` is importable but not functional, so a Lock would be a
correctness mechanism resting on a primitive the runtime does not honour. A
mechanism that imports cleanly and silently fails to exclude is worse than one
that fails loudly.

It was also unnecessary, and the reason is the real reason to keep it that way:
an isolate runs a single-threaded event loop, and the acquire check plus the
increment contain **no await**. With no suspension point, no other coroutine
can run between the check and the increment, so the critical section is already
atomic with respect to other requests in this isolate. Adding a lock would add
an unsupported primitive without adding exclusion.

The same rule is the design constraint on every method here: ``acquire`` and
``release`` are fully synchronous, and all awaited work happens strictly AFTER
a successful acquire and strictly BEFORE a release.

WHY MODULE SCOPE AND NOT AN INSTANCE ATTRIBUTE
---------------------------------------------
Cloudflare's ``WorkerEntrypoint`` lifecycle constructs a NEW class instance per
invocation. An admission guard on ``self`` would therefore be a per-request
object with an empty counter every time, bounding nothing. The same is true of
anything cached in an instance attribute.

What may live at module scope is deliberately narrow: binding-INDEPENDENT
counters. Binding-derived clients (D1, R2, Vectorize, KV) must NOT be cached
globally — Cloudflare can reuse an isolate across a binding-only change, and a
globally cached client would then be stale. Lightweight wrappers built per
request are acceptable and are what the entrypoint does.

WHY THE LIMIT IS A PARAMETER AND NOT CONSTRUCTOR STATE
-------------------------------------------------------
A module-scope object must not permanently bind itself to whichever limit the
first request happened to carry. The current limit is read and validated per
request and passed into ``acquire``, so an operator changing configuration takes
effect immediately rather than after the isolate recycles.

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
capacity that does not describe it.
"""

from __future__ import annotations

from .transport import TransportError, TransportErrorCode

#: The only public message a saturation refusal ever carries. Deliberately
#: generic: it names no capacity figure and no resource identity. Operational
#: counts belong in sanitized telemetry (#56/#57), not in a caller-visible
#: error string.
PUBLIC_SATURATION_MESSAGE = "knowledge service is saturated; retry shortly"

DEFAULT_ADMISSION_LIMIT = 8


class KnowledgeSaturatedError(Exception):
    """Raised when this isolate is already at its knowledge admission limit.

    Retryable by construction: no work was started, nothing was partially
    applied, and a later attempt is expected to succeed once in-flight work
    drains. ``TransportError`` is a model rather than an exception, so the
    wire-shaped failure travels as :attr:`error`.

    ``limit`` and :attr:`in_flight` are INTERNAL diagnostic detail for
    operator-facing logs. They are deliberately not part of
    :attr:`error`; see :data:`PUBLIC_SATURATION_MESSAGE`.
    """

    def __init__(self, *, limit: int, in_flight: int) -> None:
        super().__init__(PUBLIC_SATURATION_MESSAGE)
        self.limit = limit
        self.in_flight = in_flight
        self.error = TransportError(
            code=TransportErrorCode.SATURATED,
            message=PUBLIC_SATURATION_MESSAGE,
            retryable=True,
        )

    @property
    def code(self) -> TransportErrorCode:
        return self.error.code

    @property
    def retryable(self) -> bool:
        return self.error.retryable


class AdmissionLease:
    """Proof that one operation holds a slot.

    Released exactly once. Every exit path in the request handler must reach
    ``release`` (normally via ``finally``), because admission is only a bound
    if slots actually come back.
    """

    __slots__ = ("_state", "_released")

    def __init__(self, state: AdmissionState) -> None:
        self._state = state
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Return the slot. Synchronous; contains no await."""
        if self._released:
            raise RuntimeError("knowledge admission lease was already released")
        self._released = True
        self._state.release()

    def __enter__(self) -> AdmissionLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class AdmissionState:
    """Isolate-shared in-flight accounting.

    Bind on nothing. Holds counters only. The protected state is
    ``in_flight``, ``peak_in_flight``, ``admitted_count`` and
    ``refused_count`` — isolate-shared operational counters, never user or
    request data.
    """

    __slots__ = ("_in_flight", "_peak_in_flight", "_admitted_count",
                 "_refused_count", "_last_limit")

    def __init__(self) -> None:
        self._in_flight = 0
        self._peak_in_flight = 0
        self._admitted_count = 0
        self._refused_count = 0
        self._last_limit: int | None = None

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def peak_in_flight(self) -> int:
        return self._peak_in_flight

    @property
    def admitted_count(self) -> int:
        return self._admitted_count

    @property
    def refused_count(self) -> int:
        return self._refused_count

    @property
    def last_limit(self) -> int | None:
        """The limit most recently applied, for operator diagnostics only."""
        return self._last_limit

    def acquire(self, current_limit: int) -> AdmissionLease:
        """Admit one operation, or raise :class:`KnowledgeSaturatedError`.

        Fully synchronous. The comparison and the increment happen in one
        uninterrupted block with no await, so no other coroutine in this
        isolate can observe a free slot that is no longer free. This is the
        entire exclusion mechanism and it is why there is no lock.
        """
        limit = validate_limit(current_limit)
        if self._in_flight >= limit:
            self._refused_count += 1
            raise KnowledgeSaturatedError(limit=limit, in_flight=self._in_flight)
        self._in_flight += 1
        self._admitted_count += 1
        self._last_limit = limit
        if self._in_flight > self._peak_in_flight:
            self._peak_in_flight = self._in_flight
        return AdmissionLease(self)

    def release(self) -> None:
        """Return one slot. Fully synchronous; contains no await."""
        if self._in_flight <= 0:
            raise RuntimeError("knowledge admission released without being held")
        self._in_flight -= 1

    def snapshot(self) -> dict[str, int | None]:
        """Operator-facing counters. Candidate telemetry input for #57.

        Deliberately a separate call from the public error surface so
        sanitized metrics can carry these without a caller ever seeing them.
        """
        return {
            "in_flight": self._in_flight,
            "peak_in_flight": self._peak_in_flight,
            "admitted_count": self._admitted_count,
            "refused_count": self._refused_count,
            "last_limit": self._last_limit,
        }

    def _reset_for_tests(self) -> None:
        """Reset counters. Refuses while a slot is held.

        Test-only. The refusal is the point: a reset that could silently mask
        a leaked slot would let a real leak pass, so if something is still
        holding admission the reset fails loudly instead.
        """
        if self._in_flight != 0:
            raise RuntimeError(
                "cannot reset admission state while a slot is still held"
            )
        self._peak_in_flight = 0
        self._admitted_count = 0
        self._refused_count = 0
        self._last_limit = None


def validate_limit(value: object) -> int:
    """Accept a positive integer limit, and only that.

    Read per request so an operator's configuration change takes effect
    immediately rather than being frozen into module state by whichever
    request happened to arrive first.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("admission limit must be a positive integer")
    return value


#: The single isolate-shared admission state. Module scope, so it is shared by
#: every WorkerEntrypoint instance the same isolate handles, and it holds no
#: binding-derived objects.
ISOLATE_ADMISSION = AdmissionState()


def acquire_admission(current_limit: int) -> AdmissionLease:
    """Acquire against the isolate-shared state.

    Synchronous, by design: the entrypoint calls this AFTER cheap validation
    and BEFORE any awaited work, so a malformed or unauthenticated request
    never consumes a knowledge-operation slot.
    """
    return ISOLATE_ADMISSION.acquire(current_limit)


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
    "AdmissionLease",
    "AdmissionState",
    "DEFAULT_ADMISSION_LIMIT",
    "ISOLATE_ADMISSION",
    "KnowledgeSaturatedError",
    "PUBLIC_SATURATION_MESSAGE",
    "acquire_admission",
    "is_retryable_saturation",
    "is_retryable_saturation_error",
    "validate_limit",
]
