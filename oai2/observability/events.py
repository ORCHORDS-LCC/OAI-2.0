"""One structured trace/event schema (WI-OBS-001, #56).

WHAT THIS IS
------------
A single envelope that model, tool, retrieval, vision, agent, verifier and
knowledge activity can all be recorded into, so one task can be reconstructed
from ordered events and the evidence they point at.

WHAT THIS IS NOT
----------------
It is not a metrics backend, not a transport, and not a log format. It is the
shape, and the validation rules, that a recorder and a consumer agree on.

THE FOUR DECISIONS THAT MATTER
-------------------------------

1. ``event_type`` is a validated STRING, not a closed enum.
   REQ-OBS-016 says an unknown future event type must not corrupt an existing
   trace. A closed ``StrEnum`` cannot express that: parsing a trace written by
   a newer producer would raise, and the whole trace would be lost rather than
   the one unknown event. So the wire field is a bounded string, and
   :class:`EventType` is the *authoring* vocabulary, not a gate. A reader
   filters with :attr:`TraceEvent.is_known_type` and keeps going.

   This is deliberate and it has a cost: a typo in an event type is accepted
   rather than rejected. The cost is paid back at the consumer, which can
   always say "unknown", and not at the producer, whose entire trace would
   otherwise be unreadable.

2. ``category`` is carried explicitly rather than derived from ``event_type``.
   Deriving it would mean a new type has no category until this module learns
   about it, which is exactly the coupling REQ-OBS-016 asks to avoid. Carrying
   it keeps an unknown event classifiable at read time.

3. Versioning rejects a different MAJOR and accepts any MINOR.
   A different major may have changed field meaning, so accepting it would be
   guessing. A different minor is additive by definition, so refusing it would
   break forward compatibility for no safety gain.

4. Capacity figures are allowed here and only here.
   ``oai2.knowledge.admission`` keeps ``limit``/``in_flight`` off every
   caller-visible surface, because a saturation response must not disclose
   capacity. Sanitized operational counters are a legitimate TELEMETRY payload,
   and this is the telemetry surface. :class:`AdmissionCounters` is the one
   sanctioned way to carry them; the error path never reads it.

PUBLIC SAFETY
-------------
No account ids, database ids, bucket names, index names, endpoints or
credentials. Every free-form string is length-bounded, and ``extra="forbid"``
throughout, so a widened payload cannot smuggle extra state — the same rule
``oai2/knowledge/transport.py`` states for the wire contract.

WHAT DOES NOT EXIST YET
-----------------------
Two of the knowledge events have no emitter in the current codebase:
``knowledge.retry`` (there is no retry loop — backoff is deliberately the
client's, see ``deploy/cloudflare/knowledge_worker/entry.py``) and
``knowledge.kv.degraded`` (the KV cache currently swallows every exception and
cannot distinguish "degraded" from "miss"). The event types are defined because
the schema is a shared vocabulary, but defining an event type is not a claim
that anything emits it. Wiring them is separate work.
"""

from __future__ import annotations

import re
import uuid
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Schema version of this envelope. Bump the MAJOR when a field changes
#: meaning; bump the MINOR when a field is added.
OBSERVABILITY_SCHEMA_VERSION: Final[str] = "1.0"

#: Majors this build can read. See decision 3 in the module docstring.
SUPPORTED_SCHEMA_MAJORS: Final[frozenset[str]] = frozenset({"1"})

#: Trace ids are namespaced, like every other identifier this repo mints
#: (``ko_`` for knowledge, ``oai2v1-`` for vector ids). A prefix makes a trace
#: id self-describing in a log line, and makes it impossible to paste a
#: knowledge id where a trace id belongs.
TRACE_ID_PREFIX: Final[str] = "trc_"

MAX_TRACE_ID_LENGTH: Final[int] = 128
MAX_EVENT_TYPE_LENGTH: Final[int] = 64
MAX_IDENTIFIER_LENGTH: Final[int] = 128
MAX_DETAIL_LENGTH: Final[int] = 1000

#: Dotted lowercase path, e.g. ``knowledge.request.started``. Constrained so a
#: value is a namespace plus a name, which is what makes grouping and prefix
#: filtering meaningful, and bounded so it cannot carry payload.
_EVENT_TYPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
)

_TRACE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"^{re.escape(TRACE_ID_PREFIX)}[0-9a-f]{{8,48}}$"
)


class EventCategory(StrEnum):
    """The event classes REQ-OBS-013 requires to be distinguishable.

    Carried on every event rather than derived, so a type this build has never
    seen still arrives with a usable class. See decision 2 in the docstring.
    """

    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    VISION = "vision"
    AGENT = "agent"
    VERIFIER = "verifier"
    KNOWLEDGE = "knowledge"


class EventType(StrEnum):
    """The AUTHORING vocabulary of known event types.

    Not a gate. :attr:`TraceEvent.event_type` is a string precisely so that a
    producer newer than this reader can add types without invalidating the
    trace. Membership here only drives :attr:`TraceEvent.is_known_type`, which
    consumers use to separate what they understand from what they must pass
    through untouched.
    """

    # --- CFOPS knowledge transport -------------------------------------
    KNOWLEDGE_REQUEST_STARTED = "knowledge.request.started"
    KNOWLEDGE_REQUEST_COMPLETED = "knowledge.request.completed"
    KNOWLEDGE_SATURATED = "knowledge.saturated"
    KNOWLEDGE_DEPENDENCY_FAILURE = "knowledge.dependency.failure"
    KNOWLEDGE_INTEGRITY_FAILURE = "knowledge.integrity.failure"
    KNOWLEDGE_CONFLICT = "knowledge.conflict"
    KNOWLEDGE_KV_DEGRADED = "knowledge.kv.degraded"
    KNOWLEDGE_RETRY = "knowledge.retry"

    # --- model ----------------------------------------------------------
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_ERROR = "model.error"

    # --- tool -----------------------------------------------------------
    TOOL_DISPATCH = "tool.dispatch"
    TOOL_RESULT = "tool.result"

    # --- retrieval ------------------------------------------------------
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_RESULT = "retrieval.result"

    # --- vision ---------------------------------------------------------
    VISION_INPUT = "vision.input"
    VISION_OUTPUT = "vision.output"

    # --- agent ----------------------------------------------------------
    AGENT_STEP = "agent.step"
    AGENT_TURN = "agent.turn"

    # --- verifier -------------------------------------------------------
    VERIFIER_ASSESSMENT = "verifier.assessment"
    VERIFIER_REJECTION = "verifier.rejection"


#: The eight event types the CFOPS knowledge contract requires. Pinned as a
#: module constant so a rename has to be deliberate and is visible in review.
KNOWLEDGE_EVENT_TYPES: Final[tuple[EventType, ...]] = (
    EventType.KNOWLEDGE_REQUEST_STARTED,
    EventType.KNOWLEDGE_REQUEST_COMPLETED,
    EventType.KNOWLEDGE_SATURATED,
    EventType.KNOWLEDGE_DEPENDENCY_FAILURE,
    EventType.KNOWLEDGE_INTEGRITY_FAILURE,
    EventType.KNOWLEDGE_CONFLICT,
    EventType.KNOWLEDGE_KV_DEGRADED,
    EventType.KNOWLEDGE_RETRY,
)

_KNOWN_EVENT_TYPE_VALUES: Final[frozenset[str]] = frozenset(
    item.value for item in EventType
)


class AdmissionCounters(BaseModel):
    """Sanitized admission counters.

    Permitted here, and only here. ``KnowledgeSaturatedError`` keeps these on
    the exception for operator logs and deliberately off the wire, because a
    refusal must not disclose capacity to a caller. This model is the telemetry
    surface, which is the one place they belong. It carries no identity of any
    resource: counts only.
    """

    model_config = ConfigDict(extra="forbid")

    in_flight: int = Field(ge=0)
    peak_in_flight: int = Field(ge=0)
    admitted_count: int = Field(ge=0)
    refused_count: int = Field(ge=0)


class TraceEvent(BaseModel):
    """One event in one trace.

    Construct through :meth:`start` or the ``knowledge_*`` helpers rather than
    by hand: the ``seq`` and ``schema_version`` are the schema's job, not
    every call site's.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=OBSERVABILITY_SCHEMA_VERSION)
    trace_id: str = Field(min_length=1, max_length=MAX_TRACE_ID_LENGTH)
    seq: int = Field(ge=0)
    category: EventCategory
    event_type: str = Field(min_length=1, max_length=MAX_EVENT_TYPE_LENGTH)
    timestamp: float = Field(ge=0)

    # --- optional correlation ------------------------------------------
    #: Correlates the events of one logical request. Reuses the transport's
    #: own ``request_id`` rather than minting a second identity, so an event
    #: can be tied back to the response a caller already holds.
    request_id: str | None = Field(default=None, min_length=1,
                                   max_length=MAX_IDENTIFIER_LENGTH)
    #: Evidence produced or consumed by this event (REQ-OBS-014).
    evidence_ids: tuple[str, ...] = ()
    #: Sanitized operational counters. See :class:`AdmissionCounters`.
    admission: AdmissionCounters | None = None

    # --- optional outcome ----------------------------------------------
    #: An outcome vocabulary, e.g. a ``TransportErrorCode`` value. A string
    #: rather than an enum for the same forward-compatibility reason as
    #: ``event_type``.
    outcome: str | None = Field(default=None, min_length=1,
                                max_length=MAX_IDENTIFIER_LENGTH)
    retryable: bool | None = None
    attempt: int | None = Field(default=None, ge=1)
    detail: str | None = Field(default=None, max_length=MAX_DETAIL_LENGTH)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        major, _, _minor = self.schema_version.partition(".")
        if major not in SUPPORTED_SCHEMA_MAJORS:
            raise ValueError(
                f"unsupported trace schema major version {major!r}; this build "
                f"reads {sorted(SUPPORTED_SCHEMA_MAJORS)}"
            )
        if not _EVENT_TYPE_PATTERN.match(self.event_type):
            raise ValueError(
                "event_type must be a dotted lowercase path such as "
                "'knowledge.request.started'"
            )
        if not _TRACE_ID_PATTERN.match(self.trace_id):
            raise ValueError(
                f"trace_id must be {TRACE_ID_PREFIX!r} followed by 8-48 hex "
                "characters; use new_trace_id()"
            )
        for evidence_id in self.evidence_ids:
            if not evidence_id.strip() or len(evidence_id) > MAX_IDENTIFIER_LENGTH:
                raise ValueError(
                    "evidence_ids must be non-empty and at most "
                    f"{MAX_IDENTIFIER_LENGTH} characters"
                )
        return self

    @property
    def is_known_type(self) -> bool:
        """Whether this build's vocabulary contains the type.

        False does NOT make the event invalid. A consumer that does not
        recognise a type should keep the event, count it as unknown, and carry
        on — discarding the trace because of one new type is the failure
        REQ-OBS-016 exists to prevent.
        """
        return self.event_type in _KNOWN_EVENT_TYPE_VALUES

    @classmethod
    def start(
        cls,
        *,
        trace_id: str,
        seq: int,
        category: EventCategory,
        event_type: EventType | str,
        timestamp: float,
        **kwargs: object,
    ) -> TraceEvent:
        """Build an event with the schema version and bounds applied."""
        value = event_type.value if isinstance(event_type, EventType) else event_type
        return cls(
            trace_id=trace_id,
            seq=seq,
            category=category,
            event_type=value,
            timestamp=timestamp,
            **kwargs,  # type: ignore[arg-type]
        )


def new_trace_id() -> str:
    """Mint a fresh trace id.

    Prefixed and hex, so it is self-describing in a log line and cannot be
    confused with a knowledge id (``ko_``) or a vector id (``oai2v1-``).
    """
    return f"{TRACE_ID_PREFIX}{uuid.uuid4().hex[:32]}"


class Trace:
    """An ordered collection of events sharing one trace id.

    Enforces REQ-OBS-012 at insertion rather than leaving it to every call
    site: a trace whose sequence numbers go backwards cannot be replayed, and
    discovering that during an incident is the worst possible time.
    """

    __slots__ = ("_events", "_trace_id")

    def __init__(self, trace_id: str | None = None) -> None:
        self._trace_id = trace_id or new_trace_id()
        self._events: list[TraceEvent] = []

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    @property
    def next_seq(self) -> int:
        return len(self._events)

    def append(self, event: TraceEvent) -> TraceEvent:
        """Append, enforcing trace identity and monotonic sequencing."""
        if event.trace_id != self._trace_id:
            raise ValueError(
                f"event belongs to trace {event.trace_id!r}, not {self._trace_id!r}"
            )
        if event.seq != self.next_seq:
            raise ValueError(
                f"event seq must be {self.next_seq}, got {event.seq}; a trace "
                "whose sequence is not monotonic cannot be replayed"
            )
        self._events.append(event)
        return event

    def record(
        self,
        *,
        category: EventCategory,
        event_type: EventType | str,
        timestamp: float,
        **kwargs: object,
    ) -> TraceEvent:
        """Build the next event in this trace and append it."""
        return self.append(
            TraceEvent.start(
                trace_id=self._trace_id,
                seq=self.next_seq,
                category=category,
                event_type=event_type,
                timestamp=timestamp,
                **kwargs,
            )
        )

    def by_category(self, category: EventCategory) -> tuple[TraceEvent, ...]:
        return tuple(e for e in self._events if e.category is category)

    def of_type(self, event_type: EventType | str) -> tuple[TraceEvent, ...]:
        value = event_type.value if isinstance(event_type, EventType) else event_type
        return tuple(e for e in self._events if e.event_type == value)

    def evidence_ids(self) -> tuple[str, ...]:
        """Every evidence id referenced by this trace, in first-seen order.

        Duplicates are collapsed because a verifier assessing the same
        evidence twice is not two pieces of evidence.
        """
        seen: dict[str, None] = {}
        for event in self._events:
            for evidence_id in event.evidence_ids:
                seen.setdefault(evidence_id, None)
        return tuple(seen)

    def unknown_types(self) -> tuple[str, ...]:
        """Types this build does not recognise, in order of first appearance."""
        seen: dict[str, None] = {}
        for event in self._events:
            if not event.is_known_type:
                seen.setdefault(event.event_type, None)
        return tuple(seen)

    def as_dicts(self) -> list[dict[str, object]]:
        """JSON-ready representation, in sequence order."""
        return [e.model_dump(mode="json") for e in self._events]


# ---------------------------------------------------------------------------
# The CFOPS knowledge events
#
# Thin constructors over the same envelope, so the eight event names are
# written once and the field mapping is testable without a live request.
# ---------------------------------------------------------------------------


def knowledge_request_started(
    trace: Trace, *, request_id: str, operation: str, timestamp: float
) -> TraceEvent:
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_STARTED,
        timestamp=timestamp,
        request_id=request_id,
        detail=operation,
    )


def knowledge_request_completed(
    trace: Trace,
    *,
    request_id: str,
    timestamp: float,
    ok: bool,
    outcome: str | None = None,
    retryable: bool | None = None,
) -> TraceEvent:
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_REQUEST_COMPLETED,
        timestamp=timestamp,
        request_id=request_id,
        outcome=outcome or ("ok" if ok else "error"),
        retryable=retryable,
    )


def knowledge_saturated(
    trace: Trace,
    *,
    request_id: str,
    timestamp: float,
    counters: AdmissionCounters,
) -> TraceEvent:
    """A capacity refusal.

    The counters ride here because this is telemetry. They are NOT added to
    the refusal's message, which stays the fixed generic string — see
    ``oai2.knowledge.admission.PUBLIC_SATURATION_MESSAGE``.
    """
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_SATURATED,
        timestamp=timestamp,
        request_id=request_id,
        outcome="saturated",
        retryable=True,
        admission=counters,
    )


def knowledge_failure(
    trace: Trace,
    *,
    event_type: EventType,
    request_id: str,
    timestamp: float,
    outcome: str | None = None,
    retryable: bool | None = None,
) -> TraceEvent:
    """Dependency / integrity / conflict, which differ only in type."""
    if event_type not in (
        EventType.KNOWLEDGE_DEPENDENCY_FAILURE,
        EventType.KNOWLEDGE_INTEGRITY_FAILURE,
        EventType.KNOWLEDGE_CONFLICT,
    ):
        raise ValueError(f"{event_type!r} is not a knowledge failure type")
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=event_type,
        timestamp=timestamp,
        request_id=request_id,
        outcome=outcome,
        retryable=retryable,
    )


def knowledge_kv_degraded(
    trace: Trace, *, request_id: str, timestamp: float, detail: str
) -> TraceEvent:
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_KV_DEGRADED,
        timestamp=timestamp,
        request_id=request_id,
        outcome="degraded",
        retryable=True,
        detail=detail,
    )


def knowledge_retry(
    trace: Trace,
    *,
    request_id: str,
    timestamp: float,
    attempt: int,
    outcome: str | None = None,
) -> TraceEvent:
    return trace.record(
        category=EventCategory.KNOWLEDGE,
        event_type=EventType.KNOWLEDGE_RETRY,
        timestamp=timestamp,
        request_id=request_id,
        attempt=attempt,
        outcome=outcome,
    )


__all__ = [
    "MAX_EVENT_TYPE_LENGTH",
    "MAX_IDENTIFIER_LENGTH",
    "OBSERVABILITY_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_MAJORS",
    "TRACE_ID_PREFIX",
    "AdmissionCounters",
    "EventCategory",
    "EventType",
    "KNOWLEDGE_EVENT_TYPES",
    "Trace",
    "TraceEvent",
    "knowledge_failure",
    "knowledge_kv_degraded",
    "knowledge_request_completed",
    "knowledge_request_started",
    "knowledge_retry",
    "knowledge_saturated",
    "new_trace_id",
]
