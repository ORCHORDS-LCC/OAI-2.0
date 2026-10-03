"""One structured trace/event schema (WI-OBS-001, #56).

Public-safe by construction: no account ids, database ids, bucket or index
names, endpoints or credentials. See ``events.py`` for the schema and, more
importantly, for the reasoning behind its four load-bearing decisions;
``recorder.py`` for the request-scoped recording and the containment boundary
that keeps a broken observer from changing request semantics; ``sink.py`` for
the deliberately minimal output contract.
"""

from __future__ import annotations

from .events import (
    KNOWLEDGE_EVENT_TYPES,
    OBSERVABILITY_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_MAJORS,
    TRACE_ID_PREFIX,
    AdmissionCounters,
    EventCategory,
    EventType,
    Trace,
    TraceEvent,
    knowledge_failure,
    knowledge_kv_degraded,
    knowledge_request_completed,
    knowledge_request_started,
    knowledge_retry,
    knowledge_saturated,
    new_trace_id,
)
from .metrics import (
    KNOWN_OUTCOMES,
    MAX_DIMENSION_KEYS,
    MetricsAggregator,
)
from .recorder import TraceRecorder, new_request_recorder
from .sink import CollectingEventSink, EventSink, NullEventSink

__all__ = [
    "KNOWN_OUTCOMES",
    "KNOWLEDGE_EVENT_TYPES",
    "MAX_DIMENSION_KEYS",
    "OBSERVABILITY_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_MAJORS",
    "TRACE_ID_PREFIX",
    "AdmissionCounters",
    "CollectingEventSink",
    "EventCategory",
    "EventSink",
    "EventType",
    "MetricsAggregator",
    "NullEventSink",
    "Trace",
    "TraceEvent",
    "TraceRecorder",
    "knowledge_failure",
    "knowledge_kv_degraded",
    "knowledge_request_completed",
    "knowledge_request_started",
    "knowledge_retry",
    "knowledge_saturated",
    "new_request_recorder",
    "new_trace_id",
]
