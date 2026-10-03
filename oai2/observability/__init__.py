"""One structured trace/event schema (WI-OBS-001, #56).

Public-safe by construction: no account ids, database ids, bucket or index
names, endpoints or credentials. See ``events.py`` for the schema and, more
importantly, for the reasoning behind its four load-bearing decisions.
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

__all__ = [
    "KNOWLEDGE_EVENT_TYPES",
    "OBSERVABILITY_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_MAJORS",
    "TRACE_ID_PREFIX",
    "AdmissionCounters",
    "EventCategory",
    "EventType",
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
