"""One structured trace/event schema (WI-OBS-001, #56).

SAFETY — THE HONEST VERSION
---------------------------
A previous revision of this docstring said "public-safe by construction". That
was **false**, and it is corrected here.

``TraceEvent`` carries ``detail``, ``request_id``, ``outcome`` and
``evidence_ids``. Those are length-bounded and ``extra="forbid"`` is enforced,
which prevents STRUCTURAL expansion — a widened payload cannot attach new
fields. Neither of those REDACTS anything. A bounded string can still be a
secret.

What this schema actually guarantees:

* no account ids, database ids, bucket or index names, endpoints or
  credentials are produced BY THIS MODULE;
* every free-form string is length-bounded and the model is closed.

What it does NOT guarantee, and what PRODUCERS MUST NOT do:

* copy prompt bodies, source document content, or retrieval topics/queries;
* copy tool input or output bodies;
* copy raw exception text (driver messages carry endpoints and namespaces);
* emit secrets, credentials, or private binding/resource identifiers;
* put any of the above into ``detail`` or ``outcome``.

The mechanism is a **minimization contract on producers**, plus a sanitized
retention boundary in ``metrics.MetricRecord``, which is what keeps the metrics
store from becoming an archive of whatever any future producer chose to emit.

See ``events.py`` for the schema and the reasoning behind its load-bearing
decisions; ``recorder.py`` for request-scoped recording and the containment
boundary that keeps a broken observer from changing request semantics;
``sink.py`` for the minimal output contract; ``metrics.py`` for the derived,
redacted, bounded counter store.
"""

from __future__ import annotations

from .events import (
    KNOWLEDGE_CACHE_EVENT_TYPES,
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
    DIMENSION_KEYS,
    KNOWN_OUTCOMES,
    MAX_DIMENSION_KEYS,
    MetricRecord,
    MetricsAggregator,
    RequestLatency,
    assert_no_identifier_cardinality,
)
from .recorder import TraceRecorder, new_request_recorder
from .sink import CollectingEventSink, EventSink, NullEventSink

__all__ = [
    "DIMENSION_KEYS",
    "KNOWN_OUTCOMES",
    "KNOWLEDGE_CACHE_EVENT_TYPES",
    "KNOWLEDGE_EVENT_TYPES",
    "MAX_DIMENSION_KEYS",
    "OBSERVABILITY_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_MAJORS",
    "TRACE_ID_PREFIX",
    "KNOWLEDGE_CACHE_EVENT_TYPES",
    "AdmissionCounters",
    "CollectingEventSink",
    "EventCategory",
    "EventSink",
    "EventType",
    "MetricRecord",
    "MetricsAggregator",
    "NullEventSink",
    "RequestLatency",
    "assert_no_identifier_cardinality",
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
