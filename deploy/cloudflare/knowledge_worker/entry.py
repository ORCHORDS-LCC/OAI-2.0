"""Cloudflare Python Worker entrypoint for the OAI-2.0 knowledge transport.

Deployment identifiers and KNOWLEDGE_AUTH_TOKEN are supplied by Wrangler
bindings/secrets. The request body's auth field is ignored and replaced only
after bearer-token authentication succeeds.

TWO LIFECYCLE FACTS THIS MODULE IS BUILT AROUND
-----------------------------------------------
1. ``WorkerEntrypoint`` constructs a NEW instance per invocation. Nothing on
   ``self`` survives a request, so nothing that must span requests may live
   there — not admission state, and not a component cache. The
   ``_knowledge_components`` attribute is an intra-request memo only.
2. Module-scope Python state persists for the life of one isolate, and is
   shared by every invocation that isolate handles. That is the correct scope
   for the admission counters in ``oai2.knowledge.admission``, and the correct
   scope for nothing else here: binding-derived clients are never cached
   globally, because Cloudflare can reuse an isolate across a binding-only
   change and a cached client would then be stale.

SCHEMA IS NOT APPLIED HERE
-------------------------
This request path does NOT create or migrate tables. Schema is a deployment
prerequisite, applied out of band by the provisioning step in
``oai2.knowledge.cloudflare_provisioning`` (see #261 and #19). Applying it per
request was both an ordinary side effect of serving traffic and ineffective as
caching, because a per-invocation instance memo is rebuilt every request
anyway. A request that reaches a missing schema fails clearly instead.
"""

from __future__ import annotations

import hmac
import time
from typing import ClassVar

from pydantic import ValidationError
from workers import Response, WorkerEntrypoint

from oai2.knowledge import (
    KnowledgeTransportRequest,
    KnowledgeTransportResponse,
    TransportAuthContext,
    TransportError,
    TransportErrorCode,
    build_cloudflare_knowledge_components,
)
from oai2.knowledge.admission import (
    DEFAULT_ADMISSION_LIMIT,
    ISOLATE_ADMISSION,
    KnowledgeSaturatedError,
    acquire_admission,
    validate_limit,
)
from oai2.knowledge.observability import (
    emit_completed,
    emit_internal_failure,
    emit_request_started,
    emit_saturated,
)
from oai2.observability import EventSink, TraceRecorder, new_request_recorder
from oai2.observability.health import ISOLATE_RECORDER_HEALTH
from oai2.observability.isolate import (
    DEFAULT_METRICS_RECENT_EVENTS,
    ISOLATE_METRICS,
    clamp_recent_events,
)

#: Env var carrying the per-isolate knowledge admission limit. Read and
#: validated on EVERY request, then passed into the module-scope admission
#: state. The state is never permanently bound to the first request's value, so
#: a configuration change takes effect immediately rather than at the next
#: isolate recycle.
ADMISSION_LIMIT_ENV = "KNOWLEDGE_MAX_IN_FLIGHT"

#: Env vars controlling local, bounded, in-process metrics.
#:
#: Read on EVERY request for the same reason the admission limit is: Cloudflare
#: can reuse an isolate across a binding-only change, so a value resolved once
#: and pinned would leave the isolate running under whatever the first request
#: happened to see. Turning metrics off has to actually turn them off.
#:
#: Neither value is a secret, and neither becomes a metric dimension. The
#: aggregator's dimensions come from closed vocabularies in the event schema;
#: configuration is an input to whether counting happens at all, never a label
#: on what was counted.
METRICS_ENABLED_ENV = "KNOWLEDGE_METRICS_ENABLED"
METRICS_RETENTION_ENV = "KNOWLEDGE_METRICS_MAX_RECENT"

#: Exact, closed truthy vocabulary. Anything else is OFF.
#:
#: Falling back to OFF for an unrecognised value is the safe direction: a typo
#: in configuration must not silently start retaining request activity that
#: nobody asked to retain, and must certainly not fail the request.
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


class Default(WorkerEntrypoint):
    #: Explicitly installed event sink, class-level. ``None`` means "none was
    #: installed", and the sink is then resolved from configuration per
    #: request — see :func:`_resolve_sink`.
    #:
    #: It is a CLASS attribute because a sink is configuration, not per-request
    #: state: unlike admission counters, a sink holds nothing that must be
    #: shared or must NOT be shared between invocations. Setting it is an
    #: escape hatch for an embedding host and for tests; the SHIPPED path
    #: configures metrics through the environment, which is what makes the
    #: instrumentation reachable without anyone editing this file.
    event_sink: ClassVar[EventSink | None] = None

    async def fetch(self, request):
        # ---- Cheap validation. None of this consumes an admission slot. ----
        if str(request.method).upper() != "POST":
            return Response.json(
                _error_payload(
                    "invalid-request",
                    TransportErrorCode.VALIDATION,
                    "knowledge transport accepts POST only",
                ),
                status=405,
            )

        expected_token = str(self.env.KNOWLEDGE_AUTH_TOKEN)
        authorization = request.headers.get("Authorization")
        provided_token = _bearer_token(authorization)
        if (
            provided_token is None
            or not expected_token
            or not hmac.compare_digest(provided_token, expected_token)
        ):
            return Response.json(
                _error_payload(
                    "unauthorized",
                    TransportErrorCode.AUTHORIZATION,
                    "authentication failed",
                ),
                status=401,
            )

        try:
            payload = await request.json()
        except Exception:
            return Response.json(
                _error_payload(
                    "invalid-json",
                    TransportErrorCode.VALIDATION,
                    "request body must be valid JSON",
                ),
                status=400,
            )

        if not isinstance(payload, dict):
            return Response.json(
                _error_payload(
                    "invalid-request",
                    TransportErrorCode.VALIDATION,
                    "request body must be a JSON object",
                ),
                status=400,
            )

        payload = dict(payload)
        payload["auth"] = TransportAuthContext(
            subject="private-worker-token",
            capabilities=("knowledge.read", "knowledge.write"),
        ).model_dump(mode="json")

        try:
            transport_request = KnowledgeTransportRequest.model_validate(payload)
        except ValidationError as exc:
            request_id = _safe_request_id(payload.get("request_id"))
            return Response.json(
                _error_payload(
                    request_id,
                    TransportErrorCode.VALIDATION,
                    _validation_message(exc),
                ),
                status=400,
            )

        # ---- ONE trace for this accepted request. ----
        # Minted here, once, after validation and before admission. Every layer
        # below receives this recorder; none of them mints a trace of its own,
        # which is what makes the entrypoint/admission/transport/runtime events
        # one correlated trace instead of four unrelated ones.
        #
        # Scope is narrow on purpose: only requests that got as far as a valid
        # KnowledgeTransportRequest are traced. A malformed body or a failed
        # auth is a transport-boundary reject, not a knowledge operation, and
        # counting it as one would make "requests started" disagree with what
        # the runtime actually attempted.
        #
        # The operation is set HERE so every knowledge event in the trace is
        # groupable by it without any layer having to remember to pass it.
        recorder: TraceRecorder = new_request_recorder(
            sink=_resolve_sink(self.env, type(self)),
            request_id=transport_request.request_id,
            operation=transport_request.operation.value,
            # So a sink that breaks is counted somewhere that outlives the
            # request. Never routed back through the sink itself.
            health=ISOLATE_RECORDER_HEALTH,
        )
        recorder.start_clock()
        emit_request_started(
            recorder,
            operation=transport_request.operation.value,
            timestamp=_now(),
        )

        # ---- Admission, BEFORE any component init or awaited dependency. ----
        # Acquiring here means an unauthenticated, malformed or invalid
        # request never occupies a knowledge-operation slot, while everything
        # genuinely expensive is inside the bound.
        #
        # The exception taxonomy below is EXACT. A previous revision caught
        # `Exception` and reported every outcome as a retryable SATURATED 503,
        # which is how a broken admission primitive would be advertised to
        # callers as routine backpressure. Only KnowledgeSaturatedError is
        # saturation; everything else is an internal fault.
        try:
            lease = acquire_admission(_admission_limit(self.env))
        except KnowledgeSaturatedError as exc:
            emit_saturated(
                recorder, timestamp=_now(), snapshot=ISOLATE_ADMISSION.snapshot()
            )
            emit_completed(
                recorder,
                timestamp=_now(),
                ok=False,
                outcome=TransportErrorCode.SATURATED.value,
                retryable=True,
            )
            saturated = _saturated_response(transport_request.request_id, exc)
            return Response.json(
                saturated.model_dump(mode="json"),
                status=_status_for_error(saturated),
            )
        except ValueError:
            # A misconfigured limit is an OPERATOR error: the isolate is not
            # full, it is misconfigured. Fail closed, and say so without
            # pretending to be backpressure. This is not a runtime fault, so
            # no internal-failure event — the completion outcome carries it.
            emit_completed(
                recorder,
                timestamp=_now(),
                ok=False,
                outcome=TransportErrorCode.INTERNAL.value,
                retryable=False,
            )
            return Response.json(
                _error_payload(
                    transport_request.request_id,
                    TransportErrorCode.INTERNAL,
                    "knowledge Worker admission is misconfigured",
                ),
                status=500,
            )
        except Exception:
            # The admission primitive itself failed. That is a bug here, not
            # backpressure and not a caller error. Reported as internal and
            # explicitly NOT as saturation.
            emit_internal_failure(
                recorder, timestamp=_now(), request_id=transport_request.request_id
            )
            emit_completed(
                recorder,
                timestamp=_now(),
                ok=False,
                outcome=TransportErrorCode.INTERNAL.value,
                retryable=False,
            )
            return Response.json(
                _error_payload(
                    transport_request.request_id,
                    TransportErrorCode.INTERNAL,
                    "knowledge Worker admission failed",
                ),
                status=500,
            )

        # ---- Inside the bound. Build and execution are SEPARATE boundaries. ----
        # `lease.release()` in `finally` on every path: success, NOT_FOUND,
        # conflict, integrity failure, dependency exception, internal
        # exception and cancellation all return the slot. Admission is only a
        # bound if slots actually come back.
        #
        # The two boundaries have DIFFERENT taxonomies, and conflating them was
        # a real defect:
        #
        # * `_components()` runs with ensure_schema=False, which is local
        #   object assembly and validation. It makes NO remote dependency call,
        #   so there is nothing that could be "unavailable". An exception from
        #   it is an operator or configuration fault -> INTERNAL.
        # * `transport.handle()` converts KNOWN dependency, conflict and
        #   integrity errors into typed responses itself. So an exception
        #   ESCAPING it is by definition not one of those: it is an unexpected
        #   local fault -> INTERNAL, never UNAVAILABLE_DEPENDENCY. Labelling
        #   it a dependency outage sends an operator to D1 for a bug here.
        #
        # A typed UNAVAILABLE_DEPENDENCY *response* is already correct and
        # already emitted a dependency event inside the transport, so this
        # branch adds nothing for it — otherwise one outage counts twice.
        try:
            try:
                components = await self._components()
            except Exception:
                emit_internal_failure(
                    recorder,
                    timestamp=_now(),
                    request_id=transport_request.request_id,
                )
                response = _internal_response(
                    transport_request.request_id,
                    "knowledge Worker component initialization failed",
                )
            else:
                try:
                    response = await components.transport.handle(
                        transport_request, trace=recorder
                    )
                except Exception:
                    emit_internal_failure(
                        recorder,
                        timestamp=_now(),
                        request_id=transport_request.request_id,
                    )
                    response = _internal_response(
                        transport_request.request_id,
                        "knowledge Worker transport failed unexpectedly",
                    )
        finally:
            lease.release()

        # EXACTLY ONE completion, here, for every admitted request. The
        # transport emits failure DETAIL only; it never emits a completion, so
        # this cannot be duplicated by a success and a failure path racing.
        error = response.error
        emit_completed(
            recorder,
            timestamp=_now(),
            ok=response.ok,
            outcome=(
                "ok" if response.ok
                else (error.code.value if error is not None else "internal")
            ),
            retryable=error.retryable if error is not None else False,
        )

        status = 200 if response.ok else _status_for_error(response)
        return Response.json(response.model_dump(mode="json"), status=status)

    async def _components(self):
        # INTRA-REQUEST memo only. WorkerEntrypoint builds a new instance per
        # invocation, so this is not a cross-request cache and must not be
        # treated as one. It exists to avoid building the component set twice
        # within a single call graph, nothing more.
        existing = getattr(self, "_knowledge_components", None)
        if existing is not None:
            return existing

        components = await build_cloudflare_knowledge_components(
            d1=self.env.DB,
            r2=self.env.KNOWLEDGE_R2,
            vectorize=self.env.KNOWLEDGE_VECTORIZE,
            kv=self.env.KNOWLEDGE_KV,
            embedding_version=str(self.env.EMBEDDING_VERSION),
            embedding_digest=str(self.env.EMBEDDING_DIGEST),
            embedding_provider=None,
            require_auth=True,
            # NOT a request side effect. Schema is provisioned out of band; see
            # oai2.knowledge.cloudflare_provisioning and #261.
            ensure_schema=False,
        )
        self._knowledge_components = components
        return components


def _resolve_sink(env: object, owner: type) -> EventSink:
    """The sink for this request: explicit installation, else configuration.

    An explicitly installed class sink wins and configuration is not even
    consulted, so a host that installs one is never surprised by a Worker
    variable quietly taking it over. With nothing installed — the shipped
    shape — the sink comes from configuration, and the default is OFF.
    """
    installed = getattr(owner, "event_sink", None)
    if installed is not None:
        return installed
    enabled, max_recent = _metrics_config(env)
    return ISOLATE_METRICS.sink_for(enabled=enabled, max_recent_events=max_recent)


def _metrics_config(env: object) -> tuple[bool, int]:
    """Read the local metrics configuration for THIS request.

    Two independent decisions, and both are resolved defensively:

    * enabled — a closed truthy vocabulary. An unrecognised value is OFF,
      because a typo must not start retaining request activity nobody asked
      to retain, and must never fail the request.
    * retention — an integer, clamped to the range this build supports, so no
      configuration value turns a bounded store into an unbounded one. An
      unusable value falls back to the default rather than guessing.

    Neither value is retained as a dimension and neither is a secret.
    """
    raw_enabled = getattr(env, METRICS_ENABLED_ENV, None)
    enabled = (
        isinstance(raw_enabled, str)
        and raw_enabled.strip().lower() in _TRUTHY_ENV_VALUES
    )

    raw_recent = getattr(env, METRICS_RETENTION_ENV, None)
    try:
        max_recent = clamp_recent_events(
            int(str(raw_recent).strip())
            if raw_recent not in (None, "")
            else DEFAULT_METRICS_RECENT_EVENTS
        )
    except (TypeError, ValueError):
        max_recent = DEFAULT_METRICS_RECENT_EVENTS
    return enabled, max_recent


def _internal_response(request_id: str, message: str) -> KnowledgeTransportResponse:
    """A non-retryable INTERNAL response. Used by both inner boundaries."""
    return KnowledgeTransportResponse(
        request_id=request_id,
        ok=False,
        error=TransportError(
            code=TransportErrorCode.INTERNAL,
            message=message,
            retryable=False,
        ),
    )


def _now() -> float:
    """Wall clock for event timestamps. Module scope, no binding, no state."""
    return time.time()


def _admission_limit(env: object) -> int:
    """Read and validate the CURRENT limit for this request.

    Read per request so an operator's change takes effect immediately. A
    missing or unusable value falls back to the default rather than guessing a
    larger number, and a value that is present but invalid is surfaced as an
    operator error by the caller.
    """
    raw = getattr(env, ADMISSION_LIMIT_ENV, None)
    if raw is None or raw == "":
        return DEFAULT_ADMISSION_LIMIT
    if isinstance(raw, bool):
        raise ValueError("admission limit must be a positive integer")
    if isinstance(raw, int):
        return validate_limit(raw)
    try:
        return validate_limit(int(str(raw).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError("admission limit must be a positive integer") from exc


def _saturated_response(
    request_id: str, exc: Exception
) -> KnowledgeTransportResponse:
    from oai2.knowledge.admission import KnowledgeSaturatedError

    if not isinstance(exc, KnowledgeSaturatedError):  # pragma: no cover
        raise TypeError("expected KnowledgeSaturatedError")
    return KnowledgeTransportResponse(
        request_id=request_id,
        ok=False,
        error=exc.error,
    )


def _saturated_payload(
    request_id: str, exc: Exception
) -> dict[str, object]:
    return _saturated_response(request_id, exc).model_dump(mode="json")


def _bearer_token(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    prefix = "Bearer "
    if not text.startswith(prefix):
        return None
    token = text[len(prefix) :]
    return token if token else None


def _safe_request_id(value: object) -> str:
    if isinstance(value, str) and 1 <= len(value) <= 128:
        return value
    return "invalid-request"


def _validation_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "request validation failed"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "request validation failed"))
    text = f"{location}: {message}" if location else message
    return text[:1000]


def _error_payload(
    request_id: str,
    code: TransportErrorCode,
    message: str,
) -> dict[str, object]:
    return KnowledgeTransportResponse(
        request_id=request_id,
        ok=False,
        error=TransportError(
            code=code,
            message=message,
            retryable=False,
        ),
    ).model_dump(mode="json")


#: Public HTTP status per transport error code.
#:
#: SATURATED -> 503, deliberately, and NOT 429:
#:
#: 429 Too Many Requests is about the CLIENT's request rate and, by long
#: standing convention, carries a Retry-After tied to a quota window that the
#: server owns. Isolate saturation is the opposite: a single isolate is out of
#: in-flight capacity, nothing about the client's rate is wrong, and there is
#: no quota window to reset. A client that treats this as 429 may honour a
#: "retry after the quota resets" policy and give up entirely, which is the
#: worst possible response to transient capacity pressure. Conflating the two
#: would also blur exactly the distinction the separate body code exists to
#: preserve.
#:
#: NO Retry-After header is emitted. Retry-After carries a deadline; this
#: refusal has no defensible deadline because the drain time depends on the
#: in-flight work's own latency, which the Worker does not know. Inventing one
#: would tell a caller to retry sooner than the slot can actually free. The
#: `retryable: true` body field is the honest signal; backoff is the client's.
#:
#: 503 is shared with UNAVAILABLE_DEPENDENCY, which is intended: both mean
#: "temporarily unable, retry with backoff". The BODY code is what
#: distinguishes them, and it does: "saturated" is a capacity refusal with no
#: dependency involved, "unavailable_dependency" is a real dependency fault.
_STATUS_BY_CODE: dict[TransportErrorCode, int] = {
    TransportErrorCode.AUTHORIZATION: 403,
    TransportErrorCode.VALIDATION: 400,
    TransportErrorCode.NOT_FOUND: 404,
    TransportErrorCode.CONFLICT: 409,
    TransportErrorCode.UNAVAILABLE_DEPENDENCY: 503,
    TransportErrorCode.INTEGRITY: 502,
    TransportErrorCode.INTERNAL: 500,
    TransportErrorCode.SATURATED: 503,
}


def _status_for_error(response: KnowledgeTransportResponse) -> int:
    if response.error is None:
        return 500
    try:
        return _STATUS_BY_CODE[response.error.code]
    except KeyError:
        # Fail closed rather than raising out of the handler. An unmapped code
        # is a defect, and a 500 is a far better answer than a 500 from an
        # unhandled KeyError with no body.
        return 500
