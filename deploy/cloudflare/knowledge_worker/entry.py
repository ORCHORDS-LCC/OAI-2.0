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
    acquire_admission,
    validate_limit,
)

#: Env var carrying the per-isolate knowledge admission limit. Read and
#: validated on EVERY request, then passed into the module-scope admission
#: state. The state is never permanently bound to the first request's value, so
#: a configuration change takes effect immediately rather than at the next
#: isolate recycle.
ADMISSION_LIMIT_ENV = "KNOWLEDGE_MAX_IN_FLIGHT"


class Default(WorkerEntrypoint):
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

        # ---- Admission, BEFORE any component init or awaited dependency. ----
        # Acquiring here means an unauthenticated, malformed or invalid
        # request never occupies a knowledge-operation slot, while everything
        # genuinely expensive is inside the bound.
        try:
            lease = acquire_admission(_admission_limit(self.env))
        except ValueError:
            # A misconfigured limit is an operator error, not caller error, and
            # it is not a saturation: the isolate is not full, it is
            # misconfigured. Fail closed without pretending to be backpressure.
            return Response.json(
                _error_payload(
                    transport_request.request_id,
                    TransportErrorCode.INTERNAL,
                    "knowledge Worker admission is misconfigured",
                ),
                status=500,
            )
        except Exception as exc:  # KnowledgeSaturatedError
            return Response.json(
                _saturated_payload(transport_request.request_id, exc),
                status=_status_for_error(_saturated_response(
                    transport_request.request_id, exc
                )),
            )

        # ---- Everything below is inside the bound, and must release. ----
        # `finally`, so success, NOT_FOUND, conflict, integrity failure,
        # dependency exception, internal exception and cancellation all return
        # the slot. Admission is only a bound if slots actually come back.
        try:
            try:
                components = await self._components()
                response = await components.transport.handle(transport_request)
            except Exception:
                response = KnowledgeTransportResponse(
                    request_id=transport_request.request_id,
                    ok=False,
                    error=TransportError(
                        code=TransportErrorCode.UNAVAILABLE_DEPENDENCY,
                        message=(
                            "knowledge Worker dependency initialization failed"
                        ),
                        retryable=True,
                    ),
                )
        finally:
            lease.release()

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
