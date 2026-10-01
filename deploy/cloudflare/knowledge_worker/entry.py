"""Cloudflare Python Worker entrypoint for the OAI-2.0 knowledge transport.

Deployment identifiers and KNOWLEDGE_AUTH_TOKEN are supplied by Wrangler
bindings/secrets. The request body's auth field is ignored and replaced only
after bearer-token authentication succeeds.
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


class Default(WorkerEntrypoint):
    async def fetch(self, request):
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

        try:
            components = await self._components()
            response = await components.transport.handle(transport_request)
        except Exception:
            response = KnowledgeTransportResponse(
                request_id=transport_request.request_id,
                ok=False,
                error=TransportError(
                    code=TransportErrorCode.UNAVAILABLE_DEPENDENCY,
                    message="knowledge Worker dependency initialization failed",
                    retryable=True,
                ),
            )

        status = 200 if response.ok else _status_for_error(response)
        return Response.json(response.model_dump(mode="json"), status=status)

    async def _components(self):
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
            ensure_schema=True,
        )
        self._knowledge_components = components
        return components


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


def _status_for_error(response: KnowledgeTransportResponse) -> int:
    if response.error is None:
        return 500
    return {
        TransportErrorCode.AUTHORIZATION: 403,
        TransportErrorCode.VALIDATION: 400,
        TransportErrorCode.NOT_FOUND: 404,
        TransportErrorCode.CONFLICT: 409,
        TransportErrorCode.UNAVAILABLE_DEPENDENCY: 503,
        TransportErrorCode.INTEGRITY: 502,
        TransportErrorCode.INTERNAL: 500,
    }[response.error.code]
