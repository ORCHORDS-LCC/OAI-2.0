"""Authenticated, versioned Worker transport handler for async knowledge runtime.

The handler is HTTP/framework-neutral: a Cloudflare Worker entrypoint can parse
JSON into KnowledgeTransportRequest, call this service, then serialize the
KnowledgeTransportResponse. Authentication secrets remain outside this module;
only the normalized TransportAuthContext/capabilities enter the handler.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Protocol

from ..observability import TraceRecorder
from .abstraction import RetrievalRequest
from .admission import KnowledgeSaturatedError
from .cloudflare_runtime import (
    AsyncCloudflareKnowledgeRuntime,
    KnowledgeConflictError,
    KnowledgeIntegrityError,
)
from .observability import (
    emit_conflict,
    emit_dependency_failure,
    emit_integrity_failure,
    emit_internal_failure,
)
from .transport import (
    KnowledgeTransportRequest,
    KnowledgeTransportResponse,
    TransportError,
    TransportErrorCode,
    TransportOperation,
)


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> Sequence[float]: ...


class KnowledgeWorkerTransport:
    """Dispatch versioned knowledge operations to the async Cloudflare runtime."""

    def __init__(
        self,
        runtime: AsyncCloudflareKnowledgeRuntime,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        require_auth: bool = True,
    ) -> None:
        if not isinstance(require_auth, bool):
            raise ValueError("require_auth must be a boolean")
        self._runtime = runtime
        self._embedding_provider = embedding_provider
        self._require_auth = require_auth

    async def handle(
        self,
        request: KnowledgeTransportRequest,
        *,
        now: float | None = None,
        trace: TraceRecorder | None = None,
    ) -> KnowledgeTransportResponse:
        """Handle one request.

        ``trace`` is the request's existing recorder, passed down from the
        transport boundary. It is NOT minted here: one trace per accepted
        request, so the entrypoint, admission, transport and runtime events
        correlate. When a caller supplies none — a direct ``handle()`` call
        outside the Worker — failure events are simply not emitted, because
        there is no trace to correlate them into. Minting one here would
        produce a second, unrelated trace for the same logical request.

        This method emits failure DETAIL only. ``knowledge.request.completed``
        belongs to the entrypoint, which owns the lifecycle and emits it
        exactly once regardless of which layer saw the failure.
        """
        timestamp = time.time() if now is None else _finite_non_negative(now, "now")

        required_capability = (
            "knowledge.write"
            if request.operation is TransportOperation.PUT
            else "knowledge.read"
        )
        auth_error = self._authorize(
            request,
            required_capability=required_capability,
            now=timestamp,
        )
        if auth_error is not None:
            return self._failure(
                request,
                TransportErrorCode.AUTHORIZATION,
                auth_error,
                retryable=False,
            )

        try:
            if request.operation is TransportOperation.PUT:
                assert request.knowledge is not None
                vector = await self._embed(request.knowledge.content)
                revision = await self._runtime.put(
                    request.knowledge,
                    vector=vector,
                    now=timestamp,
                    trace=trace,
                )
                return KnowledgeTransportResponse(
                    request_id=request.request_id,
                    ok=True,
                    corpus_revision=revision,
                    knowledge=request.knowledge,
                )

            if request.operation is TransportOperation.GET:
                assert request.knowledge_id is not None
                obj = await self._runtime.get(request.knowledge_id, trace=trace)
                if obj is None:
                    return self._failure(
                        request,
                        TransportErrorCode.NOT_FOUND,
                        "knowledge object was not found",
                        retryable=False,
                    )
                return KnowledgeTransportResponse(
                    request_id=request.request_id,
                    ok=True,
                    knowledge=obj,
                )

            assert request.operation is TransportOperation.RETRIEVE
            assert request.topic is not None
            vector = await self._embed(request.topic)
            result = await self._runtime.retrieve(
                RetrievalRequest(
                    topic=request.topic,
                    limit=request.limit,
                    min_authority=request.min_authority,
                    include_status=request.include_status,
                ),
                query_vector=vector,
                trace=trace,
            )
            return KnowledgeTransportResponse(
                request_id=request.request_id,
                ok=True,
                objects=result.objects,
            )
        except KnowledgeSaturatedError as exc:
            # REQ-CFOPS-013: admitted and then refused. Carries its own
            # transport error so the SATURATED code and its retryability reach
            # the caller unchanged rather than being collapsed into a generic
            # dependency failure.
            return self._failure(
                request,
                exc.error.code,
                exc.error.message,
                retryable=exc.error.retryable,
            )
        except KnowledgeConflictError as exc:
            emit_conflict(
                trace, timestamp=timestamp, request_id=request.request_id
            )
            return self._failure(
                request,
                TransportErrorCode.CONFLICT,
                str(exc),
                retryable=True,
            )
        except KnowledgeIntegrityError as exc:
            emit_integrity_failure(
                trace, timestamp=timestamp, request_id=request.request_id
            )
            return self._failure(
                request,
                TransportErrorCode.INTEGRITY,
                str(exc),
                retryable=False,
            )
        except RuntimeError as exc:
            emit_dependency_failure(
                trace, timestamp=timestamp, request_id=request.request_id
            )
            return self._failure(
                request,
                TransportErrorCode.UNAVAILABLE_DEPENDENCY,
                str(exc),
                retryable=True,
            )
        except Exception:
            # Deliberately NOT knowledge.dependency.failure. An unexpected
            # exception is our bug, not an infrastructure fault, and reporting
            # it as a dependency failure would send an on-call engineer to D1
            # instead of to a stack trace. It gets its own event type.
            emit_internal_failure(
                trace, timestamp=timestamp, request_id=request.request_id
            )
            return self._failure(
                request,
                TransportErrorCode.INTERNAL,
                "knowledge transport operation failed",
                retryable=False,
            )

    async def _embed(self, text: str) -> Sequence[float] | None:
        if self._embedding_provider is None:
            return None
        vector = await self._embedding_provider.embed(text)
        if isinstance(vector, (str, bytes, bytearray)) or not vector:
            raise RuntimeError("embedding provider returned an invalid vector")
        return vector

    def _authorize(
        self,
        request: KnowledgeTransportRequest,
        *,
        required_capability: str,
        now: float,
    ) -> str | None:
        auth = request.auth
        if auth is None:
            return "authentication is required" if self._require_auth else None
        if auth.expires_at is not None:
            if not math.isfinite(float(auth.expires_at)):
                return "authentication context has invalid expiry"
            if now >= auth.expires_at:
                return "authentication context is expired"
        if required_capability not in auth.capabilities:
            return f"missing required capability: {required_capability}"
        return None

    @staticmethod
    def _failure(
        request: KnowledgeTransportRequest,
        code: TransportErrorCode,
        message: str,
        *,
        retryable: bool,
    ) -> KnowledgeTransportResponse:
        safe_message = message.strip() or "knowledge transport operation failed"
        return KnowledgeTransportResponse(
            request_id=request.request_id,
            ok=False,
            error=TransportError(
                code=code,
                message=safe_message[:1000],
                retryable=retryable,
            ),
        )


def _finite_non_negative(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


__all__ = [
    "EmbeddingProvider",
    "KnowledgeWorkerTransport",
]
