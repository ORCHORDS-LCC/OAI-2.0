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

from .cloudflare_runtime import (
    AsyncCloudflareKnowledgeRuntime,
    KnowledgeConflictError,
    KnowledgeIntegrityError,
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
    ) -> KnowledgeTransportResponse:
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
                )
                return KnowledgeTransportResponse(
                    request_id=request.request_id,
                    ok=True,
                    corpus_revision=revision,
                    knowledge=request.knowledge,
                )

            if request.operation is TransportOperation.GET:
                assert request.knowledge_id is not None
                obj = await self._runtime.get(request.knowledge_id)
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
                request.to_retrieval_request(),
                query_vector=vector,
            )
            return KnowledgeTransportResponse(
                request_id=request.request_id,
                ok=True,
                objects=result.objects,
            )
        except KnowledgeConflictError as exc:
            return self._failure(
                request,
                TransportErrorCode.CONFLICT,
                str(exc),
                retryable=True,
            )
        except KnowledgeIntegrityError as exc:
            return self._failure(
                request,
                TransportErrorCode.INTEGRITY,
                str(exc),
                retryable=False,
            )
        except RuntimeError as exc:
            return self._failure(
                request,
                TransportErrorCode.UNAVAILABLE_DEPENDENCY,
                str(exc),
                retryable=True,
            )
        except Exception:
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
        if auth.expires_at is not None and now >= auth.expires_at:
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
