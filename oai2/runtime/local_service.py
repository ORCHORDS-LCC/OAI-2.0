"""FastAPI binding for the deterministic local-service lifecycle.

Only health/readiness and access-policy wiring live here. Inference/session
protocol routes stay in their owning modules so this layer does not invent a
second wire contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse

from .service import ServiceCompatibility, ServiceLifecycle
from .service_security import AccessPolicy


def create_local_service_app(
    *,
    lifecycle: ServiceLifecycle,
    actual_compatibility: ServiceCompatibility,
    access_policy: AccessPolicy,
) -> FastAPI:
    """Build the local service health/readiness surface."""

    bearer = HTTPBearer(auto_error=False)

    async def require_access(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(bearer),
        ],
    ) -> None:
        token = None
        if credentials is not None and credentials.scheme.lower() == "bearer":
            token = credentials.credentials
        if not access_policy.authorize(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        lifecycle.start(actual_compatibility)
        try:
            yield
        finally:
            lifecycle.request_shutdown(cancel_active=True)

    app = FastAPI(
        title="OAI-2.0 Local Service",
        lifespan=lifespan,
        dependencies=[Depends(require_access)],
    )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        health = lifecycle.health
        return JSONResponse(
            status_code=200 if health.live else 503,
            content={
                "state": health.state.value,
                "live": health.live,
                "ready": health.ready,
                "active_sessions": health.active_sessions,
                "failure_kind": health.failure_kind,
            },
        )

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        health = lifecycle.health
        return JSONResponse(
            status_code=200 if health.ready else 503,
            content={
                "state": health.state.value,
                "ready": health.ready,
                "failure_kind": health.failure_kind,
            },
        )

    return app


__all__ = ["create_local_service_app"]
