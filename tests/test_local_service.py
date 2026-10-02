from __future__ import annotations

from fastapi.testclient import TestClient

from oai2.runtime.local_service import create_local_service_app
from oai2.runtime.service import ServiceCompatibility, ServiceLifecycle, ServiceState
from oai2.runtime.service_security import AccessPolicy


def _compat(*, model: str = "model-v1") -> ServiceCompatibility:
    return ServiceCompatibility(
        config_digest="cfg-v1",
        model_id=model,
        schema_version="schema-v1",
    )


def test_loopback_health_and_readiness_follow_lifespan() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    app = create_local_service_app(
        lifecycle=lifecycle,
        actual_compatibility=_compat(),
        access_policy=AccessPolicy(bind_host="127.0.0.1"),
    )

    with TestClient(app) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        assert health.status_code == 200
        assert health.json()["live"] is True
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

    assert lifecycle.state is ServiceState.STOPPED


def test_failed_start_reports_not_live_or_ready() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    app = create_local_service_app(
        lifecycle=lifecycle,
        actual_compatibility=_compat(model="wrong"),
        access_policy=AccessPolicy(bind_host="127.0.0.1"),
    )

    with TestClient(app) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        assert health.status_code == 503
        assert health.json()["failure_kind"] == "compatibility_mismatch"
        assert ready.status_code == 503
        assert ready.json()["ready"] is False


def test_non_loopback_service_requires_valid_bearer_for_health_surface() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    app = create_local_service_app(
        lifecycle=lifecycle,
        actual_compatibility=_compat(),
        access_policy=AccessPolicy(
            bind_host="0.0.0.0",
            bearer_token="secret-token",
        ),
    )

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 401
        assert client.get(
            "/healthz",
            headers={"Authorization": "Bearer wrong"},
        ).status_code == 401
        response = client.get(
            "/healthz",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200
        assert response.json()["ready"] is True
