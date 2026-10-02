from __future__ import annotations

import pytest

from oai2.runtime.service import (
    ServiceCompatibility,
    ServiceLifecycle,
    ServiceState,
)


def _compat(
    *,
    config: str = "cfg-v1",
    model: str = "model-v1",
    schema: str = "schema-v1",
) -> ServiceCompatibility:
    return ServiceCompatibility(
        config_digest=config,
        model_id=model,
        schema_version=schema,
    )


def test_health_and_readiness_are_separate() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())

    assert lifecycle.health.live is True
    assert lifecycle.health.ready is False
    assert lifecycle.health.state is ServiceState.STARTING

    assert lifecycle.start(_compat()) is True
    assert lifecycle.health.live is True
    assert lifecycle.health.ready is True
    assert lifecycle.health.state is ServiceState.READY


def test_initialization_compatibility_failure_is_not_ready() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())

    assert lifecycle.start(_compat(model="wrong-model")) is False
    assert lifecycle.health.live is False
    assert lifecycle.health.ready is False
    assert lifecycle.health.state is ServiceState.FAILED
    assert lifecycle.health.failure_kind == "compatibility_mismatch"


def test_drain_waits_for_active_sessions_then_stops() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    lifecycle.start(_compat())
    lifecycle.session_started("s1")
    lifecycle.session_started("s2")

    active = lifecycle.request_shutdown(cancel_active=False)

    assert active == ("s1", "s2")
    assert lifecycle.state is ServiceState.DRAINING
    assert lifecycle.health.ready is False
    with pytest.raises(RuntimeError, match="ready service"):
        lifecycle.session_started("late")

    assert lifecycle.session_finished("s1") is True
    assert lifecycle.state is ServiceState.DRAINING
    assert lifecycle.session_finished("s2") is True
    assert lifecycle.state is ServiceState.STOPPED


def test_cancel_shutdown_returns_active_sessions_and_stops_immediately() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    lifecycle.start(_compat())
    lifecycle.session_started("s2")
    lifecycle.session_started("s1")

    active = lifecycle.request_shutdown(cancel_active=True)

    assert active == ("s1", "s2")
    assert lifecycle.state is ServiceState.STOPPED
    assert lifecycle.health.active_sessions == 0


def test_restart_revalidates_compatibility() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    lifecycle.start(_compat())
    lifecycle.request_shutdown(cancel_active=True)

    assert lifecycle.restart(_compat(schema="schema-v2")) is False
    assert lifecycle.state is ServiceState.FAILED
    assert lifecycle.restart(_compat()) is True
    assert lifecycle.state is ServiceState.READY


def test_failure_metadata_does_not_require_exception_or_secret_payload() -> None:
    lifecycle = ServiceLifecycle(expected=_compat())
    lifecycle.mark_failed("runtime_init_failed")

    health = lifecycle.health
    assert health.failure_kind == "runtime_init_failed"
    assert not hasattr(health, "message")
    assert not hasattr(health, "exception")
