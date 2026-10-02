"""Local service lifecycle, readiness and graceful-shutdown primitives.

The module is transport-neutral so FastAPI/Unix-socket/embedded hosts can share
one deterministic lifecycle contract. It intentionally exposes public-safe
health metadata rather than exception text, config values or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ServiceState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(slots=True, frozen=True)
class ServiceCompatibility:
    config_digest: str
    model_id: str
    schema_version: str

    def __post_init__(self) -> None:
        for name in ("config_digest", "model_id", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty normalized string")


@dataclass(slots=True, frozen=True)
class ServiceHealth:
    state: ServiceState
    live: bool
    ready: bool
    active_sessions: int
    failure_kind: str | None = None


class ServiceLifecycle:
    """Deterministic lifecycle shared by local service transports."""

    def __init__(self, *, expected: ServiceCompatibility) -> None:
        self._expected = expected
        self._state = ServiceState.STARTING
        self._active_sessions: set[str] = set()
        self._failure_kind: str | None = None

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def health(self) -> ServiceHealth:
        return ServiceHealth(
            state=self._state,
            live=self._state not in {ServiceState.FAILED, ServiceState.STOPPED},
            ready=self._state is ServiceState.READY,
            active_sessions=len(self._active_sessions),
            failure_kind=self._failure_kind,
        )

    def start(self, actual: ServiceCompatibility) -> bool:
        if self._state not in {ServiceState.STARTING, ServiceState.STOPPED}:
            raise RuntimeError(f"cannot start service from {self._state.value}")
        self._state = ServiceState.STARTING
        self._failure_kind = None
        self._active_sessions.clear()
        if actual != self._expected:
            self._state = ServiceState.FAILED
            self._failure_kind = "compatibility_mismatch"
            return False
        self._state = ServiceState.READY
        return True

    def restart(self, actual: ServiceCompatibility) -> bool:
        if self._state not in {ServiceState.STOPPED, ServiceState.FAILED}:
            raise RuntimeError("restart requires stopped or failed service")
        return self.start(actual)

    def mark_failed(self, failure_kind: str) -> None:
        if not isinstance(failure_kind, str) or not failure_kind or failure_kind != failure_kind.strip():
            raise ValueError("failure_kind must be a non-empty normalized string")
        self._state = ServiceState.FAILED
        self._failure_kind = failure_kind
        self._active_sessions.clear()

    def session_started(self, session_id: str) -> None:
        if self._state is not ServiceState.READY:
            raise RuntimeError("new sessions require ready service")
        _validate_identity(session_id, "session_id")
        if session_id in self._active_sessions:
            raise ValueError(f"session already active: {session_id}")
        self._active_sessions.add(session_id)

    def session_finished(self, session_id: str) -> bool:
        removed = session_id in self._active_sessions
        self._active_sessions.discard(session_id)
        if self._state is ServiceState.DRAINING and not self._active_sessions:
            self._state = ServiceState.STOPPED
        return removed

    def request_shutdown(self, *, cancel_active: bool) -> tuple[str, ...]:
        if self._state is ServiceState.STOPPED:
            return ()
        if self._state is ServiceState.FAILED:
            self._state = ServiceState.STOPPED
            return ()
        self._state = ServiceState.DRAINING
        active = tuple(sorted(self._active_sessions))
        if cancel_active:
            self._active_sessions.clear()
            self._state = ServiceState.STOPPED
        elif not active:
            self._state = ServiceState.STOPPED
        return active


def _validate_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


__all__ = [
    "ServiceState",
    "ServiceCompatibility",
    "ServiceHealth",
    "ServiceLifecycle",
]
