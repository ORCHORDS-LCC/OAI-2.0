"""Local service access policy and multi-client session isolation."""

from __future__ import annotations

import hmac
import ipaddress
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class AccessPolicy:
    bind_host: str
    bearer_token: str | None = None
    max_sessions_per_client: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.bind_host, str) or not self.bind_host.strip():
            raise ValueError("bind_host must be a non-empty string")
        if self.bearer_token is not None and not self.bearer_token:
            raise ValueError("bearer_token cannot be empty")
        if (
            isinstance(self.max_sessions_per_client, bool)
            or not isinstance(self.max_sessions_per_client, int)
            or self.max_sessions_per_client <= 0
        ):
            raise ValueError("max_sessions_per_client must be a positive integer")
        if not self.is_loopback and self.bearer_token is None:
            raise ValueError("non-loopback exposure requires bearer_token")

    @property
    def is_loopback(self) -> bool:
        host = self.bind_host.strip().strip("[]")
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def authorize(self, presented_token: str | None) -> bool:
        if self.bearer_token is None:
            return self.is_loopback
        if presented_token is None:
            return False
        return hmac.compare_digest(self.bearer_token, presented_token)

    def public_summary(self) -> dict[str, object]:
        return {
            "bind_host": self.bind_host,
            "loopback": self.is_loopback,
            "auth_required": self.bearer_token is not None,
            "max_sessions_per_client": self.max_sessions_per_client,
        }


@dataclass(slots=True, frozen=True)
class SessionRecord:
    client_id: str
    session_id: str


class IsolatedSessionRegistry:
    """Own sessions by client identity and fail closed on cross-client access."""

    def __init__(self, *, max_sessions_per_client: int = 4) -> None:
        if (
            isinstance(max_sessions_per_client, bool)
            or not isinstance(max_sessions_per_client, int)
            or max_sessions_per_client <= 0
        ):
            raise ValueError("max_sessions_per_client must be a positive integer")
        self._max_sessions_per_client = max_sessions_per_client
        self._sessions: dict[str, SessionRecord] = {}

    def create(self, *, client_id: str, session_id: str) -> SessionRecord:
        _identity(client_id, "client_id")
        _identity(session_id, "session_id")
        if session_id in self._sessions:
            raise ValueError("session_id already exists")
        owned = sum(1 for item in self._sessions.values() if item.client_id == client_id)
        if owned >= self._max_sessions_per_client:
            raise RuntimeError("client session limit reached")
        record = SessionRecord(client_id=client_id, session_id=session_id)
        self._sessions[session_id] = record
        return record

    def require_owned(self, *, client_id: str, session_id: str) -> SessionRecord:
        _identity(client_id, "client_id")
        _identity(session_id, "session_id")
        record = self._sessions.get(session_id)
        if record is None or record.client_id != client_id:
            raise PermissionError("session unavailable")
        return record

    def remove(self, *, client_id: str, session_id: str) -> bool:
        try:
            self.require_owned(client_id=client_id, session_id=session_id)
        except PermissionError:
            return False
        del self._sessions[session_id]
        return True

    def count_for_client(self, client_id: str) -> int:
        _identity(client_id, "client_id")
        return sum(1 for item in self._sessions.values() if item.client_id == client_id)


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty normalized string")
    return value


__all__ = [
    "AccessPolicy",
    "SessionRecord",
    "IsolatedSessionRegistry",
]
