from __future__ import annotations

import pytest

from oai2.runtime.service_security import AccessPolicy, IsolatedSessionRegistry


def test_loopback_may_run_without_auth_token() -> None:
    policy = AccessPolicy(bind_host="127.0.0.1")
    assert policy.is_loopback is True
    assert policy.authorize(None) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "service.internal"])
def test_non_loopback_requires_explicit_auth(host: str) -> None:
    with pytest.raises(ValueError, match="requires bearer_token"):
        AccessPolicy(bind_host=host)


def test_bearer_auth_and_public_summary_do_not_expose_secret() -> None:
    policy = AccessPolicy(bind_host="0.0.0.0", bearer_token="super-secret-token")

    assert policy.authorize("super-secret-token") is True
    assert policy.authorize("wrong") is False
    assert policy.authorize(None) is False
    summary = policy.public_summary()
    assert summary["auth_required"] is True
    assert "bearer_token" not in summary
    assert "super-secret-token" not in repr(summary)


def test_clients_cannot_access_each_others_sessions() -> None:
    sessions = IsolatedSessionRegistry()
    sessions.create(client_id="client-a", session_id="session-a")
    sessions.create(client_id="client-b", session_id="session-b")

    assert sessions.require_owned(client_id="client-a", session_id="session-a").session_id == "session-a"
    with pytest.raises(PermissionError, match="session unavailable"):
        sessions.require_owned(client_id="client-a", session_id="session-b")
    with pytest.raises(PermissionError, match="session unavailable"):
        sessions.require_owned(client_id="client-b", session_id="session-a")


def test_session_limit_is_per_client() -> None:
    sessions = IsolatedSessionRegistry(max_sessions_per_client=1)
    sessions.create(client_id="client-a", session_id="a1")
    sessions.create(client_id="client-b", session_id="b1")

    with pytest.raises(RuntimeError, match="session limit"):
        sessions.create(client_id="client-a", session_id="a2")
    assert sessions.count_for_client("client-a") == 1
    assert sessions.count_for_client("client-b") == 1


def test_cross_client_remove_fails_closed() -> None:
    sessions = IsolatedSessionRegistry()
    sessions.create(client_id="owner", session_id="s1")

    assert sessions.remove(client_id="other", session_id="s1") is False
    assert sessions.require_owned(client_id="owner", session_id="s1").session_id == "s1"
    assert sessions.remove(client_id="owner", session_id="s1") is True
