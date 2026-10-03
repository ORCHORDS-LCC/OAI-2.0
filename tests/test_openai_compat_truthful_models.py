"""Tests for the truthful model-discovery contract (WI-HOST-003 / #258).

The defect these cover is a user-visible one: a client fetched the model
list, found no limits, and filled in its own. The UI then showed a context
window and output ceiling the server had never agreed to. The fix is for
the server to publish what it enforces, and to say "unknown" rather than a
plausible number when it does not know.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from oai2.server.openai_compat_app import MAX_OUTPUT_TOKENS, create_app


class _StubRuntime:
    """Stands in for the hot runtime so no model is downloaded.

    The discovery contract under test is a property of the HTTP surface, not
    of any model, and `create_app` loads eagerly — so a real load here would
    make the tests depend on a Hub fetch.
    """

    def __init__(self, spec: Any, **_: Any) -> None:
        self.spec = spec
        self.prefix_cache = None
        self.gate_digest = False
        self.load_seconds = 0.0

    def generate(self, request: Any) -> Any:
        from oai2.core import Status
        from oai2.runtime.inference import InferenceResponse

        return InferenceResponse(
            text="ok",
            tokens=1,
            elapsed_ms=0.0,
            device="stub",
            status=Status.EXPERIMENTAL,
            finish_reason="stop",
        )

    def load(self) -> None:
        """`create_app` loads eagerly; the stub is already 'loaded'."""
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the eager model load with a stub for the whole module."""
    from oai2.server import openai_compat_app as mod

    monkeypatch.setattr(mod, "MLXHotRuntime", _StubRuntime, raising=False)


def _client(**kw) -> TestClient:
    return TestClient(create_app(model_id="oai-2.0", **kw))


def test_model_listing_publishes_the_enforced_output_ceiling(stub_runtime: None) -> None:
    client = _client()
    caps = client.post("/v1/models").json()["data"][0]["capabilities"]
    # This must be the number the server actually rejects above, not a
    # separate advertised figure.
    assert caps["max_output_tokens"] == MAX_OUTPUT_TOKENS


def test_unknown_context_window_is_reported_as_unknown_not_invented(stub_runtime: None) -> None:
    """The fabrication this prevents: a plausible default presented as fact."""
    client = _client()
    caps = client.post("/v1/models").json()["data"][0]["capabilities"]
    assert caps["context_window"] is None
    assert caps["context_window_known"] is False


def test_declared_context_window_is_published_and_marked_known(stub_runtime: None) -> None:
    client = _client(context_window=8192)
    caps = client.post("/v1/models").json()["data"][0]["capabilities"]
    assert caps["context_window"] == 8192
    assert caps["context_window_known"] is True


def test_tool_capability_reflects_real_wiring(stub_runtime: None) -> None:
    """Tools are dropped when disabled, so claiming support would be false."""
    assert (
        _client(with_tools=True).post("/v1/models").json()["data"][0]["capabilities"]["tools"]
        is True
    )
    assert (
        _client(with_tools=False).post("/v1/models").json()["data"][0]["capabilities"]["tools"]
        is False
    )


def test_vision_is_declared_absent_rather_than_assumed(stub_runtime: None) -> None:
    """No verified vision path exists, so the honest value is False."""
    caps = _client().post("/v1/models").json()["data"][0]["capabilities"]
    assert caps["vision"] is False


def test_published_output_ceiling_is_actually_enforced(stub_runtime: None) -> None:
    """A published limit the server will not reject is worse than none."""
    body = {
        "model": "oai-2.0",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": MAX_OUTPUT_TOKENS + 1,
    }
    response = _client().post("/v1/chat/completions", json=body)
    assert response.status_code == 422  # pydantic bound, matching the advertised value


def test_declared_context_window_is_enforced_not_merely_advertised(stub_runtime: None) -> None:
    client = _client(context_window=4096)
    body = {
        "model": "oai-2.0",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4096,
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert "4096" in response.json()["detail"]


def test_no_context_declared_means_no_output_side_refusal(stub_runtime: None) -> None:
    """Behaviour is unchanged when the operator has declared no window.

    Defaulting to an invented limit would reject requests the backend can
    in fact serve, which is a regression dressed up as honesty.
    """
    client = _client()
    body = {
        "model": "oai-2.0",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4096,
    }
    assert client.post("/v1/chat/completions", json=body).status_code != 400


def test_model_identity_is_preserved(stub_runtime: None) -> None:
    """#237 pins a single `oai-2.0` contract; adding limits must not change it."""
    entry = _client().post("/v1/models").json()["data"][0]
    assert entry["id"] == "oai-2.0"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "oai2-local"


@pytest.mark.parametrize("field", ["max_output_tokens", "context_window", "tools", "vision"])
def test_every_capability_field_is_present_even_when_unknown(
    field: str, stub_runtime: None
) -> None:
    """Absent-vs-unknown must be a value, never a missing key.

    A client that finds a missing key fills it in. A client that finds
    `null` has to treat it as unknown.
    """
    caps = _client().post("/v1/models").json()["data"][0]["capabilities"]
    assert field in caps
