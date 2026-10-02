"""Unit tests for the public-gateway runtime.

All tests use ``httpx.MockTransport`` so no live HTTP call is ever
made — the runner-free ``scripts/verify.py`` cycle covers the
runtime end-to-end without leaving the host.
"""

from __future__ import annotations

import json

import httpx
import pytest

from oai2.core import Status
from oai2.runtime import (
    DEFAULT_GATEWAY_BASE_URL,
    DEFAULT_GATEWAY_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayConfig,
    GatewayConfigError,
    GatewayRuntime,
    GatewayRuntimeError,
    InferenceRequest,
    PlaceholderRuntime,
    load_gateway_config_from_env,
)


def _config(**overrides) -> GatewayConfig:
    payload = {
        "base_url": "https://gateway.example.test",
        "api_key": "test-token-xyz",
        "model": "oai-2.0",
        "timeout_seconds": 5.0,
    }
    payload.update(overrides)
    return GatewayConfig(**payload)


def test_gateway_config_repr_redacts_api_key() -> None:
    cfg = _config()
    rendered = repr(cfg)
    assert "test-token-xyz" not in rendered
    assert "<redacted" in rendered


def test_load_gateway_config_from_env_returns_none_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_MODEL", raising=False)
    assert load_gateway_config_from_env() is None


def test_load_gateway_config_from_env_uses_defaults_when_only_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.delenv("OAI2_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_MODEL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_TIMEOUT_SECONDS", raising=False)
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.base_url == DEFAULT_GATEWAY_BASE_URL
    assert cfg.model == DEFAULT_GATEWAY_MODEL
    assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert cfg.api_key == "k"


def test_load_gateway_config_from_env_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://api.orchords.com/")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-2.0")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.base_url == "https://api.orchords.com"


def test_load_gateway_config_from_env_rejects_bad_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("OAI2_GATEWAY_TIMEOUT_SECONDS", "not-a-number")
    cfg = load_gateway_config_from_env()
    assert cfg is not None
    assert cfg.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_gateway_runtime_requires_api_key() -> None:
    with pytest.raises(GatewayConfigError):
        GatewayRuntime(GatewayConfig(base_url="x", api_key="", model="oai-2.0"))


def test_gateway_runtime_from_env_raises_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    with pytest.raises(GatewayConfigError):
        GatewayRuntime.from_env()


def test_gateway_runtime_sends_chat_completion_request() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "oai-2.0",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello back"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
            },
        )

    transport = httpx.MockTransport(handler)
    cfg = _config()
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=transport,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    runtime = GatewayRuntime(cfg, client=client)
    try:
        response = runtime.generate(
            InferenceRequest(
                prompt="ping",
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                seed=42,
                stop=("\n\n<",),
            )
        )
    finally:
        runtime.close()

    assert captured["path"] == "/v1/chat/completions"
    assert captured["auth"] == "Bearer test-token-xyz"
    body = captured["body"]
    assert body["model"] == "oai-2.0"
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["seed"] == 42
    assert body["stop"] == ["\n\n<"]
    assert response.text == "hello back"
    assert response.tokens == 4
    assert response.status is Status.EXPERIMENTAL
    assert response.device.startswith("gateway:")
    assert response.elapsed_ms >= 0
    assert any(n.startswith("model=") for n in response.notes)


def test_gateway_runtime_extracts_content_from_message_list_payload() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "part-one "},
                                {"type": "text", "text": "part-two"},
                            ],
                        },
                    }
                ]
            },
        )

    runtime = GatewayRuntime(
        _config(),
        client=httpx.Client(
            base_url="https://gateway.example.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        response = runtime.generate(InferenceRequest(prompt="anything"))
    finally:
        runtime.close()
    assert response.text == "part-one part-two"
    assert response.tokens >= 1


def test_gateway_runtime_raises_on_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "missing bearer token", "type": "auth"}},
        )

    runtime = GatewayRuntime(
        _config(),
        client=httpx.Client(
            base_url="https://gateway.example.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert exc_info.value.status_code == 401
    assert "test-token-xyz" not in str(exc_info.value)


def test_gateway_runtime_raises_on_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection-refused-by-mock-transport")

    runtime = GatewayRuntime(
        _config(),
        client=httpx.Client(
            base_url="https://gateway.example.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        with pytest.raises(GatewayRuntimeError) as exc_info:
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert exc_info.value.status_code == 0
    # The transport error is wrapped, the underlying message is preserved
    # at least partially through the redaction filter, and the raw token
    # never leaks into the user-facing string.
    rendered = str(exc_info.value)
    assert "ConnectError" in rendered
    assert "test-token-xyz" not in rendered


def test_gateway_runtime_raises_on_malformed_payload() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    runtime = GatewayRuntime(
        _config(),
        client=httpx.Client(
            base_url="https://gateway.example.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        with pytest.raises(GatewayRuntimeError):
            runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()


def test_placeholder_runtime_remains_default_when_gateway_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity-check: leaving the gateway key unset keeps PlaceholderRuntime usable."""

    from oai2.runtime import default_runtime

    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    rt = default_runtime()
    assert isinstance(rt, PlaceholderRuntime)


def test_gateway_runtime_uses_request_model_when_provided() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ]
            },
        )

    cfg = _config(model="oai-2.0")
    runtime = GatewayRuntime(
        cfg,
        client=httpx.Client(
            base_url=cfg.base_url,
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        from oai2.runtime import ModelSpec

        response = runtime.generate(
            InferenceRequest(
                prompt="hi",
                model=ModelSpec(name="oai-1.0"),
            )
        )
    finally:
        runtime.close()
    assert captured["body"]["model"] == "oai-1.0"
    assert response.text == "ok"


def test_gateway_runtime_falls_back_to_config_model() -> None:
    """Without a request-supplied model, the configured model id is used."""

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ]
            },
        )

    cfg = _config(model="oai-2.0")
    runtime = GatewayRuntime(
        cfg,
        client=httpx.Client(
            base_url=cfg.base_url,
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        response = runtime.generate(InferenceRequest(prompt="hi"))
    finally:
        runtime.close()
    assert captured["body"]["model"] == "oai-2.0"
    assert response.text == "ok"
