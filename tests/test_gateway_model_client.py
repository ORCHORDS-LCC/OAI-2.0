"""Unit tests for the GatewayModelClient adapter.

These tests cover the q-pipe ``ModelClient`` seam contract:
``client.chat(messages, *, max_tokens, temperature) -> reply``. All
HTTP is mocked via ``httpx.MockTransport`` so the runner-free
``scripts/verify.py`` cycle exercises the adapter without a live
endpoint.
"""

from __future__ import annotations

import json

import httpx
import pytest

from oai2.runtime import (
    ChatReply,
    GatewayConfig,
    GatewayModelClient,
    GatewayRuntime,
    GatewayRuntimeError,
)


def _config() -> GatewayConfig:
    return GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="test-token-xyz",
        model="oai-1.2",
        timeout_seconds=5.0,
    )


def _runtime_with(handler) -> GatewayRuntime:
    cfg = _config()
    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=transport,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    return GatewayRuntime(cfg, client=client)


def _chat_completion_response(content: str = "ok") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "model": "oai-1.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    )


# ---------------------------------------------------------------------------
# Structural / q-pipe compatibility
# ---------------------------------------------------------------------------


def test_chat_reply_dataclass_exposes_qpipe_compatible_fields() -> None:
    reply = ChatReply(content="hi")
    assert reply.content == "hi"
    assert reply.finish_reason == "stop"
    assert reply.reasoning == ""
    assert reply.tool_calls == ()
    assert reply.raw is None


def test_gateway_model_client_is_a_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    with GatewayModelClient(runtime) as client:
        reply = client.chat([{"role": "user", "content": "hello"}])
    assert reply.content == "ok"


# ---------------------------------------------------------------------------
# Message flattening
# ---------------------------------------------------------------------------


def test_chat_flattens_single_user_message_into_prompt() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat([{"role": "user", "content": "ping"}])
    finally:
        runtime.close()

    body = captured["body"]
    assert body["messages"] == [{"role": "user", "content": "user: ping\n"}]


def test_chat_flattens_multi_message_transcript_with_role_tags() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "bye"},
            ]
        )
    finally:
        runtime.close()

    prompt = captured["body"]["messages"][0]["content"]
    assert prompt == (
        "system: be terse\n"
        "user: hello\n"
        "assistant: hi\n"
        "user: bye\n"
    )


def test_chat_skips_empty_message_content() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat(
            [
                {"role": "user", "content": "real"},
                {"role": "user", "content": ""},
                {"role": "user", "content": None},
            ]
        )
    finally:
        runtime.close()

    prompt = captured["body"]["messages"][0]["content"]
    assert prompt == "user: real\n"


def test_chat_with_no_messages_falls_back_to_user_marker() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat([])
    finally:
        runtime.close()

    prompt = captured["body"]["messages"][0]["content"]
    assert prompt == "user: \n"


# ---------------------------------------------------------------------------
# Parameter forwarding
# ---------------------------------------------------------------------------


def test_chat_forwards_max_tokens_and_temperature() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat(
            [{"role": "user", "content": "x"}],
            max_tokens=77,
            temperature=0.25,
        )
    finally:
        runtime.close()

    body = captured["body"]
    assert body["max_tokens"] == 77
    assert body["temperature"] == pytest.approx(0.25)


def test_chat_uses_defaults_when_params_omitted() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        client.chat([{"role": "user", "content": "x"}])
    finally:
        runtime.close()

    body = captured["body"]
    assert body["max_tokens"] == 256  # DEFAULT_MAX_TOKENS
    assert body["temperature"] == pytest.approx(0.7)  # DEFAULT_TEMPERATURE


# ---------------------------------------------------------------------------
# Reply construction
# ---------------------------------------------------------------------------


def test_chat_returns_chat_reply_with_runtime_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response("reply-text")

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        reply = client.chat([{"role": "user", "content": "x"}])
    finally:
        runtime.close()

    assert isinstance(reply, ChatReply)
    assert reply.content == "reply-text"
    assert reply.finish_reason == "stop"
    assert reply.raw is not None
    assert reply.raw["device"] == "gateway:https://gateway.example.test"
    assert reply.raw["status"] == "EXPERIMENTAL"


def test_chat_extracts_content_from_message_list_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "from-list-payload"},
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        reply = client.chat([{"role": "user", "content": "x"}])
    finally:
        runtime.close()

    assert reply.content == "from-list-payload"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_chat_propagates_gateway_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        with pytest.raises(GatewayRuntimeError) as excinfo:
            client.chat([{"role": "user", "content": "x"}])
        assert excinfo.value.status_code == 503
        assert "test-token-xyz" not in str(excinfo.value)
    finally:
        runtime.close()


def test_chat_propagates_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection-refused-by-mock-transport")

    runtime = _runtime_with(handler)
    try:
        client = GatewayModelClient(runtime)
        with pytest.raises(GatewayRuntimeError) as excinfo:
            client.chat([{"role": "user", "content": "x"}])
        assert excinfo.value.status_code == 0
        assert "test-token-xyz" not in str(excinfo.value)
    finally:
        runtime.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_is_safe_to_call_twice() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response()

    runtime = _runtime_with(handler)
    client = GatewayModelClient(runtime)
    client.close()
    client.close()  # should not raise


def test_close_closes_underlying_runtime_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the adapter must close the runtime's owned httpx client.

    GatewayRuntime.close() only closes the client when the runtime owns
    it (i.e. it was constructed without an injected client). This test
    exercises that owned-client path so we can assert the close
    propagated end-to-end. The test monkeypatches ``httpx.Client`` so
    the owned client picks up a mock transport — we still get a fully
    owned runtime without making a real HTTP call.
    """

    cfg = _config()
    transport = httpx.MockTransport(
        lambda request: _chat_completion_response()
    )
    captured: dict = {}
    real_client = httpx.Client

    def fake_client(**kwargs):
        captured["kwargs"] = kwargs
        # Strip base_url so the MockTransport intercepts the call.
        kwargs.pop("base_url", None)
        kwargs["transport"] = transport
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)
    runtime = GatewayRuntime(cfg)
    inner = runtime._client  # type: ignore[attr-defined]
    assert not inner.is_closed
    GatewayModelClient(runtime).close()
    assert inner.is_closed
