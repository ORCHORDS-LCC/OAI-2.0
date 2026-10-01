"""Keep upstream completion outcomes intact through the gateway adapter (#238)."""

from typing import Any

import httpx
import pytest

from oai2.runtime.gateway_model_client import GatewayModelClient
from oai2.runtime.gateway_runtime import GatewayConfig, GatewayRuntime, GatewayRuntimeError
from oai2.runtime.inference import InferenceRequest, InferenceResponse


def _payload(reason: str | None, content: str | None = "partial") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": reason}],
        "usage": {"completion_tokens": 7},
    }


def _run(response: httpx.Response, entry: str = "adapter") -> Any:
    config = GatewayConfig("https://gateway.example.test", "test-token-xyz", "test-model")
    with httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(lambda request: response),
    ) as http:
        runtime = GatewayRuntime(config, client=http)
        if entry == "runtime":
            return runtime.generate(InferenceRequest(prompt="hello"))
        return GatewayModelClient(runtime).chat([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
@pytest.mark.parametrize("reason", ["stop", "length", "content_filter", "provider_specific", None])
def test_finish_reason_is_preserved_not_invented(entry: str, reason: str | None) -> None:
    reply = _run(httpx.Response(200, json=_payload(reason)), entry)
    assert reply.finish_reason == reason


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
def test_missing_finish_reason_stays_unknown(entry: str) -> None:
    payload = _payload(None)
    del payload["choices"][0]["finish_reason"]
    assert _run(httpx.Response(200, json=payload), entry).finish_reason is None


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
def test_no_final_answer_at_token_limit_is_not_reported_as_stop(entry: str) -> None:
    reply = _run(httpx.Response(200, json=_payload("length", None)), entry)
    assert reply.finish_reason == "length"
    assert getattr(reply, "text", getattr(reply, "content", None)) == ""
    if entry == "runtime":
        assert reply.tokens == 7


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
def test_tool_only_reply_preserves_call_ids_names_and_argument_strings(entry: str) -> None:
    calls = [
        {"id": "call_read", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"src/app.py"}'}},
        {"id": "call_test", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}},
    ]
    payload = _payload("tool_calls", None)
    payload["choices"][0]["message"]["tool_calls"] = calls
    reply = _run(httpx.Response(200, json=payload), entry)
    assert reply.finish_reason == "tool_calls"
    assert reply.tool_calls == tuple(calls)
    assert getattr(reply, "text", getattr(reply, "content", None)) == ""


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
def test_absent_tools_do_not_create_tool_calls(entry: str) -> None:
    assert _run(httpx.Response(200, json=_payload("stop")), entry).tool_calls == ()


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
@pytest.mark.parametrize(
    "payload",
    [
        [], None, {"choices": []}, {"choices": [None]}, {"choices": [{}]},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": True}]},
        {"choices": [{"message": {"content": None, "tool_calls": "bad"}, "finish_reason": "tool_calls"}]},
        {"choices": [{"message": {"content": None, "tool_calls": [1]}, "finish_reason": "tool_calls"}]},
        {"choices": [{"message": {"content": None, "tool_calls": []}, "finish_reason": "tool_calls"}]},
    ],
)
def test_malformed_success_envelopes_raise_gateway_error(entry: str, payload: Any) -> None:
    # Passing raw 'null' avoids HTTPX's json=None meaning 'no JSON argument'.
    response = httpx.Response(200, content="null") if payload is None else httpx.Response(200, json=payload)
    with pytest.raises(GatewayRuntimeError):
        _run(response, entry)


@pytest.mark.parametrize("entry", ["runtime", "adapter"])
def test_invalid_json_uses_gateway_error_contract(entry: str) -> None:
    with pytest.raises(GatewayRuntimeError, match="JSON"):
        _run(httpx.Response(200, text="not-json"), entry)


def test_non_gateway_response_defaults_do_not_assert_completion() -> None:
    response = InferenceResponse("placeholder", 1, 0.0, "offline")
    assert response.finish_reason is None
    assert response.tool_calls == ()
