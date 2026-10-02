"""End-to-end tests through the real ZCode request path.

`oai2/server/openai_compat_app.py` is the endpoint ZCode actually speaks:
`POST /v1/chat/completions` with OpenAI-shaped messages. These tests drive
that HTTP surface (via the app's own test client, with the model load stubbed
out) rather than calling the composer in isolation, so they cover translation,
template validation, prefix identity and response shape together.

The runtime is stubbed because loading an MLX model here would both cost
seconds and contend with the #240 owner's hardware measurements. What that
means for evidence is stated explicitly at the bottom of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from oai2.core import Status


@dataclass
class _StubHotRuntime:
    """Stands in for MLXHotRuntime: records requests, returns a canned reply."""

    script: list[dict[str, Any]] = field(default_factory=list)
    requests: list[Any] = field(default_factory=list)
    load_seconds: float | None = 0.01
    model_id: str = "stub-model"
    _loaded: bool = False

    def load(self) -> None:
        self._loaded = True

    def generate(self, request: Any) -> Any:
        from oai2.runtime.inference import InferenceResponse

        self.requests.append(request)
        step = self.script.pop(0) if self.script else {"text": "ok"}
        return InferenceResponse(
            text=step.get("text", ""),
            tokens=7,
            elapsed_ms=1.0,
            device="stub",
            status=Status.EXPERIMENTAL,
            finish_reason=step.get("finish_reason", "stop"),
            tool_calls=tuple(step.get("tool_calls", ())),
            notes=list(step.get("notes", ())),
        )

    def close(self) -> None:
        self._loaded = False


def _client(runtime: _StubHotRuntime) -> TestClient:
    """Build the real app with the model load replaced by a stub."""
    from oai2.server import openai_compat_app as mod

    real_hot = mod.MLXHotRuntime

    class _Patched(_StubHotRuntime):
        def __init__(self, spec: Any, *, model_id: str = "", **_: Any) -> None:
            super().__init__(model_id=model_id or "stub-model")
            self._shared = runtime

        def generate(self, request: Any) -> Any:
            # Delegate to the outer stub so its script and recording are used.
            return self._shared.generate(request)

    mod.MLXHotRuntime = _Patched  # type: ignore[misc]
    try:
        app = mod.create_app(model_id="stub-model", with_tools=True)
    finally:
        mod.MLXHotRuntime = real_hot  # type: ignore[misc]
    return TestClient(app)


BASE_MESSAGES = [
    {"role": "system", "content": "You are ZCode, a coding agent."},
    {"role": "user", "content": "fix the login redirect bug"},
]

TOOL_CALL = {
    "id": "call_abc",
    "type": "function",
    "function": {"name": "Read", "arguments": '{"path": "app.py"}'},
}


def _post(client: TestClient, messages: list[dict[str, Any]], **kw: Any) -> Any:
    payload = {"model": "stub-model", "messages": messages, **kw}
    return client.post("/v1/chat/completions", json=payload)


class TestZCodePathCarriesPrefixIdentity:
    def test_prefix_digest_is_returned_and_forwarded(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        resp = _post(client, BASE_MESSAGES)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prefix_digest"]
        assert rt.requests[0].prefix_digest == body["prefix_digest"]

    def test_digest_tracks_the_callers_system_prompt(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        a = _post(client, BASE_MESSAGES).json()["prefix_digest"]
        b = _post(
            client,
            [
                {"role": "system", "content": "You are ZCode, different."},
                {"role": "user", "content": "fix the login redirect bug"},
            ],
        ).json()["prefix_digest"]
        assert a != b

    def test_digest_tracks_the_advertised_tool_schema(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        a = _post(client, BASE_MESSAGES, tools=[{"type": "function", "function": {"name": "Read"}}])
        b = _post(
            client,
            BASE_MESSAGES,
            tools=[
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Bash"}},
            ],
        )
        assert a.json()["prefix_digest"] != b.json()["prefix_digest"]

    def test_turn_content_does_not_change_the_prefix_digest(self):
        """Per-turn content is not prefix; it must not force a cache miss."""
        rt = _StubHotRuntime()
        client = _client(rt)
        a = _post(client, BASE_MESSAGES).json()["prefix_digest"]
        b = _post(
            client,
            [*BASE_MESSAGES, {"role": "assistant", "content": "done"}],
        ).json()["prefix_digest"]
        assert a == b


class TestZCodePathPreservesCallerOwnedState:
    def test_messages_reach_the_runtime_unmodified(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        conv = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "file contents"},
            {"role": "user", "content": "U2"},
        ]
        _post(client, conv)
        assert rt.requests[0].messages == conv

    def test_user_goal_survives(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        _post(client, BASE_MESSAGES)
        assert rt.requests[0].messages[1]["content"] == "fix the login redirect bug"

    def test_structured_tool_calls_are_returned_unchanged(self):
        rt = _StubHotRuntime(script=[{"finish_reason": "tool_calls", "tool_calls": [TOOL_CALL]}])
        client = _client(rt)
        body = _post(
            client, BASE_MESSAGES, tools=[{"type": "function", "function": {"name": "Read"}}]
        ).json()
        call = body["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "call_abc"
        assert call["function"]["name"] == "Read"
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_no_lessons_are_injected_twice(self):
        """This endpoint is a transport; it must add no retrieved content."""
        rt = _StubHotRuntime()
        client = _client(rt)
        _post(client, BASE_MESSAGES)
        assert rt.requests[0].messages == BASE_MESSAGES
        assert not any(
            "Retrieved evidence" in str(m.get("content", "")) for m in rt.requests[0].messages
        )


class TestZCodePathTemplateSafety:
    def test_system_message_after_a_turn_is_rejected_with_400(self):
        """The llama.cpp qwen3.8 shape must fail as a 400, not a traceback."""
        rt = _StubHotRuntime()
        client = _client(rt)
        bad = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U1"},
            {"role": "system", "content": "LATE SYSTEM"},
        ]
        resp = _post(client, bad)
        assert resp.status_code == 400
        assert "follows the first non-system turn" in resp.json()["detail"]

    def test_a_valid_conversation_is_not_rejected(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        conv = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A", "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "R"},
        ]
        assert _post(client, conv).status_code == 200

    def test_developer_role_is_treated_as_part_of_the_leading_block(self):
        rt = _StubHotRuntime()
        client = _client(rt)
        conv = [
            {"role": "system", "content": "SYS"},
            {"role": "developer", "content": "DEV"},
            {"role": "user", "content": "U"},
        ]
        assert _post(client, conv).status_code == 200


class TestZCodePathHonestAccounting:
    def test_prompt_tokens_are_reported_as_unknown_not_zero(self):
        """A hardcoded 0 was a false reading of the wire."""
        rt = _StubHotRuntime()
        client = _client(rt)
        usage = _post(client, BASE_MESSAGES).json()["usage"]
        assert usage["prompt_tokens"] is None
        assert usage["completion_tokens"] == 7

    def test_completion_tokens_still_reported(self):
        rt = _StubHotRuntime(script=[{"text": "hi"}])
        client = _client(rt)
        assert _post(client, BASE_MESSAGES).json()["usage"]["completion_tokens"] == 7


# ---------------------------------------------------------------------------
# Evidence boundary
# ---------------------------------------------------------------------------
# These tests prove the *transport contract*: translation, prefix identity,
# template validation, message fidelity, and response shape, against the real
# HTTP surface. They do NOT prove real-model resistance to prompt injection,
# and they do not measure TTFT/prefill/decode — the runtime is stubbed. Those
# remain pending and are reported as such on #186.
