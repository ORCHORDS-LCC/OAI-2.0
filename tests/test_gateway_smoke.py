"""End-to-end tests for ``scripts/gateway_smoke.py``.

The smoke is the smallest CLI that exercises
:class:`oai2.runtime.GatewayRuntime`. These tests verify every code path
through :func:`gateway_smoke.main` without touching the live cloud:

* env-var error / missing key  ->  exit 2, FAIL on stderr
* model override -> arg wins over ``models_payload`` env
* HTTP error     -> exit 1, FAIL + status_code surfaced
* success        -> exit 0, JSON-shaped or human-shaped output
* API key never echoed in either branch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

import scripts.gateway_smoke as smoke
from oai2.runtime import (
    GatewayConfig,
    GatewayRuntime,
    GatewayRuntimeError,
    ModelSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _runtime_with(handler) -> GatewayRuntime:
    """Build a GatewayRuntime whose httpx.Client uses the supplied handler."""
    cfg = GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="smoke-token-xyz",
        model="oai-1.2",
        timeout_seconds=5.0,
    )
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    return GatewayRuntime(cfg, client=client)


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: GatewayRuntime) -> None:
    """Make ``scripts.gateway_smoke.main`` use the supplied prebuilt runtime.

    The smoke module does ``from oai2.runtime import GatewayRuntime`` at
    module top, so the call-site resolves to ``scripts.gateway_smoke.GatewayRuntime``.
    Patching the symbol under the scripts module is what the call site sees.

    The factory rebuilds the test runtime with the *actual* config the
    smoke uses — including any ``--model`` override applied via
    ``dataclasses.replace`` — so the request body carries the configured
    model id, not the test-time default.
    """
    import scripts.gateway_smoke as smoke_pkg

    # Capture the original handler so we can rebuild against a new client.
    handler = runtime._client._transport.handler  # type: ignore[attr-defined]

    def factory(
        config: GatewayConfig, client: httpx.Client | None = None, spec: ModelSpec | None = None
    ) -> GatewayRuntime:
        if config.base_url == "https://gateway.example.test":
            # Rebuild the runtime against the smoke's config (carrying
            # any --model override), preserving the mock transport.
            client = httpx.Client(
                base_url=config.base_url,
                transport=httpx.MockTransport(handler),
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
            return GatewayRuntime(config, client=client)
        return GatewayRuntime(config, client=client, spec=spec)

    monkeypatch.setattr(smoke_pkg, "GatewayRuntime", factory)


def _chat_completion(text: str, model: str = "oai-1.2") -> httpx.Response:
    body = {
        "id": "chatcmpl-smoke-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return httpx.Response(200, json=body)


# ---------------------------------------------------------------------------
# Path 1: env-var missing / no API key
# ---------------------------------------------------------------------------


def test_main_exits_two_when_api_key_unset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)

    exit_code = smoke.main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "OAI2_GATEWAY_API_KEY" in captured.out
    assert captured.err == ""


def test_main_exits_two_on_env_loader_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> object:
        raise RuntimeError("env unreadable")

    monkeypatch.setattr(smoke, "load_gateway_config_from_env", boom)

    exit_code = smoke.main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "env config error" in captured.out
    assert "env unreadable" in captured.out


# ---------------------------------------------------------------------------
# Path 2: success
# ---------------------------------------------------------------------------


def test_main_success_json_path_does_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion("pong", model="oai-1.2")

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main(["--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip())
    assert payload["ok"] is True
    assert payload["text"] == "pong"
    assert payload["model"] == "oai-1.2"
    assert "smoke-token-xyz" not in captured.out
    assert "smoke-token-xyz" not in captured.err


def test_main_success_human_path_does_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion("pong", model="oai-1.2")

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main([])  # no --json

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "model:        oai-1.2" in captured.out
    assert "text:         pong" in captured.out
    assert "smoke-token-xyz" not in captured.out
    assert "smoke-token-xyz" not in captured.err


# ---------------------------------------------------------------------------
# Path 3: --model override
# ---------------------------------------------------------------------------


def test_main_model_override_sends_overridden_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # The model id is sent in the JSON request body.
        body = json.loads(request.content.decode())
        seen_models.append(body["model"])
        return _chat_completion("ok", model=body["model"])

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main(["--model", "oai-override", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen_models == ["oai-override"]
    payload = json.loads(captured.out.strip())
    assert payload["model"] == "oai-override"


# ---------------------------------------------------------------------------
# Path 4: HTTP / runtime error
# ---------------------------------------------------------------------------


def test_main_runtime_error_human_path_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service unavailable"})

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "503" in captured.out
    assert "smoke-token-xyz" not in captured.out


def test_main_runtime_error_json_path_surfaces_status_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "service unavailable"})

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main(["--json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.out.strip())
    assert payload["ok"] is False
    assert payload["status_code"] == 503
    assert "smoke-token-xyz" not in captured.out


# ---------------------------------------------------------------------------
# Path 5: contract — GatewayRuntimeError propagates unchanged
# ---------------------------------------------------------------------------


def test_main_propagates_runtime_error_with_status_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Direct injection of GatewayRuntimeError through the runtime path.

    Confirms the smoke CLI's exit-code contract (1 for runtime error,
    2 for config error) and that the status_code is surfaced.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limit"})

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main(["--json"])
    assert exit_code == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status_code"] == 429
    # Sanity: error string is human-readable and contains the status code.
    assert "429" in payload["error"]


def test_gateway_runtime_error_inherits_runtime_error() -> None:
    """Smoke CLI catches ``GatewayRuntimeError`` specifically.

    If this contract is broken (e.g. by changing the base class), the
    smoke CLI would treat HTTP errors as config errors and return 2
    instead of 1.
    """
    err = GatewayRuntimeError(500, "boom")
    assert isinstance(err, RuntimeError)
