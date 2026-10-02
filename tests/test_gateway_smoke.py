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

import contextlib
import io
import json
import runpy
import sys
import warnings
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
SCRIPT = ROOT / "scripts" / "gateway_smoke.py"


def _runtime_with(handler) -> GatewayRuntime:
    """Build a GatewayRuntime whose httpx.Client uses the supplied handler."""
    cfg = GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="smoke-token-xyz",
        model="oai-2.0",
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


def _chat_completion(text: str, model: str = "oai-2.0") -> httpx.Response:
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
        return _chat_completion("pong", model="oai-2.0")

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
    assert payload["model"] == "oai-2.0"
    assert "smoke-token-xyz" not in captured.out
    assert "smoke-token-xyz" not in captured.err


def test_main_success_human_path_does_not_leak_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion("pong", model="oai-2.0")

    runtime = _runtime_with(handler)
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "smoke-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")

    exit_code = smoke.main([])  # no --json

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "model:        oai-2.0" in captured.out
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


# ---------------------------------------------------------------------------
# Path 6: __main__ boundary (runpy-driven)
# ---------------------------------------------------------------------------
#
# Same pattern as the slice-19 boundary test in tests/test_backend_smoke.py:
# drive the actual ``if __name__ == "__main__":`` block of the script via
# ``runpy.run_module(...)`` so a refactor that turns the boundary into a
# no-op cannot silently regress. Unlike backend_smoke.py, gateway_smoke.py's
# boundary is just ``raise SystemExit(main())`` — all exception translation
# is handled inside main(), so this test only needs to verify the success
# path (exit 0) plus a source-pin for the literal boundary string.


def test_gateway_smoke_main_boundary_exits_zero_via_runpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the actual ``__main__`` block via runpy and assert exit 0.

    The happy-path ``main()`` returns 0 (see Path 1 in this file), and the
    ``__main__`` boundary is ``raise SystemExit(main())`` — so a successful
    runpy execution should propagate ``SystemExit(0)`` and emit the
    human-path renderer output on stdout (since no ``--json`` flag is passed
    to the re-executed script).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "pong"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    cfg = GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="runpy-token-xyz",
        model="orchordsai-m3",
        timeout_seconds=5.0,
    )
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    runtime = GatewayRuntime(cfg, client=client)

    # The script's ``main()`` does ``with GatewayRuntime(config) as runtime_ctx:``.
    # The GatewayRuntime identifier in main()'s frame resolves through
    # ``oai2.runtime.GatewayRuntime`` (re-exported from ``oai2.runtime.__init__``).
    # Patching smoke_pkg.GatewayRuntime is NOT enough: ``runpy.run_module``
    # reloads the script module from disk and re-executes
    # ``from oai2.runtime import GatewayRuntime``, which re-binds to the
    # original class via ``oai2.runtime.__init__.py``. Patching the source
    # class directly (``oai2.runtime.gateway_runtime.GatewayRuntime``) is
    # also not enough — the ``__init__`` re-export already holds the
    # original class reference. The patch MUST hit the attribute on the
    # re-exporting package (``oai2.runtime.GatewayRuntime``) so the
    # re-executed ``from oai2.runtime import GatewayRuntime`` resolves to
    # our lambda. This is the same cache limitation documented in slice 19
    # for the backend_smoke failure-path runpy test.
    monkeypatch.setattr("oai2.runtime.GatewayRuntime", lambda config: runtime)
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "runpy-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "orchordsai-m3")

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code: int | None = None
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"'scripts\.gateway_smoke' found in sys\.modules",
                category=RuntimeWarning,
            )
            prior_argv = sys.argv
            sys.argv = ["gateway_smoke"]
            try:
                runpy.run_module(
                    "scripts.gateway_smoke",
                    run_name="__main__",
                    alter_sys=True,
                )
            except SystemExit as exc:
                exit_code = exc.code if isinstance(exc.code, int) else (
                    int(exc.code) if exc.code is not None else 0
                )
            else:
                exit_code = 0
            finally:
                sys.argv = prior_argv

    assert exit_code == 0, (
        f"expected exit 0 from runpy-driven __main__ boundary, got {exit_code!r}\n"
        f"stdout={stdout.getvalue()!r}\nstderr={stderr.getvalue()!r}"
    )
    captured = stdout.getvalue()
    # No --json flag, so the human-path renderer is used (see _render_human).
    # Assert on the same shape the existing Path-1 happy-path test pins:
    # model line, base_url line, status_code: 200, text: pong.
    assert "model:" in captured
    assert "orchordsai-m3" in captured
    assert "status_code:  200" in captured
    assert "text:         pong" in captured
    assert "runpy-token-xyz" not in captured


def test_gateway_smoke_main_boundary_pins_exit_translation_source() -> None:
    """Source-pin the ``__main__`` block to ``raise SystemExit(main())``.

    Unlike ``scripts/backend_smoke.py``, this script has no exception
    translation in its boundary — ``main()`` catches all expected
    exceptions and returns 0/1/2 ints. The literal
    ``raise SystemExit(main())`` form must remain so the boundary keeps
    propagating main()'s int return as the process exit code.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source
