"""Integration tests for scripts/bench.py --backend=gateway.

These tests drive ``run_one_gateway`` and ``main()`` with a mocked
``httpx`` transport so the bench harness can be exercised end-to-end
without a live endpoint. They pin:

- The ``--backend=gateway`` SKIP behaviour when no API key is set.
- Cold + warm metrics separation (warm-up run excluded from the
  measured run; recorded separately as ``warm_run_seconds``).
- ``prefill_seconds`` == ``end_to_end_seconds`` and ``decode_seconds``
  == 0 by convention for the gateway backend.
- Memory metrics are ``None`` (model runs in the cloud).
- ``summary_<tag>.json`` is written and tagged with ``backend``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

import scripts.bench as bench
from oai2.runtime import GatewayConfig, GatewayRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-bench",
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


def _runtime_with(handler) -> GatewayRuntime:
    """Build a GatewayRuntime that uses a MockTransport for ``handler``.

    The runtime is constructed with an explicit ``httpx.Client`` whose
    transport is the MockTransport, so no module-level monkeypatching is
    needed. The caller is responsible for closing the runtime.
    """
    cfg = GatewayConfig(
        base_url="https://gateway.example.test",
        api_key="bench-token-xyz",
        model="oai-1.2",
        timeout_seconds=5.0,
    )
    client = httpx.Client(
        base_url=cfg.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {cfg.api_key}"},
    )
    return GatewayRuntime(cfg, client=client)


def _patch_gateway_runtime(monkeypatch: pytest.MonkeyPatch, runtime: GatewayRuntime) -> None:
    """Make ``scripts.bench.run_one_gateway`` use the supplied ``runtime``.

    ``scripts.bench.run_one_gateway`` does
    ``from oai2.runtime import GatewayConfig, GatewayModelClient, GatewayRuntime``
    locally. Because ``oai2.runtime.__init__`` re-exports
    ``GatewayRuntime`` from ``gateway_runtime``, the symbol the local
    import binds to lives in ``oai2.runtime``'s module dict — not in
    ``gateway_runtime``. We patch ``oai2.runtime.GatewayRuntime``
    directly so the local import sees our factory.
    """
    import oai2.runtime as oai2_runtime_pkg

    def factory(config, **kwargs):
        # Match any config that points at the test environment. The
        # base_url is the unique identifier; api_key varies across tests
        # (token-leak test uses a different key to assert redaction).
        if config.base_url == "https://gateway.example.test":
            return runtime
        return GatewayRuntime(config, **kwargs)

    monkeypatch.setattr(oai2_runtime_pkg, "GatewayRuntime", factory)


# ---------------------------------------------------------------------------
# SKIP behaviour
# ---------------------------------------------------------------------------


def test_main_skips_gateway_backend_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OAI2_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_MODEL", raising=False)
    monkeypatch.delenv("OAI2_GATEWAY_TIMEOUT_SECONDS", raising=False)

    exit_code = bench.main(
        [
            "--backend",
            "gateway",
            "--repetitions",
            "1",
            "--prompt-tokens",
            "16",
            "--no-warmup",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "SKIP gateway-bench" in captured.err
    assert "OAI2_GATEWAY_API_KEY" in captured.err


# ---------------------------------------------------------------------------
# Metrics: cold/warm split + gateway conventions
# ---------------------------------------------------------------------------


def test_run_one_gateway_records_warmup_and_measured_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up is excluded from measured runs but recorded in its own field."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return _chat_completion("ok")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    prompt = "summarize this module " * 20
    metrics = bench.run_one_gateway(
        base_url="https://gateway.example.test",
        api_key="bench-token-xyz",
        model="oai-1.2",
        prompt=prompt,
        prompt_label="~16tokens (8words)",
        max_tokens=64,
        timeout_seconds=5.0,
        warm=True,
    )

    # 1 warm-up call + 1 measured call = 2 total HTTP requests.
    assert call_count["n"] == 2
    assert metrics.warm_run_seconds is not None
    assert metrics.warm_run_seconds > 0
    assert metrics.end_to_end_seconds is not None
    assert metrics.prefill_seconds == metrics.end_to_end_seconds
    assert metrics.decode_seconds == 0.0
    assert metrics.prefill_tokens_per_second is not None
    assert metrics.device == "gateway:https://gateway.example.test"
    assert metrics.model == "gateway:oai-1.2"
    assert "prefill==end_to_end" in " ".join(metrics.notes)


def test_run_one_gateway_no_warmup_makes_only_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return _chat_completion("ok")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    metrics = bench.run_one_gateway(
        base_url="https://gateway.example.test",
        api_key="bench-token-xyz",
        model="oai-1.2",
        prompt="hi",
        prompt_label="~16tokens",
        max_tokens=64,
        timeout_seconds=5.0,
        warm=False,
    )

    assert call_count["n"] == 1
    assert metrics.warm_run_seconds is None
    assert metrics.compile_seconds is None
    assert metrics.end_to_end_seconds is not None
    assert metrics.end_to_end_seconds > 0


def test_run_one_gateway_memory_metrics_are_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion("ok")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    metrics = bench.run_one_gateway(
        base_url="https://gateway.example.test",
        api_key="bench-token-xyz",
        model="oai-1.2",
        prompt="hi",
        prompt_label="~16tokens",
        max_tokens=64,
        timeout_seconds=5.0,
        warm=False,
    )

    assert metrics.peak_memory_gb is None
    assert metrics.active_memory_gb is None
    assert metrics.cache_memory_gb is None
    assert metrics.load_seconds is None


def test_run_one_gateway_propagates_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    metrics = bench.run_one_gateway(
        base_url="https://gateway.example.test",
        api_key="bench-token-xyz",
        model="oai-1.2",
        prompt="hi",
        prompt_label="~16tokens",
        max_tokens=64,
        timeout_seconds=5.0,
        warm=False,
    )

    assert metrics.end_to_end_seconds is None
    assert metrics.prefill_seconds is None
    assert metrics.decode_seconds is None
    assert metrics.generation_tokens is None
    assert any("generate-failed" in note for note in metrics.notes)
    # The token MUST NOT appear in any note.
    assert not any("bench-token-xyz" in note for note in metrics.notes)


def test_run_one_gateway_api_key_does_not_leak_into_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion("ok")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    metrics = bench.run_one_gateway(
        base_url="https://gateway.example.test",
        api_key="decoy-token-xyz",
        model="oai-1.2",
        prompt="hi",
        prompt_label="~16tokens",
        max_tokens=64,
        timeout_seconds=5.0,
        warm=True,
    )

    serialized = json.dumps(asdict(metrics), default=str)
    assert "decoy-token-xyz" not in serialized
    assert metrics.output_excerpt == "ok"


# ---------------------------------------------------------------------------
# End-to-end main() with mocked transport
# ---------------------------------------------------------------------------


def test_main_gateway_backend_writes_summary_with_backend_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion("ok")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    # Use the runtime's already-built config so main()'s load-from-env
    # sees the matching token + base URL.
    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", runtime._config.api_key)
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", runtime._config.base_url)

    exit_code = bench.main(
        [
            "--backend",
            "gateway",
            "--model",
            "oai-1.2",
            "--repetitions",
            "2",
            "--prompt-tokens",
            "16",
            "--no-warmup",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    summary_files = list(tmp_path.glob("summary_*.json"))
    assert len(summary_files) == 1
    summary = json.loads(summary_files[0].read_text())
    assert summary["backend"] == "gateway"
    assert summary["model"] == "oai-1.2"
    assert summary["repetitions"] == 2
    assert len(summary["configs"]) == 1
    config = summary["configs"][0]
    assert len(config["runs"]) == 2
    for run in config["runs"]:
        assert run["device"] == "gateway:https://gateway.example.test"
        assert run["prefill_seconds"] == run["end_to_end_seconds"]
        assert run["decode_seconds"] == 0.0
        assert run["peak_memory_gb"] is None
        assert run["load_seconds"] is None
        assert run["warm_run_seconds"] is None  # --no-warmup

    # Per-run json files exist
    run_files = list(tmp_path.glob("bench_*.json"))
    assert len(run_files) == 2
