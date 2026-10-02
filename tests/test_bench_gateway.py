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

import contextlib
import io
import json
import runpy
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

import scripts.bench as bench
from oai2.runtime import GatewayConfig, GatewayRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench.py"


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


# ---------------------------------------------------------------------------
# __main__ boundary (runpy-driven)
# ---------------------------------------------------------------------------
#
# Same pattern as slices 19 and 21: drive the actual
# ``if __name__ == "__main__":`` block of scripts/bench.py via
# ``runpy.run_module(...)`` so a refactor that turns the boundary into a
# no-op cannot silently regress. The bench CLI's boundary is just
# ``raise SystemExit(main())`` — same shape as scripts/gateway_smoke.py.
#
# The patch chain reuses the same ``oai2.runtime.GatewayRuntime`` factory
# from ``_patch_gateway_runtime`` above, which is what survives the
# re-executed from-import chain in ``scripts.bench.run_one_gateway``.
# The script writes per-run and summary JSON artifacts to ``--out-dir``;
# we redirect that to a tmp dir so the test doesn't pollute the real
# ``evals/benchmarks/`` directory.


def test_bench_main_boundary_exits_zero_via_runpy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Drive the actual ``__main__`` block via runpy and assert exit 0.

    The happy-path ``main()`` returns 0 once the bench loop finishes and
    the summary file is written. The ``__main__`` boundary is
    ``raise SystemExit(main())`` — so a successful runpy execution
    should propagate ``SystemExit(0)`` and emit the bench progress +
    ``wrote summary: ...`` line on stderr.
    """
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return _chat_completion("ok")

    runtime = _runtime_with(handler)
    _patch_gateway_runtime(monkeypatch, runtime)

    monkeypatch.setenv("OAI2_GATEWAY_API_KEY", "runpy-bench-token-xyz")
    monkeypatch.setenv("OAI2_GATEWAY_BASE_URL", "https://gateway.example.test")
    monkeypatch.setenv("OAI2_GATEWAY_MODEL", "oai-1.2")

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code: int | None = None

    prior_argv = sys.argv
    sys.argv = [
        "bench",
        "--backend",
        "gateway",
        "--model",
        "oai-1.2",
        "--repetitions",
        "1",
        "--prompt-tokens",
        "16",
        "--no-warmup",
        "--out-dir",
        str(tmp_path),
        "--tag",
        "runpy-boundary",
    ]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"'scripts\.bench' found in sys\.modules",
                    category=RuntimeWarning,
                )
                try:
                    runpy.run_module(
                        "scripts.bench",
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
    err = stderr.getvalue()
    assert "benchmark: backend=gateway model=oai-1.2" in err
    assert "wrote summary:" in err

    # Per-run + summary artifacts were written to tmp_path, not the real
    # evals/benchmarks/ directory. The script tags with
    # 'runpy-boundary' because we passed --tag explicitly.
    run_files = sorted(tmp_path.glob("bench_*.json"))
    assert len(run_files) == 1
    summary_files = sorted(tmp_path.glob("summary_runpy-boundary.json"))
    assert len(summary_files) == 1
    summary = json.loads(summary_files[0].read_text())
    assert summary["backend"] == "gateway"
    assert summary["model"] == "oai-1.2"
    assert summary["repetitions"] == 1
    assert summary["tag"] == "runpy-boundary"

    # 1 measured call (--no-warmup). No warm-up call.
    assert call_count["n"] == 1

    # Sanity: the test environment token never leaked into any artifact.
    for path in (*run_files, *summary_files):
        text = path.read_text()
        assert "runpy-bench-token-xyz" not in text


def test_bench_main_boundary_pins_exit_translation_source() -> None:
    """Source-pin the ``__main__`` block to ``raise SystemExit(main())``.

    Same as the gateway_smoke slice-21 source-pin: the literal
    ``raise SystemExit(main())`` form must remain so ``main()``'s int
    return (0 only — the bench script doesn't return 1 or 2 today) keeps
    propagating as the process exit code.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source
