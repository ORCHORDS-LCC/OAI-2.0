"""Tests for the MLX hot runtime (WI-PERF-003 / #240)."""

from __future__ import annotations

from pathlib import Path

import pytest

from oai2.runtime.inference import InferenceRequest
from oai2.runtime.mlx_hot_runtime import MLXHotRuntime
from oai2.runtime.model import ModelSpec
from oai2.runtime.prefix_kv_cache import PrefixKVCache


def test_mlx_hot_runtime_unloaded_state() -> None:
    spec = ModelSpec(name="dummy")
    runtime = MLXHotRuntime(spec, model_id="dummy")
    assert runtime.model_id == "dummy"
    assert runtime.load_seconds is None
    assert runtime.STATUS.value == "EXPERIMENTAL"


def test_mlx_hot_runtime_close_clears_handle() -> None:
    spec = ModelSpec(name="dummy")
    runtime = MLXHotRuntime(spec, model_id="dummy")
    runtime.close()
    assert runtime.load_seconds is None


def _hf_cache_has(model_id: str) -> bool:
    parts = model_id.split("/")
    if len(parts) != 2:
        return False
    org, name = parts
    snapshots = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{org}--{name}"
        / "snapshots"
    )
    if not snapshots.is_dir():
        return False
    return any(s.is_dir() for s in snapshots.iterdir())


def _smoke_model_id() -> str:
    candidates = [
        "mlx-community/SmolLM-135M-Instruct-4bit",
        "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    ]
    for c in candidates:
        if _hf_cache_has(c):
            return c
    pytest.skip("no cached MLX smoke-test model available")


def test_mlx_hot_runtime_load_and_generate() -> None:
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    model_id = _smoke_model_id()
    spec = ModelSpec(name=model_id)
    runtime = MLXHotRuntime(spec, model_id=model_id)
    try:
        runtime.load()
        assert runtime.load_seconds is not None
        assert runtime.load_seconds > 0.0

        request = InferenceRequest(
            prompt="The capital of France is",
            max_tokens=8,
            temperature=0.0,
        )
        response = runtime.generate(request)
        assert response.tokens > 0
        assert response.elapsed_ms > 0.0
        joined = "|".join(response.notes)
        assert "prefill_seconds=" in joined
        assert "decode_seconds=" in joined
    finally:
        runtime.close()


def test_prefix_cache_reuses_state_for_identical_prompt() -> None:
    model_id = _smoke_model_id()
    runtime = MLXHotRuntime(
        ModelSpec(name=model_id),
        model_id=model_id,
        prefix_cache=PrefixKVCache(max_entries=8),
    )
    runtime.load()
    request = InferenceRequest(
        prompt="Count: one two three four five. " * 4,
        max_tokens=8,
        prefix_digest="test-digest",
    )
    first = runtime.generate(request)
    second = runtime.generate(request)
    assert first.text == second.text
    joined = "\n".join(second.notes)
    assert "prefix_matched=" in joined
    assert "prefix_matched=0" not in joined
    metrics = runtime.prefix_cache.metrics
    assert metrics.misses == 1
    assert metrics.hits == 1
    runtime.close()
