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
        # Explicitly greedy. The runtime now honours the request's
        # temperature instead of silently generating greedily, so "the same
        # prompt gives the same answer" is a property of temperature=0, not
        # of the prefix cache alone.
        temperature=0.0,
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


def test_real_model_shared_prefix_across_different_suffixes_is_a_miss() -> None:
    """Config D now reports a miss, and that is the correct answer.

    This test previously asserted a HIT when two different prompts shared a
    leading token run. Measured against installed mlx_lm 0.32, restoring that
    run and trimming the stored state back to it corrupts the model's state:
    ``KVCache.trim`` decrements the offset without shrinking the key arrays,
    so the next model call masks over a stale sequence length, and a request
    following an unrelated one returns a continuation of the PREVIOUS prompt.

    The runtime therefore reuses only a stored entry that is a strict prefix
    of the query, which needs no trim. Config D's cross-sample reuse stays
    BLOCKED until mlx_lm exposes a safe way to truncate cached state; see
    tests/test_runtime_prefix_cache_correctness.py and #240.

    The suffix sharing is still real, and the test still proves the misses
    are counted honestly rather than being reported as reuse.
    """
    model_id = _smoke_model_id()
    runtime = MLXHotRuntime(
        ModelSpec(name=model_id),
        model_id=model_id,
        prefix_cache=PrefixKVCache(max_entries=8),
    )
    runtime.load()
    base = "You are a helpful counting assistant. " * 4
    first = runtime.generate(
        InferenceRequest(
            prompt=base + "Alpha suffix.",
            max_tokens=8,
            temperature=0.0,
            prefix_digest="d1",
        )
    )
    second = runtime.generate(
        InferenceRequest(
            prompt=base + "Beta suffix.",
            max_tokens=8,
            temperature=0.0,
            prefix_digest="d2",
        )
    )
    assert "prefix_matched=0" in "\n".join(second.notes)
    metrics = runtime.prefix_cache.metrics
    assert metrics.misses == 2
    assert metrics.hits == 0
    assert first.tokens > 0 and second.tokens > 0
    runtime.close()


def test_real_model_unrelated_prompts_miss() -> None:
    """Changed project state = changed tokens: an unrelated prompt must miss."""
    model_id = _smoke_model_id()
    runtime = MLXHotRuntime(
        ModelSpec(name=model_id),
        model_id=model_id,
        prefix_cache=PrefixKVCache(max_entries=8),
    )
    runtime.load()
    first = runtime.generate(
        InferenceRequest(prompt="Zebra herds migrate across the savannah. " * 4, max_tokens=8, prefix_digest="zebra")
    )
    second = runtime.generate(
        InferenceRequest(prompt="Apple orchards bloom in early spring. " * 4, max_tokens=8, prefix_digest="apple")
    )
    joined = "\n".join(second.notes)
    assert "prefix_matched=0" in joined
    metrics = runtime.prefix_cache.metrics
    assert metrics.misses == 2
    assert metrics.hits == 0
    assert first.tokens > 0 and second.tokens > 0
    runtime.close()
