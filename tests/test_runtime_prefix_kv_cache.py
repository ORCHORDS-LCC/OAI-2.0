"""Tests for the WI-PERF-003 / #240 digest-keyed prefix KV-state cache."""

from __future__ import annotations

import pytest

from oai2.runtime import PrefixCacheEntry, PrefixCacheMetrics, PrefixKVCache
from oai2.runtime.prefix_kv_cache import PrefixKVCache as DirectPrefixKVCache


class _FakeLayer:
    def __init__(self) -> None:
        self._state: object = None

    @property
    def state(self) -> object:
        return self._state

    @state.setter
    def state(self, value: object) -> None:
        self._state = value


def test_package_export_identity() -> None:
    assert PrefixKVCache is DirectPrefixKVCache
    assert PrefixCacheEntry is not None
    assert PrefixCacheMetrics is not None


def test_store_lookup_and_metrics() -> None:
    layer = _FakeLayer()
    cache = PrefixKVCache()
    assert cache.store([1, 2, 3], ["state-a"]) == 3
    assert layer.state is None

    matched = cache.lookup_and_apply([1, 2, 3, 9, 9], [layer])
    assert matched == 3
    assert layer.state == "state-a"

    metrics = cache.metrics
    assert (metrics.lookups, metrics.hits, metrics.misses, metrics.restores) == (1, 1, 0, 1)
    assert metrics.stored_entries == 1


def test_miss_when_no_stored_prefix_matches() -> None:
    cache = PrefixKVCache()
    layer = _FakeLayer()
    assert cache.lookup_and_apply([7, 8], [layer]) == 0
    assert layer.state is None
    metrics = cache.metrics
    assert (metrics.lookups, metrics.hits, metrics.misses) == (1, 0, 1)


def test_longest_stored_prefix_wins() -> None:
    cache = PrefixKVCache()
    layer = _FakeLayer()
    cache.store([1, 2], ["short"])
    cache.store([1, 2, 3], ["long"])
    assert cache.lookup_and_apply([1, 2, 3, 4], [layer]) == 3
    assert layer.state == "long"


def test_entry_longer_than_query_never_matches() -> None:
    cache = PrefixKVCache()
    cache.store([1, 2, 3, 4], ["state"])
    assert cache.lookup_and_apply([1, 2, 3], [_FakeLayer()]) == 0


def test_eviction_counts_as_invalidation() -> None:
    cache = PrefixKVCache(max_entries=1)
    cache.store([1], ["a"])
    cache.store([2], ["b"])
    metrics = cache.metrics
    assert metrics.stored_entries == 1
    assert metrics.invalidations == 1
    assert cache.lookup_and_apply([1, 5], [_FakeLayer()]) == 0


def test_invalidate_all_drops_every_entry() -> None:
    cache = PrefixKVCache()
    cache.store([1, 2], ["a"])
    cache.store([3, 4], ["b"])
    assert cache.invalidate_all() == 2
    assert cache.metrics.invalidations == 2
    assert cache.lookup_and_apply([1, 2, 9], [_FakeLayer()]) == 0


def test_invalid_construction_and_empty_inputs() -> None:
    with pytest.raises(ValueError):
        PrefixKVCache(max_entries=0)
    cache = PrefixKVCache()
    with pytest.raises(ValueError):
        cache.store([], ["state"])
    assert cache.lookup_and_apply([], [_FakeLayer()]) == 0


def test_lookup_refreshes_recency_for_eviction() -> None:
    cache = PrefixKVCache(max_entries=2)
    cache.store([1], ["a"])
    cache.store([2], ["b"])
    layer = _FakeLayer()
    assert cache.lookup_and_apply([1, 9], [layer]) == 1
    cache.store([3], ["c"])  # evicts [2], keeps refreshed [1]
    assert cache.lookup_and_apply([1, 9], [layer]) == 1
    assert cache.lookup_and_apply([2, 9], [_FakeLayer()]) == 0


def test_real_model_prefix_restore_skips_prefix_prefill() -> None:
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load("mlx-community/SmolLM-135M-Instruct-4bit")
    prefix_ids = tokenizer.encode("You are a helpful assistant. " * 4)[:48]
    suffix_ids = tokenizer.encode("Count: one, two,")[-8:]

    warm_layers = make_prompt_cache(model)
    model(mx.array([prefix_ids]), cache=warm_layers)
    mx.eval(warm_layers[0].state[0])

    cache = PrefixKVCache()
    cache.store(prefix_ids, [layer.state for layer in warm_layers])

    restored_layers = make_prompt_cache(model)
    matched = cache.lookup_and_apply([*prefix_ids, *suffix_ids], restored_layers)
    assert matched == len(prefix_ids)

    generated = 0
    for response in stream_generate(
        model,
        tokenizer,
        prompt=suffix_ids,
        max_tokens=4,
        prompt_cache=restored_layers,
    ):
        generated = response.generation_tokens
    assert generated >= 1
    metrics = cache.metrics
    assert (metrics.hits, metrics.misses, metrics.restores) == (1, 0, 1)
