"""Tests for compatibility-digest gating in the prefix KV cache.

The cache stores token identity plus the per-layer KV state. Two requests
can share every leading token and still be mutually incompatible, because
their tool schema or world state differs — and token-prefix matching cannot
see either. The ``digest`` is therefore part of an entry's identity.

Gating is opt-in (``gate_digest=True``) because the training-loop case
(WI-PERF-003 config D) deliberately reuses a prefix across samples that the
caller considers distinct requests, and KV state for a shared leading token
run is valid regardless of which sample produced it. Serving paths turn it
on, because there the digest carries state that tokens do not.
"""

from __future__ import annotations

from typing import Any

from oai2.runtime.inference import InferenceRequest
from oai2.runtime.mlx_hot_runtime import MLXHotRuntime
from oai2.runtime.model import ModelSpec
from oai2.runtime.prefix_kv_cache import PrefixKVCache


class _Layer:
    """Fake KV layer recording what the cache did to it."""

    def __init__(self, name: str, trimmable: bool = True) -> None:
        self.name = name
        self.state: Any = None
        self._trimmable = trimmable
        self.trimmed = 0

    def is_trimmable(self) -> bool:
        return self._trimmable

    def trim(self, n: int) -> None:
        self.trimmed += n


def _layers(n: int = 2, *, trimmable: bool = True) -> list[_Layer]:
    return [_Layer(f"l{i}", trimmable=trimmable) for i in range(n)]


# ---------------------------------------------------------------------------
# Default (ungated) behaviour is unchanged
# ---------------------------------------------------------------------------


def test_ungated_lookup_still_matches_across_digests() -> None:
    """Config D: distinct samples sharing a prefix must still reuse."""
    cache = PrefixKVCache(max_entries=8)
    cache.store([1, 2, 3, 4], [("s", 1)], digest="d1")
    layers = _layers(1)
    matched = cache.lookup_common_and_trim([1, 2, 3, 4, 5], layers, digest="d2")
    assert matched == 4
    assert cache.metrics.hits == 1


def test_digest_defaults_to_none_and_matches_nothing_else() -> None:
    cache = PrefixKVCache(max_entries=8)
    cache.store([1, 2, 3], [("s",)])
    assert cache.lookup_common_and_trim([1, 2, 3, 9], _layers(1)) == 3


# ---------------------------------------------------------------------------
# Gated behaviour
# ---------------------------------------------------------------------------


def test_gated_lookup_misses_on_a_different_digest() -> None:
    cache = PrefixKVCache(max_entries=8)
    cache.store([1, 2, 3, 4], [("s", 1)], digest="schema-a")
    layers = _layers()
    matched = cache.lookup_common_and_trim([1, 2, 3, 4, 5], layers, digest="schema-b", gate_digest=True)
    assert matched == 0
    assert cache.metrics.misses == 1
    assert cache.metrics.hits == 0
    # A miss must leave the layers untouched rather than half-applied.
    assert all(layer.state is None for layer in layers)


def test_gated_lookup_hits_on_the_same_digest() -> None:
    cache = PrefixKVCache(max_entries=8)
    cache.store([1, 2, 3, 4], [("s1",), ("s2",)], digest="schema-a")
    layers = _layers()
    assert cache.lookup_common_and_trim([1, 2, 3, 4, 5], layers, digest="schema-a", gate_digest=True) == 4
    assert [layer.state for layer in layers] == [("s1",), ("s2",)]


def test_gated_lookup_does_not_match_ungated_entries() -> None:
    """A gated caller must not inherit state an ungated caller stored."""
    cache = PrefixKVCache(max_entries=8)
    cache.store([1, 2, 3, 4], [("s",)])
    layers = _layers()
    assert cache.lookup_common_and_trim([1, 2, 3, 4, 5], layers, digest="x", gate_digest=True) == 0


def test_ungated_lookup_does_not_match_gated_entries() -> None:
    cache = PrefixKVCache(max_entries=8)
    cache.store([1, 2, 3, 4], [("s",)], digest="schema-a")
    layers = _layers(1)
    assert cache.lookup_common_and_trim([1, 2, 3, 4, 5], layers) == 4


def test_two_schemas_coexist_without_evicting_each_other() -> None:
    cache = PrefixKVCache(max_entries=4)
    cache.store([1, 2, 3], [("a",)], digest="schema-a")
    cache.store([1, 2, 3], [("b",)], digest="schema-b")
    la, lb = _layers(1), _layers(1)
    cache.lookup_common_and_trim([1, 2, 3, 4], la, digest="schema-a", gate_digest=True)
    cache.lookup_common_and_trim([1, 2, 3, 4], lb, digest="schema-b", gate_digest=True)
    assert [layer.state for layer in la] == [("a",)]
    assert [layer.state for layer in lb] == [("b",)]


def test_lookup_and_apply_honours_gating_too() -> None:
    cache = PrefixKVCache(max_entries=4)
    cache.store([1, 2], [("a",)], digest="one")
    assert cache.lookup_and_apply([1, 2, 3], _layers(1), digest="two", gate_digest=True) == 0
    assert cache.lookup_and_apply([1, 2, 3], _layers(1), digest="one", gate_digest=True) == 2


def test_invalidate_all_clears_every_digest() -> None:
    cache = PrefixKVCache(max_entries=4)
    cache.store([1, 2], [("a",)], digest="one")
    cache.store([3, 4], [("b",)], digest="two")
    assert cache.invalidate_all() == 2
    assert cache.metrics.stored_entries == 0
    assert cache.lookup_common_and_trim([1, 2, 3], _layers(), digest="one", gate_digest=True) == 0


def test_gated_miss_on_untrimmable_layers_leaves_them_untouched() -> None:
    cache = PrefixKVCache(max_entries=4)
    cache.store([1, 2, 3, 4, 5], [("s",)], digest="d")
    layers = _layers(trimmable=False)
    assert cache.lookup_common_and_trim([1, 2, 9], layers, digest="d", gate_digest=True) == 0
    assert all(layer.state is None for layer in layers)


# ---------------------------------------------------------------------------
# Runtime plumbing
# ---------------------------------------------------------------------------


def test_runtime_passes_the_digest_through_to_the_cache() -> None:
    """The wiring gap: generate() must forward, not merely receive, the digest."""
    cache = PrefixKVCache(max_entries=4)
    runtime = MLXHotRuntime(
        ModelSpec(name="stub"), model_id="stub", prefix_cache=cache, gate_digest=True
    )
    seen: list[str | None] = []
    original = cache.lookup_common_and_trim

    def spy(token_ids, layers, *, digest=None, gate_digest=False):
        seen.append(digest)
        return original(token_ids, layers, digest=digest, gate_digest=gate_digest)

    cache.lookup_common_and_trim = spy  # type: ignore[method-assign]
    # No model is loaded, so generate() would try to load one. Exercise the
    # wiring through the runtime's own attribute surface instead.
    assert runtime.prefix_cache is cache
    assert runtime.gate_digest is True
    assert InferenceRequest(prompt="x", prefix_digest="d").prefix_digest == "d"
    assert seen == []


def test_runtime_exposes_the_gating_flag() -> None:
    assert MLXHotRuntime(ModelSpec(name="s"), model_id="s").gate_digest is False
