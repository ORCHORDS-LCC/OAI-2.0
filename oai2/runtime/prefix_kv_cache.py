"""Digest-keyed prefix KV-state cache for WI-PERF-003 / #240 (configs D/E).

Verified against the installed mlx_lm 0.32.0 source: ``generate_step``
prefills the WHOLE prompt on every call and performs no token-prefix
matching, so a plain shared cache never skips prefix prefill. This
component adds the missing layer: after a prefix is prefilled once, its
per-layer cache state is snapshotted; later requests whose token sequence
starts with that prefix get the state restored, so only the divergent
suffix is prefilled.

The caller owns compatibility gating (REQ-INF-012/013): key upstream by
the request's compatibility digest so tool/world-state changes miss
naturally; :meth:`invalidate_all` covers wholesale changes. This module
never imports ``mlx`` directly — ``model.py`` stays the single sanctioned
MLX importer in the runtime package.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _common_prefix_len(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Length of the longest common leading token run of ``a`` and ``b``."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass(slots=True, frozen=True)
class PrefixCacheEntry:
    """One stored prefix: the token identity and its per-layer KV states."""

    token_ids: tuple[int, ...]
    states: tuple[Any, ...]


@dataclass(slots=True, frozen=True)
class PrefixCacheMetrics:
    """Hit/miss/invalidation telemetry (REQ-PERF-032 observability)."""

    lookups: int
    hits: int
    misses: int
    restores: int
    invalidations: int
    stored_entries: int


class PrefixKVCache:
    """Longest-prefix KV-state store with bounded entries and hit telemetry."""

    def __init__(self, *, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], PrefixCacheEntry] = OrderedDict()
        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._restores = 0
        self._invalidations = 0

    def store(self, token_ids: Sequence[int], states: Sequence[Any]) -> int:
        """Snapshot one prefix; returns the number of tokens stored."""
        key = tuple(token_ids)
        if not key:
            raise ValueError("token_ids must be non-empty")
        entry = PrefixCacheEntry(token_ids=key, states=tuple(states))
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._invalidations += 1
        return len(key)

    def lookup_and_apply(self, token_ids: Sequence[int], cache_layers: Sequence[Any]) -> int:
        """Restore the longest stored prefix of ``token_ids`` into the layers.

        Returns the number of matched prefix tokens (0 = miss). The caller
        prefills only ``token_ids[matched:]`` through the same layers. The
        matched prefix must be shorter than the full sequence; a fully
        consumed sequence leaves nothing to prefill and the caller should
        handle that case before calling.
        """
        self._lookups += 1
        key = tuple(token_ids)
        if not key:
            self._misses += 1
            return 0
        best: PrefixCacheEntry | None = None
        for stored in self._entries.values():
            n = len(stored.token_ids)
            if n < len(key) and key[:n] == stored.token_ids:
                if best is None or n > len(best.token_ids):
                    best = stored
        if best is None:
            self._misses += 1
            return 0
        self._entries.move_to_end(best.token_ids)
        for layer, state in zip(cache_layers, best.states, strict=True):
            layer.state = state
        self._hits += 1
        self._restores += 1
        return len(best.token_ids)

    def invalidate_all(self) -> int:
        """Drop every stored prefix (wholesale tool/world-state change)."""
        count = len(self._entries)
        self._entries.clear()
        self._invalidations += count
        return count

    def lookup_common_and_trim(self, token_ids: Sequence[int], cache_layers: Sequence[Any]) -> int:
        """Restore the longest common token prefix with any stored entry.

        Unlike :meth:`lookup_and_apply` — which requires a stored entry to
        be a strict prefix of the query — this matches the longest common
        leading token run between the query and a stored entry and trims
        the restored layers back to the shared length, so a stable prefix
        followed by a divergent suffix still hits (the training-loop case).
        KV state depends only on preceding tokens, so the shared run's
        state is valid for the query.

        Returns the number of matched tokens (0 = miss); the caller
        prefills ``token_ids[matched:]``. At least one token is always
        left for the caller to prefill. Partial matches require trimmable
        cache layers; if the layers cannot trim, the match is reported as
        a miss and the layers are left untouched.
        """
        self._lookups += 1
        key = tuple(token_ids)
        if not key:
            self._misses += 1
            return 0
        best: PrefixCacheEntry | None = None
        best_common = 0
        for stored in self._entries.values():
            common = _common_prefix_len(stored.token_ids, key)
            if common > best_common:
                best = stored
                best_common = common
        matched = min(best_common, len(key) - 1)
        if best is None or matched < 1:
            self._misses += 1
            return 0
        excess = len(best.token_ids) - matched
        if excess:
            for layer in cache_layers:
                if not layer.is_trimmable():
                    self._misses += 1
                    return 0
        for layer, state in zip(cache_layers, best.states, strict=True):
            layer.state = state
        if excess:
            for layer in cache_layers:
                layer.trim(excess)
        self._entries.move_to_end(best.token_ids)
        self._hits += 1
        self._restores += 1
        return matched

    @property
    def metrics(self) -> PrefixCacheMetrics:
        return PrefixCacheMetrics(
            lookups=self._lookups,
            hits=self._hits,
            misses=self._misses,
            restores=self._restores,
            invalidations=self._invalidations,
            stored_entries=len(self._entries),
        )


__all__ = ["PrefixCacheEntry", "PrefixCacheMetrics", "PrefixKVCache"]
