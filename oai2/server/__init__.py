"""Hot-residency local inference server surfaces for the OAI-2.0 runtime.

Two surfaces live here:

- :func:`create_openai_compat_app` (from
  :mod:`oai2.server.openai_compat_app`) — an OpenAI
  ``/v1/chat/completions`` surface backed by the canonical
  :class:`oai2.runtime.MLXHotRuntime`. This is the surface a generic
  OpenAI client (e.g. ZCode pointed at a local endpoint) uses to
  reach the live MLX weights without going through the upstream
  orchords.com gateway.

- :class:`MLXHotRuntime` is re-exported from
  :mod:`oai2.runtime.mlx_hot_runtime` for convenience.
"""

from __future__ import annotations

from .openai_compat_app import create_app as create_openai_compat_app

__all__ = ["create_openai_compat_app", "MLXHotRuntime"]

#: Resolved on first access. ``oai2.runtime.mlx_hot_runtime`` imports
#: ``mlx_lm`` at module level; re-exporting it eagerly would make
#: ``import oai2.server`` require an Apple GPU stack. See the same rule in
#: :mod:`oai2.runtime`.
_LAZY_EXPORTS = frozenset({"MLXHotRuntime"})


def __getattr__(name: str) -> object:
    """Resolve a lazily-exported name (PEP 562)."""
    if name in _LAZY_EXPORTS:
        from oai2.runtime import mlx_hot_runtime

        return getattr(mlx_hot_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_EXPORTS)
