"""Model specification and MLX environment probe.

This module is the **only** module in the runtime that imports MLX.
Everything else uses the abstract :class:`InferenceRuntime`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status
from ..model import NumericalOperation, check_numerics


class ModelSpec(BaseModel):
    """A reference to a model artifact the runtime can load.

    The long-term target is **large total specialist capacity with small
    dynamic active compute** (see
    ``docs/agent-architecture/ARCHITECTURE_TARGET.md``). ``ModelSpec``
    is a thin reference; routing decisions live in
    :mod:`oai2.reasoning.modes`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    revision: str = "main"
    quantization: str | None = None  # e.g. "4bit", "8bit".
    local_path: str | None = None
    # Total specialist capacity (parameters). Optional — used for logging
    # and routing hints, not enforced.
    total_params: int | None = Field(default=None, ge=0)
    # Average active parameters for the default routing profile.
    active_params: int | None = Field(default=None, ge=0)

    @property
    def status(self) -> Status:
        # Until a model is checked in, the runtime is PROPOSED.
        return Status.PROPOSED


def discover_default_device() -> str:
    """Return ``mlx.core.default_device()`` as a stable string."""
    try:
        import mlx.core as mx  # local import: keep this module dependency-free at parse time.
    except Exception as exc:  # pragma: no cover - depends on install
        return f"unavailable:{type(exc).__name__}"
    return str(mx.default_device())


def _reset_peak_memory() -> None:
    """Reset MLX peak-memory accounting; no-op when MLX is unavailable.

    Package-internal helper so sibling runtime modules that must not
    import ``mlx`` directly can still reset peak-memory accounting;
    ``model.py`` stays the single sanctioned MLX importer.
    """
    try:
        import mlx.core as mx  # local import: keep this module dependency-free at parse time.
    except Exception:  # pragma: no cover - depends on install
        return
    mx.reset_peak_memory()


def _prefill_tokens(model: Any, cache_layers: Any, token_ids: list[int]) -> None:
    """Prefill ``token_ids`` into ``cache_layers`` without generating.

    Package-internal helper so sibling runtime modules that must not
    import ``mlx`` directly can drive explicit prefill (e.g. to snapshot
    a prefix's KV state); ``model.py`` stays the single sanctioned MLX
    importer in the runtime package.
    """
    if not token_ids:
        return
    try:
        import mlx.core as mx  # local import: keep this module dependency-free at parse time.
    except Exception:  # pragma: no cover - depends on install
        return
    model(mx.array([token_ids]), cache=cache_layers)
    first_state = cache_layers[0].state
    if isinstance(first_state, tuple) and first_state:
        mx.eval(first_state[0])


def smoke_check() -> tuple[bool, str]:
    """Return ``(ok, device_or_error)`` for the installed MLX stack.

    This is intentionally tiny: it does a 1x1 matmul and confirms MLX can
    talk to the default device. Used by tests and a future ``bench`` CLI.
    """
    try:
        import mlx.core as mx

        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([[1.0], [1.0], [1.0]])
        out = (a @ b).item()
        assert isinstance(out, float)
        check_numerics(
            [float(out)],
            operation=NumericalOperation.LONG_CONTEXT_REDUCTION,
            stage="runtime.smoke.matmul_reduction",
        )
        device = str(mx.default_device())
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"
    if out != 6.0:
        return False, f"unexpected result: {out}"
    return True, device


__all__ = ["ModelSpec", "discover_default_device", "smoke_check"]
