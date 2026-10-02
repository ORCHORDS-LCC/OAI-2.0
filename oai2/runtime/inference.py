"""Inference runtime (scaffold).

Defines the public request/response shape and the :class:`InferenceRuntime`
abstract base. A live implementation loads a model on MLX and serves
tokens; the reference implementation here records a deterministic
response so the rest of the agent can be exercised offline.

Status: PROPOSED for end-to-end live inference. The runtime imports
successfully and the default-device probe is wired to the installed
MLX stack; actual generation requires a checked-in model spec.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import Status
from .model import ModelSpec, discover_default_device


class InferenceRequest(BaseModel):
    """Public shape for a single inference call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=32_768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: tuple[str, ...] = Field(default_factory=tuple)
    seed: int | None = None
    model: ModelSpec | None = None
    # If True, ask the runtime to use speculative decoding if available.
    speculative: bool = False
    # Optional OpenAI-style chat history. When provided, the gateway
    # runtime POSTs ``messages`` instead of flattening ``prompt``.
    messages: list[dict[str, Any]] = Field(default_factory=list)
    # Optional OpenAI-style tool definitions (the wire shape that
    # makes oai-2.0 actually invoke tools instead of writing essays).
    tools: list[dict[str, Any]] = Field(default_factory=list)
    # ``"auto"`` lets the model decide; ``"none"`` forbids tool calls;
    # ``"any"`` requires at least one; or a specific ``{"name": "..."}``.
    tool_choice: str | dict[str, Any] | None = None


@dataclass(slots=True)
class InferenceResponse:
    text: str
    tokens: int
    elapsed_ms: float
    device: str
    status: Status = Status.EXPERIMENTAL
    notes: list[str] = field(default_factory=list)
    # None means the upstream runtime did not supply a completion reason.
    finish_reason: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()


class InferenceRuntime(ABC):
    """Abstract runtime — concrete subclasses own a model + tokenizer."""

    STATUS: Status = Status.PROPOSED

    def __init__(self, spec: ModelSpec | None = None) -> None:
        self.spec = spec or ModelSpec(name="placeholder")

    @property
    def device(self) -> str:
        return discover_default_device()

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse: ...


class PlaceholderRuntime(InferenceRuntime):
    """Deterministic offline runtime for tests and CI.

    Echoes a stable prefix plus the prompt length and device so callers
    can verify shape without loading any weights.
    """

    STATUS = Status.EXPERIMENTAL

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        start = time.perf_counter()
        device = self.device
        text = f"[placeholder:{device}] prompt_len={len(request.prompt)}"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResponse(
            text=text,
            tokens=len(text.split()),
            elapsed_ms=elapsed_ms,
            device=device,
            status=Status.EXPERIMENTAL,
            notes=["placeholder runtime — no model loaded"],
        )


def default_runtime() -> InferenceRuntime:
    """Return the runtime the agent should use by default."""
    return PlaceholderRuntime()


def select_runtime_from_env() -> InferenceRuntime:
    """Pick the best available :class:`InferenceRuntime` from the environment.

    Returns a :class:`GatewayRuntime` when ``OAI2_GATEWAY_API_KEY`` is set,
    otherwise returns a :class:`PlaceholderRuntime`. Never raises on a
    missing or partial key — callers in CI / offline / no-token
    environments keep working with a deterministic offline runtime.

    This is the public entry point for callers that want the highest
    available fidelity without coupling to a specific backend.
    Use :func:`default_runtime` for the static default (always
    :class:`PlaceholderRuntime`, regardless of environment).
    """

    # Local import: ``gateway_runtime`` imports ``InferenceRuntime``
    # from this module at module load time, so we resolve it lazily.
    from .gateway_runtime import GatewayConfigError, GatewayRuntime

    try:
        return GatewayRuntime.from_env()
    except GatewayConfigError:
        return PlaceholderRuntime()


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token count for callers that need a placeholder."""
    return max(1, len(text.split()))


def stop_sequences() -> Sequence[str]:
    return ("\n\n<", "<|end|>", "</s>")


__all__ = [
    "InferenceRequest",
    "InferenceResponse",
    "InferenceRuntime",
    "PlaceholderRuntime",
    "default_runtime",
    "estimate_tokens",
    "select_runtime_from_env",
    "stop_sequences",
]
