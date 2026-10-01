"""Apple Silicon MLX inference runtime (scaffold only).

Status: PROPOSED for end-to-end live inference. The module structure,
imports and smoke-tests for the installed MLX stack are wired and
verified. The actual model loader, generator, and speculative decoding
loop are placeholders pending a checked-in reference model.
"""

from __future__ import annotations

from .inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    PlaceholderRuntime,
    default_runtime,
)
from .model import ModelSpec, discover_default_device, smoke_check

__all__ = [
    "InferenceRequest",
    "InferenceResponse",
    "InferenceRuntime",
    "PlaceholderRuntime",
    "default_runtime",
    "ModelSpec",
    "discover_default_device",
    "smoke_check",
]
