"""Apple Silicon MLX inference runtime (scaffold only).

Status: PROPOSED for end-to-end live inference. The module structure,
imports and smoke-tests for the installed MLX stack are wired and
verified. The actual model loader, generator, and speculative decoding
loop are placeholders pending a checked-in reference model.
"""

from __future__ import annotations

from .admission import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionQueue,
    AdmissionReason,
    AdmissionRequest,
    CapacitySnapshot,
)
from .admission_scheduler import (
    AdmissionBatchController,
    AdmissionScheduledRequest,
    AdmissionSchedulerMetrics,
)
from .inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    PlaceholderRuntime,
    default_runtime,
)
from .model import ModelSpec, discover_default_device, smoke_check
from .scheduler import (
    BatchPlan,
    SafeBatchScheduler,
    ScheduledRequest,
    SchedulerMetrics,
    SessionCompatibilityKey,
)

__all__ = [
    "AdmissionBatchController",
    "AdmissionScheduledRequest",
    "AdmissionSchedulerMetrics",
    "AdmissionAction",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionQueue",
    "AdmissionReason",
    "AdmissionRequest",
    "CapacitySnapshot",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceRuntime",
    "PlaceholderRuntime",
    "default_runtime",
    "ModelSpec",
    "discover_default_device",
    "smoke_check",
    "SessionCompatibilityKey",
    "ScheduledRequest",
    "SchedulerMetrics",
    "BatchPlan",
    "SafeBatchScheduler",
]
