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
    AdmissionDecisionTrace,
    AdmissionScheduledRequest,
    AdmissionSchedulerMetrics,
    AdmissionTelemetry,
)
from .gateway_model_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ChatReply,
    GatewayModelClient,
)
from .gateway_runtime import (
    DEFAULT_GATEWAY_BASE_URL,
    DEFAULT_GATEWAY_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayConfig,
    GatewayConfigError,
    GatewayRuntime,
    GatewayRuntimeError,
    load_gateway_config_from_env,
)
from .inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceRuntime,
    PlaceholderRuntime,
    default_runtime,
    select_runtime_from_env,
)
from .model import ModelSpec, discover_default_device, smoke_check
from .scheduler import (
    BatchPlan,
    SafeBatchScheduler,
    ScheduledRequest,
    SchedulerMetrics,
    SessionCompatibilityKey,
)
from .service import ServiceCompatibility, ServiceHealth, ServiceLifecycle, ServiceState
from .service_security import AccessPolicy, IsolatedSessionRegistry, SessionRecord

__all__ = [
    "AdmissionBatchController",
    "AdmissionDecisionTrace",
    "AdmissionScheduledRequest",
    "AdmissionSchedulerMetrics",
    "AdmissionTelemetry",
    "AdmissionAction",
    "AdmissionDecision",
    "AdmissionPolicy",
    "AdmissionQueue",
    "AdmissionReason",
    "AdmissionRequest",
    "CapacitySnapshot",
    "AccessPolicy",
    "ChatReply",
    "DEFAULT_GATEWAY_BASE_URL",
    "DEFAULT_GATEWAY_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "GatewayConfig",
    "GatewayConfigError",
    "GatewayModelClient",
    "GatewayRuntime",
    "GatewayRuntimeError",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceRuntime",
    "IsolatedSessionRegistry",
    "PlaceholderRuntime",
    "ServiceCompatibility",
    "ServiceHealth",
    "ServiceLifecycle",
    "ServiceState",
    "SessionRecord",
    "default_runtime",
    "load_gateway_config_from_env",
    "select_runtime_from_env",
    "ModelSpec",
    "discover_default_device",
    "smoke_check",
    "SessionCompatibilityKey",
    "ScheduledRequest",
    "SchedulerMetrics",
    "BatchPlan",
    "SafeBatchScheduler",
]
