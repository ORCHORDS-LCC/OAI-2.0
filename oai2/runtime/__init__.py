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
from .gateway_models import (
    KNOWN_CLOUD_MODELS,
    STRICT_CLOUD_MODEL_ID,
    CloudDiscoveryError,
    CloudModelError,
    ModelProbe,
    UnknownModelError,
    WorkingModelResolution,
    discover_cloud_models,
    probe_model,
    resolve_working_model,
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
from .local_service import create_local_service_app
from .model import ModelSpec, discover_default_device, smoke_check
from .residency import (
    CapacityTrendPoint,
    ResidencyAccountant,
    ResidencyOutcome,
    ResidencyRecord,
    ResidencySummary,
)
from .scheduler import (
    BatchPlan,
    SafeBatchScheduler,
    ScheduledRequest,
    SchedulerMetrics,
    SessionCompatibilityKey,
)
from .service import ServiceCompatibility, ServiceHealth, ServiceLifecycle, ServiceState
from .service_binding import (
    BatchExecutionResult,
    BatchInferenceSurface,
    BatchSurfaceMetrics,
    create_batch_inference_app,
)
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
    "CapacityTrendPoint",
    "ResidencyAccountant",
    "ResidencyOutcome",
    "ResidencyRecord",
    "ResidencySummary",
    "AccessPolicy",
    "ChatReply",
    "CloudDiscoveryError",
    "CloudModelError",
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
    "ServiceCompatibility",
    "ServiceHealth",
    "ServiceLifecycle",
    "ServiceState",
    "SessionRecord",
    "create_local_service_app",
    "KNOWN_CLOUD_MODELS",
    "STRICT_CLOUD_MODEL_ID",
    "ModelProbe",
    "PlaceholderRuntime",
    "default_runtime",
    "discover_cloud_models",
    "load_gateway_config_from_env",
    "probe_model",
    "resolve_working_model",
    "select_runtime_from_env",
    "ModelSpec",
    "discover_default_device",
    "smoke_check",
    "SessionCompatibilityKey",
    "ScheduledRequest",
    "SchedulerMetrics",
    "BatchPlan",
    "SafeBatchScheduler",
    "BatchExecutionResult",
    "BatchInferenceSurface",
    "BatchSurfaceMetrics",
    "create_batch_inference_app",
    "UnknownModelError",
    "WorkingModelResolution",
]
