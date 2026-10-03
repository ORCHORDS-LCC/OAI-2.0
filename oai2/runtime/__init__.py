"""Apple Silicon MLX inference runtime (scaffold only).

Status: PROPOSED for end-to-end live inference. The module structure,
imports and smoke-tests for the installed MLX stack are wired and
verified. The actual model loader, generator, and speculative decoding
loop are placeholders pending a checked-in reference model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Binds the name for type checkers only. At runtime it resolves through
    # the module ``__getattr__`` below, which is what keeps the MLX backend
    # out of the import graph.
    from .mlx_hot_runtime import MLXHotRuntime

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
    TemplateRenderError,
    default_runtime,
    select_runtime_from_env,
)
from .local_service import create_local_service_app
from .model import ModelSpec, discover_default_device, smoke_check
from .prefix_kv_cache import PrefixCacheEntry, PrefixCacheMetrics, PrefixKVCache
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
from .tool_calls import (
    ToolCallParseError,
    normalise_tool_choice,
    parse_tool_calls,
    render_tool_result,
    validate_tool_calls,
)

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
    "MLXHotRuntime",
    "ServiceCompatibility",
    "ServiceHealth",
    "ServiceLifecycle",
    "ServiceState",
    "SessionRecord",
    "TemplateRenderError",
    "ToolCallParseError",
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
    "PrefixCacheEntry",
    "PrefixCacheMetrics",
    "PrefixKVCache",
    "discover_default_device",
    "smoke_check",
    "normalise_tool_choice",
    "parse_tool_calls",
    "render_tool_result",
    "validate_tool_calls",
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

#: Names resolved on first access rather than at package import.
#:
#: ``mlx_hot_runtime`` imports ``mlx_lm`` at module level, so importing it
#: eagerly made ``import oai2.runtime`` — and every package that depends on
#: it, including :mod:`oai2.server` and :mod:`oai2.agents` — require an
#: Apple GPU stack. MLX is a *backend*, and selecting a backend is a decision
#: the caller makes, not one this package should make at import time.
#: :mod:`oai2.runtime.model` already keeps its MLX imports function-local for
#: exactly this reason; this extends the same rule to the package boundary.
_LAZY_EXPORTS = frozenset({"MLXHotRuntime"})


def __getattr__(name: str) -> object:
    """Resolve a lazily-exported name (PEP 562).

    Reaching ``MLXHotRuntime`` without MLX installed raises ``ImportError``,
    which names the real missing dependency, rather than ``AttributeError``,
    which would report a missing export and hide the actual cause.
    """
    if name in _LAZY_EXPORTS:
        from . import mlx_hot_runtime

        return getattr(mlx_hot_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_EXPORTS)
