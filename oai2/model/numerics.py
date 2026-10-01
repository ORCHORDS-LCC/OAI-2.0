"""Versioned numerical-safety policy and backend-neutral sentinels.

The reference policy is deliberately conservative. Optimized/backend-specific
precision choices must be validated by WI-NUM-002 before promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

NUMERICAL_POLICY_VERSION = "1"


class NumericalOperation(StrEnum):
    ATTENTION_SOFTMAX = "attention_softmax"
    NORMALIZATION = "normalization"
    ROUTER_PROBABILITIES = "router_probabilities"
    LOSS = "loss"
    QUANTIZE_DEQUANTIZE = "quantize_dequantize"
    LONG_CONTEXT_REDUCTION = "long_context_reduction"


@dataclass(slots=True, frozen=True)
class PrecisionRule:
    compute_dtype: str
    storage_dtype: str
    accumulation_dtype: str

    def __post_init__(self) -> None:
        for name in ("compute_dtype", "storage_dtype", "accumulation_dtype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


REFERENCE_PRECISION_RULES: dict[NumericalOperation, PrecisionRule] = {
    NumericalOperation.ATTENTION_SOFTMAX: PrecisionRule("fp32", "bf16", "fp32"),
    NumericalOperation.NORMALIZATION: PrecisionRule("fp32", "bf16", "fp32"),
    NumericalOperation.ROUTER_PROBABILITIES: PrecisionRule("fp32", "bf16", "fp32"),
    NumericalOperation.LOSS: PrecisionRule("fp32", "fp32", "fp32"),
    NumericalOperation.QUANTIZE_DEQUANTIZE: PrecisionRule("fp32", "int4_or_int8", "fp32"),
    NumericalOperation.LONG_CONTEXT_REDUCTION: PrecisionRule("fp32", "bf16", "fp32"),
}


@dataclass(slots=True, frozen=True)
class NumericalCheckResult:
    policy_version: str
    operation: NumericalOperation
    stage: str
    checked_count: int
    max_abs_observed: float
    enabled: bool


class NumericalFailure(RuntimeError):
    """Structured failure containing metadata only, never raw tensor values."""

    def __init__(
        self,
        *,
        operation: NumericalOperation,
        stage: str,
        reason: str,
        index: int | None = None,
    ) -> None:
        self.operation = operation
        self.stage = stage
        self.reason = reason
        self.index = index
        suffix = "" if index is None else f" at index {index}"
        super().__init__(f"{operation.value}/{stage}: {reason}{suffix}")


def precision_rule(operation: NumericalOperation) -> PrecisionRule:
    return REFERENCE_PRECISION_RULES[operation]


def check_numerics(
    values: list[float] | tuple[float, ...],
    *,
    operation: NumericalOperation,
    stage: str,
    max_abs: float | None = None,
    enabled: bool = True,
) -> NumericalCheckResult:
    """Fail closed on non-finite or out-of-range values when enabled."""
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    if max_abs is not None:
        if (
            isinstance(max_abs, bool)
            or not isinstance(max_abs, (int, float))
            or not math.isfinite(float(max_abs))
            or float(max_abs) <= 0.0
        ):
            raise ValueError("max_abs must be finite and > 0")
    if not enabled:
        return NumericalCheckResult(
            policy_version=NUMERICAL_POLICY_VERSION,
            operation=operation,
            stage=stage,
            checked_count=0,
            max_abs_observed=0.0,
            enabled=False,
        )

    peak = 0.0
    for index, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise NumericalFailure(
                operation=operation,
                stage=stage,
                reason="non_numeric_value",
                index=index,
            )
        value = float(raw)
        if not math.isfinite(value):
            raise NumericalFailure(
                operation=operation,
                stage=stage,
                reason="non_finite_value",
                index=index,
            )
        magnitude = abs(value)
        peak = max(peak, magnitude)
        if max_abs is not None and magnitude > float(max_abs):
            raise NumericalFailure(
                operation=operation,
                stage=stage,
                reason="range_exceeded",
                index=index,
            )

    return NumericalCheckResult(
        policy_version=NUMERICAL_POLICY_VERSION,
        operation=operation,
        stage=stage,
        checked_count=len(values),
        max_abs_observed=peak,
        enabled=True,
    )


def extreme_value_fixtures() -> dict[NumericalOperation, tuple[float, ...]]:
    """Finite stress fixtures for the reference/debug path.

    Deliberate NaN/Inf injection is tested separately so these fixtures remain
    valid reference inputs rather than guaranteed failures.
    """
    return {
        NumericalOperation.ATTENTION_SOFTMAX: (-80.0, 0.0, 80.0),
        NumericalOperation.NORMALIZATION: (-1.0e12, 0.0, 1.0e12),
        NumericalOperation.ROUTER_PROBABILITIES: (0.0, 1.0e-30, 1.0),
        NumericalOperation.LOSS: (0.0, 1.0e-20, 1.0e6),
        NumericalOperation.QUANTIZE_DEQUANTIZE: (-127.0, 0.0, 127.0),
        NumericalOperation.LONG_CONTEXT_REDUCTION: (-1.0e8, 1.0e-8, 1.0e8),
    }


__all__ = [
    "NUMERICAL_POLICY_VERSION",
    "NumericalOperation",
    "PrecisionRule",
    "REFERENCE_PRECISION_RULES",
    "NumericalCheckResult",
    "NumericalFailure",
    "precision_rule",
    "check_numerics",
    "extreme_value_fixtures",
]
