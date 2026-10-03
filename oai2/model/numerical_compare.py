"""Reference-vs-optimized numerical comparison and fallback policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .numerics import NumericalOperation


class NumericalFallback(StrEnum):
    USE_REFERENCE = "use_reference"
    REJECT = "reject"


@dataclass(slots=True, frozen=True)
class NumericalToleranceProfile:
    profile_id: str
    operation: NumericalOperation
    dtype: str
    shape_class: str
    max_abs_error: float
    max_rel_error: float
    max_capability_regression: float = 0.0
    require_finite_state_match: bool = True
    fallback: NumericalFallback = NumericalFallback.USE_REFERENCE

    def __post_init__(self) -> None:
        for name in ("profile_id", "dtype", "shape_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("max_abs_error", "max_rel_error", "max_capability_regression"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and >= 0")


@dataclass(slots=True, frozen=True)
class NumericalArtifactIdentity:
    model_version: str
    runtime_version: str
    backend: str
    config_id: str

    def __post_init__(self) -> None:
        for name in ("model_version", "runtime_version", "backend", "config_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(slots=True, frozen=True)
class NumericalComparison:
    profile_id: str
    identity: NumericalArtifactIdentity
    sample_count: int
    max_abs_error: float
    max_rel_error: float
    finite_state_match: bool
    capability_regression: float
    passed: bool
    failures: tuple[str, ...]
    fallback: NumericalFallback


@dataclass(slots=True, frozen=True)
class NumericalSelection:
    selected_path: str
    used_fallback: bool
    reason: str


def compare_numerical_paths(
    reference: list[float] | tuple[float, ...],
    optimized: list[float] | tuple[float, ...],
    *,
    profile: NumericalToleranceProfile,
    identity: NumericalArtifactIdentity,
    reference_capability_score: float | None = None,
    optimized_capability_score: float | None = None,
) -> NumericalComparison:
    """Compare one optimized path against its declared reference/oracle."""
    if len(reference) != len(optimized):
        raise ValueError("reference and optimized sequences must have equal length")
    if not reference:
        raise ValueError("at least one numerical sample is required")

    abs_errors: list[float] = []
    rel_errors: list[float] = []
    finite_match = True
    for ref_raw, opt_raw in zip(reference, optimized, strict=True):
        ref = float(ref_raw)
        opt = float(opt_raw)
        ref_finite = math.isfinite(ref)
        opt_finite = math.isfinite(opt)
        if ref_finite != opt_finite:
            finite_match = False
        if not ref_finite or not opt_finite:
            continue
        abs_error = abs(opt - ref)
        abs_errors.append(abs_error)
        denominator = max(abs(ref), 1.0e-12)
        rel_errors.append(abs_error / denominator)

    max_abs = max(abs_errors, default=0.0)
    max_rel = max(rel_errors, default=0.0)
    capability_regression = _capability_regression(
        reference_capability_score,
        optimized_capability_score,
    )

    failures: list[str] = []
    if profile.require_finite_state_match and not finite_match:
        failures.append("finite_state_mismatch")
    # NO COMPARABLE SAMPLES IS NOT A PASS. Non-finite samples are skipped
    # above, so when EVERY sample is non-finite the error lists are empty and
    # the `default=0.0` above reports a PERFECT match for a comparison that
    # never happened. `finite_state_match` does not catch this: two NaNs
    # agree on being non-finite, so `finite_match` stays True.
    #
    # This matters because a NaN-producing kernel is not "numerically
    # identical" — it is broken, and the evidence artifact would otherwise be
    # indistinguishable from a genuinely good one (both report
    # max_abs_error=0.0 and passed=True), which is what REQ-NUM-024 forbids.
    if not abs_errors:
        failures.append("no_comparable_samples")
    if max_abs > profile.max_abs_error:
        failures.append("absolute_error")
    if max_rel > profile.max_rel_error:
        failures.append("relative_error")
    if capability_regression > profile.max_capability_regression:
        failures.append("capability_regression")

    return NumericalComparison(
        profile_id=profile.profile_id,
        identity=identity,
        sample_count=len(reference),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        finite_state_match=finite_match,
        capability_regression=capability_regression,
        passed=not failures,
        failures=tuple(failures),
        fallback=profile.fallback,
    )


def select_numerical_path(
    comparison: NumericalComparison,
    *,
    optimized_path: str,
    reference_path: str,
) -> NumericalSelection:
    """Select the optimized path only when the comparison passed."""
    if not optimized_path.strip() or not reference_path.strip():
        raise ValueError("path names must be non-empty")
    if comparison.passed:
        return NumericalSelection(
            selected_path=optimized_path,
            used_fallback=False,
            reason="within_tolerance",
        )
    if comparison.fallback is NumericalFallback.USE_REFERENCE:
        return NumericalSelection(
            selected_path=reference_path,
            used_fallback=True,
            reason="out_of_tolerance",
        )
    raise RuntimeError(
        "optimized numerical path failed tolerance and fallback policy is reject"
    )


def _capability_regression(
    reference_score: float | None,
    optimized_score: float | None,
) -> float:
    if reference_score is None and optimized_score is None:
        return 0.0
    if reference_score is None or optimized_score is None:
        raise ValueError("capability scores must be provided together")
    for name, value in (
        ("reference_capability_score", reference_score),
        ("optimized_capability_score", optimized_score),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be between 0 and 1")
    return max(0.0, float(reference_score) - float(optimized_score))


__all__ = [
    "NumericalFallback",
    "NumericalToleranceProfile",
    "NumericalArtifactIdentity",
    "NumericalComparison",
    "NumericalSelection",
    "compare_numerical_paths",
    "select_numerical_path",
]
