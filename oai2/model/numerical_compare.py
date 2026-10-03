"""Reference-vs-optimized numerical comparison and fallback policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

from .numerics import NumericalOperation
from .tolerance_matrix import canonical_policy, policy_violations


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
    # Digest of the exact sample pair that produced this comparison.
    # Without it the artifact is not evidence of anything in particular:
    # comparing [1.0, 2.0, 3.0] and comparing [-98765.4321, 1e30, 4.2]
    # produced byte-identical artifacts -- same sample_count, same 0.0
    # errors, same verdict -- so a reader could not tell which inputs a
    # numerical failure came from. REQ-NUM-006 asks for reproducibility
    # from model/config/INPUT identity, and the input half was missing.
    input_digest: str
    # Which CANONICAL policy the declared profile was measured against.
    # `profile_id` is chosen by the caller and therefore proves nothing: the
    # same caller-supplied id can be attached to a profile declaring
    # max_abs_error=1e9 and to one declaring 1e-6, producing artifacts that
    # agree on every field a reader would check. Recording the policy that
    # actually applied means an artifact is falsifiable against a declared
    # baseline rather than against the caller's own description of it.
    tolerance_policy_id: str
    # The specific axes on which the declared profile was looser than
    # `tolerance_policy_id` required. Empty when the profile complies. A bare
    # `tolerance_policy` failure kind says only that something was wrong; this
    # says which number a reader has to go argue with.
    tolerance_policy_violations: tuple[str, ...]
    sample_count: int
    max_abs_error: float
    max_rel_error: float
    finite_state_match: bool
    # Whether capability was actually MEASURED, as opposed to omitted.
    # `_capability_regression` returns 0.0 when both scores are None, so
    # before this field existed an artifact for a comparison that never
    # measured capability was identical in every field to one that measured
    # 0.9 -> 0.9 and passed. A reader could not tell whether the profile's
    # `max_capability_regression` gate had been exercised at all. REQ-NUM-004
    # asks for capability/routing operations to be stress-tested; an
    # unexercised gate must not be readable as a clean one.
    capability_measured: bool
    capability_regression: float
    passed: bool
    failures: tuple[str, ...]
    fallback: NumericalFallback


@dataclass(slots=True, frozen=True)
class NumericalSelection:
    selected_path: str
    used_fallback: bool
    reason: str


def _samples_digest(
    reference: list[float] | tuple[float, ...],
    optimized: list[float] | tuple[float, ...],
) -> str:
    """Digest the exact sample pair, so the artifact says WHAT was compared.

    The two sides are fed in separately and the side name is mixed in, so
    swapping reference and optimized is a different digest rather than the
    same comparison described backwards. ``repr`` of a float round-trips
    exactly and renders nan/inf distinctly, so no sample is silently
    normalised away.
    """
    digest = hashlib.sha256()
    for side, values in (("reference", reference), ("optimized", optimized)):
        digest.update(side.encode("utf-8"))
        digest.update(b"\x1e")
        for raw in values:
            digest.update(repr(float(raw)).encode("ascii"))
            digest.update(b",")
        digest.update(b"\x1d")
    return f"sha256:{digest.hexdigest()}"


def _value_state(value: float) -> str:
    """Classify a sample so two values are only "equal" when they really are.

    "Both non-finite" is not one claim, it is four: finite, NaN, +inf, -inf.
    Comparing only finiteness makes a reference that saturated to +inf look
    identical to an optimized path that saturated to -inf, which is a
    sign-flipped kernel rather than a matching one.
    """
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return "finite"


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
        # A sample is a match only if it is the SAME KIND of value. Comparing
        # finiteness alone let +inf/-inf and NaN/+inf through as identical,
        # because both pairs are "non-finite on both sides".
        if _value_state(ref) != _value_state(opt):
            finite_match = False
        if not (math.isfinite(ref) and math.isfinite(opt)):
            continue
        abs_error = abs(opt - ref)
        abs_errors.append(abs_error)
        denominator = max(abs(ref), 1.0e-12)
        rel_errors.append(abs_error / denominator)

    max_abs = max(abs_errors, default=0.0)
    max_rel = max(rel_errors, default=0.0)
    capability_measured = (
        reference_capability_score is not None and optimized_capability_score is not None
    )
    capability_regression = _capability_regression(
        reference_capability_score,
        optimized_capability_score,
    )

    failures: list[str] = []
    # A DECLARED TOLERANCE IS NOT EVIDENCE OF ONE. `NumericalToleranceProfile`
    # is caller-supplied, so before the canonical matrix existed a caller could
    # declare max_abs_error=1e9 / max_capability_regression=1.0 /
    # require_finite_state_match=False for ROUTER_PROBABILITIES and promote a
    # kernel whose router probabilities were wrong by 0.4 absolute with total
    # capability loss: `passed=True, failures=()`, at any speedup. Each
    # declaration was individually legal and collectively meaningless.
    #
    # This check is reported FIRST because it invalidates the frame the other
    # failures are measured in -- the numbers below were compared against a
    # gate that should not have existed, so a reader must not reach
    # `absolute_error` and conclude the profile was meaningfully tight.
    #
    # Only WIDENING is a violation. A caller may always demand more accuracy
    # than the matrix requires, so this can only ever make the gate stricter
    # than the caller intended, never looser.
    policy_violation_details = policy_violations(
        operation=profile.operation,
        dtype=profile.dtype,
        max_abs_error=profile.max_abs_error,
        max_rel_error=profile.max_rel_error,
        max_capability_regression=profile.max_capability_regression,
        require_finite_state_match=profile.require_finite_state_match,
    )
    if policy_violation_details:
        failures.append("tolerance_policy")
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
        input_digest=_samples_digest(reference, optimized),
        tolerance_policy_id=canonical_policy(profile.operation, profile.dtype).policy_id,
        tolerance_policy_violations=policy_violation_details,
        sample_count=len(reference),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        finite_state_match=finite_match,
        capability_measured=capability_measured,
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
