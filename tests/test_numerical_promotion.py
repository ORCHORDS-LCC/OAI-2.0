from __future__ import annotations

import math

import pytest

from oai2.model import (
    NumericalArtifactIdentity,
    NumericalFallback,
    NumericalOperation,
    NumericalToleranceProfile,
)
from oai2.model.numerical_promotion import (
    NumericalCandidateKind,
    evaluate_numerical_candidate,
)


def _profile(
    *,
    fallback: NumericalFallback = NumericalFallback.USE_REFERENCE,
) -> NumericalToleranceProfile:
    return NumericalToleranceProfile(
        profile_id="norm-fp16-small-v1",
        operation=NumericalOperation.NORMALIZATION,
        dtype="fp16",
        shape_class="small",
        max_abs_error=1.0e-3,
        max_rel_error=1.0e-3,
        max_capability_regression=0.01,
        fallback=fallback,
    )


def _identity(backend: str) -> NumericalArtifactIdentity:
    return NumericalArtifactIdentity(
        model_version="model-v1",
        runtime_version="runtime-v1",
        backend=backend,
        config_id="cfg-v1",
    )


@pytest.mark.parametrize(
    "kind",
    [
        NumericalCandidateKind.BACKEND,
        NumericalCandidateKind.EXPORT,
        NumericalCandidateKind.KERNEL,
    ],
)
def test_each_optimized_candidate_kind_consumes_tolerance_profile(
    kind: NumericalCandidateKind,
) -> None:
    result = evaluate_numerical_candidate(
        [1.0, 2.0, 3.0],
        [1.0001, 2.0001, 2.9999],
        candidate_kind=kind,
        profile=_profile(),
        identity=_identity(f"{kind.value}-candidate"),
        optimized_path=f"{kind.value}-optimized",
        reference_path="reference",
        reference_capability_score=0.90,
        optimized_capability_score=0.895,
        speedup_ratio=1.4,
    )

    assert result.eligible is True
    assert result.candidate_kind is kind
    assert result.selection.selected_path == f"{kind.value}-optimized"
    assert result.selection.used_fallback is False
    assert result.tolerance_profile_id == "norm-fp16-small-v1"
    assert result.identity.backend == f"{kind.value}-candidate"
    assert result.speedup_ratio == 1.4


def test_large_speedup_cannot_override_numerical_failure() -> None:
    result = evaluate_numerical_candidate(
        [1.0, 2.0],
        [1.5, 2.0],
        candidate_kind=NumericalCandidateKind.KERNEL,
        profile=_profile(),
        identity=_identity("custom-metal"),
        optimized_path="custom-metal-fast",
        reference_path="mlx-reference",
        speedup_ratio=10.0,
    )

    assert result.eligible is False
    assert result.speedup_ratio == 10.0
    assert result.selection.used_fallback is True
    assert result.selection.selected_path == "mlx-reference"
    assert "absolute_error" in result.comparison.failures


def test_capability_regression_blocks_numerically_identical_candidate() -> None:
    result = evaluate_numerical_candidate(
        [1.0, 2.0],
        [1.0, 2.0],
        candidate_kind=NumericalCandidateKind.EXPORT,
        profile=_profile(),
        identity=_identity("int4-export"),
        optimized_path="int4-export",
        reference_path="fp32-reference",
        reference_capability_score=0.95,
        optimized_capability_score=0.80,
        speedup_ratio=3.0,
    )

    assert result.eligible is False
    assert result.selection.used_fallback is True
    assert result.comparison.failures == ("capability_regression",)


def test_reject_policy_blocks_promotion_instead_of_selecting_fallback() -> None:
    with pytest.raises(RuntimeError, match="fallback policy is reject"):
        evaluate_numerical_candidate(
            [1.0],
            [2.0],
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(fallback=NumericalFallback.REJECT),
            identity=_identity("unsafe-backend"),
            optimized_path="unsafe",
            reference_path="safe",
            speedup_ratio=5.0,
        )


def test_speedup_ratio_is_optional_but_must_be_positive_when_present() -> None:
    result = evaluate_numerical_candidate(
        [1.0],
        [1.0],
        candidate_kind=NumericalCandidateKind.BACKEND,
        profile=_profile(),
        identity=_identity("backend"),
        optimized_path="backend",
        reference_path="reference",
    )
    assert result.speedup_ratio is None

    with pytest.raises(ValueError, match="speedup_ratio"):
        evaluate_numerical_candidate(
            [1.0],
            [1.0],
            candidate_kind=NumericalCandidateKind.BACKEND,
            profile=_profile(),
            identity=_identity("backend"),
            optimized_path="backend",
            reference_path="reference",
            speedup_ratio=0.0,
        )



def test_numerical_promotion_exports_from_model_package() -> None:
    from oai2.model import NumericalCandidateKind as ExportedKind
    from oai2.model import NumericalPromotionEvidence as ExportedEvidence
    from oai2.model import evaluate_numerical_candidate as ExportedEvaluate

    assert ExportedKind is NumericalCandidateKind
    from oai2.model.numerical_promotion import NumericalPromotionEvidence
    assert ExportedEvidence is NumericalPromotionEvidence
    assert ExportedEvaluate is evaluate_numerical_candidate


def test_a_kernel_producing_only_nan_is_not_promoted() -> None:
    """REQ-NUM-024: speed cannot promote a path never shown equivalent.

    This is the consequence that matters. Pre-fix, an all-NaN optimized path
    reported `max_abs_error=0.0`, `passed=True` and `eligible=True` with a
    10x speedup -- an evidence artifact indistinguishable from a genuinely
    good kernel, which is the false-success shape the promotion gate exists
    to prevent.
    """
    evidence = evaluate_numerical_candidate(
        [1.0, 2.0, 3.0],
        [float("nan")] * 3,
        candidate_kind=NumericalCandidateKind.KERNEL,
        profile=_profile(),
        identity=_identity("llamacpp"),
        optimized_path="llamacpp:q4",
        reference_path="mlx:fp16",
        speedup_ratio=10.0,
    )
    assert not evidence.eligible
    assert evidence.selection.used_fallback
    assert "no_comparable_samples" in evidence.comparison.failures


def test_both_sides_nonfinite_is_not_promoted() -> None:
    """The exact boundary: a NaN reference AND a NaN optimized path.

    `finite_state_match` stays True here (two NaNs agree on being
    non-finite), so the finite-state gate cannot catch this case on its own.
    """
    nan = [float("nan")] * 3
    evidence = evaluate_numerical_candidate(
        nan,
        nan,
        candidate_kind=NumericalCandidateKind.KERNEL,
        profile=_profile(),
        identity=_identity("llamacpp"),
        optimized_path="llamacpp:q4",
        reference_path="mlx:fp16",
        speedup_ratio=10.0,
    )
    assert not evidence.eligible


def test_a_genuinely_good_kernel_is_still_promoted() -> None:
    """Opposite-direction guard at the gate itself.

    The fix must reject only the unmeasured comparison, never a real one.
    """
    evidence = evaluate_numerical_candidate(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        candidate_kind=NumericalCandidateKind.KERNEL,
        profile=_profile(),
        identity=_identity("llamacpp"),
        optimized_path="llamacpp:q4",
        reference_path="mlx:fp16",
        speedup_ratio=10.0,
    )
    assert evidence.eligible
    assert not evidence.selection.used_fallback


def test_a_sign_flipped_infinity_kernel_is_not_promoted() -> None:
    """End-to-end: a fast kernel that flips a saturated sign must not win.

    Before the fix this reported eligible=True, used_fallback=False,
    max_abs_error=0.0 and a 10x speedup, because both samples are
    "non-finite" and the error lists stayed empty. That is a broken kernel
    indistinguishable from a good one in the evidence artifact, which is what
    REQ-NUM-024 forbids.
    """
    evidence = evaluate_numerical_candidate(
        [1.0, math.inf, 2.0],
        [1.0, -math.inf, 2.0],
        candidate_kind=NumericalCandidateKind.KERNEL,
        profile=_profile(),
        identity=_identity("llamacpp"),
        optimized_path="llamacpp:q4",
        reference_path="mlx:fp16",
        speedup_ratio=10.0,
    )
    assert not evidence.eligible
    assert evidence.selection.used_fallback
    assert "finite_state_mismatch" in evidence.comparison.failures
    assert evidence.comparison.max_abs_error == 0.0


def test_a_nan_versus_infinity_kernel_is_not_promoted() -> None:
    """NaN and +/-inf are different failure modes, not one claim."""
    for reference, optimized in (
        ([1.0, math.nan, 2.0], [1.0, math.inf, 2.0]),
        ([1.0, math.inf, 2.0], [1.0, math.nan, 2.0]),
    ):
        evidence = evaluate_numerical_candidate(
            reference,
            optimized,
            candidate_kind=NumericalCandidateKind.KERNEL,
            profile=_profile(),
            identity=_identity("llamacpp"),
            optimized_path="llamacpp:q4",
            reference_path="mlx:fp16",
            speedup_ratio=10.0,
        )
        assert not evidence.eligible, (reference, optimized)
        assert evidence.selection.used_fallback


def test_a_genuine_infinity_match_is_still_promoted() -> None:
    """Opposite-direction guard at the gate: real agreement is not a defect.

    A reference and an optimized path that BOTH saturate to +inf are making
    the same claim, and a profile that allows non-finite states must keep
    allowing them.
    """
    evidence = evaluate_numerical_candidate(
        [1.0, math.inf, 2.0],
        [1.0, math.inf, 2.0],
        candidate_kind=NumericalCandidateKind.KERNEL,
        profile=_profile(),
        identity=_identity("llamacpp"),
        optimized_path="llamacpp:q4",
        reference_path="mlx:fp16",
        speedup_ratio=10.0,
    )
    assert evidence.eligible, evidence.comparison.failures
    assert not evidence.selection.used_fallback
