"""Reference-vs-optimized numerical comparison tests for WI-NUM-002."""

from __future__ import annotations

import math

import pytest

from oai2.model import (
    NumericalArtifactIdentity,
    NumericalFallback,
    NumericalOperation,
    NumericalToleranceProfile,
    compare_numerical_paths,
    extreme_value_fixtures,
    select_numerical_path,
)


def _identity() -> NumericalArtifactIdentity:
    return NumericalArtifactIdentity(
        model_version="model-v1",
        runtime_version="runtime-v1",
        backend="optimized-backend",
        config_id="cfg-v1",
    )


def _profile(
    *,
    operation: NumericalOperation = NumericalOperation.NORMALIZATION,
    fallback: NumericalFallback = NumericalFallback.USE_REFERENCE,
) -> NumericalToleranceProfile:
    return NumericalToleranceProfile(
        profile_id=f"{operation.value}-fp16-small-v1",
        operation=operation,
        dtype="fp16",
        shape_class="small",
        max_abs_error=1.0e-3,
        max_rel_error=1.0e-3,
        max_capability_regression=0.01,
        fallback=fallback,
    )


def test_known_good_optimized_path_passes_tolerance_and_capability() -> None:
    result = compare_numerical_paths(
        [1.0, 2.0, 3.0],
        [1.0001, 2.0001, 2.9999],
        profile=_profile(),
        identity=_identity(),
        reference_capability_score=0.90,
        optimized_capability_score=0.895,
    )
    assert result.passed
    assert result.identity.backend == "optimized-backend"
    assert result.profile_id == "normalization-fp16-small-v1"


def test_over_aggressive_path_falls_back_to_reference() -> None:
    result = compare_numerical_paths(
        [1.0, 2.0],
        [1.5, 2.0],
        profile=_profile(),
        identity=_identity(),
    )
    assert not result.passed
    assert "absolute_error" in result.failures
    selection = select_numerical_path(
        result,
        optimized_path="mlx-int4",
        reference_path="mlx-fp32",
    )
    assert selection.used_fallback
    assert selection.selected_path == "mlx-fp32"


def test_nonfinite_disagreement_blocks_optimized_path() -> None:
    result = compare_numerical_paths(
        [1.0, 2.0],
        [1.0, float("nan")],
        profile=_profile(),
        identity=_identity(),
    )
    assert not result.finite_state_match
    assert "finite_state_mismatch" in result.failures


def test_capability_regression_blocks_numerically_close_path() -> None:
    result = compare_numerical_paths(
        [1.0, 2.0],
        [1.0, 2.0],
        profile=_profile(),
        identity=_identity(),
        reference_capability_score=0.90,
        optimized_capability_score=0.80,
    )
    assert not result.passed
    assert result.failures == ("capability_regression",)


def test_long_context_extreme_fixture_is_represented() -> None:
    values = extreme_value_fixtures()[
        NumericalOperation.LONG_CONTEXT_REDUCTION
    ]
    result = compare_numerical_paths(
        values,
        values,
        profile=_profile(
            operation=NumericalOperation.LONG_CONTEXT_REDUCTION
        ),
        identity=_identity(),
    )
    assert result.passed
    assert result.sample_count == len(values)


def test_reject_policy_raises_instead_of_falling_back() -> None:
    result = compare_numerical_paths(
        [1.0],
        [2.0],
        profile=_profile(fallback=NumericalFallback.REJECT),
        identity=_identity(),
    )
    with pytest.raises(RuntimeError, match="fallback policy is reject"):
        select_numerical_path(
            result,
            optimized_path="custom-metal",
            reference_path="mlx-reference",
        )


def test_numerical_comparison_exports_from_model_package() -> None:
    from oai2.model import (
        NumericalArtifactIdentity as ExportedArtifactIdentity,
    )
    from oai2.model import (
        NumericalComparison as ExportedComparison,
    )
    from oai2.model import (
        NumericalFallback as ExportedFallback,
    )
    from oai2.model import (
        NumericalSelection as ExportedSelection,
    )
    from oai2.model import (
        NumericalToleranceProfile as ExportedToleranceProfile,
    )
    from oai2.model import (
        compare_numerical_paths as ExportedCompare,
    )
    from oai2.model import (
        select_numerical_path as ExportedSelect,
    )
    from oai2.model.numerical_compare import (
        NumericalArtifactIdentity as SourceArtifactIdentity,
    )
    from oai2.model.numerical_compare import (
        NumericalComparison as SourceComparison,
    )
    from oai2.model.numerical_compare import (
        NumericalFallback as SourceFallback,
    )
    from oai2.model.numerical_compare import (
        NumericalSelection as SourceSelection,
    )
    from oai2.model.numerical_compare import (
        NumericalToleranceProfile as SourceToleranceProfile,
    )
    from oai2.model.numerical_compare import (
        compare_numerical_paths as SourceCompare,
    )
    from oai2.model.numerical_compare import (
        select_numerical_path as SourceSelect,
    )

    assert ExportedArtifactIdentity is SourceArtifactIdentity
    assert ExportedComparison is SourceComparison
    assert ExportedFallback is SourceFallback
    assert ExportedSelection is SourceSelection
    assert ExportedToleranceProfile is SourceToleranceProfile
    assert ExportedCompare is SourceCompare
    assert ExportedSelect is SourceSelect


# ---------------------------------------------------------------------------
# An absent measurement is not a clean measurement.
# ---------------------------------------------------------------------------


def test_all_nonfinite_samples_do_not_report_a_perfect_match() -> None:
    """Every sample non-finite compared nothing and must not pass.

    Non-finite samples are skipped, so the error lists stay empty and
    `max(abs_errors, default=0.0)` reports a PERFECT match for a comparison
    that never happened. `finite_state_match` does not catch it: two NaNs
    agree on being non-finite, so it stays True.
    """
    result = compare_numerical_paths(
        [float("nan")] * 3,
        [float("nan")] * 3,
        profile=_profile(),
        identity=_identity(),
    )
    assert not result.passed
    assert "no_comparable_samples" in result.failures
    # The reported errors are still 0.0 — which is exactly why the failure
    # list has to carry this, and why the artifact must not be read as
    # "numerically identical" on its numbers alone.
    assert result.max_abs_error == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_no_comparable_samples_is_reported_for_any_nonfinite_value(value: float) -> None:
    result = compare_numerical_paths(
        [value] * 2,
        [value] * 2,
        profile=_profile(),
        identity=_identity(),
    )
    assert not result.passed
    assert "no_comparable_samples" in result.failures


def test_mixed_finite_and_nonfinite_still_passes() -> None:
    """Opposite-direction guard: the fix must not fail a real comparison.

    A fixture containing some non-finite samples and some real ones DOES
    compare the real ones, so it must still be eligible. A blanket
    "any non-finite input is rejected" fix would pass the test above and
    break this one.
    """
    result = compare_numerical_paths(
        [1.0, 2.0, 3.0, float("nan")],
        [1.0, 2.0, 3.0, float("nan")],
        profile=_profile(),
        identity=_identity(),
    )
    assert result.passed
    assert result.failures == ()
    assert result.sample_count == 4


def test_a_perfect_real_match_is_still_reported_as_zero_error() -> None:
    """The 0.0 error of a real perfect match must remain distinguishable
    from the 0.0 error of a comparison that never happened — by the failure
    list, since the numbers alone cannot tell them apart."""
    perfect = compare_numerical_paths(
        [1.0, 2.0], [1.0, 2.0], profile=_profile(), identity=_identity()
    )
    vacuous = compare_numerical_paths(
        [float("nan")], [float("nan")], profile=_profile(), identity=_identity()
    )
    assert perfect.max_abs_error == vacuous.max_abs_error == 0.0
    assert perfect.passed and not vacuous.passed


# ---------------------------------------------------------------------------
# "Both non-finite" is not the same claim as "the same non-finite value".
# ---------------------------------------------------------------------------


def test_opposite_signed_infinities_are_not_a_match() -> None:
    """+inf against -inf is a sign-flipped kernel, not a matching one.

    Non-finite samples are skipped when errors are accumulated, and
    `finite_state_match` only asks "are both non-finite?". A reference that
    saturated to +inf against an optimized path that saturated to -inf
    therefore reported max_abs_error=0.0, finite_state_match=True and
    passed -- a catastrophic sign flip declared numerically identical.
    """
    result = compare_numerical_paths(
        [1.0, math.inf],
        [1.0, -math.inf],
        profile=_profile(),
        identity=_identity(),
    )
    assert not result.passed
    assert not result.finite_state_match
    assert "finite_state_mismatch" in result.failures
    assert result.max_abs_error == 0.0  # the failure list has to carry this


def test_nan_against_infinity_is_not_a_match() -> None:
    """NaN and +/-inf are different failure modes and must not be equated."""
    for reference, optimized in (
        ([1.0, math.nan], [1.0, math.inf]),
        ([1.0, math.nan], [1.0, -math.inf]),
        ([1.0, math.inf], [1.0, math.nan]),
        ([1.0, -math.inf], [1.0, math.nan]),
    ):
        result = compare_numerical_paths(
            reference, optimized, profile=_profile(), identity=_identity()
        )
        assert not result.passed, (reference, optimized)
        assert not result.finite_state_match, (reference, optimized)
        assert "finite_state_mismatch" in result.failures


def test_same_non_finite_value_still_matches() -> None:
    """Opposite-direction guard: the fix must not reject genuine agreement.

    NaN/NaN and +inf/+inf ARE the same claim, and a profile that permits
    them must keep permitting them.
    """
    for value in (math.nan, math.inf, -math.inf):
        result = compare_numerical_paths(
            [1.0, value], [1.0, value], profile=_profile(), identity=_identity()
        )
        assert result.finite_state_match, value
        assert result.passed, (value, result.failures)


def test_mixed_fixture_still_compares_the_finite_samples() -> None:
    """A correctly-matched non-finite pair alongside real samples still passes."""
    result = compare_numerical_paths(
        [1.0, math.inf, 2.0, 3.0],
        [1.0, math.inf, 2.0, 3.0],
        profile=_profile(),
        identity=_identity(),
    )
    assert result.passed, result.failures
    assert result.sample_count == 4


def test_a_real_error_still_fails_alongside_a_matching_infinity() -> None:
    """The infinity must not mask a genuine tolerance breach on a finite sample."""
    result = compare_numerical_paths(
        [1.0, math.inf, 2.0],
        [1.0, math.inf, 9.0],
        profile=_profile(),
        identity=_identity(),
    )
    assert not result.passed
    assert "absolute_error" in result.failures
    assert "relative_error" in result.failures


def test_finite_state_match_is_reported_false_for_sign_flips_only() -> None:
    """State identity, not merely finiteness, drives the flag."""
    flipped = compare_numerical_paths(
        [math.inf], [math.inf - math.inf * 2],  # -inf
        profile=_profile(),
        identity=_identity(),
    )
    assert not flipped.finite_state_match
    # A real finite comparison is unaffected.
    fine = compare_numerical_paths(
        [1.0, 2.0], [1.0, 2.0], profile=_profile(), identity=_identity()
    )
    assert fine.finite_state_match


# ---------------------------------------------------------------------------
# The artifact must identify WHAT was compared (REQ-NUM-006).
# ---------------------------------------------------------------------------


def test_different_inputs_do_not_produce_an_identical_artifact() -> None:
    """Two different sample sets must not yield the same evidence.

    Before the input digest existed, comparing `[1.0, 2.0, 3.0]` and
    comparing `[-98765.4321, 1e30, 4.2]` produced artifacts identical in
    every field: same sample_count, same 0.0 errors, same verdict. An
    artifact that cannot say which inputs produced it cannot make a
    numerical failure reproducible, which is what REQ-NUM-006 requires.
    """
    easy = compare_numerical_paths(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], profile=_profile(), identity=_identity()
    )
    hard = compare_numerical_paths(
        [-98765.4321, 1e30, 4.2],
        [-98765.4321, 1e30, 4.2],
        profile=_profile(),
        identity=_identity(),
    )
    assert easy.input_digest != hard.input_digest


def test_digest_is_deterministic_for_the_same_inputs() -> None:
    """Re-running the same comparison must reproduce the same digest."""
    first = compare_numerical_paths(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], profile=_profile(), identity=_identity()
    )
    second = compare_numerical_paths(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], profile=_profile(), identity=_identity()
    )
    assert first.input_digest == second.input_digest
    assert first.input_digest.startswith("sha256:")


def test_digest_distinguishes_the_reference_from_the_optimized_side() -> None:
    """Swapping the sides is a different comparison, not the same one."""
    forward = compare_numerical_paths(
        [1.0, 2.0], [1.5, 2.5], profile=_profile(), identity=_identity()
    )
    swapped = compare_numerical_paths(
        [1.5, 2.5], [1.0, 2.0], profile=_profile(), identity=_identity()
    )
    assert forward.input_digest != swapped.input_digest


def test_digest_distinguishes_a_different_optimized_result() -> None:
    """Same reference, different optimized output -> different evidence."""
    good = compare_numerical_paths(
        [1.0, 2.0], [1.0, 2.0], profile=_profile(), identity=_identity()
    )
    bad = compare_numerical_paths(
        [1.0, 2.0], [1.0, 9.0], profile=_profile(), identity=_identity()
    )
    assert good.input_digest != bad.input_digest


def test_digest_covers_non_finite_samples() -> None:
    """NaN/inf inputs still have to be distinguishable from finite ones."""
    nonfinite = compare_numerical_paths(
        [1.0, math.inf], [1.0, math.inf], profile=_profile(), identity=_identity()
    )
    finite = compare_numerical_paths(
        [1.0, 2.0], [1.0, 2.0], profile=_profile(), identity=_identity()
    )
    flipped = compare_numerical_paths(
        [1.0, math.inf], [1.0, -math.inf], profile=_profile(), identity=_identity()
    )
    assert len({nonfinite.input_digest, finite.input_digest, flipped.input_digest}) == 3
