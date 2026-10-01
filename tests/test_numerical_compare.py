"""Reference-vs-optimized numerical comparison tests for WI-NUM-002."""

from __future__ import annotations

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
        NumericalComparison as ExportedComparison,
        NumericalFallback as ExportedFallback,
        NumericalSelection as ExportedSelection,
        NumericalToleranceProfile as ExportedToleranceProfile,
        compare_numerical_paths as ExportedCompare,
        select_numerical_path as ExportedSelect,
    )
    from oai2.model.numerical_compare import (
        NumericalArtifactIdentity as SourceArtifactIdentity,
        NumericalComparison as SourceComparison,
        NumericalFallback as SourceFallback,
        NumericalSelection as SourceSelection,
        NumericalToleranceProfile as SourceToleranceProfile,
        compare_numerical_paths as SourceCompare,
        select_numerical_path as SourceSelect,
    )

    assert ExportedArtifactIdentity is SourceArtifactIdentity
    assert ExportedComparison is SourceComparison
    assert ExportedFallback is SourceFallback
    assert ExportedSelection is SourceSelection
    assert ExportedToleranceProfile is SourceToleranceProfile
    assert ExportedCompare is SourceCompare
    assert ExportedSelect is SourceSelect
