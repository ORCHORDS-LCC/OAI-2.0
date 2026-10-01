"""Numerical policy/sentinel tests for WI-NUM-001."""

from __future__ import annotations

import math

import pytest

from oai2.model import (
    NUMERICAL_POLICY_VERSION,
    NumericalFailure,
    NumericalOperation,
    REFERENCE_PRECISION_RULES,
    check_numerics,
    extreme_value_fixtures,
    precision_rule,
)


def test_reference_precision_policy_covers_every_operation() -> None:
    assert NUMERICAL_POLICY_VERSION == "1"
    assert set(REFERENCE_PRECISION_RULES) == set(NumericalOperation)
    for operation in NumericalOperation:
        rule = precision_rule(operation)
        assert rule.compute_dtype
        assert rule.storage_dtype
        assert rule.accumulation_dtype


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_fail_with_structured_metadata(bad: float) -> None:
    with pytest.raises(NumericalFailure) as exc_info:
        check_numerics(
            [0.0, bad, 1.0],
            operation=NumericalOperation.ATTENTION_SOFTMAX,
            stage="decode.attention",
        )
    failure = exc_info.value
    assert failure.operation is NumericalOperation.ATTENTION_SOFTMAX
    assert failure.stage == "decode.attention"
    assert failure.reason == "non_finite_value"
    assert failure.index == 1
    assert "nan" not in str(failure).lower()
    assert "inf" not in str(failure).lower()


def test_range_sentinel_rejects_deliberate_overflow_fixture() -> None:
    with pytest.raises(NumericalFailure, match="range_exceeded"):
        check_numerics(
            [0.0, 1.0e9],
            operation=NumericalOperation.LOSS,
            stage="train.loss",
            max_abs=1.0e6,
        )


def test_all_reference_extreme_fixtures_are_finite() -> None:
    fixtures = extreme_value_fixtures()
    assert set(fixtures) == set(NumericalOperation)
    for operation, values in fixtures.items():
        result = check_numerics(
            values,
            operation=operation,
            stage=f"fixture.{operation.value}",
        )
        assert result.checked_count == len(values)
        assert math.isfinite(result.max_abs_observed)


def test_debug_sentinel_can_be_disabled_without_reading_values() -> None:
    result = check_numerics(
        [float("nan")],
        operation=NumericalOperation.NORMALIZATION,
        stage="decode.norm",
        enabled=False,
    )
    assert not result.enabled
    assert result.checked_count == 0


def test_numerical_policy_exports_from_model_package() -> None:
    from oai2.model.numerics import NumericalFailure as SourceFailure
    from oai2.model.numerics import NumericalOperation as SourceOperation

    assert NumericalFailure is SourceFailure
    assert NumericalOperation is SourceOperation
