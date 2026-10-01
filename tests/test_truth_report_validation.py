"""Behavioral regression coverage for WI-TRUTH-002 report integrity."""

from dataclasses import replace
from typing import Any

import pytest

from oai2.evals.truth import (
    TruthCaseClass,
    TruthOutcome,
    TruthPromotionBudget,
    TruthReport,
    TruthSample,
    evaluate_truth_candidate_promotion,
    evaluate_truth_promotion,
    summarize_truth,
)


def _sample(outcome: TruthOutcome = TruthOutcome.SUPPORTED) -> TruthSample:
    return TruthSample(
        "case-1", TruthCaseClass.FAILING_VERIFICATION, outcome, "policy-1", "runtime-1", "source-1"
    )


def _report() -> TruthReport:
    return summarize_truth([_sample()])


def _budget() -> TruthPromotionBudget:
    return TruthPromotionBudget("strict-1", 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("field", ["outcome", "case_class"])
@pytest.mark.parametrize("invalid", ["not-a-known-label", "", None, True, 7])
def test_sample_rejects_unknown_taxonomy(field: str, invalid: Any) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_sample(), **{field: invalid})


def test_sample_accepts_and_normalizes_known_serialized_labels() -> None:
    values: dict[str, Any] = {"case_class": "failing_verification", "outcome": "false_success"}
    sample = replace(_sample(), **values)
    assert sample.case_class is TruthCaseClass.FAILING_VERIFICATION
    assert sample.outcome is TruthOutcome.FALSE_SUCCESS
    result = evaluate_truth_promotion(summarize_truth([sample]), _budget())
    assert not result.passed
    assert result.failures == ("false_success_rate",)


@pytest.mark.parametrize("field", [f"{outcome.value}_rate" for outcome in TruthOutcome])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 1.1, True, "0", None])
def test_gate_rejects_invalid_report_rates(field: str, invalid: Any) -> None:
    report = replace(_report(), **{field: invalid})
    with pytest.raises(ValueError, match=field):
        evaluate_truth_promotion(report, _budget())


@pytest.mark.parametrize("count", [0, -1, True, 1.0, "1", None])
def test_gate_rejects_invalid_sample_count(count: Any) -> None:
    report = replace(_report(), sample_count=count)
    with pytest.raises(ValueError, match="sample_count"):
        evaluate_truth_promotion(report, _budget())


@pytest.mark.parametrize("field", ["by_outcome", "by_case_class"])
@pytest.mark.parametrize("invalid", [{}, {"unknown": 1}, None])
def test_gate_rejects_invalid_histograms(field: str, invalid: Any) -> None:
    report = replace(_report(), **{field: invalid})
    with pytest.raises(ValueError, match=field):
        evaluate_truth_promotion(report, _budget())


@pytest.mark.parametrize(
    ("field", "label"),
    [("by_outcome", TruthOutcome.SUPPORTED), ("by_case_class", TruthCaseClass.FAILING_VERIFICATION)],
)
@pytest.mark.parametrize("count", [-1, True, 1.0, "1", 2])
def test_gate_rejects_invalid_histogram_counts(field: str, label: Any, count: Any) -> None:
    report = replace(_report(), **{field: {label: count}})
    with pytest.raises(ValueError, match=field):
        evaluate_truth_promotion(report, _budget())


def test_gate_rejects_zero_rate_hiding_a_false_success_count() -> None:
    report = summarize_truth([_sample(TruthOutcome.FALSE_SUCCESS)])
    report = replace(report, false_success_rate=0.0)
    with pytest.raises(ValueError, match="false_success_rate"):
        evaluate_truth_promotion(report, _budget())


def test_gate_rechecks_histograms_after_report_construction() -> None:
    report = _report()
    report.by_outcome.clear()
    report.by_outcome[TruthOutcome.FALSE_SUCCESS] = 1
    with pytest.raises(ValueError, match="rate"):
        evaluate_truth_promotion(report, _budget())


def test_combined_promotion_cannot_bypass_invalid_truth_report() -> None:
    report = replace(_report(), false_success_rate=float("nan"))
    with pytest.raises(ValueError, match="false_success_rate"):
        evaluate_truth_candidate_promotion(
            report,
            _budget(),
            baseline_verified_task_rate=1.0,
            candidate_verified_task_rate=1.0,
            max_verified_task_regression=0.0,
            speedup_ratio=100.0,
        )


@pytest.mark.parametrize("outcome", list(TruthOutcome))
def test_valid_reports_preserve_existing_promotion_decisions(outcome: TruthOutcome) -> None:
    result = evaluate_truth_promotion(summarize_truth([_sample(outcome)]), _budget())
    expected_pass = outcome in (TruthOutcome.SUPPORTED, TruthOutcome.CORRECT_ABSTENTION)
    assert result.passed is expected_pass
    assert result.failures == (() if expected_pass else (f"{outcome.value}_rate",))


def test_valid_mixed_report_preserves_fractional_rates_and_serialized_counts() -> None:
    samples = [
        replace(_sample(outcome), case_id=f"case-{index}")
        for index, outcome in enumerate(TruthOutcome)
    ]
    report = summarize_truth(samples)
    counts: dict[str, Any] = {
        "by_outcome": {key.value: value for key, value in report.by_outcome.items()},
        "by_case_class": {key.value: value for key, value in report.by_case_class.items()},
    }
    report = replace(report, **counts)
    budget = TruthPromotionBudget("fractional-1", *(1 / 7 for _ in range(5)))
    assert evaluate_truth_promotion(report, budget).passed
