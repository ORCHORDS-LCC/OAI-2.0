from __future__ import annotations

import pytest

from oai2.evals import CapabilityScore, SuiteReport
from oai2.evals.regression import (
    CapabilityRegressionThreshold,
    HeldOutPromotionBudget,
    evaluate_held_out_promotion,
)


def _score(case_id: str, capability: str, *, passed: bool) -> CapabilityScore:
    return CapabilityScore(
        case_id=case_id,
        capability=capability,
        score=1.0 if passed else 0.0,
        passed=passed,
        matched_pattern="ok" if passed else None,
        forbidden_matched=(),
        notes="" if passed else "failed",
    )


def _report(
    suite_id: str,
    capability: str,
    outcomes: tuple[tuple[str, bool], ...],
) -> SuiteReport:
    scores = [_score(case_id, capability, passed=passed) for case_id, passed in outcomes]
    return SuiteReport(
        suite_id=suite_id,
        capability=capability,
        runtime="held-out-runtime",
        n_cases=len(scores),
        n_passed=sum(score.passed for score in scores),
        scores=scores,
    )


def _budget() -> HeldOutPromotionBudget:
    return HeldOutPromotionBudget(
        version="held-out-v1",
        thresholds=(
            CapabilityRegressionThreshold(
                capability="coding",
                max_pass_rate_regression=0.10,
                max_mean_score_regression=0.10,
            ),
            CapabilityRegressionThreshold(
                capability="verification",
                max_pass_rate_regression=0.0,
                max_mean_score_regression=0.0,
            ),
        ),
        min_abstention_accuracy=0.90,
        max_false_success_rate=0.05,
    )


def test_material_critical_regression_is_not_hidden_by_other_gain() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True), ("c2", False))),
        _report("verification", "verification", (("v1", True), ("v2", True))),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True), ("c2", True))),
        _report("verification", "verification", (("v1", True), ("v2", False))),
    ]

    result = evaluate_held_out_promotion(
        baseline_reports=baseline,
        candidate_reports=candidate,
        budget=_budget(),
        training_case_ids={"train-1", "train-2"},
        candidate_abstention_accuracy=1.0,
        candidate_false_success_rate=0.0,
    )

    assert result.passed is False
    assert "verification:pass_rate_regression" in result.failures
    assert "verification:mean_score_regression" in result.failures
    coding = next(item for item in result.capability_regressions if item.capability == "coding")
    assert coding.candidate_pass_rate > coding.baseline_pass_rate


def test_non_regressing_control_candidate_passes() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True), ("c2", False))),
        _report("verification", "verification", (("v1", True), ("v2", True))),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True), ("c2", True))),
        _report("verification", "verification", (("v1", True), ("v2", True))),
    ]

    result = evaluate_held_out_promotion(
        baseline_reports=baseline,
        candidate_reports=candidate,
        budget=_budget(),
        training_case_ids={"train-1"},
        candidate_abstention_accuracy=0.95,
        candidate_false_success_rate=0.0,
    )
    assert result.passed is True
    assert result.failures == ()


def test_train_held_out_overlap_is_rejected_before_promotion() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]

    with pytest.raises(ValueError, match="overlap training/tuning"):
        evaluate_held_out_promotion(
            baseline_reports=baseline,
            candidate_reports=candidate,
            budget=_budget(),
            training_case_ids={"c1"},
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        )


@pytest.mark.parametrize(
    ("abstention", "false_success", "expected_failure"),
    [
        (0.80, 0.0, "abstention_accuracy"),
        (1.0, 0.10, "false_success_rate"),
    ],
)
def test_abstention_and_false_success_are_first_class_gates(
    abstention: float,
    false_success: float,
    expected_failure: str,
) -> None:
    baseline = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    candidate = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]

    result = evaluate_held_out_promotion(
        baseline_reports=baseline,
        candidate_reports=candidate,
        budget=_budget(),
        training_case_ids=set(),
        candidate_abstention_accuracy=abstention,
        candidate_false_success_rate=false_success,
    )
    assert result.passed is False
    assert expected_failure in result.failures


def test_candidate_and_baseline_must_use_identical_held_out_cases() -> None:
    baseline = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    candidate = [
        _report("coding", "coding", (("c2", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    with pytest.raises(ValueError, match="identical held-out case IDs"):
        evaluate_held_out_promotion(
            baseline_reports=baseline,
            candidate_reports=candidate,
            budget=_budget(),
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        )


def test_every_candidate_capability_requires_explicit_threshold() -> None:
    budget = HeldOutPromotionBudget(
        version="partial",
        thresholds=(
            CapabilityRegressionThreshold(
                capability="coding",
                max_pass_rate_regression=0.0,
                max_mean_score_regression=0.0,
            ),
        ),
        min_abstention_accuracy=0.0,
        max_false_success_rate=1.0,
    )
    reports = [
        _report("coding", "coding", (("c1", True),)),
        _report("verification", "verification", (("v1", True),)),
    ]
    with pytest.raises(ValueError, match="missing capability thresholds: verification"):
        evaluate_held_out_promotion(
            baseline_reports=reports,
            candidate_reports=reports,
            budget=budget,
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        )
