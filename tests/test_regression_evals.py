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



def test_held_out_regression_gate_exports_from_evals_package() -> None:
    from oai2.evals import CapabilityRegression as ExportedRegression
    from oai2.evals import CapabilityRegressionThreshold as ExportedThreshold
    from oai2.evals import HeldOutPromotionBudget as ExportedBudget
    from oai2.evals import HeldOutPromotionEvaluation as ExportedEvaluation
    from oai2.evals import evaluate_held_out_promotion as ExportedEvaluate
    from oai2.evals.regression import (
        CapabilityRegression,
        HeldOutPromotionEvaluation,
    )

    assert ExportedRegression is CapabilityRegression
    assert ExportedThreshold is CapabilityRegressionThreshold
    assert ExportedBudget is HeldOutPromotionBudget
    assert ExportedEvaluation is HeldOutPromotionEvaluation
    assert ExportedEvaluate is evaluate_held_out_promotion


# --- fail-closed reporting ---------------------------------------------------
#
# The promotion gate decides whether a candidate ships. It fails a capability
# only when the regression *exceeds* a tolerance, so a non-finite operand made
# every comparison False and silently disabled that check. Paired with a pass
# count larger than the case count, a candidate that failed every held-out
# case was promoted and the evidence recorded a pass rate of 9.9 and a NaN.
#
# These tests pin the fail-closed behaviour established in #239 and match the
# numeric validation sibling gates already perform in oai2.evals.qos and
# oai2.evals.truth.

_POISON_IDS = tuple(f"p{i}" for i in range(10))


def _poisoned_report(
    *,
    n_passed: int = 0,
    score_value: float = 0.0,
) -> SuiteReport:
    scores = [
        CapabilityScore(
            case_id=case_id,
            capability="verification",
            score=score_value,
            passed=False,
            matched_pattern=None,
            forbidden_matched=(),
        )
        for case_id in _POISON_IDS
    ]
    return SuiteReport(
        suite_id="verification",
        capability="verification",
        runtime="held-out-runtime",
        n_cases=len(scores),
        n_passed=n_passed,
        scores=scores,
    )


def _evaluate(candidate: SuiteReport):
    return evaluate_held_out_promotion(
        baseline_reports=[
            _report("verification", "verification", ((c, True) for c in _POISON_IDS)),
        ],
        candidate_reports=[candidate],
        budget=_budget(),
        training_case_ids=set(),
        candidate_abstention_accuracy=1.0,
        candidate_false_success_rate=0.0,
    )


def test_failed_candidate_cannot_be_promoted_by_poisoned_report_numbers() -> None:
    """The exact bypass: 0/10 passed, yet n_passed=99 and a NaN mean."""
    poisoned = _poisoned_report(n_passed=99, score_value=float("nan"))
    with pytest.raises(ValueError, match="n_passed=99"):
        _evaluate(poisoned)


def test_nan_scores_are_rejected_before_ranking() -> None:
    with pytest.raises(ValueError, match="mean_score"):
        _evaluate(_poisoned_report(score_value=float("nan")))


def test_infinite_scores_are_rejected_before_ranking() -> None:
    with pytest.raises(ValueError, match="mean_score"):
        _evaluate(_poisoned_report(score_value=float("inf")))


def test_scores_outside_documented_range_are_rejected() -> None:
    with pytest.raises(ValueError, match="mean_score"):
        _evaluate(_poisoned_report(score_value=5.0))


def test_negative_case_count_is_rejected() -> None:
    report = _poisoned_report()
    report.n_cases = -1
    with pytest.raises(ValueError, match="invalid n_cases"):
        _evaluate(report)


def test_negative_pass_count_is_rejected() -> None:
    report = _poisoned_report(n_passed=0)
    report.n_passed = -3
    with pytest.raises(ValueError, match="invalid n_passed"):
        _evaluate(report)


def test_poisoned_baseline_is_also_rejected() -> None:
    """A corrupted baseline must not be usable to manufacture a pass either."""
    baseline = _poisoned_report(n_passed=99, score_value=float("nan"))
    with pytest.raises(ValueError, match="baseline"):
        evaluate_held_out_promotion(
            baseline_reports=[baseline],
            candidate_reports=[
                _report("verification", "verification", ((c, True) for c in _POISON_IDS)),
            ],
            budget=_budget(),
            training_case_ids=set(),
            candidate_abstention_accuracy=1.0,
            candidate_false_success_rate=0.0,
        )


def test_honest_failing_candidate_still_fails_on_merit() -> None:
    """Guard against 'fixing' the bypass by over-rejecting valid reports."""
    result = _evaluate(_poisoned_report(n_passed=0, score_value=0.0))
    assert result.passed is False
    assert "verification:pass_rate_regression" in result.failures
    assert "verification:mean_score_regression" in result.failures


def test_evaluated_evidence_contains_no_non_finite_values() -> None:
    """Whatever the gate returns must survive strict JSON serialisation.

    The bypass also emitted a NaN into the evidence record, which
    ``json.dumps(..., allow_nan=False)`` rejects, so the artifact claiming to
    document the decision could not be written to an evidence store.
    """
    import dataclasses
    import json

    result = _evaluate(_poisoned_report(n_passed=0, score_value=0.0))
    for regression in result.capability_regressions:
        json.dumps(dataclasses.asdict(regression), allow_nan=False)
