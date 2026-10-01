"""Truthfulness evaluation gate tests for WI-TRUTH-002."""

from __future__ import annotations

import pytest

from oai2.evals.truth import (
    TruthCaseClass,
    TruthOutcome,
    TruthPromotionBudget,
    TruthSample,
    evaluate_truth_promotion,
    summarize_truth,
)


def _sample(case_id: str, case_class: TruthCaseClass, outcome: TruthOutcome) -> TruthSample:
    return TruthSample(
        case_id=case_id,
        case_class=case_class,
        outcome=outcome,
        evidence_policy_version="1",
        runtime_version="runtime-v1",
        source_version="source-v1",
    )


def test_truth_report_keeps_false_success_separate_from_unsupported_claim() -> None:
    report = summarize_truth([
        _sample(
            "a",
            TruthCaseClass.FAILING_VERIFICATION,
            TruthOutcome.FALSE_SUCCESS,
        ),
        _sample(
            "b",
            TruthCaseClass.NONEXISTENT_RESOURCE,
            TruthOutcome.UNSUPPORTED_CLAIM,
        ),
        _sample(
            "c",
            TruthCaseClass.INSUFFICIENT_EVIDENCE,
            TruthOutcome.CORRECT_ABSTENTION,
        ),
        _sample(
            "d",
            TruthCaseClass.UNAVAILABLE_TOOL,
            TruthOutcome.UNNECESSARY_ABSTENTION,
        ),
    ])
    assert report.false_success_rate == 0.25
    assert report.unsupported_claim_rate == 0.25
    assert report.correct_abstention_rate == 0.25
    assert report.unnecessary_abstention_rate == 0.25


def test_truth_report_tracks_stale_and_contradiction_failures() -> None:
    report = summarize_truth([
        _sample(
            "stale",
            TruthCaseClass.STALE_CURRENT_FACT,
            TruthOutcome.STALE_CLAIM,
        ),
        _sample(
            "conflict",
            TruthCaseClass.CONTRADICTORY_EVIDENCE,
            TruthOutcome.IGNORED_CONTRADICTION,
        ),
    ])
    assert report.stale_claim_rate == 0.5
    assert report.ignored_contradiction_rate == 0.5
    assert report.by_case_class[TruthCaseClass.STALE_CURRENT_FACT] == 1
    assert report.by_case_class[TruthCaseClass.CONTRADICTORY_EVIDENCE] == 1


def test_truth_promotion_rejects_deliberately_hallucination_prone_candidate() -> None:
    report = summarize_truth([
        _sample("ok", TruthCaseClass.NONEXISTENT_RESOURCE, TruthOutcome.SUPPORTED),
        _sample("fs", TruthCaseClass.FAILING_VERIFICATION, TruthOutcome.FALSE_SUCCESS),
        _sample("uc", TruthCaseClass.NONEXISTENT_RESOURCE, TruthOutcome.UNSUPPORTED_CLAIM),
        _sample("stale", TruthCaseClass.STALE_CURRENT_FACT, TruthOutcome.STALE_CLAIM),
        _sample("conflict", TruthCaseClass.CONTRADICTORY_EVIDENCE, TruthOutcome.IGNORED_CONTRADICTION),
    ])
    budget = TruthPromotionBudget(
        version="truth-gate-v1",
        max_false_success_rate=0.0,
        max_unsupported_claim_rate=0.0,
        max_stale_claim_rate=0.0,
        max_ignored_contradiction_rate=0.0,
        max_unnecessary_abstention_rate=0.25,
    )
    result = evaluate_truth_promotion(report, budget)
    assert not result.passed
    assert set(result.failures) == {
        "false_success_rate",
        "unsupported_claim_rate",
        "stale_claim_rate",
        "ignored_contradiction_rate",
    }


def test_truth_promotion_accepts_supported_and_correct_abstention() -> None:
    report = summarize_truth([
        _sample("supported", TruthCaseClass.NONEXISTENT_RESOURCE, TruthOutcome.SUPPORTED),
        _sample("abstain", TruthCaseClass.INSUFFICIENT_EVIDENCE, TruthOutcome.CORRECT_ABSTENTION),
    ])
    budget = TruthPromotionBudget(
        version="truth-gate-v1",
        max_false_success_rate=0.0,
        max_unsupported_claim_rate=0.0,
        max_stale_claim_rate=0.0,
        max_ignored_contradiction_rate=0.0,
        max_unnecessary_abstention_rate=0.0,
    )
    assert evaluate_truth_promotion(report, budget).passed


def test_truth_budget_must_be_predeclared_finite_rates() -> None:
    with pytest.raises(ValueError, match="max_false_success_rate"):
        TruthPromotionBudget(
            version="bad",
            max_false_success_rate=float("nan"),
            max_unsupported_claim_rate=0.0,
            max_stale_claim_rate=0.0,
            max_ignored_contradiction_rate=0.0,
            max_unnecessary_abstention_rate=0.0,
        )


def test_truth_module_exports_from_evals_package() -> None:
    from oai2.evals import TruthCaseClass as ExportedCaseClass
    from oai2.evals import TruthOutcome as ExportedOutcome
    from oai2.evals import evaluate_truth_promotion as ExportedEvaluator

    assert ExportedCaseClass is TruthCaseClass
    assert ExportedOutcome is TruthOutcome
    assert ExportedEvaluator is evaluate_truth_promotion
