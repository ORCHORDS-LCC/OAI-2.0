"""Held-out capability regression and promotion gates for WI-EVAL-002.

This layer compares baseline and candidate capability reports on the same
held-out case IDs, rejects train/held-out overlap, applies explicit tolerances
per capability class, and treats abstention/false-success as first-class gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import SuiteReport


@dataclass(slots=True, frozen=True)
class CapabilityRegressionThreshold:
    capability: str
    max_pass_rate_regression: float
    max_mean_score_regression: float

    def __post_init__(self) -> None:
        if not isinstance(self.capability, str) or not self.capability.strip():
            raise ValueError("capability must be a non-empty string")
        _rate(self.max_pass_rate_regression, "max_pass_rate_regression")
        _rate(self.max_mean_score_regression, "max_mean_score_regression")


@dataclass(slots=True, frozen=True)
class HeldOutPromotionBudget:
    version: str
    thresholds: tuple[CapabilityRegressionThreshold, ...]
    min_abstention_accuracy: float
    max_false_success_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if not self.thresholds:
            raise ValueError("at least one capability threshold is required")
        capabilities = [item.capability for item in self.thresholds]
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("capability thresholds must be unique")
        _rate(self.min_abstention_accuracy, "min_abstention_accuracy")
        _rate(self.max_false_success_rate, "max_false_success_rate")


@dataclass(slots=True, frozen=True)
class CapabilityRegression:
    capability: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_regression: float
    baseline_mean_score: float
    candidate_mean_score: float
    mean_score_regression: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class HeldOutPromotionEvaluation:
    budget_version: str
    passed: bool
    failures: tuple[str, ...]
    capability_regressions: tuple[CapabilityRegression, ...]


def evaluate_held_out_promotion(
    *,
    baseline_reports: tuple[SuiteReport, ...] | list[SuiteReport],
    candidate_reports: tuple[SuiteReport, ...] | list[SuiteReport],
    budget: HeldOutPromotionBudget,
    training_case_ids: set[str] | frozenset[str],
    candidate_abstention_accuracy: float,
    candidate_false_success_rate: float,
) -> HeldOutPromotionEvaluation:
    """Evaluate a candidate against baseline on isolated held-out cases."""
    abstention = _rate(candidate_abstention_accuracy, "candidate_abstention_accuracy")
    false_success = _rate(candidate_false_success_rate, "candidate_false_success_rate")

    baseline = _reports_by_capability(baseline_reports, "baseline")
    candidate = _reports_by_capability(candidate_reports, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate capabilities must match")

    baseline_case_ids = _case_ids(baseline_reports)
    candidate_case_ids = _case_ids(candidate_reports)
    if baseline_case_ids != candidate_case_ids:
        raise ValueError("baseline and candidate must use identical held-out case IDs")
    overlap = candidate_case_ids.intersection(training_case_ids)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ValueError(f"held-out cases overlap training/tuning inputs: {joined}")

    failures: list[str] = []
    regressions: list[CapabilityRegression] = []
    threshold_by_capability = {item.capability: item for item in budget.thresholds}

    missing_thresholds = set(candidate) - set(threshold_by_capability)
    if missing_thresholds:
        joined = ", ".join(sorted(missing_thresholds))
        raise ValueError(f"missing capability thresholds: {joined}")

    for capability in sorted(candidate):
        threshold = threshold_by_capability[capability]
        baseline_pass, baseline_mean = _aggregate_reports(baseline[capability])
        candidate_pass, candidate_mean = _aggregate_reports(candidate[capability])
        pass_regression = max(baseline_pass - candidate_pass, 0.0)
        mean_regression = max(baseline_mean - candidate_mean, 0.0)

        capability_failures: list[str] = []
        if pass_regression > threshold.max_pass_rate_regression:
            capability_failures.append("pass_rate_regression")
        if mean_regression > threshold.max_mean_score_regression:
            capability_failures.append("mean_score_regression")
        if capability_failures:
            failures.extend(
                f"{capability}:{failure}" for failure in capability_failures
            )

        regressions.append(
            CapabilityRegression(
                capability=capability,
                baseline_pass_rate=baseline_pass,
                candidate_pass_rate=candidate_pass,
                pass_rate_regression=pass_regression,
                baseline_mean_score=baseline_mean,
                candidate_mean_score=candidate_mean,
                mean_score_regression=mean_regression,
                passed=not capability_failures,
                failures=tuple(capability_failures),
            )
        )

    if abstention < budget.min_abstention_accuracy:
        failures.append("abstention_accuracy")
    if false_success > budget.max_false_success_rate:
        failures.append("false_success_rate")

    return HeldOutPromotionEvaluation(
        budget_version=budget.version,
        passed=not failures,
        failures=tuple(failures),
        capability_regressions=tuple(regressions),
    )


def _reports_by_capability(
    reports: tuple[SuiteReport, ...] | list[SuiteReport],
    name: str,
) -> dict[str, list[SuiteReport]]:
    if not reports:
        raise ValueError(f"{name}_reports must not be empty")
    out: dict[str, list[SuiteReport]] = {}
    for report in reports:
        out.setdefault(report.capability, []).append(report)
    return out


def _case_ids(reports: tuple[SuiteReport, ...] | list[SuiteReport]) -> set[str]:
    ids: set[str] = set()
    for report in reports:
        for score in report.scores:
            if score.case_id in ids:
                raise ValueError(f"duplicate held-out case ID: {score.case_id}")
            ids.add(score.case_id)
    if not ids:
        raise ValueError("held-out reports must contain per-case scores")
    return ids


def _aggregate_reports(reports: list[SuiteReport]) -> tuple[float, float]:
    total_cases = sum(report.n_cases for report in reports)
    if total_cases <= 0:
        raise ValueError("capability reports must contain cases")
    passed = sum(report.n_passed for report in reports)
    weighted_score = sum(report.mean_score * report.n_cases for report in reports)
    return passed / total_cases, weighted_score / total_cases


def _rate(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


__all__ = [
    "CapabilityRegressionThreshold",
    "HeldOutPromotionBudget",
    "CapabilityRegression",
    "HeldOutPromotionEvaluation",
    "evaluate_held_out_promotion",
]
