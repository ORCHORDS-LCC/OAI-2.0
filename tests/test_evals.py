"""Tests for the capability-eval scaffold."""

from __future__ import annotations

from oai2.evals import (
    BUILTIN_SUITES_NAMES,
    CapabilityScore,
    CapabilitySuite,
    builtin_suite,
    builtin_suites,
    run_suite,
)
from oai2.runtime import PlaceholderRuntime


def test_all_builtin_suites_construct() -> None:
    names = set(BUILTIN_SUITES_NAMES)
    assert {"coding", "tool_use", "bug_diagnosis", "reasoning",
            "verification", "vision", "orchestration"} <= names


def test_suite_has_cases() -> None:
    suite = builtin_suite("coding")
    assert isinstance(suite, CapabilitySuite)
    assert suite.suite_id == "coding_basic"
    assert len(suite.cases) >= 3
    for case in suite.cases:
        assert case.case_id
        assert case.prompt
        assert case.expected_patterns


def test_run_suite_against_placeholder_scores_zero_for_text_only() -> None:
    suite = builtin_suite("coding")
    rt = PlaceholderRuntime()
    report = run_suite(suite, rt)
    assert report.suite_id == "coding_basic"
    assert report.n_cases == len(suite.cases)
    assert report.runtime == "PlaceholderRuntime"
    # Placeholder never matches expected patterns, so all cases score 0.0.
    assert report.pass_rate == 0.0
    assert report.mean_score == 0.0
    for score in report.scores:
        assert isinstance(score, CapabilityScore)
        assert score.score == 0.0
        assert score.passed is False


def test_run_suite_rejects_unknown_scorer() -> None:
    bad = CapabilitySuite(
        suite_id="x",
        capability="x",
        description="x",
        cases=[],
        scorer="no_such_scorer",
    )
    try:
        run_suite(bad, PlaceholderRuntime())
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown scorer")


def test_builtin_suites_iterate() -> None:
    suites = list(builtin_suites())
    assert len(suites) == len(BUILTIN_SUITES_NAMES)
    for s in suites:
        assert s.suite_id
