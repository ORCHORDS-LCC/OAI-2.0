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
    assert {
        "coding",
        "tool_use",
        "bug_diagnosis",
        "reasoning",
        "verification",
        "vision",
        "orchestration",
        "multi_file_reasoning",
        "abstention",
        "conflicting_evidence",
    } <= names


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


def test_evals_package_full_surface_identity() -> None:
    """Every name in ``oai2/evals/__init__.py`` ``__all__`` is importable
    from ``oai2.evals`` and aliases its source-of-truth.

    Closes a coverage gap where ``oai2.evals`` declared 17 ``__all__``
    entries (9 capability-eval scaffold symbols defined inline plus 8
    truth re-exports from ``oai2.evals.truth``) but the only sibling
    identity test
    (``tests/test_truth_evals.py::test_truth_module_exports_from_evals_package``)
    pinned the 8 truth re-exports only. The 9 capability-eval symbols
    ``BUILTIN_SUITES_NAMES``, ``CapabilityCase``, ``CapabilityScore``,
    ``CapabilitySuite``, ``SCORERS``, ``SuiteReport``, ``builtin_suite``,
    ``builtin_suites``, ``run_suite`` were not pinned at the package
    surface, leaving a 53 percent coverage gap on the package public
    contract.

    The test imports all 17 names from ``oai2.evals`` (the import itself
    fails if any declared ``__all__`` entry is missing or mis-named), then
    asserts the package-surface reference is identical to the canonical
    source-of-truth (the same ``oai2.evals.__init__`` for capability-eval
    inline symbols, or ``oai2.evals.truth`` for truth re-exports).
    """
    import oai2.evals as evals_pkg

    from oai2.evals import (
        # Capability-eval (9): defined inline in oai2/evals/__init__.py.
        BUILTIN_SUITES_NAMES as _PkgBuiltInSuitesNames,
        CapabilityCase as _PkgCapabilityCase,
        CapabilityScore as _PkgCapabilityScore,
        CapabilitySuite as _PkgCapabilitySuite,
        SCORERS as _PkgScorers,
        SuiteReport as _PkgSuiteReport,
        builtin_suite as _PkgBuiltinSuite,
        builtin_suites as _PkgBuiltinSuites,
        run_suite as _PkgRunSuite,
        # Truth re-exports (8): sourced from oai2.evals.truth.
        TruthCaseClass as _PkgTruthCaseClass,
        TruthOutcome as _PkgTruthOutcome,
        TruthPromotionBudget as _PkgTruthPromotionBudget,
        TruthPromotionEvaluation as _PkgTruthPromotionEvaluation,
        TruthReport as _PkgTruthReport,
        TruthSample as _PkgTruthSample,
        evaluate_truth_promotion as _PkgEvaluateTruthPromotion,
        summarize_truth as _PkgSummarizeTruth,
    )
    from oai2.evals.truth import (
        TruthCaseClass as _SrcTruthCaseClass,
        TruthOutcome as _SrcTruthOutcome,
        TruthPromotionBudget as _SrcTruthPromotionBudget,
        TruthPromotionEvaluation as _SrcTruthPromotionEvaluation,
        TruthReport as _SrcTruthReport,
        TruthSample as _SrcTruthSample,
        evaluate_truth_promotion as _SrcEvaluateTruthPromotion,
        summarize_truth as _SrcSummarizeTruth,
    )

    # Capability-eval symbols: the package-surface name is the same object
    # as the locally-imported name (both resolve to the binding in
    # ``oai2/evals/__init__.py``).
    assert evals_pkg.BUILTIN_SUITES_NAMES is _PkgBuiltInSuitesNames
    assert evals_pkg.CapabilityCase is _PkgCapabilityCase
    assert evals_pkg.CapabilityScore is _PkgCapabilityScore
    assert evals_pkg.CapabilitySuite is _PkgCapabilitySuite
    assert evals_pkg.SCORERS is _PkgScorers
    assert evals_pkg.SuiteReport is _PkgSuiteReport
    assert evals_pkg.builtin_suite is _PkgBuiltinSuite
    assert evals_pkg.builtin_suites is _PkgBuiltinSuites
    assert evals_pkg.run_suite is _PkgRunSuite

    # Truth re-exports: the package-surface name must be the same object
    # as the source-of-truth defined in ``oai2.evals.truth``.
    assert evals_pkg.TruthCaseClass is _SrcTruthCaseClass
    assert evals_pkg.TruthOutcome is _SrcTruthOutcome
    assert evals_pkg.TruthPromotionBudget is _SrcTruthPromotionBudget
    assert evals_pkg.TruthPromotionEvaluation is _SrcTruthPromotionEvaluation
    assert evals_pkg.TruthReport is _SrcTruthReport
    assert evals_pkg.TruthSample is _SrcTruthSample
    assert evals_pkg.evaluate_truth_promotion is _SrcEvaluateTruthPromotion
    assert evals_pkg.summarize_truth is _SrcSummarizeTruth


def test_every_builtin_case_declares_test_procedure_and_evidence() -> None:
    for suite in builtin_suites():
        for case in suite.cases:
            assert case.preconditions
            assert case.procedure
            assert case.expected_evidence
            assert all(item.strip() for item in case.preconditions)
            assert all(item.strip() for item in case.procedure)
            assert all(item.strip() for item in case.expected_evidence)


def test_missing_required_capability_classes_are_machine_verifiable() -> None:
    assert builtin_suite("multi_file_reasoning").scorer == "regex_all"
    assert builtin_suite("abstention").scorer == "regex_all"
    assert builtin_suite("conflicting_evidence").scorer == "regex_all"
    assert builtin_suite("multi_file_reasoning").cases[0].expected_patterns
    assert builtin_suite("abstention").cases[0].forbidden_patterns
    assert len(builtin_suite("conflicting_evidence").cases[0].expected_patterns) >= 2


def test_regex_all_requires_every_expected_pattern() -> None:
    from types import SimpleNamespace

    from oai2.evals import CapabilityCase, SCORERS

    case = CapabilityCase(
        case_id="all-patterns",
        capability="verification",
        prompt="p",
        expected_patterns=(r"alpha", r"beta"),
    )
    scorer = SCORERS["regex_all"]

    passed = scorer(case, SimpleNamespace(text="alpha beta"))  # type: ignore[arg-type]
    assert passed.passed is True
    assert passed.score == 1.0

    missing = scorer(case, SimpleNamespace(text="alpha"))  # type: ignore[arg-type]
    assert missing.passed is False
    assert missing.notes == "missing-required-pattern"


def test_regex_all_honors_forbidden_patterns() -> None:
    from types import SimpleNamespace

    from oai2.evals import CapabilityCase, SCORERS

    case = CapabilityCase(
        case_id="forbidden",
        capability="abstention",
        prompt="p",
        expected_patterns=(r"not verified",),
        forbidden_patterns=(r"verified working",),
    )
    score = SCORERS["regex_all"](
        case,
        SimpleNamespace(text="not verified; verified working"),  # type: ignore[arg-type]
    )
    assert score.passed is False
    assert score.notes == "forbidden-match"


def test_all_declared_evals_exports_are_importable() -> None:
    import oai2.evals as evals_pkg

    for name in evals_pkg.__all__:
        assert hasattr(evals_pkg, name), name
