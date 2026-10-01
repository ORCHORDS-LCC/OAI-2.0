"""OAI-2.0 capability-eval scaffold.

Public surface
--------------

- :class:`CapabilityCase` — a single prompt + expected-output pattern.
- :class:`CapabilitySuite` — a named set of cases for one capability.
- :class:`CapabilityScore` — the score returned by a scorer for a case.
- :class:`SuiteReport` — aggregated results for a whole suite.
- :func:`run_suite` — run a suite against an :class:`InferenceRuntime`
  and produce a :class:`SuiteReport`.
- :func:`builtin_suites` — the canonical offline suites that come with
: :mod:`oai2`.

The harness is **model-agnostic** and **runtime-agnostic**. It runs the
same cases against the :class:`PlaceholderRuntime` in CI and against a
real MLX runtime on a developer's machine. No live model is required
to use it offline; the offline scorer validates that cases are well
formed and that the placeholder produces parseable, non-trivial output.

When wired to a real :class:`InferenceRuntime`, the same harness reports
per-case and aggregate scores for capability tracking over time.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..runtime.inference import InferenceRequest, InferenceResponse, InferenceRuntime
from .truth import (
    TruthCaseClass,
    TruthOutcome,
    TruthPromotionBudget,
    TruthPromotionEvaluation,
    TruthCandidatePromotionEvaluation,
    TruthReport,
    TruthSample,
    evaluate_truth_candidate_promotion,
    evaluate_truth_promotion,
    summarize_truth,
)
from .truth_runner import (
    CandidateRunner,
    CandidateTruthInput,
    CandidateTruthResponse,
    HiddenVerifier,
    TruthCase,
    TruthRunResult,
    TruthVerifierVerdict,
    classify_truth_outcome,
    run_held_out_truth_cases,
)


class Scorer(Protocol):
    """A pure function from (case, response) to :class:`CapabilityScore`."""

    def __call__(
        self, case: CapabilityCase, response: InferenceResponse
    ) -> CapabilityScore: ...


@dataclass(slots=True, frozen=True)
class CapabilityCase:
    """One capability eval prompt + expected patterns.

    Multiple ``expected_patterns`` are OR-combined: the case passes if
    any of them appears in the response text.
    """

    case_id: str
    capability: str  # e.g. "coding", "tool_use".
    prompt: str
    expected_patterns: tuple[str, ...] = ()
    # Optional forbidden patterns — case fails if any appear.
    forbidden_patterns: tuple[str, ...] = ()
    # Optional structural check (run before regex checks).
    must_contain_action_token: str | None = None  # e.g. "<READ F:".
    tags: tuple[str, ...] = ()
    notes: str = ""


class CapabilitySuite(BaseModel):
    """A named bundle of capability cases."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    capability: str
    description: str
    cases: list[CapabilityCase] = Field(default_factory=list)
    # Scorer name (resolved against the SCORERS registry).
    scorer: str = "regex_or"


@dataclass(slots=True, frozen=True)
class CapabilityScore:
    case_id: str
    capability: str
    score: float  # 0.0 .. 1.0
    passed: bool
    matched_pattern: str | None
    forbidden_matched: tuple[str, ...]
    notes: str = ""


@dataclass(slots=True)
class SuiteReport:
    suite_id: str
    capability: str
    runtime: str
    n_cases: int
    n_passed: int
    scores: list[CapabilityScore] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.n_cases == 0 else self.n_passed / self.n_cases

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "capability": self.capability,
            "runtime": self.runtime,
            "n_cases": self.n_cases,
            "n_passed": self.n_passed,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "scores": [
                {
                    "case_id": s.case_id,
                    "capability": s.capability,
                    "score": s.score,
                    "passed": s.passed,
                    "matched_pattern": s.matched_pattern,
                    "forbidden_matched": list(s.forbidden_matched),
                    "notes": s.notes,
                }
                for s in self.scores
            ],
        }


# --- Built-in scorers --------------------------------------------------------


def _regex_or_scorer(case: CapabilityCase, response: InferenceResponse) -> CapabilityScore:
    """Pass if any ``expected_pattern`` appears in response text.

    A response of "OK" or empty text is treated as 0.0 — the placeholder
    cannot actually answer these. Real MLX runs are expected to score
    >0.5 once the model is wired up.
    """
    text = response.text or ""
    matched: str | None = None
    forbidden = tuple(p for p in case.forbidden_patterns if re.search(p, text))
    if not text.strip():
        return CapabilityScore(
            case_id=case.case_id,
            capability=case.capability,
            score=0.0,
            passed=False,
            matched_pattern=None,
            forbidden_matched=forbidden,
            notes="empty response",
        )
    for pat in case.expected_patterns:
        if re.search(pat, text):
            matched = pat
            break
    # Structural check: action-token presence.
    token_ok = True
    if case.must_contain_action_token is not None:
        token_ok = case.must_contain_action_token in text
    passed = matched is not None and not forbidden and token_ok
    return CapabilityScore(
        case_id=case.case_id,
        capability=case.capability,
        score=1.0 if passed else 0.0,
        passed=passed,
        matched_pattern=matched,
        forbidden_matched=forbidden,
        notes="" if passed else (
            "forbidden-match" if forbidden else
            "missing-action-token" if not token_ok else
            "no-pattern-match"
        ),
    )


SCORERS: dict[str, Callable[[CapabilityCase, InferenceResponse], CapabilityScore]] = {
    "regex_or": _regex_or_scorer,
}


# --- Suite runner ------------------------------------------------------------


def run_suite(
    suite: CapabilitySuite,
    runtime: InferenceRuntime,
    *,
    request_factory: Callable[[CapabilityCase], InferenceRequest] | None = None,
    max_tokens: int = 256,
) -> SuiteReport:
    """Run a suite against an :class:`InferenceRuntime` and score each case."""
    if suite.scorer not in SCORERS:
        raise KeyError(f"unknown scorer: {suite.scorer}")
    scorer = SCORERS[suite.scorer]
    factory = request_factory or (
        lambda case: InferenceRequest(prompt=case.prompt, max_tokens=max_tokens)
    )
    report = SuiteReport(
        suite_id=suite.suite_id,
        capability=suite.capability,
        runtime=type(runtime).__name__,
        n_cases=len(suite.cases),
        n_passed=0,
    )
    for case in suite.cases:
        response = runtime.generate(factory(case))
        score = scorer(case, response)
        report.scores.append(score)
        if score.passed:
            report.n_passed += 1
    return report


# --- Built-in suites ---------------------------------------------------------


def _coding_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="sum-n-natural",
            capability="coding",
            prompt=(
                "Write a Python function `sum_n(n: int) -> int` that returns "
                "the sum of the first n natural numbers. Include a short "
                "docstring. Do not use any imports."
            ),
            expected_patterns=(
                r"\bdef\s+sum_n\s*\(\s*n\s*:\s*int",
                r"return\s+n\s*\*\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2",
                r"return\s+sum\s*\(",
            ),
            tags=("python", "numeric"),
        ),
        CapabilityCase(
            case_id="factorial-iterative",
            capability="coding",
            prompt=(
                "Write a Python function `factorial(n: int) -> int` that "
                "computes n! iteratively. Include a short docstring."
            ),
            expected_patterns=(
                r"\bdef\s+factorial\s*\(\s*n\s*:\s*int",
                r"return\s+1",
            ),
            tags=("python", "numeric"),
        ),
        CapabilityCase(
            case_id="fizzbuzz",
            capability="coding",
            prompt=(
                "Write a Python function `fizzbuzz(n: int) -> list[str]` "
                "that returns the fizzbuzz sequence up to n. Include a "
                "short docstring."
            ),
            expected_patterns=(
                r"\bdef\s+fizzbuzz\s*\(",
                r"\b3\b",
                r"\b5\b",
            ),
            tags=("python", "string"),
        ),
    ]
    return CapabilitySuite(
        suite_id="coding_basic",
        capability="coding",
        description="Small Python functions: arithmetic, list ops.",
        cases=cases,
    )


def _tool_use_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="emit-read-token",
            capability="tool_use",
            prompt=(
                "To read the file `src/foo.py`, emit a single READ action "
                "token in the form `<READ F:src/foo.py>`."
            ),
            expected_patterns=(r"<READ\s+F:src/foo\.py>",),
            must_contain_action_token="<READ F:src/foo.py>",
        ),
        CapabilityCase(
            case_id="emit-test-token",
            capability="tool_use",
            prompt=(
                "To run the test selector `test_login`, emit a single TEST "
                "action token in the form `<TEST T:test_login>`."
            ),
            expected_patterns=(r"<TEST\s+T:test_login>",),
            must_contain_action_token="<TEST T:test_login>",
        ),
        CapabilityCase(
            case_id="emit-image-token",
            capability="tool_use",
            prompt=(
                "To reference screenshot `home_view`, emit an IMAGE action "
                "token in the form `<IMAGE I:home_view>`."
            ),
            expected_patterns=(r"<IMAGE\s+I:home_view>",),
            must_contain_action_token="<IMAGE I:home_view>",
        ),
    ]
    return CapabilitySuite(
        suite_id="tool_use_action_tokens",
        capability="tool_use",
        description=(
            "Emit correctly-formed compact action tokens for READ, TEST, "
            "IMAGE."
        ),
        cases=cases,
    )


def _bug_diagnosis_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="off-by-one",
            capability="bug_diagnosis",
            prompt=(
                "The following Python function should sum elements of a list "
                "but returns 0 for non-empty inputs. Diagnose the bug.\n\n"
                "```python\ndef total(xs):\n    s = 0\n"
                "    for i in range(len(xs) - 1):\n"
                "        s += xs[i]\n    return s\n```"
            ),
            expected_patterns=(
                r"range\s*\(\s*len\s*\(\s*xs\s*\),\s*\)",
                r"range\s*\(\s*len\s*\(\s*xs\s*\)\s*-\s*1\s*\)",
                r"off[- ]by[- ]one",
                r"last element",
            ),
            tags=("python",),
        ),
        CapabilityCase(
            case_id="mutable-default",
            capability="bug_diagnosis",
            prompt=(
                "Diagnose the bug in this Python function:\n\n"
                "```python\ndef append_to(item, lst=[]):\n"
                "    lst.append(item)\n    return lst\n```"
            ),
            expected_patterns=(
                r"mutable default",
                r"None",
                r"\blst\s*=\s*None\b",
            ),
            tags=("python",),
        ),
    ]
    return CapabilitySuite(
        suite_id="bug_diagnosis_basic",
        capability="bug_diagnosis",
        description="Diagnose common Python bugs.",
        cases=cases,
    )


def _reasoning_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="arithmetic-3digit",
            capability="reasoning",
            prompt="Compute 123 * 456 and return only the integer answer.",
            expected_patterns=(r"\b56088\b",),
            tags=("arithmetic",),
        ),
        CapabilityCase(
            case_id="modular-arithmetic",
            capability="reasoning",
            prompt="Compute (17 ** 5) % 23 and return only the integer answer.",
            expected_patterns=(r"\b\d+\b",),
            tags=("arithmetic",),
        ),
    ]
    return CapabilitySuite(
        suite_id="reasoning_basic",
        capability="reasoning",
        description="Numeric reasoning.",
        cases=cases,
    )


def _verification_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="evidence-class",
            capability="verification",
            prompt=(
                "A claim is supported by (a) a literal string from a source "
                "file the agent read, (b) a captured log line, (c) a "
                "human-supplied spec excerpt. Identify which evidence class "
                "each item is from these classes: USER_INTENT, REPO_SOURCE, "
                "CHANGE_HISTORY, DETERMINISTIC, RUNTIME_OBS, VISUAL_OBS, "
                "EXTERNAL, HYPOTHESIS. Reply with one class per line."
            ),
            expected_patterns=(
                r"REPO_SOURCE",
                r"RUNTIME_OBS",
                r"USER_INTENT",
            ),
            tags=("taxonomy",),
        ),
    ]
    return CapabilitySuite(
        suite_id="verification_taxonomy",
        capability="verification",
        description="Evidence-class taxonomy recall.",
        cases=cases,
    )


def _vision_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="describe-structure",
            capability="vision",
            prompt=(
                "Describe the visual structure of a screenshot: header bar "
                "with title, two-column body, footer with two buttons."
            ),
            expected_patterns=(
                r"header",
                r"footer",
                r"two columns|two-column",
            ),
            tags=("ui",),
        ),
    ]
    return CapabilitySuite(
        suite_id="vision_basic",
        capability="vision",
        description="Visual structure recall.",
        cases=cases,
    )


def _orchestration_suite() -> CapabilitySuite:
    cases = [
        CapabilityCase(
            case_id="plan-decompose",
            capability="orchestration",
            prompt=(
                "Decompose this task into ordered steps: 'Add a `/healthz` "
                "endpoint to a FastAPI app, including a test.' Provide a "
                "numbered list with at least three steps."
            ),
            expected_patterns=(
                r"\b1[\.\),]\s",
                r"\b2[\.\)]\s",
                r"\b3[\.\)]\s",
                r"healthz|/healthz",
                r"test|pytest",
            ),
            tags=("planning",),
        ),
    ]
    return CapabilitySuite(
        suite_id="orchestration_basic",
        capability="orchestration",
        description="Plan decomposition.",
        cases=cases,
    )


_BUILTIN_SUITES: dict[str, Callable[[], CapabilitySuite]] = {
    "coding": _coding_suite,
    "tool_use": _tool_use_suite,
    "bug_diagnosis": _bug_diagnosis_suite,
    "reasoning": _reasoning_suite,
    "verification": _verification_suite,
    "vision": _vision_suite,
    "orchestration": _orchestration_suite,
}


def builtin_suite(name: str) -> CapabilitySuite:
    if name not in _BUILTIN_SUITES:
        raise KeyError(f"unknown builtin suite: {name}")
    return _BUILTIN_SUITES[name]()


def builtin_suites() -> Iterable[CapabilitySuite]:
    return (factory() for factory in _BUILTIN_SUITES.values())


__all__ = [
    "BUILTIN_SUITES_NAMES",
    "CapabilityCase",
    "CapabilityScore",
    "CapabilitySuite",
    "SCORERS",
    "SuiteReport",
    "builtin_suite",
    "builtin_suites",
    "run_suite",
    "TruthCaseClass",
    "TruthOutcome",
    "TruthPromotionBudget",
    "TruthPromotionEvaluation",
    "TruthCandidatePromotionEvaluation",
    "TruthReport",
    "TruthSample",
    "evaluate_truth_candidate_promotion",
    "evaluate_truth_promotion",
    "summarize_truth",
    "TruthCase",
    "CandidateTruthInput",
    "CandidateTruthResponse",
    "TruthVerifierVerdict",
    "CandidateRunner",
    "HiddenVerifier",
    "TruthRunResult",
    "run_held_out_truth_cases",
    "classify_truth_outcome",
]


BUILTIN_SUITES_NAMES = tuple(_BUILTIN_SUITES.keys())
