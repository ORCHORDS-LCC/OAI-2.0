"""Tests for the declared NORMAL service budget and its evaluation.

The point of these tests is that the budget must be able to *fail*. A gate
that passes whatever it is given cannot gate anything, and the failure mode
that matters most here is a fast-but-wrong configuration being waved through
because every latency number is comfortably inside its limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from declare_normal_budget import (  # noqa: E402
    CONFIG_ID,
    NORMAL_BUDGET,
    TARGET_HARDWARE,
    build_samples,
)

from oai2.evals.qos import (  # noqa: E402
    BudgetKind,
    WorkloadClass,
    WorkloadSample,
    evaluate_budget,
    summarize_samples,
)


def _sample(**kw) -> WorkloadSample:
    base = {
        "workload": WorkloadClass.NORMAL,
        "target_hardware": TARGET_HARDWARE,
        "config_id": CONFIG_ID,
        "ttft_ms": 16.0,
        "first_useful_action_ms": 101.0,
        "end_to_end_ms": 600.0,
        "declared_success": True,
        "verified_success": True,
    }
    base.update(kw)
    return WorkloadSample(**base)  # type: ignore[arg-type]


def test_budget_is_a_service_budget_not_a_research_target() -> None:
    """A research target is an aspiration; only a service budget can gate."""
    assert NORMAL_BUDGET.kind is BudgetKind.SERVICE_BUDGET
    assert NORMAL_BUDGET.workload is WorkloadClass.NORMAL
    assert NORMAL_BUDGET.version.strip()
    assert NORMAL_BUDGET.config_id == CONFIG_ID


def test_correctness_limits_are_not_permissive() -> None:
    """A budget that tolerates wrong answers cannot enforce useful work."""
    assert NORMAL_BUDGET.min_verified_success_rate >= 0.80
    assert NORMAL_BUDGET.max_false_success_rate <= 0.05


def test_a_fully_correct_configuration_passes_the_budget() -> None:
    """The gate must be satisfiable, or it is not a gate."""
    samples = [_sample() for _ in range(20)]
    result = evaluate_budget(summarize_samples(samples), NORMAL_BUDGET)
    assert result.passed, result.failures


def test_a_fast_but_wrong_configuration_fails_the_budget() -> None:
    """The failure mode this whole budget exists to catch.

    Every latency field is comfortably inside its limit, so a gate that only
    watched latency would wave this through. Correctness is what rejects it.
    """
    samples = [_sample(verified_success=False) for _ in range(20)]
    report = summarize_samples(samples)
    result = evaluate_budget(report, NORMAL_BUDGET)

    # Latency is fine...
    assert report.first_useful_action_ms.p95 <= NORMAL_BUDGET.first_useful_action_p95_ms
    assert report.end_to_end_ms.p95 <= NORMAL_BUDGET.end_to_end_p95_ms
    # ...and the configuration is still rejected.
    assert not result.passed
    assert "verified_success_rate" in result.failures
    assert "false_success_rate" in result.failures
    # And the latency fields are not among the failures.
    assert "first_useful_action_p95" not in result.failures
    assert "end_to_end_p95" not in result.failures


def test_a_slow_but_correct_configuration_also_fails() -> None:
    """The gate must not be a correctness-only gate either."""
    samples = [_sample(first_useful_action_ms=5000.0, end_to_end_ms=9000.0) for _ in range(20)]
    result = evaluate_budget(summarize_samples(samples), NORMAL_BUDGET)
    assert not result.passed
    assert "first_useful_action_p95" in result.failures


def test_the_measured_evidence_fails_on_correctness_not_latency() -> None:
    """The committed measurement must fail, and for the right reason.

    This pins the headline result: the NORMAL lane meets every latency
    budget and misses every correctness budget. If a later change made the
    measured evidence pass, that would mean the measurement changed — not
    that the budget became stricter.
    """
    samples, censored = build_samples(cap_seconds=6.0)
    assert len(samples) == 12
    assert len(censored) == 8

    report = summarize_samples(samples)
    result = evaluate_budget(report, NORMAL_BUDGET)

    assert not result.passed
    assert set(result.failures) == {"verified_success_rate", "false_success_rate"}
    # Latency is inside budget even though correctness is not.
    assert set(result.failures).isdisjoint(
        {"first_useful_action_p95", "end_to_end_p95", "end_to_end_p99"}
    )
    assert report.verified_success_rate == pytest.approx(4 / 12, abs=1e-9)


def test_never_solved_cases_are_recorded_as_false_success_not_dropped() -> None:
    """Dropping the 8 unsolved cases would report a 100% success rate.

    That is the single easiest way to make this suite look good, so it is
    pinned: every case is present, unsolved ones are false successes, and
    they are flagged as censored lower bounds rather than given an invented
    time.
    """
    samples, censored = build_samples(cap_seconds=6.0)
    assert len(samples) == 12
    assert all(s.declared_success for s in samples)
    false_successes = [s for s in samples if s.false_success]
    assert len(false_successes) == 8
    assert all(s.deadline_missed for s in false_successes)
    # Censored entries are marked as lower bounds, not presented as exact,
    # and there is exactly one per false success.
    assert all(c["first_useful_action_is_lower_bound"] for c in censored)
    assert all(c["case_id"] for c in censored)
    assert len(censored) == len(false_successes) == 8
