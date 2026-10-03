from __future__ import annotations

import pytest

from oai2.evals.qos import (
    BudgetKind,
    WorkloadBudget,
    WorkloadClass,
    WorkloadSample,
    evaluate_budget,
    evaluate_promotion,
    summarize_samples,
    useful_work_rank_key,
)


def _sample(
    *,
    ttft: float,
    useful: float,
    total: float,
    declared: bool = True,
    verified: bool = True,
    actions: int = 1,
    tps: float = 100.0,
    tool_ms: float = 0.0,
    retrieval_ms: float = 0.0,
    vision_ms: float = 0.0,
    build_test_ms: float = 0.0,
    deadline_missed: bool = False,
) -> WorkloadSample:
    return WorkloadSample(
        workload=WorkloadClass.NORMAL,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        ttft_ms=ttft,
        first_useful_action_ms=useful,
        end_to_end_ms=total,
        declared_success=declared,
        verified_success=verified,
        verified_actions=actions,
        generated_tokens=32,
        decode_tokens_per_second=tps,
        tool_ms=tool_ms,
        retrieval_ms=retrieval_ms,
        vision_ms=vision_ms,
        build_test_ms=build_test_ms,
        deadline_missed=deadline_missed,
    )


def test_canonical_workload_classes_cover_required_modes() -> None:
    assert {member.value for member in WorkloadClass} == {
        "FURIOUS",
        "NORMAL",
        "DEEP",
        "SWARM",
        "VISION",
        "RETRIEVAL_HEAVY",
    }


def test_summary_separates_ttft_from_first_useful_action_and_tail() -> None:
    report = summarize_samples(
        [
            _sample(ttft=10, useful=200, total=400, tps=300),
            _sample(ttft=12, useful=220, total=420, tps=310),
            _sample(ttft=11, useful=240, total=450, tps=320),
            _sample(ttft=9, useful=260, total=480, tps=330),
        ]
    )
    assert report.ttft_ms.p95 < 13
    assert report.first_useful_action_ms.p95 > 250
    assert report.end_to_end_ms.p99 > report.end_to_end_ms.p95
    assert report.decode_tokens_per_second.mean == pytest.approx(315.0)
    assert report.verified_success_rate == 1.0


def test_summary_reports_component_latency_decomposition() -> None:
    report = summarize_samples(
        [
            _sample(
                ttft=10,
                useful=100,
                total=400,
                tool_ms=40,
                retrieval_ms=80,
                vision_ms=0,
                build_test_ms=120,
            ),
            _sample(
                ttft=12,
                useful=120,
                total=500,
                tool_ms=60,
                retrieval_ms=120,
                vision_ms=20,
                build_test_ms=180,
            ),
        ]
    )
    assert report.tool_ms.mean == pytest.approx(50.0)
    assert report.retrieval_ms.mean == pytest.approx(100.0)
    assert report.vision_ms.mean == pytest.approx(10.0)
    assert report.build_test_ms.mean == pytest.approx(150.0)
    assert report.tool_ms.count == 2
    assert report.build_test_ms.p95 > 170


def test_false_success_and_verified_useful_work_are_reported() -> None:
    report = summarize_samples(
        [
            _sample(ttft=5, useful=20, total=100, declared=True, verified=False, actions=0),
            _sample(ttft=15, useful=25, total=100, declared=True, verified=True, actions=2),
        ]
    )
    assert report.false_success_rate == 0.5
    assert report.verified_success_rate == 0.5
    assert report.verified_actions_per_second == pytest.approx(10.0)


def test_useful_work_rank_penalizes_fast_talking_slow_action() -> None:
    fast_talking = summarize_samples([_sample(ttft=5, useful=500, total=700, tps=500)])
    useful_sooner = summarize_samples([_sample(ttft=30, useful=100, total=300, tps=100)])
    assert fast_talking.ttft_ms.p50 < useful_sooner.ttft_ms.p50
    assert useful_work_rank_key(useful_sooner) < useful_work_rank_key(fast_talking)


def test_budget_evaluation_is_versioned_and_kind_is_explicit() -> None:
    report = summarize_samples(
        [
            _sample(ttft=10, useful=100, total=200),
            _sample(ttft=10, useful=120, total=240, declared=True, verified=False),
        ]
    )
    budget = WorkloadBudget(
        version="normal-service-v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.SERVICE_BUDGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=150,
        end_to_end_p95_ms=300,
        end_to_end_p99_ms=350,
        max_false_success_rate=0.1,
        min_verified_success_rate=0.9,
    )
    result = evaluate_budget(report, budget)
    assert result.budget_version == "normal-service-v1"
    assert result.budget_kind is BudgetKind.SERVICE_BUDGET
    assert result.passed is False
    assert result.failures == ("false_success_rate", "verified_success_rate")

    research = WorkloadBudget(
        version="normal-research-v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.RESEARCH_TARGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=150,
        end_to_end_p95_ms=300,
        end_to_end_p99_ms=350,
        max_false_success_rate=0.5,
        min_verified_success_rate=0.5,
    )
    assert evaluate_budget(report, research).budget_kind is BudgetKind.RESEARCH_TARGET


def test_mixed_identity_and_invalid_budget_fail_closed() -> None:
    a = _sample(ttft=10, useful=20, total=30)
    b = WorkloadSample(
        workload=WorkloadClass.DEEP,
        target_hardware="mac-studio-m5",
        config_id="deep-v1",
        ttft_ms=10,
        first_useful_action_ms=20,
        end_to_end_ms=30,
        declared_success=True,
        verified_success=True,
    )
    with pytest.raises(ValueError, match="all samples must share"):
        summarize_samples([a, b])

    with pytest.raises(ValueError, match="p99"):
        WorkloadBudget(
            version="bad",
            workload=WorkloadClass.NORMAL,
            kind=BudgetKind.SERVICE_BUDGET,
            target_hardware="mac-studio-m5",
            config_id="normal-v1",
            first_useful_action_p95_ms=100,
            end_to_end_p95_ms=200,
            end_to_end_p99_ms=100,
            max_false_success_rate=0.1,
            min_verified_success_rate=0.9,
        )



def test_deadline_miss_rate_is_reported_and_budgeted() -> None:
    report = summarize_samples(
        [
            _sample(ttft=10, useful=50, total=100, deadline_missed=False),
            _sample(ttft=10, useful=60, total=120, deadline_missed=True),
        ]
    )
    assert report.deadline_miss_rate == 0.5

    budget = WorkloadBudget(
        version="deadline-v1",
        workload=WorkloadClass.NORMAL,
        kind=BudgetKind.SERVICE_BUDGET,
        target_hardware="mac-studio-m5",
        config_id="normal-v1",
        first_useful_action_p95_ms=200,
        end_to_end_p95_ms=300,
        end_to_end_p99_ms=350,
        max_false_success_rate=1.0,
        min_verified_success_rate=0.0,
        max_deadline_miss_rate=0.25,
    )
    result = evaluate_budget(report, budget)
    assert result.passed is False
    assert "deadline_miss_rate" in result.failures


def test_promotion_rejects_faster_mean_when_tail_regresses() -> None:
    baseline = summarize_samples(
        [
            _sample(ttft=20, useful=80, total=100, tps=100),
            _sample(ttft=20, useful=90, total=110, tps=100),
            _sample(ttft=20, useful=100, total=120, tps=100),
            _sample(ttft=20, useful=100, total=130, tps=100),
        ]
    )
    candidate = summarize_samples(
        [
            _sample(ttft=5, useful=50, total=60, tps=500),
            _sample(ttft=5, useful=50, total=60, tps=500),
            _sample(ttft=5, useful=50, total=60, tps=500),
            _sample(ttft=5, useful=50, total=250, tps=500),
        ]
    )
    assert candidate.decode_tokens_per_second.mean > baseline.decode_tokens_per_second.mean
    result = evaluate_promotion(candidate, baseline)
    assert result.passed is False
    assert "p95_regression" in result.failures or "p99_regression" in result.failures


def test_promotion_rejects_deadline_miss_regression() -> None:
    baseline = summarize_samples(
        [
            _sample(ttft=10, useful=50, total=100, deadline_missed=False),
            _sample(ttft=10, useful=50, total=100, deadline_missed=False),
        ]
    )
    candidate = summarize_samples(
        [
            _sample(ttft=10, useful=50, total=90, deadline_missed=False),
            _sample(ttft=10, useful=50, total=90, deadline_missed=True),
        ]
    )
    result = evaluate_promotion(candidate, baseline)
    assert result.passed is False
    assert result.failures == ("deadline_miss_regression",)


class TestPromotionGatesCorrectnessNotJustLatency:
    """`evaluate_promotion` decides promotion, so it must see correctness.

    Before this, the gate checked p95, p99 and deadline misses only, while
    its sibling `evaluate_budget` checked `false_success_rate` and
    `verified_success_rate`. A candidate at identical latency that is never
    verified-correct and reports a false success every time was promoted:

        evaluate_promotion -> passed=True  failures=()
        evaluate_budget    -> passed=False failures=('false_success_rate',
                                                     'verified_success_rate')

    The gate that makes the promotion decision was the one gate that could
    not see it.
    """

    @staticmethod
    def _good() -> list[WorkloadSample]:
        return [
            _sample(ttft=10, useful=50, total=100, declared=True, verified=True),
            _sample(ttft=10, useful=50, total=100, declared=True, verified=True),
        ]

    def test_promotion_rejects_a_false_success_regression(self) -> None:
        """Same latency, but every answer is a declared, unverified success."""
        baseline = summarize_samples(self._good())
        candidate = summarize_samples(
            [
                _sample(ttft=10, useful=50, total=100, declared=True, verified=False),
                _sample(ttft=10, useful=50, total=100, declared=True, verified=False),
            ]
        )
        assert candidate.end_to_end_ms.p95 == baseline.end_to_end_ms.p95
        assert candidate.false_success_rate == 1.0
        result = evaluate_promotion(candidate, baseline)
        assert result.passed is False
        assert "false_success_regression" in result.failures

    def test_promotion_rejects_a_verified_success_regression(self) -> None:
        """A candidate that stops producing verified work at all.

        Uses `declared=False` so the false-success rate stays at 0.0 and this
        isolates the verified-success check from the one above.
        """
        baseline = summarize_samples(self._good())
        candidate = summarize_samples(
            [
                _sample(ttft=10, useful=50, total=100, declared=False, verified=False),
                _sample(ttft=10, useful=50, total=100, declared=False, verified=False),
            ]
        )
        assert candidate.false_success_rate == 0.0
        assert candidate.verified_success_rate == 0.0
        result = evaluate_promotion(candidate, baseline)
        assert result.passed is False
        assert "verified_success_regression" in result.failures

    def test_opposite_direction_equal_correctness_still_promotes(self) -> None:
        """Guard: a purely faster candidate is unaffected.

        A correctness check that fired on anything other than an actual
        regression would block every optimisation, which is how a safety gate
        becomes an obstacle and gets disabled.
        """
        baseline = summarize_samples(self._good())
        candidate = summarize_samples(
            [
                _sample(ttft=5, useful=25, total=50, declared=True, verified=True),
                _sample(ttft=5, useful=25, total=50, declared=True, verified=True),
            ]
        )
        result = evaluate_promotion(candidate, baseline)
        assert result.passed is True
        assert result.failures == ()

    def test_opposite_direction_correctness_improvement_promotes(self) -> None:
        """Guard: a candidate that is more correct must not be blocked."""
        baseline = summarize_samples(
            [
                _sample(ttft=10, useful=50, total=100, declared=False, verified=False),
                _sample(ttft=10, useful=50, total=100, declared=True, verified=False),
            ]
        )
        candidate = summarize_samples(self._good())
        result = evaluate_promotion(candidate, baseline)
        assert result.passed is True
        assert result.failures == ()

    def test_latency_regression_is_still_reported_alongside_correctness(self) -> None:
        """Guard: the new checks compose with the existing ones, not replace them."""
        baseline = summarize_samples(self._good())
        candidate = summarize_samples(
            [
                _sample(ttft=400, useful=400, total=900, declared=True, verified=False),
                _sample(ttft=400, useful=400, total=900, declared=True, verified=False),
            ]
        )
        result = evaluate_promotion(candidate, baseline)
        assert result.passed is False
        assert "p95_regression" in result.failures
        assert "false_success_regression" in result.failures
