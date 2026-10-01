from __future__ import annotations

import pytest

from oai2.evals.qos import (
    BudgetKind,
    WorkloadBudget,
    WorkloadClass,
    WorkloadSample,
    evaluate_budget,
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
