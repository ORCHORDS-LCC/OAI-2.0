"""Deterministic end-to-end service-quality metrics.

The helpers in this module deliberately keep model throughput separate from
user-visible usefulness. They describe workload identity, collect per-run
latency/useful-work evidence, summarize tail distributions, and evaluate a
versioned caller-supplied budget without inventing hardware targets.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import StrEnum


class WorkloadClass(StrEnum):
    """Canonical QoS workload classes owned by WP-76."""

    FURIOUS = "FURIOUS"
    NORMAL = "NORMAL"
    DEEP = "DEEP"
    SWARM = "SWARM"
    VISION = "VISION"
    RETRIEVAL_HEAVY = "RETRIEVAL_HEAVY"


class BudgetKind(StrEnum):
    """Keep aspirational research targets distinct from measured service gates."""

    RESEARCH_TARGET = "research_target"
    SERVICE_BUDGET = "service_budget"


@dataclass(slots=True, frozen=True)
class WorkloadBudget:
    """Versioned tail/useful-work limits for one hardware/config identity."""

    version: str
    workload: WorkloadClass
    kind: BudgetKind
    target_hardware: str
    config_id: str
    first_useful_action_p95_ms: float
    end_to_end_p95_ms: float
    end_to_end_p99_ms: float
    max_false_success_rate: float
    min_verified_success_rate: float

    def __post_init__(self) -> None:
        for name in ("version", "target_hardware", "config_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "first_useful_action_p95_ms",
            "end_to_end_p95_ms",
            "end_to_end_p99_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.end_to_end_p99_ms < self.end_to_end_p95_ms:
            raise ValueError("end_to_end_p99_ms must be >= end_to_end_p95_ms")
        for name in ("max_false_success_rate", "min_verified_success_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(slots=True, frozen=True)
class WorkloadSample:
    """One replayed workload with end-to-end and useful-work evidence."""

    workload: WorkloadClass
    target_hardware: str
    config_id: str
    ttft_ms: float
    first_useful_action_ms: float
    end_to_end_ms: float
    declared_success: bool
    verified_success: bool
    verified_actions: int = 0
    generated_tokens: int = 0
    decode_tokens_per_second: float = 0.0
    tool_ms: float = 0.0
    retrieval_ms: float = 0.0
    vision_ms: float = 0.0
    build_test_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("target_hardware", "config_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "ttft_ms",
            "first_useful_action_ms",
            "end_to_end_ms",
            "decode_tokens_per_second",
            "tool_ms",
            "retrieval_ms",
            "vision_ms",
            "build_test_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        for name in ("verified_actions", "generated_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.ttft_ms > self.end_to_end_ms:
            raise ValueError("ttft_ms cannot exceed end_to_end_ms")
        if self.first_useful_action_ms > self.end_to_end_ms:
            raise ValueError("first_useful_action_ms cannot exceed end_to_end_ms")

    @property
    def false_success(self) -> bool:
        return self.declared_success and not self.verified_success


@dataclass(slots=True, frozen=True)
class Distribution:
    count: int
    mean: float
    variance: float
    p50: float
    p95: float
    p99: float


@dataclass(slots=True, frozen=True)
class WorkloadReport:
    workload: WorkloadClass
    target_hardware: str
    config_id: str
    sample_count: int
    ttft_ms: Distribution
    first_useful_action_ms: Distribution
    end_to_end_ms: Distribution
    decode_tokens_per_second: Distribution
    tool_ms: Distribution
    retrieval_ms: Distribution
    vision_ms: Distribution
    build_test_ms: Distribution
    verified_success_rate: float
    false_success_rate: float
    verified_actions_per_second: float


@dataclass(slots=True, frozen=True)
class BudgetEvaluation:
    budget_version: str
    budget_kind: BudgetKind
    passed: bool
    failures: tuple[str, ...]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> Distribution:
    if not values:
        raise ValueError("at least one value is required")
    return Distribution(
        count=len(values),
        mean=statistics.fmean(values),
        variance=statistics.pvariance(values) if len(values) > 1 else 0.0,
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        p99=_percentile(values, 0.99),
    )


def summarize_samples(samples: list[WorkloadSample]) -> WorkloadReport:
    """Aggregate one homogeneous workload/config sample set."""
    if not samples:
        raise ValueError("at least one workload sample is required")

    first = samples[0]
    identity = (first.workload, first.target_hardware, first.config_id)
    for sample in samples[1:]:
        if (sample.workload, sample.target_hardware, sample.config_id) != identity:
            raise ValueError("all samples must share workload, hardware, and config identity")

    total_seconds = sum(sample.end_to_end_ms for sample in samples) / 1000.0
    verified_actions = sum(sample.verified_actions for sample in samples)
    n = len(samples)
    return WorkloadReport(
        workload=first.workload,
        target_hardware=first.target_hardware,
        config_id=first.config_id,
        sample_count=n,
        ttft_ms=_distribution([sample.ttft_ms for sample in samples]),
        first_useful_action_ms=_distribution(
            [sample.first_useful_action_ms for sample in samples]
        ),
        end_to_end_ms=_distribution([sample.end_to_end_ms for sample in samples]),
        decode_tokens_per_second=_distribution(
            [sample.decode_tokens_per_second for sample in samples]
        ),
        tool_ms=_distribution([sample.tool_ms for sample in samples]),
        retrieval_ms=_distribution([sample.retrieval_ms for sample in samples]),
        vision_ms=_distribution([sample.vision_ms for sample in samples]),
        build_test_ms=_distribution([sample.build_test_ms for sample in samples]),
        verified_success_rate=sum(sample.verified_success for sample in samples) / n,
        false_success_rate=sum(sample.false_success for sample in samples) / n,
        verified_actions_per_second=(
            verified_actions / total_seconds if total_seconds > 0.0 else 0.0
        ),
    )


def evaluate_budget(report: WorkloadReport, budget: WorkloadBudget) -> BudgetEvaluation:
    """Evaluate a report against the exact versioned budget identity."""
    if (
        report.workload != budget.workload
        or report.target_hardware != budget.target_hardware
        or report.config_id != budget.config_id
    ):
        raise ValueError("report and budget identity do not match")

    failures: list[str] = []
    if report.first_useful_action_ms.p95 > budget.first_useful_action_p95_ms:
        failures.append("first_useful_action_p95")
    if report.end_to_end_ms.p95 > budget.end_to_end_p95_ms:
        failures.append("end_to_end_p95")
    if report.end_to_end_ms.p99 > budget.end_to_end_p99_ms:
        failures.append("end_to_end_p99")
    if report.false_success_rate > budget.max_false_success_rate:
        failures.append("false_success_rate")
    if report.verified_success_rate < budget.min_verified_success_rate:
        failures.append("verified_success_rate")
    return BudgetEvaluation(
        budget_version=budget.version,
        budget_kind=budget.kind,
        passed=not failures,
        failures=tuple(failures),
    )


def useful_work_rank_key(report: WorkloadReport) -> tuple[float, float, float, float]:
    """Lower is better; usefulness/tail latency outrank raw token throughput."""
    return (
        report.false_success_rate,
        -report.verified_success_rate,
        report.first_useful_action_ms.p95,
        report.end_to_end_ms.p95,
    )


__all__ = [
    "BudgetEvaluation",
    "BudgetKind",
    "Distribution",
    "WorkloadBudget",
    "WorkloadClass",
    "WorkloadReport",
    "WorkloadSample",
    "evaluate_budget",
    "summarize_samples",
    "useful_work_rank_key",
]
