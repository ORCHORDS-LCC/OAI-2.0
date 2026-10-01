"""Versioned claim taxonomy, evidence policy, freshness and invalidation rules.

WI-TRUTH-001 requires evidence requirements to be explicit rather than inferred
from nearby claims. This module keeps claim-class policy separate from the
EvidenceGraph container and makes state/freshness invalidation deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .evidence import Evidence, EvidenceClass

EVIDENCE_POLICY_VERSION = "1"


class ClaimClass(StrEnum):
    REPOSITORY_STATE = "repository_state"
    TOOL_RUNTIME_OBSERVATION = "tool_runtime_observation"
    EXTERNAL_CURRENT_FACT = "external_current_fact"
    EXTERNAL_STABLE_FACT = "external_stable_fact"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"
    PLAN = "plan"
    TARGET = "target"
    PREFERENCE = "preference"
    HYPOTHETICAL = "hypothetical"


class PolicyDecision(StrEnum):
    NOT_REQUIRED = "not_required"
    NEEDS_EVIDENCE = "needs_evidence"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    STALE = "stale"
    CONFLICTING = "conflicting"


@dataclass(slots=True, frozen=True)
class EvidenceRequirement:
    accepted_classes: tuple[EvidenceClass, ...]
    min_support: int = 1
    state_scoped: bool = False
    freshness_required: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.min_support, bool) or not isinstance(self.min_support, int):
            raise ValueError("min_support must be a non-negative integer")
        if self.min_support < 0:
            raise ValueError("min_support must be a non-negative integer")

    @property
    def evidence_required(self) -> bool:
        return self.min_support > 0


@dataclass(slots=True, frozen=True)
class EvidenceBinding:
    evidence: Evidence
    supports: bool = True
    state_version: str | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.supports, bool):
            raise ValueError("supports must be a boolean")
        for name in ("state_version", "content_hash"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty normalized string")


@dataclass(slots=True, frozen=True)
class EvidenceRejection:
    evidence_id: str
    reason: str


@dataclass(slots=True, frozen=True)
class EvidenceAssessment:
    policy_version: str
    claim_class: ClaimClass
    decision: PolicyDecision
    accepted_support_ids: tuple[str, ...] = ()
    accepted_refute_ids: tuple[str, ...] = ()
    rejected: tuple[EvidenceRejection, ...] = ()


_RULES: dict[ClaimClass, EvidenceRequirement] = {
    ClaimClass.REPOSITORY_STATE: EvidenceRequirement(
        accepted_classes=(
            EvidenceClass.REPO_SOURCE,
            EvidenceClass.CHANGE_HISTORY,
            EvidenceClass.DETERMINISTIC,
        ),
        state_scoped=True,
    ),
    ClaimClass.TOOL_RUNTIME_OBSERVATION: EvidenceRequirement(
        accepted_classes=(
            EvidenceClass.RUNTIME_OBS,
            EvidenceClass.DETERMINISTIC,
        ),
        state_scoped=True,
    ),
    ClaimClass.EXTERNAL_CURRENT_FACT: EvidenceRequirement(
        accepted_classes=(EvidenceClass.EXTERNAL,),
        freshness_required=True,
    ),
    ClaimClass.EXTERNAL_STABLE_FACT: EvidenceRequirement(
        accepted_classes=(EvidenceClass.EXTERNAL,),
    ),
    ClaimClass.INFERENCE: EvidenceRequirement(
        accepted_classes=(
            EvidenceClass.HYPOTHESIS,
            EvidenceClass.DETERMINISTIC,
        ),
    ),
    ClaimClass.ASSUMPTION: EvidenceRequirement(accepted_classes=(), min_support=0),
    ClaimClass.PLAN: EvidenceRequirement(accepted_classes=(), min_support=0),
    ClaimClass.TARGET: EvidenceRequirement(accepted_classes=(), min_support=0),
    ClaimClass.PREFERENCE: EvidenceRequirement(accepted_classes=(), min_support=0),
    ClaimClass.HYPOTHETICAL: EvidenceRequirement(accepted_classes=(), min_support=0),
}


class ClaimEvidencePolicy:
    """Evaluate evidence against the versioned WI-TRUTH-001 policy."""

    def __init__(self, *, current_external_max_age_seconds: float) -> None:
        self.current_external_max_age_seconds = _positive_finite(
            current_external_max_age_seconds,
            "current_external_max_age_seconds",
        )

    @property
    def version(self) -> str:
        return EVIDENCE_POLICY_VERSION

    def rule_for(self, claim_class: ClaimClass) -> EvidenceRequirement:
        return _RULES[claim_class]

    def assess(
        self,
        claim_class: ClaimClass,
        bindings: tuple[EvidenceBinding, ...] | list[EvidenceBinding],
        *,
        now: float | None = None,
        current_state_version: str | None = None,
    ) -> EvidenceAssessment:
        rule = self.rule_for(claim_class)
        if not rule.evidence_required:
            return EvidenceAssessment(
                policy_version=self.version,
                claim_class=claim_class,
                decision=PolicyDecision.NOT_REQUIRED,
            )

        timestamp: float | None = None
        if rule.freshness_required:
            if now is None:
                raise ValueError("now is required for freshness-scoped claims")
            timestamp = _non_negative_finite(now, "now")

        state_version: str | None = None
        if rule.state_scoped:
            if (
                not isinstance(current_state_version, str)
                or not current_state_version
                or current_state_version != current_state_version.strip()
            ):
                raise ValueError(
                    "current_state_version is required for state-scoped claims"
                )
            state_version = current_state_version

        supporting: list[str] = []
        refuting: list[str] = []
        rejected: list[EvidenceRejection] = []
        freshness_rejected = False

        for binding in bindings:
            evidence = binding.evidence
            evidence_id = str(evidence.id)

            if evidence.cls not in rule.accepted_classes:
                rejected.append(
                    EvidenceRejection(
                        evidence_id=evidence_id,
                        reason="evidence_class_not_accepted",
                    )
                )
                continue

            if rule.state_scoped:
                if binding.state_version is None:
                    rejected.append(
                        EvidenceRejection(
                            evidence_id=evidence_id,
                            reason="state_version_missing",
                        )
                    )
                    continue
                if binding.state_version != state_version:
                    rejected.append(
                        EvidenceRejection(
                            evidence_id=evidence_id,
                            reason="state_version_mismatch",
                        )
                    )
                    continue

            if rule.freshness_required:
                assert timestamp is not None
                observed_at = float(evidence.observed_at)
                if not math.isfinite(observed_at) or observed_at <= 0.0:
                    freshness_rejected = True
                    rejected.append(
                        EvidenceRejection(
                            evidence_id=evidence_id,
                            reason="observed_at_missing",
                        )
                    )
                    continue
                if observed_at > timestamp:
                    freshness_rejected = True
                    rejected.append(
                        EvidenceRejection(
                            evidence_id=evidence_id,
                            reason="observed_at_in_future",
                        )
                    )
                    continue
                if timestamp - observed_at > self.current_external_max_age_seconds:
                    freshness_rejected = True
                    rejected.append(
                        EvidenceRejection(
                            evidence_id=evidence_id,
                            reason="evidence_stale",
                        )
                    )
                    continue

            if binding.supports:
                supporting.append(evidence_id)
            else:
                refuting.append(evidence_id)

        if supporting and refuting:
            decision = PolicyDecision.CONFLICTING
        elif len(supporting) >= rule.min_support:
            decision = PolicyDecision.SUPPORTED
        elif refuting:
            decision = PolicyDecision.REFUTED
        elif freshness_rejected:
            decision = PolicyDecision.STALE
        else:
            decision = PolicyDecision.NEEDS_EVIDENCE

        return EvidenceAssessment(
            policy_version=self.version,
            claim_class=claim_class,
            decision=decision,
            accepted_support_ids=tuple(supporting),
            accepted_refute_ids=tuple(refuting),
            rejected=tuple(rejected),
        )


def _non_negative_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _positive_finite(value: object, name: str) -> float:
    result = _non_negative_finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = [
    "EVIDENCE_POLICY_VERSION",
    "ClaimClass",
    "PolicyDecision",
    "EvidenceRequirement",
    "EvidenceBinding",
    "EvidenceRejection",
    "EvidenceAssessment",
    "ClaimEvidencePolicy",
]
