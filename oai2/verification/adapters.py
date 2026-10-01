"""Evidence-path adapters for repository, tool/runtime, and web claims.

These helpers are intentionally small. They translate evidence produced by real
repository/tool/web paths into the versioned WI-TRUTH-001 policy without
inventing new retrieval or execution behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import Evidence, EvidenceClass
from .policy import (
    ClaimClass,
    ClaimEvidencePolicy,
    EvidenceAssessment,
    EvidenceBinding,
)


@dataclass(slots=True, frozen=True)
class RepositoryEvidencePath:
    policy: ClaimEvidencePolicy

    def assess(
        self,
        evidence: Evidence,
        *,
        state_version: str,
        current_state_version: str,
        supports: bool = True,
        content_hash: str | None = None,
    ) -> EvidenceAssessment:
        _require_class(evidence, {
            EvidenceClass.REPO_SOURCE,
            EvidenceClass.CHANGE_HISTORY,
            EvidenceClass.DETERMINISTIC,
        }, "repository")
        return self.policy.assess(
            ClaimClass.REPOSITORY_STATE,
            [
                EvidenceBinding(
                    evidence=evidence,
                    supports=supports,
                    state_version=state_version,
                    content_hash=content_hash,
                )
            ],
            current_state_version=current_state_version,
        )


@dataclass(slots=True, frozen=True)
class ToolRuntimeEvidencePath:
    policy: ClaimEvidencePolicy

    def assess(
        self,
        evidence: Evidence,
        *,
        state_version: str,
        current_state_version: str,
        supports: bool = True,
    ) -> EvidenceAssessment:
        _require_class(
            evidence,
            {EvidenceClass.RUNTIME_OBS, EvidenceClass.DETERMINISTIC},
            "tool/runtime",
        )
        return self.policy.assess(
            ClaimClass.TOOL_RUNTIME_OBSERVATION,
            [
                EvidenceBinding(
                    evidence=evidence,
                    supports=supports,
                    state_version=state_version,
                )
            ],
            current_state_version=current_state_version,
        )


@dataclass(slots=True, frozen=True)
class WebEvidencePath:
    policy: ClaimEvidencePolicy

    def assess_current(
        self,
        evidence: Evidence,
        *,
        now: float,
        supports: bool = True,
    ) -> EvidenceAssessment:
        _require_class(evidence, {EvidenceClass.EXTERNAL}, "web")
        return self.policy.assess(
            ClaimClass.EXTERNAL_CURRENT_FACT,
            [EvidenceBinding(evidence=evidence, supports=supports)],
            now=now,
        )

    def assess_stable(
        self,
        evidence: Evidence,
        *,
        supports: bool = True,
    ) -> EvidenceAssessment:
        _require_class(evidence, {EvidenceClass.EXTERNAL}, "web")
        return self.policy.assess(
            ClaimClass.EXTERNAL_STABLE_FACT,
            [EvidenceBinding(evidence=evidence, supports=supports)],
        )


def _require_class(
    evidence: Evidence,
    accepted: set[EvidenceClass],
    path_name: str,
) -> None:
    if evidence.cls not in accepted:
        accepted_names = ", ".join(sorted(item.value for item in accepted))
        raise ValueError(
            f"{path_name} evidence path requires one of: {accepted_names}"
        )


__all__ = [
    "RepositoryEvidencePath",
    "ToolRuntimeEvidencePath",
    "WebEvidencePath",
]
