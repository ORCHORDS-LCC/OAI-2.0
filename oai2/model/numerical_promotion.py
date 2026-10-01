"""Canonical numerical promotion gate for backend/export/kernel candidates.

The gate consumes the WI-NUM-002 tolerance matrix before a candidate path can
be considered eligible. Speed is evidence only: it never overrides numerical
or capability failures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .numerical_compare import (
    NumericalArtifactIdentity,
    NumericalComparison,
    NumericalSelection,
    NumericalToleranceProfile,
    compare_numerical_paths,
    select_numerical_path,
)


class NumericalCandidateKind(StrEnum):
    BACKEND = "backend"
    EXPORT = "export"
    KERNEL = "kernel"


@dataclass(slots=True, frozen=True)
class NumericalPromotionEvidence:
    candidate_kind: NumericalCandidateKind
    comparison: NumericalComparison
    selection: NumericalSelection
    eligible: bool
    speedup_ratio: float | None = None

    @property
    def tolerance_profile_id(self) -> str:
        return self.comparison.profile_id

    @property
    def identity(self) -> NumericalArtifactIdentity:
        return self.comparison.identity


def evaluate_numerical_candidate(
    reference: list[float] | tuple[float, ...],
    optimized: list[float] | tuple[float, ...],
    *,
    candidate_kind: NumericalCandidateKind,
    profile: NumericalToleranceProfile,
    identity: NumericalArtifactIdentity,
    optimized_path: str,
    reference_path: str,
    reference_capability_score: float | None = None,
    optimized_capability_score: float | None = None,
    speedup_ratio: float | None = None,
) -> NumericalPromotionEvidence:
    """Evaluate one optimized candidate through the canonical numerical gate."""
    if not isinstance(candidate_kind, NumericalCandidateKind):
        raise ValueError("candidate_kind must be a NumericalCandidateKind")

    normalized_speedup: float | None = None
    if speedup_ratio is not None:
        if (
            isinstance(speedup_ratio, bool)
            or not isinstance(speedup_ratio, (int, float))
            or not math.isfinite(float(speedup_ratio))
            or float(speedup_ratio) <= 0.0
        ):
            raise ValueError("speedup_ratio must be finite and > 0")
        normalized_speedup = float(speedup_ratio)

    comparison = compare_numerical_paths(
        reference,
        optimized,
        profile=profile,
        identity=identity,
        reference_capability_score=reference_capability_score,
        optimized_capability_score=optimized_capability_score,
    )
    selection = select_numerical_path(
        comparison,
        optimized_path=optimized_path,
        reference_path=reference_path,
    )
    return NumericalPromotionEvidence(
        candidate_kind=candidate_kind,
        comparison=comparison,
        selection=selection,
        eligible=comparison.passed and not selection.used_fallback,
        speedup_ratio=normalized_speedup,
    )


__all__ = [
    "NumericalCandidateKind",
    "NumericalPromotionEvidence",
    "evaluate_numerical_candidate",
]
