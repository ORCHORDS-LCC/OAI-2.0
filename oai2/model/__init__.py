"""OAI-2.0 model architecture primitives."""

from .numerical_compare import (
    NumericalArtifactIdentity,
    NumericalComparison,
    NumericalFallback,
    NumericalSelection,
    NumericalToleranceProfile,
    compare_numerical_paths,
    select_numerical_path,
)
from .numerical_promotion import (
    NumericalCandidateKind,
    NumericalPromotionEvidence,
    evaluate_numerical_candidate,
)
from .numerics import (
    NUMERICAL_POLICY_VERSION,
    REFERENCE_PRECISION_RULES,
    NumericalCheckResult,
    NumericalFailure,
    NumericalOperation,
    PrecisionRule,
    check_numerics,
    extreme_value_fixtures,
    precision_rule,
)
from .tolerance_matrix import (
    TOLERANCE_MATRIX,
    TOLERANCE_POLICY_VERSION,
    DtypeClass,
    TolerancePolicy,
    canonical_policy,
    dtype_class,
    policy_violations,
)

__all__ = [
    "NumericalArtifactIdentity",
    "NumericalComparison",
    "NumericalFallback",
    "NumericalSelection",
    "NumericalToleranceProfile",
    "compare_numerical_paths",
    "select_numerical_path",
    "NumericalCandidateKind",
    "NumericalPromotionEvidence",
    "evaluate_numerical_candidate",
    "NUMERICAL_POLICY_VERSION",
    "NumericalCheckResult",
    "NumericalFailure",
    "NumericalOperation",
    "PrecisionRule",
    "REFERENCE_PRECISION_RULES",
    "check_numerics",
    "extreme_value_fixtures",
    "precision_rule",
    "TOLERANCE_MATRIX",
    "TOLERANCE_POLICY_VERSION",
    "DtypeClass",
    "TolerancePolicy",
    "canonical_policy",
    "dtype_class",
    "policy_violations",
]
